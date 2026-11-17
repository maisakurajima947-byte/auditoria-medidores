"""
Auditoría Automatizada de Medidores Electrónicos de Energía Eléctrica
Visión Computacional (OpenCV) + OCR (EasyOCR) + SQLite  |  ODS 9
Ejecutar:  streamlit run app.py
"""
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ───────────────────────── 1. CONFIGURACIÓN ─────────────────────────
DB_PATH = Path("data/auditorias.db")
DB_PATH.parent.mkdir(exist_ok=True)

OBIS_RE = re.compile(r"1[.,]\s?8[.,]\s?0")   # código OBIS 1.8.0
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Suministros de prueba: [lectura hace 60 días, lectura hace 30 días] en kWh.
# ► Ajusta estos valores para que la "última lectura" quede cerca (y por debajo)
#   del número real de tu foto de prueba.
SEED = {
    "SUM-001001": [12450.3, 12980.4],
    "SUM-002045": [8320.7, 8655.2],
    "SUM-003310": [20111.5, 20890.0],
}

MENSAJES = {
    "APROBADO": ("success", "✅ APROBADO: lectura coherente, guardada en la base de datos."),
    "ERROR_ANOMALIA": ("error", "❌ ERROR / ANOMALÍA: la lectura es menor a la anterior (posible fallo de OCR o manipulación)."),
    "ALERTA_REVISION": ("warning", "⚠️ ALERTA: REVISIÓN MANUAL REQUERIDA (incremento superior al umbral)."),
    "BAJA_CONFIANZA": ("warning", "🔁 Confianza del OCR insuficiente: tome una nueva fotografía o solicite revisión humana."),
    "SIN_LECTURA": ("warning", "🔁 No se detectó una lectura numérica válida: tome una nueva fotografía."),
}


# ───────────────────────── 2. BASE DE DATOS (SQLite) ─────────────────────────
def init_db():
    """Crea la tabla y precarga 3 suministros de prueba si está vacía."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS auditorias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suministro TEXT NOT NULL, fecha TEXT NOT NULL,
            lectura_anterior REAL, lectura_extraida REAL,
            confianza REAL, consumo REAL,
            estado TEXT NOT NULL, origen TEXT DEFAULT 'OCR')""")
        if con.execute("SELECT COUNT(*) FROM auditorias").fetchone()[0] == 0:
            now = datetime.now()
            for sid, lecturas in SEED.items():
                prev = None
                for i, val in enumerate(lecturas):
                    fecha = (now - timedelta(days=60 - 30 * i)).strftime("%Y-%m-%d %H:%M:%S")
                    consumo = None if prev is None else round(val - prev, 2)
                    con.execute(
                        "INSERT INTO auditorias (suministro,fecha,lectura_anterior,lectura_extraida,"
                        "confianza,consumo,estado,origen) VALUES (?,?,?,?,?,?,?,?)",
                        (sid, fecha, prev, val, 96.0, consumo, "APROBADO", "SEED"))
                    prev = val


def ultima_lectura(suministro):
    """Última lectura APROBADA del suministro (las anómalas no sirven de referencia)."""
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT lectura_extraida FROM auditorias WHERE suministro=? "
                          "AND estado='APROBADO' ORDER BY id DESC LIMIT 1", (suministro,)).fetchone()
    return row[0] if row else None


def guardar(suministro, anterior, actual, conf, consumo, estado, origen):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO auditorias (suministro,fecha,lectura_anterior,lectura_extraida,"
                    "confianza,consumo,estado,origen) VALUES (?,?,?,?,?,?,?,?)",
                    (suministro, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     anterior, actual, conf, consumo, estado, origen))


def cargar_df():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql("SELECT * FROM auditorias ORDER BY id DESC", con)


# ───────────────────────── 3. VISIÓN COMPUTACIONAL (OpenCV) ─────────────────────────
def preprocesar(bgr, crop):
    """Recorte ROI → grises → CLAHE → reducción de ruido → umbral adaptativo."""
    h, w = bgr.shape[:2]
    if w > 1280:                                    # normaliza resolución
        bgr = cv2.resize(bgr, (1280, int(h * 1280 / w)))
        h, w = bgr.shape[:2]
    izq, der, sup, inf = crop                       # porcentajes a recortar
    roi = bgr[int(h * sup / 100): h - int(h * inf / 100), int(w * izq / 100): w - int(w * der / 100)]
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gris)  # mitiga reflejos/sombras
    suave = cv2.bilateralFilter(clahe, 9, 75, 75)                            # quita ruido, conserva bordes
    binaria = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # une segmentos
    return roi, gris, clahe, suave, binaria


# ───────────────────────── 4. OCR (EasyOCR) ─────────────────────────
@st.cache_resource(show_spinner="Cargando modelo EasyOCR (solo la primera vez)…")
def get_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def interpretar(dets):
    """Extrae el valor kWh de las detecciones y verifica si aparece el código OBIS 1.8.0."""
    obis = any(OBIS_RE.search(t) for _, t, _ in dets)
    cands = []
    for _, texto, conf in dets:
        for m in NUM_RE.findall(texto):
            digitos = len(re.sub(r"\D", "", m))
            if digitos >= 4:                         # descarta "1.8.0" y ruido corto
                cands.append((digitos, conf, float(m.replace(",", "."))))
    if not cands:
        return None, 0.0, obis
    _, conf, valor = max(cands, key=lambda c: (c[0], c[1]))
    return valor, conf * 100, obis


def ejecutar_ocr(imagenes):
    """Prueba el OCR sobre varias versiones de la imagen y conserva la de mayor confianza."""
    reader, mejor = get_reader(), (None, 0.0, False)
    for img in imagenes:
        dets = reader.readtext(img, allowlist="0123456789.,", paragraph=False)
        valor, conf, obis = interpretar(dets)
        if valor is not None and conf >= mejor[1]:
            mejor = (valor, conf, obis)
    return mejor


# ───────────────────────── 5. AUDITORÍA LÓGICA ─────────────────────────
def auditar(actual, anterior, conf, umbral_pico, umbral_conf):
    """Devuelve (estado, consumo). Primero se valida la confianza: sin ella, las demás reglas no tienen sentido."""
    if actual is None:
        return "SIN_LECTURA", None
    if conf < umbral_conf:                                   # Regla 3
        return "BAJA_CONFIANZA", None
    if anterior is None:                                     # primera lectura = línea base
        return "APROBADO", None
    consumo = round(actual - anterior, 2)
    if consumo < 0:                                          # Regla 1
        return "ERROR_ANOMALIA", consumo
    if consumo > umbral_pico:                                # Regla 2
        return "ALERTA_REVISION", consumo
    return "APROBADO", consumo                               # Regla 4


# ───────────────────────── 6. INTERFAZ ─────────────────────────
st.set_page_config(page_title="Auditoría de Medidores", page_icon="⚡", layout="wide")
init_db()

st.title("⚡ Auditoría Automatizada de Medidores Electrónicos")
st.caption("Visión computacional + OCR · Universidad de Lima · ODS 9: Industria, innovación e infraestructura")

with st.sidebar:
    st.header("⚙️ Parámetros de auditoría")
    umbral_pico = st.number_input("Umbral de pico inusual (kWh)", 100, 10000, 1000, 100)
    umbral_conf = st.slider("Confianza mínima del OCR (%)", 10, 100, 70)
    st.markdown("**ROI (recorte, % por lado)**")
    crop = (st.slider("Izquierda", 0, 45, 0), st.slider("Derecha", 0, 45, 0),
            st.slider("Arriba", 0, 45, 0), st.slider("Abajo", 0, 45, 0))

tab1, tab2, tab3 = st.tabs(["📷 Nueva Auditoría (Escáner)", "🗄️ Historial y Base de Datos (SQLite)",
                            "📊 Dashboard de Métricas"])

# ── TAB 1: escáner ──
with tab1:
    izq, der = st.columns(2)
    with izq:
        suministro = st.text_input("N.º de suministro", "SUM-001001").strip().upper()
        previa = ultima_lectura(suministro)
        st.info(f"Última lectura aprobada: **{previa:,.1f} kWh**" if previa is not None
                else "Suministro nuevo: la primera lectura será la línea base.")
        fuente = st.radio("Fuente de la imagen", ["Subir archivo", "Cámara"], horizontal=True)
        archivo = (st.file_uploader("Fotografía del medidor", type=["jpg", "jpeg", "png"])
                   if fuente == "Subir archivo" else st.camera_input("Capturar medidor"))
        manual = st.number_input("Corrección humana (kWh, 0 = usar OCR)", 0.0, step=0.1, format="%.1f")
    with der:
        st.markdown("**Flujo:** imagen → ROI → grises → CLAHE → ruido → umbral → OCR → auditoría → SQLite")

    if st.button("🔍 Ejecutar auditoría", type="primary", disabled=archivo is None):
        bgr = cv2.imdecode(np.frombuffer(archivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        roi, gris, clahe, suave, binaria = preprocesar(bgr, crop)

        with st.spinner("Ejecutando OCR…"):
            valor, conf, obis = ejecutar_ocr([suave, binaria])
        origen = "OCR"
        if manual > 0:                                      # revisión humana
            valor, conf, origen = manual, 100.0, "MANUAL"

        estado, consumo = auditar(valor, previa, conf, umbral_pico, umbral_conf)
        guardar(suministro, previa, valor, round(conf, 1), consumo, estado, origen)

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="Imagen original (ROI)", use_container_width=True)
        c2.image(binaria, caption="Preprocesada (CLAHE + umbral adaptativo)", use_container_width=True)
        with st.expander("Ver etapas intermedias"):
            e = st.columns(3)
            e[0].image(gris, caption="1. Escala de grises", use_container_width=True)
            e[1].image(clahe, caption="2. CLAHE", use_container_width=True)
            e[2].image(suave, caption="3. Reducción de ruido", use_container_width=True)

        m = st.columns(4)
        m[0].metric("Lectura extraída", f"{valor:,.1f} kWh" if valor is not None else "—")
        m[1].metric("Confianza", f"{conf:.1f}%")
        m[2].metric("Consumo", f"{consumo:,.1f} kWh" if consumo is not None else "—")
        m[3].metric("Código OBIS 1.8.0", "Detectado" if obis else "No detectado")
        nivel, texto = MENSAJES[estado]
        getattr(st, nivel)(texto)

# ── TAB 2: historial ──
with tab2:
    df = cargar_df()
    filtro = st.multiselect("Filtrar por suministro", sorted(df["suministro"].unique()))
    vista = df[df["suministro"].isin(filtro)] if filtro else df
    st.dataframe(vista, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Exportar CSV", vista.to_csv(index=False).encode("utf-8"), "auditorias.csv", "text/csv")

# ── TAB 3: dashboard ──
with tab3:
    df = cargar_df()
    total = len(df)
    k = st.columns(4)
    k[0].metric("Auditorías", total)
    k[1].metric("% aprobadas", f"{(df['estado'].eq('APROBADO').mean() * 100):.0f}%" if total else "—")
    k[2].metric("Confianza media", f"{df['confianza'].mean():.1f}%" if total else "—")
    k[3].metric("Consumo medio", f"{df['consumo'].mean():,.1f} kWh" if df["consumo"].notna().any() else "—")
    g1, g2 = st.columns(2)
    g1.subheader("Auditorías por estado")
    g1.bar_chart(df["estado"].value_counts())
    g2.subheader("Evolución de lecturas (kWh)")
    evol = (df.dropna(subset=["lectura_extraida"]).pivot_table(
        index="fecha", columns="suministro", values="lectura_extraida").sort_index().ffill())
    g2.line_chart(evol)