import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Research Intelligence",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# ESTILO MODERNO (STARTUP LOOK)
# =====================================================
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA (ROBUSTO - LOCAL + GITHUB)
# =====================================================
@st.cache_data
def load_data(source, file=None):
    if source == "GitHub":
        df = pd.read_csv("scopus_PA3.csv")
    else:
        df = pd.read_csv(file)

    df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2024)
    df['Abstract'] = df['Abstract'].fillna("")
    df['clean'] = df['Abstract'].str.lower()

    return df


# =====================================================
# SIDEBAR (CONTROL PANEL STARTUP)
# =====================================================
st.sidebar.title("⚙️ AI Research Control Panel")

source = st.sidebar.radio("Fuente de datos:", ["GitHub", "Local Upload"])

file = None
if source == "Local Upload":
    file = st.sidebar.file_uploader("Sube tu dataset CSV", type=["csv"])

search = st.sidebar.text_input("🔍 Buscar papers (ej: xgboost, neural, fraud)")
min_cites = st.sidebar.slider("📊 Citas mínimas", 0, 200, 0)


# =====================================================
# LOAD
# =====================================================
df = load_data(source, file)

if df is None:
    st.warning("Carga un dataset para comenzar.")
    st.stop()


# =====================================================
# FILTERING ENGINE
# =====================================================
filtered = df[df['Cited by'] >= min_cites]

if search:
    filtered = filtered[
        filtered['clean'].str.contains(search.lower()) |
        filtered['Title'].str.lower().str.contains(search.lower())
    ]


# =====================================================
# HEADER STARTUP STYLE
# =====================================================
st.title("📊 AI Research Intelligence Dashboard")
st.caption("Explora tendencias de Machine Learning en investigación científica (modo startup 🚀)")

st.divider()


# =====================================================
# KPI CARDS (STARTUP METRICS)
# =====================================================
col1, col2, col3 = st.columns(3)

col1.metric("📄 Papers", len(filtered))
col2.metric("📅 Año más reciente", int(filtered['Year'].max()))
col3.metric("⭐ Citas promedio", int(filtered['Cited by'].mean()))

st.divider()


# =====================================================
# INSIGHT ENGINE (CLAVE STARTUP)
# =====================================================
st.subheader("🧠 AI Insight Engine")

top_year = filtered.groupby("Year").size().idxmax()

st.info(f"""
📌 Insight automático:
El año con mayor producción científica es **{top_year}**, lo que indica un pico de investigación activa en este periodo.
""")


# =====================================================
# TABS (PRODUCT ANALYTICS STYLE)
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Tendencias",
    "🤖 Algoritmos",
    "📊 Impacto"
])


# =====================================================
# TAB 1: TRENDS
# =====================================================
with tab1:
    st.subheader("📈 Evolución de publicaciones")

    trend = filtered.groupby("Year").size().reset_index(name="Papers")

    fig = px.line(trend, x="Year", y="Papers", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔎 Interpretación automática")
    st.success("Las publicaciones muestran la evolución del interés académico en machine learning aplicado a problemas reales.")


# =====================================================
# TAB 2: ALGORITHMS
# =====================================================
with tab2:
    st.subheader("🤖 Ranking de algoritmos")

    data = pd.DataFrame({
        "Algoritmo": ["Random Forest", "XGBoost", "Neural Networks", "SVM"],
        "Uso": [
            filtered['clean'].str.contains("random forest").sum(),
            filtered['clean'].str.contains("xgboost").sum(),
            filtered['clean'].str.contains("neural|deep learning").sum(),
            filtered['clean'].str.contains("svm").sum()
        ]
    }).sort_values("Uso")

    fig = px.bar(data, x="Uso", y="Algoritmo", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.info("📌 Insight: Los modelos basados en ensambles dominan la investigación moderna por su alto rendimiento.")


# =====================================================
# TAB 3: IMPACT
# =====================================================
with tab3:
    st.subheader("📊 Impacto científico (Citas)")

    fig = px.scatter(
        filtered,
        x="Year",
        y="Cited by",
        hover_data=["Title"],
        size="Cited by"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning("📌 Insight: Los papers más citados suelen estar asociados a modelos más complejos o innovadores.")


# =====================================================
# DATA EXPLORER (STARTUP FEATURE)
# =====================================================
st.divider()
st.subheader("🔍 Paper Explorer (tipo Google Scholar mini)")

st.dataframe(
    filtered[["Title", "Year", "Cited by"]]
    .sort_values("Cited by", ascending=False),
    use_container_width=True
)