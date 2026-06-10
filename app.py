import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

# 1. Configuración de página a pantalla completa
st.set_page_config(page_title="AI Bank Churn Analytics", page_icon="🏦", layout="wide")

# 2. Carga automática de datos
@st.cache_data
def load_data():
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    # Acortamos nombres largos para que los gráficos se vean elegantes
    df['Revista Corta'] = df['Source title'].str[:35] + '...'
    return df

def main():
    try:
        df = load_data()
    except Exception:
        st.error("🚨 Error: No se encontró el archivo scopus_PA3.csv en GitHub.")
        return

    # --- BARRA LATERAL: CENTRO DE CONTROL ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.sidebar.title("🕹️ Filtros de Negocio")
    st.sidebar.markdown("Explora las soluciones científicas:")

    # FILTRO 1: Buscador Libre (Más útil que los años)
    busqueda = st.sidebar.text_input("🔍 Buscar término (ej. Credit Card, Behaviour):", "")

    # FILTRO 2: Selector de Algoritmo (Útil para el banco)
    modelos = ["Todos", "Random Forest", "Neural Network", "SVM", "XGBoost", "Decision Tree", "Deep Learning", "SHAP"]
    modelo_seleccionado = st.sidebar.selectbox("🤖 Filtrar por Tecnología IA:", modelos)

    # Lógica de filtrado en vivo
    df_filtrado = df.copy()
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract'].fillna('').str.contains(busqueda, case=False) | 
                                  df_filtrado['Title'].fillna('').str.contains(busqueda, case=False)]
    
    if modelo_seleccionado != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Abstract'].fillna('').str.contains(modelo_seleccionado, case=False) | 
                                  df_filtrado['Author Keywords'].fillna('').str.contains(modelo_seleccionado, case=False)]

    # --- ENCABEZADO ---
    st.title("🏦 Dashboard Interactivo: Mitigación de Fuga de Clientes con IA")
    st.markdown("*Herramienta de exploración visual para equipos de Riesgo y Analítica Bancaria.*")

    if df_filtrado.empty:
        st.warning("⚠️ Tus filtros son muy específicos y no hay resultados. Prueba seleccionando 'Todos'.")
        return

    # --- KPIs DE IMPACTO ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Papers Encontrados", len(df_filtrado))
    c2.metric("⭐ Impacto (Citas Totales)", df_filtrado['Cited by'].sum())
    # Calculamos cuántos autores trabajaron en estas soluciones
    total_autores = df_filtrado['Authors'].str.count(';').sum() + len(df_filtrado)
    c3.metric("🧠 Mentes Investigando", int(total_autores))
    c4.metric("🏢 Revistas Únicas", df_filtrado['Source title'].nunique())

    st.divider()

    # --- ZONA DE GRÁFICOS ULTRA INTERACTIVOS ---
    st.markdown("### 🔭 Ecosistema Analítico (Haz clic en los gráficos)")
    colA, colB = st.columns(2)

    with colA:
        # GRÁFICO 1: Sunburst Chart (Gráfico Solar)
        # Es visualmente increíble. El usuario hace clic en un año y se expanden las revistas.
        df_filtrado['Conteo'] = 1
        fig_sunburst = px.sunburst(
            df_filtrado, 
            path=['Year', 'Revista Corta'], 
            values='Conteo',
            color='Cited by', 
            color_continuous_scale='Teal',
            title="Distribución de Publicaciones (Click para Zoom)"
        )
        fig_sunburst.update_traces(textinfo="label+percent parent")
        st.plotly_chart(fig_sunburst, use_container_width=True)

    with colB:
        # GRÁFICO 2: Dispersión con Distribuciones Marginales (Estilo Dark Mode)
        # Se ve extremadamente profesional y "hacker".
        fig_scatter = px.scatter(
            df_filtrado, x="Year", y="Cited by", color="Revista Corta",
            hover_name="Title", size_max=15, marginal_y="violin", marginal_x="box",
            title="Distribución de Impacto Científico por Año",
            template="plotly_dark" # Tema oscuro que lo hace resaltar
        )
        fig_scatter.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')))
        fig_scatter.update_layout(xaxis_type='category') # Evita que el año salga como "2025.5"
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- GRÁFICO 3: TREEMAP DE CONCEPTOS ---
    st.markdown("### 🧩 Mapa Interactivo de Algoritmos y Conceptos")
    st.markdown("Los bloques más grandes son las tecnologías con mayor tendencia en el sector bancario.")
    if 'Author Keywords' in df_filtrado.columns:
        kws = df_filtrado['Author Keywords'].dropna().str.split(';')
        lista_kws = [k.strip().title() for sub in kws for k in sub if len(k.strip()) > 3]
        conteo_kws = pd.DataFrame(Counter(lista_kws).most_common(25), columns=['Concepto', 'Frecuencia'])
        
        fig_tree = px.treemap(
            conteo_kws, path=['Concepto'], values='Frecuencia',
            color='Frecuencia', color_continuous_scale='Plasma'
        )
        fig_tree.update_traces(textinfo="label+value")
        st.plotly_chart(fig_tree, use_container_width=True)

    # --- EXPLORADOR DE DATOS FINAL ---
    st.divider()
    st.markdown("### 🗂️ Catálogo de Soluciones")
    st.dataframe(
        df_filtrado[['Title', 'Year', 'Cited by', 'Revista Corta', 'Link']], 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Year": st.column_config.NumberColumn("Año", format="%d"),
            "Cited by": st.column_config.NumberColumn("Citas"),
            "Link": st.column_config.LinkColumn("Leer Paper")
        }
    )

if __name__ == "__main__":
    main()