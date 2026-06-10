import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

# 1. Configuración de página a pantalla completa
st.set_page_config(page_title="Bank Churn AI Explorer", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# 2. Carga automática del archivo (sin que el usuario haga nada)
@st.cache_data
def load_bank_data():
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    return df

def main():
    try:
        df = load_bank_data()
    except:
        st.error("🚨 Sube tu archivo scopus_PA3.csv a GitHub primero.")
        return

    # --- BARRA LATERAL: EL CONTROL REMOTO DEL USUARIO ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.sidebar.title("🕹️ Panel de Control")
    st.sidebar.markdown("Juega con los filtros para descubrir insights.")

    # Filtro 1: Rango de años
    min_yr, max_yr = int(df['Year'].min()), int(df['Year'].max())
    rango_anios = st.sidebar.slider("📅 Selecciona el periodo:", min_yr, max_yr, (min_yr, max_yr))

    # Filtro 2: Nivel de impacto (Citas)
    min_citas = st.sidebar.slider("🔥 Mínimo de citas del artículo:", 0, int(df['Cited by'].max()), 0)

    # Filtro 3: Buscador interactivo
    palabra_clave = st.sidebar.text_input("🔍 Buscar tema (ej. Random Forest, SHAP):", "")

    # APLICAR FILTROS
    df_filtrado = df[(df['Year'] >= rango_anios[0]) & (df['Year'] <= rango_anios[1])]
    df_filtrado = df_filtrado[df_filtrado['Cited by'] >= min_citas]
    if palabra_clave:
        df_filtrado = df_filtrado[df_filtrado['Abstract'].fillna('').str.contains(palabra_clave, case=False) | 
                                  df_filtrado['Title'].fillna('').str.contains(palabra_clave, case=False)]

    # --- ENCABEZADO ---
    st.title("🏦 Explorador de IA para Mitigar Fuga de Clientes (Churn)")
    
    # --- KPIs VISUALES ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Artículos Encontrados", len(df_filtrado))
    c2.metric("⭐ Citas Totales", int(df_filtrado['Cited by'].sum()))
    c3.metric("🏆 Máximo de Citas en un Paper", int(df_filtrado['Cited by'].max()))
    c4.metric("🏢 Revistas Especializadas", df_filtrado['Source title'].nunique())
    
    st.divider()

    # Si no hay datos tras filtrar, avisar visualmente
    if df_filtrado.empty:
        st.warning("⚠️ No hay artículos que coincidan con tus filtros. Prueba ajustándolos en el panel izquierdo.")
        return

    # --- FILA 1 DE GRÁFICOS: BURBUJAS Y MAPA DE PALABRAS ---
    st.markdown("### 🔭 Panorama de la Investigación")
    colA, colB = st.columns(2)

    with colA:
        # Gráfico 1: Treemap de Palabras Clave (Super dinámico)
        if 'Author Keywords' in df_filtrado.columns:
            # Procesar las palabras clave
            kws = df_filtrado['Author Keywords'].dropna().str.split(';')
            lista_kws = [k.strip().title() for sub in kws for k in sub]
            conteo_kws = pd.DataFrame(Counter(lista_kws).most_common(20), columns=['Concepto', 'Frecuencia'])
            
            fig_tree = px.treemap(conteo_kws, path=['Concepto'], values='Frecuencia',
                                  title="Conceptos y Algoritmos Más Estudiados (Treemap)",
                                  color='Frecuencia', color_continuous_scale='Teal')
            st.plotly_chart(fig_tree, use_container_width=True)

    with colB:
        # Gráfico 2: Gráfico de Burbujas (Citas vs Año)
        fig_bubble = px.scatter(df_filtrado, x="Year", y="Cited by", size="Cited by", color="Source title",
                                hover_name="Title", size_max=45,
                                title="Impacto de los Artículos (El tamaño es el nº de citas)")
        
        # Ajustamos el eje X para que los años se vean enteros (2025, 2026)
        fig_bubble.update_layout(xaxis=dict(tickformat="d"))
        st.plotly_chart(fig_bubble, use_container_width=True)

    # --- FILA 2 DE GRÁFICOS: TENDENCIAS Y TOP REVISTAS ---
    st.markdown("### 🏆 Liderazgo Científico")
    colC, colD = st.columns([1.5, 1])

    with colC:
        # Gráfico 3: Barras Horizontales Top Papers
        df_top = df_filtrado.sort_values('Cited by', ascending=False).head(7)
        df_top['Título'] = df_top['Title'].str[:50] + "..."
        fig_bar = px.bar(df_top, x='Cited by', y='Título', orientation='h', 
                         color='Cited by', color_continuous_scale='Inferno',
                         title="Los 7 Artículos Más Influyentes (Por Citas)",
                         hover_data=['Authors', 'Year'])
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with colD:
        # Gráfico 4: Gráfico de Anillo
        top_revistas = df_filtrado['Source title'].value_counts().head(5).reset_index()
        top_revistas.columns = ['Revista', 'Artículos']
        fig_donut = px.pie(top_revistas, values='Artículos', names='Revista', hole=0.5,
                           title="Top 5 Revistas", color_discrete_sequence=px.colors.sequential.Agal)
        fig_donut.update_traces(textposition='inside', textinfo='percent')
        st.plotly_chart(fig_donut, use_container_width=True)

    # --- SECCIÓN INTERACTIVA FINAL: EXPLORADOR DE DATOS ---
    st.divider()
    st.markdown("### 🗄️ Base de Datos Interactiva")
    st.markdown("Haz clic en cualquier columna para ordenar los datos.")
    
    # Mostramos una tabla limpia y bonita
    columnas_mostrar = ['Year', 'Title', 'Authors', 'Source title', 'Cited by', 'Link']
    columnas_existentes = [c for c in columnas_mostrar if c in df_filtrado.columns]
    
    st.dataframe(
        df_filtrado[columnas_existentes],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Year": st.column_config.NumberColumn("Año", format="%d"),
            "Cited by": st.column_config.NumberColumn("Citas"),
            "Link": st.column_config.LinkColumn("Enlace a Scopus")
        }
    )

if __name__ == "__main__":
    main()