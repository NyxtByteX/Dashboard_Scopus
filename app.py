import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de la página (layout ancho para que se vea como un dashboard real)
st.set_page_config(page_title="Dashboard Churn Bancario", page_icon="📈", layout="wide")

# 2. Optimización: Caché para carga rápida
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Limpieza básica: si hay citas vacías, las convertimos en 0
    if 'Cited by' in df.columns:
        df['Cited by'] = df['Cited by'].fillna(0)
    return df

# 3. Modularización: Función principal
def main():
    # Menú lateral
    st.sidebar.title("Navegación")
    st.sidebar.markdown("Sube tu archivo CSV de Scopus para visualizar el análisis.")
    uploaded_file = st.sidebar.file_uploader("Cargar Archivo", type=["csv"])

    # Título principal
    st.title("📈 Dashboard Bibliométrico: Machine Learning en Fuga de Clientes")
    st.markdown("Análisis de literatura científica sobre *Customer Churn* en el sector bancario.")
    st.divider()

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)

            # --- SECCIÓN 1: KPIs (Métricas Clave) ---
            st.subheader("📊 Indicadores Generales")
            total_articulos = df.shape[0]
            total_citas = int(df['Cited by'].sum()) if 'Cited by' in df.columns else 0
            fuentes_unicas = df['Source title'].nunique() if 'Source title' in df.columns else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Artículos", total_articulos)
            col2.metric("Total de Citas", total_citas)
            col3.metric("Fuentes Únicas (Revistas)", fuentes_unicas)
            st.divider()

            # --- SECCIÓN 2: Gráficos Interactivos (Plotly) ---
            colA, colB = st.columns(2)

            with colA:
                st.subheader("Artículos por Año")
                if 'Year' in df.columns:
                    # Agrupamos por año
                    df_year = df['Year'].value_counts().reset_index()
                    df_year.columns = ['Año', 'Cantidad']
                    df_year = df_year.sort_values('Año')
                    
                    fig_year = px.bar(df_year, x='Año', y='Cantidad', text='Cantidad', 
                                      color='Cantidad', color_continuous_scale='Blues',
                                      title="Evolución de Publicaciones")
                    fig_year.update_traces(textposition='outside')
                    st.plotly_chart(fig_year, use_container_width=True)

            with colB:
                st.subheader("Top 5 Revistas")
                if 'Source title' in df.columns:
                    df_source = df['Source title'].value_counts().head(5).reset_index()
                    df_source.columns = ['Revista', 'Cantidad']
                    
                    fig_source = px.pie(df_source, values='Cantidad', names='Revista', 
                                        title="Distribución por Fuentes", hole=0.4)
                    st.plotly_chart(fig_source, use_container_width=True)

            # --- SECCIÓN 3: Gráfico Adicional y Dataset ---
            st.subheader("Artículos Más Citados")
            if 'Title' in df.columns and 'Cited by' in df.columns:
                df_citas = df.sort_values(by='Cited by', ascending=False).head(10)
                # Acortamos el título para que encaje bien en el gráfico
                df_citas['Título Corto'] = df_citas['Title'].str[:50] + "..."
                
                fig_citas = px.bar(df_citas, x='Cited by', y='Título Corto', orientation='h',
                                   color='Cited by', color_continuous_scale='Viridis',
                                   title="Top 10 Artículos con Mayor Impacto")
                fig_citas.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_citas, use_container_width=True)

            # Expansor para mostrar la tabla de datos (como en la imagen de tu compañero)
            with st.expander("Ver Dataset Completo"):
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("👈 Por favor, carga tu archivo CSV en el menú lateral.")

if __name__ == "__main__":
    main()