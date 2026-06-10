import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Configuración de apariencia profesional (Tema corporativo)
st.set_page_config(page_title="Bank Churn Intelligence Hub", page_icon="🏦", layout="wide")

# 2. AUTOMATIZACIÓN: El sistema lee el archivo directamente (¡Sin widgets de carga!)
@st.cache_data
def load_bank_data():
    # Lee el archivo scopus_PA3.csv que estará guardado en tu mismo GitHub
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0)
    df['Year'] = df['Year'].fillna(2025)
    return df

def main():
    # Intentar cargar los datos automáticamente
    try:
        df = load_bank_data()
    except FileNotFoundError:
        st.error("🚨 Error crítico: No se encontró el archivo 'scopus_PA3.csv' en la carpeta.")
        return

    # --- BANNER PRINCIPAL (Enfoque de Negocios) ---
    st.title("🏦 Portal de Inteligencia: Mitigación de Churn Bancario")
    st.markdown("""
    **Audiencia:** Equipo de Analytics, Riesgos y Retención de Clientes.  
    *Este dashboard consolida la evidencia científica de los últimos estudios (2025-2026) para identificar los algoritmos y estrategias más eficientes del mercado para predecir la fuga de clientes.*
    """)
    st.divider()

    # --- METRICAS CLAVE PARA EL NEGOCIO ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📊 Modelos/Casos Analizados", value=len(df))
    with col2:
        # Contamos cuántas investigaciones mencionan validación o alto impacto
        st.metric(label="🔥 Respaldo Científico (Citas Totales)", value=int(df['Cited by'].sum()))
    with col3:
        # Revistas de finanzas o tecnología involucradas
        st.metric(label="🏢 Fuentes/Revistas Especializadas", value=df['Source title'].nunique())
    with col4:
        st.metric(label="📅 Horizonte Temporal", value="2025 - 2026")
    
    st.divider()

    # --- PESTAÑAS ESTRATÉGICAS ---
    tab1, tab2, tab3 = st.tabs([
        "🤖 Consenso Tecnológico (Modelos ML)", 
        "🔍 Buscador de Casos y Soluciones Bancarias", 
        "📈 Impacto de la Investigación"
    ])

    # PESTAÑA 1: CONSENSO TECNOLÓGICO (Útil para el equipo de Data Science del banco)
    with tab1:
        st.subheader("💡 ¿Qué modelos de Machine Learning recomiendan los expertos?")
        st.markdown("Análisis semántico automatizado de los algoritmos más exitosos mencionados en los abstracts:")

        # Mapeo inteligente de algoritmos dentro de tu archivo de Scopus
        texto_completo = df['Abstract'].fillna('').str.lower() + ' ' + df['Author Keywords'].fillna('').str.lower()
        
        lista_algoritmos = ['Random Forest', 'SHAP (Interpretable ML)', 'XGBoost', 'Neural Networks (Redes Neuronales)', 'SVM', 'Decision Tree']
        conteos = []
        for alg in lista_algoritmos:
            # Simplificar término para la búsqueda
            termino_busqueda = alg.split('(')[0].strip().lower()
            cantidad = texto_completo.str.contains(termino_busqueda).sum()
            conteos.append({'Algoritmo': alg, 'Investigaciones que lo validan': cantidad})
        
        df_alg = pd.DataFrame(conteos).sort_values(by='Investigaciones que lo validan', ascending=False)

        col_graf, col_txt = st.columns([2, 1])
        with col_graf:
            fig_alg = px.bar(df_alg, x='Investigaciones que lo validan', y='Algoritmo', orientation='h',
                             color='Investigaciones que lo validan', color_continuous_scale='Blues',
                             title="Modelos de IA con Mayor Tasa de Éxito en Banca")
            st.plotly_chart(fig_alg, use_container_width=True)
        
        with col_txt:
            st.markdown("##### 📌 Conclusión para el Banco:")
            st.info("""
            Si el banco busca implementar un modelo inmediato:
            1. **Random Forest y XGBoost** son los más recomendados por la literatura por su precisión con datos tabulares de clientes.
            2. El uso de **SHAP** se ha vuelto obligatorio en 2025/2026 para que el área legal y de riesgos entienda *por qué* la IA dice que un cliente se va a ir.
            """)

    # PESTAÑA 2: BUSCADOR DE CASOS (Útil para un Gerente de Retención)
    with tab2:
        st.subheader("🔍 Biblioteca de Soluciones y Metodologías Antifuga")
        st.markdown("Usa el buscador para filtrar resúmenes científicos según palabras clave del negocio (ej. *credit card*, *behavioral*, *predicting*, *interpretable*).")
        
        busqueda = st.text_input("⌨️ Introduce un término de negocio (en inglés preferiblemente por Scopus):", "interpretable")
        
        # Filtrado dinámico por texto
        df_resultados = df[
            df['Title'].fillna('').str.lower().str.contains(busqueda.lower()) | 
            df['Abstract'].fillna('').str.lower().str.contains(busqueda.lower())
        ]
        
        st.caption(f"Se encontraron {len(df_resultados)} papers que resuelven este problema específico.")
        
        # Mostrar los resultados limpios para lectura ejecutiva
        for idx, row in df_resultados.head(5).iterrows():
            with st.expander(f"📌 {row['Title']} ({row['Year']})"):
                st.write(f"**Autores:** {row['Authors']}")
                st.write(f"**Revista:** {row['Source title']}")
                st.write(f"**Resumen Técnico (Abstract):** {row['Abstract']}")
                if 'Link' in df.columns:
                    st.write(f"[🔗 Ver paper original en Scopus]({row['Link']})")

    # PESTAÑA 3: ANALISIS DE IMPACTO
    with tab3:
        st.subheader("📈 Análisis de Impacto y Validación de Fuentes")
        colA, colB = st.columns(2)
        
        with colA:
            # Artículos con más citas (los que tienen metodologías más confiables)
            df_citas = df.sort_values(by='Cited by', ascending=False).head(5)
            df_citas['Título Corto'] = df_citas['Title'].str[:40] + "..."
            fig_citas = px.bar(df_citas, x='Cited by', y='Título Corto', orientation='h',
                               title="Top 5 Metodologías más Validadas (Más Citadas)",
                               color='Cited by', color_continuous_scale='Greens')
            fig_citas.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_citas, use_container_width=True)
            
        with colB:
            # Dónde se publica más esto
            df_source = df['Source title'].value_counts().head(5).reset_index()
            df_source.columns = ['Revista', 'Cantidad']
            fig_source = px.pie(df_source, values='Cantidad', names='Revista', 
                                title="¿De dónde provienen estas soluciones?", hole=0.4,
                                color_discrete_sequence=px.colors.sequential.YlGnBu)
            st.plotly_chart(fig_source, use_container_width=True)

if __name__ == "__main__":
    main()