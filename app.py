import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# --- 1. CONFIGURACIÓN DEL DASHBOARD ---
st.set_page_config(page_title="Bank Churn AI Hub", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# --- 2. PROCESAMIENTO AVANZADO DE DATOS ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    df['Revista Corta'] = df['Source title'].str[:35] + '...'
    df['Abstract'] = df['Abstract'].fillna('').str.lower()
    
    # 🌟 NUEVO EXTRACTOR DE PAÍSES (Más preciso para Scopus)
    def extract_countries(text):
        if pd.isna(text): return []
        text = str(text).upper()
        # Diccionario de países más comunes en investigación
        paises_map = {
            'UNITED STATES': 'United States', 'USA': 'United States', 'INDIA': 'India', 
            'CHINA': 'China', 'UNITED KINGDOM': 'United Kingdom', 'UK': 'United Kingdom',
            'SPAIN': 'Spain', 'GERMANY': 'Germany', 'FRANCE': 'France', 'CANADA': 'Canada',
            'AUSTRALIA': 'Australia', 'BRAZIL': 'Brazil', 'ITALY': 'Italy', 'MALAYSIA': 'Malaysia',
            'INDONESIA': 'Indonesia', 'SAUDI ARABIA': 'Saudi Arabia', 'TURKEY': 'Turkey',
            'PAKISTAN': 'Pakistan', 'TAIWAN': 'Taiwan', 'SOUTH KOREA': 'South Korea'
        }
        encontrados = []
        for clave, valor in paises_map.items():
            if clave in text:
                encontrados.append(valor)
        return list(set(encontrados)) # Eliminar duplicados

    df['Countries'] = df['Affiliations'].apply(extract_countries)
    return df

def main():
    try:
        df = load_and_process_data()
    except Exception:
        st.error("🚨 Sube tu base de datos scopus_PA3.csv al repositorio.")
        return

    # --- BARRA LATERAL (CONTROLES) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
        st.title("🏦 Panel Bancario")
        st.markdown("Filtra la evidencia científica:")
        
        busqueda = st.text_input("🔍 Buscar término (Ej. Credit, Loan):", "")
        min_citas = st.slider("⭐ Nivel mínimo de citas:", 0, int(df['Cited by'].max()), 0)
        
        st.markdown("---")
        st.info("💡 **Tip para el banco:** Usa este dashboard para justificar presupuestos en tecnología analizando qué hacen los líderes mundiales.")

    # --- FILTRADO EN VIVO ---
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO ---
    st.title("📊 Centro de Inteligencia: Predicción de Churn con IA")
    st.markdown("*Plataforma de apoyo a la toma de decisiones para Gerentes de Riesgo, Retención y Científicos de Datos.*")

    # --- KPIs DE ALTO VALOR ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Casos de Estudio Mapeados", len(df_filtrado))
    c2.metric("🎯 Confiabilidad (Total Citas)", df_filtrado['Cited by'].sum())
    
    todos_paises = [p for sub in df_filtrado['Countries'] for p in sub]
    c3.metric("🌍 Hubs Tecnológicos (Países)", len(set(todos_paises)))
    c4.metric("🏆 Máx. Citas en un solo modelo", df_filtrado['Cited by'].max())

    st.markdown("---")

    # =========================================================================
    # 🗂️ ORGANIZACIÓN EN PESTAÑAS (Para no saturar al usuario, pero darle MUCHA info)
    # =========================================================================
    tab1, tab2, tab3 = st.tabs([
        "🌍 1. Visión Global y Tendencias", 
        "🤖 2. Evaluación de Algoritmos IA", 
        "💼 3. Factores de Riesgo del Cliente"
    ])

    # -------------------------------------------------------------------------
    # PESTAÑA 1: VISIÓN GLOBAL (El Mapa y Revistas)
    # -------------------------------------------------------------------------
    with tab1:
        colA, colB = st.columns([1.5, 1])
        
        with colA:
            # GRÁFICO 1: MAPA MUNDIAL CORREGIDO
            st.subheader("📍 ¿De dónde provienen las soluciones antifuga?")
            if todos_paises:
                df_paises = pd.DataFrame.from_dict(Counter(todos_paises), orient='index').reset_index()
                df_paises.columns = ['País', 'Investigaciones']
                
                fig_map = px.choropleth(
                    df_paises, locations="País", locationmode="country names",
                    color="Investigaciones", hover_name="País",
                    color_continuous_scale="Blues",
                    title="Concentración Global de Desarrollo en IA"
                )
                fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True), margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("No hay datos geográficos con los filtros actuales.")

        with colB:
            # GRÁFICO 2: DONA DE FUENTES CON EXPLICACIÓN
            st.subheader("📖 Fuentes de Autoridad")
            df_fuentes = df_filtrado['Source title'].value_counts().head(5).reset_index()
            df_fuentes.columns = ['Revista', 'Cantidad']
            fig_donut = px.pie(df_fuentes, values='Cantidad', names='Revista', hole=0.4,
                               color_discrete_sequence=px.colors.sequential.Teal)
            fig_donut.update_traces(textposition='inside', textinfo='percent')
            fig_donut.update_layout(margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)
            st.caption("🔍 **Lectura:** Estas son las 5 revistas científicas a las que el banco debería suscribirse para mantenerse actualizado.")

    # -------------------------------------------------------------------------
    # PESTAÑA 2: TECNOLOGÍA (Radar y Burbujas) - PARA CIENTÍFICOS DE DATOS
    # -------------------------------------------------------------------------
    with tab2:
        colC, colD = st.columns(2)
        
        with colC:
            # GRÁFICO 3: RADAR DE ALGORITMOS (Súper Visual y Profesional)
            st.subheader("🎯 Comparativa de Adopción de Algoritmos")
            algoritmos = {'Random Forest': 'random forest', 'Redes Neuronales': 'neural network', 
                          'XGBoost': 'xgboost', 'SVM': 'svm', 'Regresión Logística': 'logistic regression'}
            
            conteos_alg = []
            for nombre, keyword in algoritmos.items():
                conteo = df_filtrado['Abstract'].str.contains(keyword).sum()
                conteos_alg.append({'Algoritmo': nombre, 'Frecuencia': conteo})
                
            df_alg = pd.DataFrame(conteos_alg)
            
            fig_radar = px.line_polar(df_alg, r='Frecuencia', theta='Algoritmo', line_close=True,
                                      title="Tecnologías Dominantes en la Literatura",
                                      template="plotly_white", color_discrete_sequence=['#1f77b4'])
            fig_radar.update_traces(fill='toself', fillcolor='rgba(31, 119, 180, 0.4)')
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("🔍 **Conclusión de Negocio:** El área sombreada más amplia indica el algoritmo más probado por la industria. Este debería ser el primer modelo que el banco intente implementar.")

        with colD:
            # GRÁFICO 4: BURBUJAS DE IMPACTO
            st.subheader("🔥 Validaciones Exitosas por Año")
            fig_bubble = px.scatter(
                df_filtrado, x="Year", y="Cited by", size="Cited by", color="Revista Corta",
                hover_name="Title", size_max=45,
                title="Modelos más citados (Burbujas grandes = Mayor éxito comprobado)"
            )
            fig_bubble.update_layout(xaxis=dict(tickmode='linear', dtick=1), showlegend=False)
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.caption("🔍 **Tip:** Pasa el ratón sobre las burbujas más grandes para ver qué modelo usar.")

    # -------------------------------------------------------------------------
    # PESTAÑA 3: NEGOCIO Y FACTORES DE RIESGO - PARA GERENTES
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("💡 ¿Qué variables causan la fuga de clientes según la ciencia?")
        st.markdown("Hemos escaneado todos los documentos para identificar cuáles son los factores del cliente más estudiados al predecir su fuga.")
        
        colE, colF = st.columns([1, 2])
        
        with colE:
            # GRÁFICO 5: BARRAS DE FACTORES DE NEGOCIO
            factores = {'Tarjetas de Crédito': 'credit', 'Comportamiento/Transacciones': 'behavio', 
                        'Factores Demográficos': 'demograph', 'Préstamos/Deudas': 'loan', 
                        'Atención al Cliente': 'service'}
            
            conteos_factores = []
            for nombre, keyword in factores.items():
                conteo = df_filtrado['Abstract'].str.contains(keyword).sum()
                conteos_factores.append({'Factor': nombre, 'Menciones': conteo})
                
            df_factores = pd.DataFrame(conteos_factores).sort_values(by='Menciones')
            
            fig_bar_factores = px.bar(df_factores, x='Menciones', y='Factor', orientation='h',
                                      color='Menciones', color_continuous_scale='Reds',
                                      title="Top Variables Predictoras")
            st.plotly_chart(fig_bar_factores, use_container_width=True)

        with colF:
            # GRÁFICO 6: TABLA DE LECTURA OBLIGATORIA
            st.markdown("### 📚 Documentos Recomendados para la Gerencia")
            st.dataframe(
                df_filtrado[['Title', 'Year', 'Cited by', 'Link']].sort_values('Cited by', ascending=False).head(8), 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Title": st.column_config.TextColumn("Investigación", width="large"),
                    "Year": "Año",
                    "Cited by": st.column_config.ProgressColumn("Nivel de Autoridad (Citas)", format="%f", min_value=0, max_value=int(df['Cited by'].max())),
                    "Link": st.column_config.LinkColumn("Enlace a Scopus")
                }
            )

if __name__ == "__main__":
    main()