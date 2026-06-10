import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DEL ENTORNO EMPRESARIAL ---
st.set_page_config(
    page_title="ChurnAI Analytics Hub", 
    page_icon="🔮", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para forzar la estética premium Dark Mode & Fuentes limpias
st.markdown("""
    <style>
        .reportview-container { background: #0E1117; }
        .stMarkdown { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        h1, h2, h3 { color: #00CED1 !important; }
        div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE PROCESAMIENTO DE DATOS ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    df['Revista Corta'] = df['Source title'].str[:35] + '...'
    df['Abstract_Clean'] = df['Abstract'].fillna('').str.lower()
    return df

def main():
    try:
        df = load_and_process_data()
    except Exception:
        st.error("🚨 Error crítico: No se encontró la base de datos de Scopus 'scopus_PA3.csv'.")
        return

    # --- BARRA LATERAL: CONTROL DE FILTROS ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=50)
        st.title("Filtros de Control")
        st.markdown("---")
        
        busqueda = st.text_input("🔍 Buscar término técnico:", "")
        min_citas = st.slider("📈 Nivel de autoridad del estudio (Citas):", 0, int(df['Cited by'].max()), 0)
        
        st.markdown("---")
        st.caption("🔒 ChurnAI Enterprise v2.1. Periodo Operativo 2025-2026.")

    # Inyección de filtros a la data analizada
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO CORPORATIVO DE LA STARTUP ---
    st.title("🔮 ChurnAI Analytics Hub")
    st.subheader("Plataforma de Onboarding Técnico para Analistas de Riesgo y Retención Bancaria")
    
    # Respuesta directa a la pregunta de investigación integrada en la interfaz
    st.markdown("""
    <div style='background-color: #161B22; padding: 20px; border-radius: 8px; border-left: 5px solid #FF1493; margin-bottom: 25px;'>
        <h4 style='color: #FF1493; margin-top:0;'>📌 PREGUNTA DE ESTRATEGIA CENTRAL (2025-2026)</h4>
        <p style='color: #E6EDF3; font-size: 1.05rem; line-height: 1.6;'>
            <b>¿Cómo optimiza el uso de machine learning la predicción de la fuga de clientes (customer churn) en el sector bancario?</b><br>
            La optimización ocurre al reemplazar modelos estadísticos estáticos por arquitecturas predictivas dinámicas. Esto permite identificar patrones invisibles en el comportamiento transaccional del cliente antes de que decida abandonar la entidad, reduciendo la tasa de deserción mediante alertas tempranas automatizadas.
        </p>
        <p style='color: #8B949E; font-size: 0.85rem; margin-bottom: 0;'>
            <b>Keywords Clave:</b> Machine learning • Customer churn • Banking • Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- PESTAÑAS OPERATIVAS ---
    tab1, tab2 = st.tabs(["📊 Simulador de Métricas y Variables", "🧠 Modelos y Evidencia Científica"])

    # =========================================================================
    # PESTAÑA 1: GRÁFICOS SOLICITADOS CON INTERACCIÓN AVANZADA
    # =========================================================================
    with tab1:
        colC, colD = st.columns(2)
        
        # COLUMNA IZQUIERDA: MÉTRICAS DE EVALUACIÓN
        with colC:
            st.markdown("### 🎯 Métricas de Evaluación")
            
            metricas_data = [
                {'Metrica': 'Accuracy', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum(), 'Desc': 'Porcentaje general de aciertos.'},
                {'Metrica': 'F1-Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum(), 'Desc': 'Balance crítico entre precisión y falsos positivos.'},
                {'Metrica': 'AUC-ROC', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum(), 'Desc': 'Capacidad del modelo para distinguir un cliente leal de uno en riesgo.'}
            ]
            df_metrics = pd.DataFrame(metricas_data)
            
            fig_radar = px.line_polar(
                df_metrics, r='Menciones', theta='Metrica', line_close=True,
                custom_data=['Desc'],
                template="plotly_dark"
            )
            fig_radar.update_traces(
                fill='toself', 
                line_color='#00CED1', 
                hovertemplate="<b>Métrica: %{theta}</b><br>Frecuencia en industria: %{r}<br><i>%{customdata[0]}</i><extra></extra>"
            )
            fig_radar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                polar=dict(radialaxis=dict(gridcolor='#30363D'), angularaxis=dict(gridcolor='#30363D'))
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            with st.expander("📖 Guía de Interpretación Exclusiva"):
                st.markdown("""
                * **¿Qué significa?** Muestra qué herramientas matemáticas prefieren los científicos de datos para certificar que un modelo de fuga es seguro y preciso.
                * **¿Qué representa?** El área cubierta del polígono representa el estándar de validación en la banca actual.
                * **¿Cómo se lee este gráfico?** Mientras más se extienda el polígono hacia un extremo, mayor es el consenso de la industria en usar esa métrica específica.
                """)
            
            # BLOQUE INTERACTIVO SOLICITADO: SIMULADOR DE CASO DE NEGOCIO
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #FF1493;'>🧪 Laboratorio de Simulación Corporativa</h4>", unsafe_allow_html=True)
            metric_choice = st.selectbox(
                "Haz clic aquí y selecciona una métrica para ver un ejemplo de aplicación en banca:",
                ["Selecciona una opción...", "F1-Score", "AUC-ROC", "Accuracy"]
            )
            
            if metric_choice == "F1-Score":
                st.info("💡 **Caso Real en Churn:** El Churn está severamente desbalanceado (ej. solo el 2% de tus clientes se va). Si usas *Accuracy*, un modelo perezoso que diga 'Nadie se irá' tendrá 98% de éxito pero perderás millones. El **F1-Score** obliga al modelo a encontrar eficazmente al 2% real sin equivocarse.")
            elif metric_choice == "AUC-ROC":
                st.info("💡 **Caso Real en Churn:** El **AUC-ROC** te ayuda a ordenar a los clientes de un banco mediante un score de 0 a 100. Permite al equipo de marketing definir el 'umbral de corte': a partir de un 85% de probabilidad de fuga, se dispara automáticamente un bono de retención.")
            elif metric_choice == "Accuracy":
                st.warning("⚠️ **Alerta de Riesgo:** En predicción de Churn Bancario, confiar ciegamente en el *Accuracy* es un error crítico de novato debido al desbalance de las clases. Se utiliza únicamente como métrica de control secundario.")

        # COLUMNA DERECHA: VARIABLES PREDICTORAS (TREEMAP)
        with colD:
            st.markdown("### 🧩 Variables Predictoras")
            
            features_data = [
                {'Categoria': 'Transacciones', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('transaction|behavio').sum(), 'Desc': 'Fluctuaciones de saldo, frecuencia de uso de apps, caída en depósitos.'},
                {'Categoria': 'Credit Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('credit score|credit history').sum(), 'Desc': 'Historial crediticio, deudas externas y comportamiento de pago.'},
                {'Categoria': 'Demografía', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('demograph|age|gender').sum(), 'Desc': 'Edad, ingresos estimados, ubicación geográfica.'}
            ]
            df_feat = pd.DataFrame(features_data)
            
            fig_tree = px.treemap(
                df_feat, path=['Categoria'], values='Menciones',
                custom_data=['Desc'], 
                color='Menciones', 
                color_continuous_scale=['#FF1493', '#00CED1'],
                template="plotly_dark"
            )
            fig_tree.update_traces(hovertemplate="<b>Módulo: %{label}</b><br>Estudios que lo respaldan: %{value}<br><i>%{customdata[0]}</i><extra></extra>")
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tree, use_container_width=True)
            
            with st.expander("📖 Guía de Interpretación Exclusiva"):
                st.markdown("""
                * **¿Qué significa?** Clasifica los grupos de datos del cliente que alimentan al algoritmo para que este aprenda a detectar anomalías.
                * **¿Qué representa?** Representa los pilares lógicos del *Feature Engineering* (Ingeniería de Variables) financiero.
                * **¿Cómo se lee este gráfico?** El volumen del cuadro es proporcional a la importancia y uso en la literatura científica. Los cuadros más grandes indican datos con mayor poder predictivo.
                """)

            # BLOQUE INTERACTIVO SOLICITADO: EJEMPLO DE DATA ENGINEERING
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #FF1493;'>🛠️ Feature Engineering Hub</h4>", unsafe_allow_html=True)
            feat_choice = st.selectbox(
                "Haz clic aquí y selecciona un tipo de variable para ver un ejemplo de arquitectura de datos:",
                ["Selecciona una opción...", "Transacciones", "Credit Score", "Demografía"]
            )
            
            if feat_choice == "Transacciones":
                st.info("⚙️ **Extracción de Variables:** La startup calcula la tasa de decaimiento del saldo (ej. si el cliente redujo sus depósitos un 40% en los últimos 45 días). Este cambio de comportamiento es el indicador más fuerte de que el cliente está usando otro banco.")
            elif feat_choice == "Credit Score":
                st.info("⚙️ **Extracción de Variables:** Un aumento repentino en las consultas de su historial crediticio sugiere que el cliente busca financiamiento externo. Si el algoritmo detecta esto combinado con inactividad transaccional, eleva el nivel de riesgo de fuga.")
            elif feat_choice == "Demografía":
                st.info("⚙️ **Extracción de Variables:** Permite segmentar el ciclo de vida del cliente. Los jóvenes adultos (20-30 años) presentan un Churn significativamente más alto debido a la volatilidad del mercado y ofertas competitivas en canales 100% digitales.")

    # =========================================================================
    # PESTAÑA 2: EVIDENCIA CIENTÍFICA (SOPORTE RIGUROSO)
    # =========================================================================
    with tab2:
        st.markdown("### 🧠 Respaldo de Investigaciones Scopus (2025-2026)")
        st.markdown("Todo algoritmo implementado en producción por la Startup está sustentado en papers indexados de alta autoridad.")
        
        df_mostrar = df_filtrado[['Title', 'Year', 'Cited by', 'Revista Corta', 'Link']].copy()
        
        st.dataframe(
            df_mostrar.sort_values('Cited by', ascending=False), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Título del Estudio Científico"),
                "Year": st.column_config.NumberColumn("Año", format="%d"),
                "Cited by": st.column_config.ProgressColumn("Impacto (Citas)", format="%d", min_value=0, max_value=int(df['Cited by'].max())),
                "Link": st.column_config.LinkColumn("Enlace Oficial Scopus")
            }
        )

if __name__ == "__main__":
    main()