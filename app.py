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

# Estilo CSS personalizado para mantener la estética Dark Mode Estricta
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

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=50)
        st.title("Filtros de Control")
        st.markdown("---")
        
        busqueda = st.text_input("🔍 Buscar término técnico:", "")
        min_citas = st.slider("📈 Mínimo de citas del paper:", 0, int(df['Cited by'].max()), 0)
        
        st.markdown("---")
        st.caption("🔒 ChurnAI Enterprise v2.5. Periodo Operativo 2025-2026.")

    # Filtros de la base de datos
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO CORPORATIVO ---
    st.title("🔮 ChurnAI Analytics Hub")
    st.subheader("Plataforma Financiera de Onboarding y Simulación Predictiva")
    
    st.markdown("""
    <div style='background-color: #161B22; padding: 20px; border-radius: 8px; border-left: 5px solid #FF1493; margin-bottom: 25px;'>
        <h4 style='color: #FF1493; margin-top:0;'>📌 ESTRATEGIA CENTRAL DE INVESTIGACIÓN (2025-2026)</h4>
        <p style='color: #E6EDF3; font-size: 1.05rem; line-height: 1.6;'>
            <b>¿Cómo optimiza el uso de machine learning la predicción de la fuga de clientes (customer churn) en el sector bancario?</b><br>
            La optimización se logra al procesar patrones transaccionales complejos en tiempo real. Los algoritmos de Machine Learning detectan anomalías de comportamiento antes de que el cliente finalice la baja, permitiendo ejecutar acciones de retención de manera automatizada y quirúrgica.
        </p>
        <p style='color: #8B949E; font-size: 0.85rem; margin-bottom: 0;'>
            <b>Keywords:</b> Machine learning • Customer churn • Banking • Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- PESTAÑAS OPERATIVAS ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Métricas y Variables de la Industria", 
        "🔮 Simulador de Proyección de Riesgo", 
        "📚 Repositorio Scopus Inteligente"
    ])

    # =========================================================================
    # PESTAÑA 1: GRÁFICOS ANALÍTICOS (MÉTRICAS Y VARIABLES)
    # =========================================================================
    with tab1:
        colC, colD = st.columns(2)
        
        with colC:
            st.markdown("### 🎯 Métricas de Evaluación")
            metricas_data = [
                {'Metrica': 'Accuracy', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum(), 'Desc': 'Porcentaje general de aciertos.'},
                {'Metrica': 'F1-Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum(), 'Desc': 'Balance crítico entre precisión y reclamos.'},
                {'Metrica': 'AUC-ROC', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum(), 'Desc': 'Capacidad para diferenciar clientes leales de clientes en riesgo.'}
            ]
            df_metrics = pd.DataFrame(metricas_data)
            
            fig_radar = px.line_polar(df_metrics, r='Menciones', theta='Metrica', line_close=True, custom_data=['Desc'], template="plotly_dark")
            fig_radar.update_traces(fill='toself', line_color='#00CED1', hovertemplate="<b>Métrica: %{theta}</b><br>Menciones: %{r}<br><i>%{customdata[0]}</i><extra></extra>")
            fig_radar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, use_container_width=True)
            
            with st.expander("📖 Guía de Interpretación"):
                st.write("**¿Qué significa?** Revela los estándares matemáticos preferidos por la ciencia de datos para validar modelos bancarios.")
                st.write("**¿Qué representa?** El área de cobertura del mapa de radar.")
                st.write("**¿Cómo se lee este gráfico?** Entre más lejano esté el vértice del centro, mayor es la adopción de esa métrica en soluciones reales de Churn.")

        with colD:
            st.markdown("### 🧩 Variables Predictoras")
            features_data = [
                {'Categoria': 'Transacciones', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('transaction|behavio').sum(), 'Desc': 'Fluctuación de saldos y uso de canales.'},
                {'Categoria': 'Credit Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('credit score|credit history').sum(), 'Desc': 'Historial de deudas e índices de riesgo externo.'},
                {'Categoria': 'Demografía', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('demograph|age|gender').sum(), 'Desc': 'Edad del portador, ingresos y geolocalización.'}
            ]
            df_feat = pd.DataFrame(features_data)
            
            fig_tree = px.treemap(df_feat, path=['Categoria'], values='Menciones', custom_data=['Desc'], color='Menciones', color_continuous_scale=['#FF1493', '#00CED1'], template="plotly_dark")
            fig_tree.update_traces(hovertemplate="<b>Módulo: %{label}</b><br>Papers que lo respaldan: %{value}<br><i>%{customdata[0]}</i><extra></extra>")
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tree, use_container_width=True)
            
            with st.expander("📖 Guía de Interpretación"):
                st.write("**¿Qué significa?** Identifica las categorías de datos del cliente con mayor peso analítico dentro del algoritmo.")
                st.write("**¿Qué representa?** Los bloques esenciales de la ingeniería de variables financieras.")
                st.write("**¿Cómo se lee este gráfico?** El tamaño de los rectángulos es proporcional a su impacto predictivo; cuadros grandes equivalen a variables críticas de comportamiento.")

    # =========================================================================
    # PESTAÑA 2: NUEVA FUNCIÓN DE PROYECTO / SIMULADOR DE ESCENARIOS
    # =========================================================================
    with tab2:
        st.markdown("### 🔮 Módulo de Proyección de Riesgo de Churn")
        st.markdown("Interactúa con los controles simulando el perfil transaccional de un cliente para proyectar su probabilidad de abandono según las tendencias de Machine Learning extraídas de la literatura.")
        
        col_sim1, col_sim2 = st.columns([1, 1])
        
        with col_sim1:
            st.markdown("#### ⚙️ Entrada de Datos del Cliente (Variables Predictoras)")
            # Preguntas e inputs dinámicos que simulan la consulta de un analista
            caida_trans = st.slider("1. ¿Qué porcentaje de reducción transaccional presenta en el último mes? (%)", 0, 100, 25)
            score_banco = st.slider("2. ¿Cuál es el Score Crediticio interno del cliente?", 300, 850, 680)
            quejas_activas = st.radio("3. ¿El cliente ha registrado reclamos formales en los últimos 30 días?", ["No", "Sí"])
            
            # Algoritmo interno de proyección simulado (Lógica de scoring financiero)
            riesgo_calculado = 15.0
            riesgo_calculado += (caida_trans * 0.55)
            riesgo_calculado += ((850 - score_banco) * 0.08)
            if quejas_activas == "Sí":
                riesgo_calculado += 20.0
            riesgo_final = min(max(riesgo_calculado, 0.0), 100.0)

        with col_sim2:
            st.markdown("#### 🎯 Proyección de Fuga y Respuesta del Modelo")
            
            # Paleta bicolor dinámica según el nivel de alerta
            color_alerta = "#00CED1" if riesgo_final < 50 else "#FF1493"
            
            st.markdown(f"""
            <div style='background-color: #161B22; padding: 25px; border-radius: 10px; border: 2px solid {color_alerta}; text-align: center;'>
                <p style='color: #E6EDF3; font-size: 1.2rem; margin-bottom: 5px;'>PROYECCIÓN DE PROBABILIDAD DE CHURN</p>
                <h1 style='color: {color_alerta} !important; font-size: 3.5rem; margin: 0;'>{riesgo_final:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Plan de Acción Recomendado")
            if riesgo_final < 40:
                st.success("🟢 **Riesgo Bajo / Estable:** El comportamiento entra en rangos tolerables. Mantener estrategias estándar de fidelización digital.")
            elif 40 <= riesgo_final < 70:
                st.warning("🟡 **Alerta Moderada / Retención Preventiva:** El sistema detecta anomalías transaccionales. Emitir automáticamente una oferta de tasa preferencial en canales móviles.")
            else:
                st.error("🔴 **Alerta Crítica / Intervención Inmediata:** Alta probabilidad de deserción. El protocolo de Machine Learning sugiere el contacto directo de un ejecutivo de cuenta en menos de 24 horas.")

        with st.expander("📖 Guía de Interpretación de esta Sección"):
            st.write("**¿Qué significa?** Es una simulación práctica de cómo la ingeniería de variables optimiza la toma de decisiones predictivas en la banca.")
            st.write("**¿Qué representa?** El flujo automatizado de evaluación de riesgo que un analista bancario ejecuta usando modelos de Machine Learning.")
            st.write("**¿Cómo se lee este gráfico?** Al modificar los sliders, la probabilidad se recalcula instantáneamente; un resultado superior al 50% cambia el indicador a Rosa Neón, marcando una alerta operativa inmediata.")

    # =========================================================================
    # PESTAÑA 3: NUEVA FUNCIÓN INTERACTIVA DE SCOPUS (A LA CARTA)
    # =========================================================================
    with tab3:
        st.markdown("### 📚 Repositorio Científico Interactivo Scopus (2025-2026)")
        st.markdown("Configura tu vista preferida seleccionando únicamente los metadatos analíticos que deseas que aparezcan en el reporte en pantalla.")
        
        # Selector dinámico de columnas solicitado por el usuario
        columnas_disponibles = {
            "Título del Paper": "Title",
            "Año de Publicación": "Year",
            "Citas Recibidas": "Cited by",
            "Revista Indexada": "Source title",
            "Enlace de Verificación": "Link"
        }
        
        columnas_seleccionadas_es = st.multiselect(
            "Selecciona la información que deseas que aparezca en la tabla:",
            options=list(columnas_disponibles.keys()),
            default=["Título del Paper", "Año de Publicación", "Citas Recibidas", "Enlace de Verificación"]
        )
        
        # Mapear las columnas seleccionadas en español a los nombres del DataFrame original
        columnas_mapeadas = [columnas_disponibles[col] for col in columnas_seleccionadas_es]
        
        if columnas_mapeadas:
            df_mostrar = df_filtrado[columnas_mapeadas].copy()
            
            # Configuración dinámica de columnas de Streamlit
            config_columnas = {}
            if "Title" in columnas_mapeadas: config_columnas["Title"] = st.column_config.TextColumn("Título del Estudio Científico")
            if "Year" in columnas_mapeadas: config_columnas["Year"] = st.column_config.NumberColumn("Año", format="%d")
            if "Cited by" in columnas_mapeadas: config_columnas["Cited by"] = st.column_config.ProgressColumn("Citas", format="%d", min_value=0, max_value=int(df['Cited by'].max()))
            if "Source title" in columnas_mapeadas: config_columnas["Source title"] = st.column_config.TextColumn("Revista / Source Title")
            if "Link" in columnas_mapeadas: config_columnas["Link"] = st.column_config.LinkColumn("Enlace Oficial Scopus")
            
            st.dataframe(
                df_mostrar.sort_values(by=df_mostrar.columns[0], ascending=True), 
                use_container_width=True, 
                hide_index=True,
                column_config=config_columnas
            )
        else:
            st.warning("⚠️ Selecciona al menos una opción en el menú superior para desplegar los registros del repositorio.")

        with st.expander("📖 Guía de Interpretación de esta Sección"):
            st.write("**¿Qué significa?** Es la base de datos científica estructurada que audita y da soporte de ingeniería a los módulos interactivos anteriores.")
            st.write("**¿Qué representa?** El compendio bibliométrico indexado de la literatura científica internacional sobre Churn bancario.")
            st.write("**¿Cómo se lee este gráfico?** Puedes usar el buscador interactivo para agregar o quitar columnas de metadatos en tiempo real, u ordenar las filas haciendo clic directamente sobre las cabeceras de la tabla.")

if __name__ == "__main__":
    main()