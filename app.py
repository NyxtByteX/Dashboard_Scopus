import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURACIÓN DEL ENTORNO EMPRESARIAL ---
st.set_page_config(
    page_title="ChurnAI Analytics Hub - Perú", 
    page_icon="🔮", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para la estética Dark Mode Estricta (Cian y Rosa Neón)
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
    # Carga de la base de datos Scopus entregada
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    df['Abstract_Clean'] = df['Abstract'].fillna('').str.lower()
    return df

def main():
    try:
        df = load_and_process_data()
    except Exception:
        st.error("🚨 Error crítico: No se encontró la base de datos de Scopus 'scopus_PA3.csv' en la raíz.")
        return

    # --- BARRA LATERAL (Filtros Globales que controlan TODO el sistema) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=50)
        st.title("Filtros Globales")
        st.markdown("---")
        
        busqueda = st.text_input("🔍 Filtrar papers por término técnico:", "")
        max_citas_posibles = int(df['Cited by'].max()) if len(df) > 0 else 100
        min_citas = st.slider("📈 Nivel de rigor científico (Mínimo citas):", 0, max_citas_posibles, 0)
        
        st.markdown("---")
        st.caption("🔒 ChurnAI Enterprise v3.0. Modelo Predictivo Conectado 2025-2026.")

    # APLICACIÓN DE FILTRADO GLOBAL DE INFORMACIÓN
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO CORPORATIVO ---
    st.title("🔮 ChurnAI Analytics Hub")
    st.subheader("Plataforma de Simulación Científica de Churn Conectada para la Banca Peruana")
    
    st.markdown("""
    <div style='background-color: #161B22; padding: 20px; border-radius: 8px; border-left: 5px solid #FF1493; margin-bottom: 25px;'>
        <h4 style='color: #FF1493; margin-top:0;'>📌 ESTRATEGIA CENTRAL DE INVESTIGACIÓN (PERÚ 2025-2026)</h4>
        <p style='color: #E6EDF3; font-size: 1.05rem; line-height: 1.6;'>
            <b>¿Cómo optimiza el uso de machine learning la predicción de la fuga de clientes (customer churn) en el sector bancario?</b><br>
            La optimización se logra al procesar patrones transaccionales en tiempo real. En el mercado peruano, este Hub conecta la evidencia de la literatura internacional indexada en Scopus con los disparadores críticos locales (billeteras digitales como Yape/Plin, portabilidades de Cuenta Sueldo y deudas reportadas en la SBS).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- CÁLCULO EN TIEMPO REAL DE LA RELEVANCIA CIENTÍFICA (EL CONECTOR DEL SISTEMA) ---
    menciones_trans = df_filtrado['Abstract_Clean'].str.contains('transaction|behavio|digital|channel').sum()
    menciones_score = df_filtrado['Abstract_Clean'].str.contains('credit score|credit history|credit|risk|sbs').sum()
    menciones_demo  = df_filtrado['Abstract_Clean'].str.contains('demograph|age|gender|income|status').sum()
    
    total_menciones = menciones_trans + menciones_score + menciones_demo
    
    # Pesos dinámicos basados en la literatura filtrada (si no hay datos, se asigna un baseline balanceado)
    if total_menciones > 0:
        peso_trans_raw = menciones_trans / total_menciones
        peso_score_raw = menciones_score / total_menciones
        peso_demo_raw  = menciones_demo / total_menciones
    else:
        peso_trans_raw, peso_score_raw, peso_demo_raw = 0.50, 0.30, 0.20

    # Normalización para la fórmula del simulador
    w_trans = peso_trans_raw * 0.8
    w_score = peso_score_raw * 0.8
    w_demo  = peso_demo_raw * 0.8

    # --- PESTAÑAS OPERATIVAS ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Métricas y Variables de la Industria", 
        "🔮 Simulador de Riesgo Conectado (Bancos Perú)", 
        "📚 Repositorio Scopus Inteligente"
    ])

    # =========================================================================
    # PESTAÑA 1: GRÁFICOS ANALÍTICOS DINÁMICOS
    # =========================================================================
    with tab1:
        st.markdown("### 🧬 Diagnóstico de Tendencias Científicas Actuales")
        st.markdown("Los gráficos a continuación reflejan de forma dinámica el peso del conocimiento científico extraído de los papers según los filtros actuales.")
        
        colC, colD = st.columns(2)
        
        with colC:
            st.markdown("#### 🎯 Métricas de Evaluación Preferidas")
            metricas_data = [
                {'Metrica': 'Accuracy', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum(), 'Desc': 'Porcentaje general de aciertos del modelo.'},
                {'Metrica': 'F1-Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum(), 'Desc': 'Equilibrio crítico entre falsos positivos y falsos negativos.'},
                {'Metrica': 'AUC-ROC', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum(), 'Desc': 'Capacidad del modelo para discriminar entre un cliente leal y uno en riesgo.'}
            ]
            df_metrics = pd.DataFrame(metricas_data)
            
            fig_radar = px.line_polar(df_metrics, r='Menciones', theta='Metrica', line_close=True, custom_data=['Desc'], template="plotly_dark")
            fig_radar.update_traces(fill='toself', line_color='#00CED1', hovertemplate="<b>Métrica: %{theta}</b><br>Menciones: %{r}<br><i>%{customdata[0]}</i><extra></extra>")
            fig_radar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, use_container_width=True)
            
            with st.expander("📖 Guía de Interpretación"):
                st.write("**¿Qué significa?** Revela los estándares matemáticos preferidos por la ciencia de datos para validar modelos bancarios.")
                st.write("**¿Qué representa?** El área de cobertura del mapa de radar.")
                st.write("**¿Cómo se lee esta sección?** Entre más lejano esté el vértice del centro, mayor es la adopción de esa métrica en soluciones reales de Churn.")

        with colD:
            st.markdown("#### 🧩 Distribución de Impacto de las Variables (Driving Forces)")
            features_data = [
                {'Categoria': 'Uso de Canales y Transacciones', 'Menciones': menciones_trans, 'Desc': 'Fluctuación de saldos en cuentas y uso de canales digitales.'},
                {'Categoria': 'Comportamiento Crediticio', 'Menciones': menciones_score, 'Desc': 'Evolución del historial crediticio y deudas en el sistema.'},
                {'Categoria': 'Perfil Demográfico y Estado', 'Menciones': menciones_demo, 'Desc': 'Atributos base del usuario y tipos de cuenta contratados.'}
            ]
            df_feat = pd.DataFrame(features_data)
            
            fig_tree = px.treemap(df_feat, path=['Categoria'], values='Menciones', custom_data=['Desc'], color='Menciones', color_continuous_scale=['#FF1493', '#00CED1'], template="plotly_dark")
            fig_tree.update_traces(hovertemplate="<b>Módulo: %{label}</b><br>Papers que lo respaldan: %{value}<br><i>%{customdata[0]}</i><extra></extra>")
            fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tree, use_container_width=True)
            
            with st.expander("📖 Guía de Interpretación"):
                st.write("**¿Qué significa?** Identifica las categorías de datos del cliente con mayor peso analítico dentro del algoritmo.")
                st.write("**¿Qué representa?** Los bloques esenciales de la ingeniería de variables financieras.")
                st.write("**¿Cómo se lee esta sección?** El tamaño de los rectángulos es proporcional a su impacto predictivo; cuadros grandes equivalen a variables críticas de comportamiento.")

    # =========================================================================
    # PESTAÑA 2: SIMULADOR DE ESCENARIOS BANCA PERÚ (CONECTADO MATEMÁTICAMENTE)
    # =========================================================================
    with tab2:
        st.markdown("### 🔮 Módulo de Proyección de Riesgo Calibrado por Evidencia Científica")
        
        # Alerta visual que demuestra la conexión real de datos
        st.info(f"🔗 **Conexión Activa:** El motor algorítmico está calculando el riesgo usando la configuración de pesos extraída de los {len(df_filtrado)} papers filtrados en este momento. "
                f"(Peso Transaccional: {w_trans*100:.1f}% | Peso Scoring: {w_score*100:.1f}% | Peso Perfil: {w_demo*100:.1f}%)")
        
        col_sim1, col_sim2 = st.columns([1, 1])
        
        with col_sim1:
            st.markdown("#### ⚙️ Variables Operativas del Cliente Peruano")
            
            banco_seleccionado = st.selectbox(
                "Selecciona el banco del ecosistema financiero local a evaluar:",
                ["Banco de Crédito del Perú (BCP)", "BBVA Perú", "Interbank", "Scotiabank Perú"]
            )
            
            # Formulación de preguntas con arraigo en la banca nacional
            caida_trans = st.slider("1. ¿Qué porcentaje de reducción presenta en su actividad digital local (Uso de Yape / Plin, transferencias interbancarias vía CCE) en el último mes? (%)", 0, 100, 25)
            score_infocorp = st.slider("2. ¿Cuál es el Score Crediticio del cliente en centrales de riesgo nacionales (Sentinel / SBS / Equifax)?", 300, 850, 680)
            portabilidad_sueldo = st.radio("3. ¿El sistema reporta solicitudes de portabilidad de Cuenta Sueldo a otro banco o retiro total de CTS?", ["No", "Sí"])
            
            # CÁLCULO DE RIESGO 100% DINÁMICO E INTERCONECTADO CON LOS GRÁFICOS
            riesgo_calculado = 10.0
            riesgo_calculado += (caida_trans * w_trans * 1.5)
            riesgo_calculado += ((850 - score_infocorp) * w_score * 0.2)
            if portabilidad_sueldo == "Sí":
                riesgo_calculado += (30.0 * (w_demo + 0.5))
            
            # Variaciones de fricción según la entidad del ecosistema nacional
            if "BCP" in banco_seleccionado:
                riesgo_calculado += 2.0  # Ajuste por alta competitividad en ecosistema Yape
            elif "BBVA" in banco_seleccionado:
                riesgo_calculado -= 1.0  # Ajuste por retención pasiva de cuentas de haberes
                
            riesgo_final = min(max(riesgo_calculado, 0.0), 100.0)

        with col_sim2:
            st.markdown(f"#### 🎯 Proyección de Fuga para {banco_seleccionado}")
            color_alerta = "#00CED1" if riesgo_final < 50 else "#FF1493"
            
            st.markdown(f"""
            <div style='background-color: #161B22; padding: 25px; border-radius: 10px; border: 2px solid {color_alerta}; text-align: center;'>
                <p style='color: #E6EDF3; font-size: 1.2rem; margin-bottom: 5px;'>PROYECCIÓN DE PROBABILIDAD DE CHURN ACTUALIZADA</p>
                <h1 style='color: {color_alerta} !important; font-size: 3.5rem; margin: 0;'>{riesgo_final:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Plan de Acción Operativo (Alineado a la Regulación SBS)")
            
            # Ejecución de planes de retención contextualizados a Perú
            if riesgo_final < 40:
                st.success(f"🟢 **Riesgo Bajo / Estable:** El cliente mantiene alta fidelidad transaccional. Continuar con pautas automáticas de venta cruzada (Campañas de Millas LATAM Pass, Puntos BBVA o Beneficios Interbank Benefit según la entidad).")
            elif 40 <= riesgo_final < 70:
                st.warning(f"🟡 **Alerta Moderada / Retención Preventiva:** Descenso marcado en canales digitales. El motor sugiere activar la exoneración automatizada de la membresía de la tarjeta de crédito u ofrecer una campaña de Tasa Preferencial de Compra de Deuda para mitigar la fuga hacia competidores.")
            else:
                st.error(f"🔴 **Alerta Crítica / Intervención Inmediata:** Alta probabilidad de pérdida del cliente. Prioridad máxima para el equipo de Telemarketing. El protocolo exige la asignación inmediata de un asesor Élite para contraofertar beneficios en la Cuenta Sueldo y retener los fondos en menos de 12 horas.")

        with st.expander("Base teórica: Explicación de las variables en el mercado peruano"):
            st.markdown("""
            * **Yape / Plin / CCE:** Herramientas fundamentales de bancarización y uso diario en Perú. La caída transaccional en estas redes es el indicador de corto plazo más potente para predecir fugas.
            * **Sentinel / Equifax / SBS:** El endeudamiento externo reportado ante la Superintendencia de Banca, Seguros y AFP (SBS) altera el score del cliente, indicando que busca productos financieros fuera de nuestra entidad.
            * **Cuenta Sueldo y CTS:** Constituyen el núcleo de la lealtad e ingresos del cliente minorista peruano. Una solicitud de portabilidad o retiro total implica una deserción inminente.
            """)

        with st.expander("📖 Guía de Interpretación de esta Sección"):
            st.write("**¿Qué significa?** Es una simulación práctica de cómo la ingeniería de variables optimiza la toma de decisiones predictivas en la banca.")
            st.write("**¿Qué representa?** El flujo automatizado de evaluación de riesgo que un analista bancario ejecuta usando modelos de Machine Learning.")
            st.write("**¿Cómo se lee esta sección?** Al modificar los sliders, la probabilidad se recalcula instantáneamente; un resultado superior al 50% cambia el indicador a Rosa Neón, marcando una alerta operativa inmediata.")

    # =========================================================================
    # PESTAÑA 3: REPOSITORIO INTERACTIVO SCOPUS (A LA CARTA)
    # =========================================================================
    with tab3:
        st.markdown("### 📚 Repositorio Científico Interactivo Scopus (2025-2026)")
        st.markdown("Configura tu vista preferida seleccionando únicamente los metadatos analíticos que deseas que aparezcan en el reporte en pantalla.")
        
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
        
        columnas_mapeadas = [columnas_disponibles[col] for col in columnas_seleccionadas_es]
        
        if columnas_mapeadas:
            df_mostrar = df_filtrado[columnas_mapeadas].copy()
            
            config_columnas = {}
            if "Title" in columnas_mapeadas: 
                config_columnas["Title"] = st.column_config.TextColumn("Título del Estudio Científico")
            if "Year" in columnas_mapeadas: 
                config_columnas["Year"] = st.column_config.NumberColumn("Año", format="%d")
            if "Cited by" in columnas_mapeadas: 
                config_columnas["Cited by"] = st.column_config.ProgressColumn("Citas", format="%d", min_value=0, max_value=max_citas_posibles)
            if "Source title" in columnas_mapeadas: 
                config_columnas["Source title"] = st.column_config.TextColumn("Revista / Source Title")
            if "Link" in columnas_mapeadas: 
                config_columnas["Link"] = st.column_config.LinkColumn("Enlace Oficial Scopus")
            
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
            st.write("**¿Cómo se lee esta sección?** Puedes usar el buscador interactivo para agregar o quitar columnas de metadatos en tiempo real, u ordenar las filas haciendo clic directamente sobre las cabeceras de la tabla.")

if __name__ == "__main__":
    main()