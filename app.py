import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURACIÓN DEL ENTORNO EMPRESARIAL ---
st.set_page_config(
    page_title="ChurnAI Horizon - Enterprise Hub", 
    page_icon="🔮", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo CSS Avanzado para romper el molde estático de Streamlit
st.markdown("""
    <style>
        .reportview-container { background: #0E1117; }
        .stMarkdown { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        h1, h2, h3 { color: #00CED1 !important; }
        div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; }
        /* Efecto tarjeta para Top Papers */
        .paper-card {
            background-color: #161B22;
            padding: 15px;
            border-radius: 8px;
            border-top: 4px solid #00CED1;
            margin-bottom: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE PROCESAMIENTO DE DATOS ---
@st.cache_data
def load_and_process_data():
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

    # --- BARRA LATERAL (Panel de Control Avanzado) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=50)
        st.title("Control de Datos")
        st.markdown("---")
        
        busqueda = st.text_input("🔍 Concepto Clave o Algoritmo (ej. Neural, XGBoost):", "")
        max_citas_posibles = int(df['Cited by'].max()) if len(df) > 0 else 100
        min_citas = st.slider("📈 Rigor del Filtro (Mínimo de Citas):", 0, max_citas_posibles, 0)
        
        st.markdown("---")
        st.caption("⚡ Powered by ChurnAI Engine v3.5 • Mercado Peruano 2026")

    # Filtrado Dinámico Global
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO CORPORATIVO ---
    st.title("🔮 ChurnAI Horizon")
    st.subheader("Plataforma Inteligente de Exploración de Datos y Simulación Predictiva Bancaria")
    
    st.markdown("""
    <div style='background-color: #161B22; padding: 20px; border-radius: 8px; border-left: 5px solid #FF1493; margin-bottom: 25px;'>
        <h4 style='color: #FF1493; margin-top:0; margin-bottom:5px;'>📌 ESTRATEGIA CENTRAL DE RETENCIÓN</h4>
        <p style='color: #E6EDF3; font-size: 1.05rem; line-height: 1.5; margin:0;'>
            <b>¿Cómo optimiza el uso de machine learning la predicción de la fuga de clientes en el sector bancario?</b><br>
            La optimización se ejecuta mediante el análisis dinámico de comportamiento. Al interconectar la Big Data de Scopus con disparadores transaccionales locales (Yape, Plin, CTS y variaciones del Score SBS), la IA automatiza la toma de decisiones críticas para congelar la fuga de capitales antes de que el cliente abandone la entidad.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- CÁLCULO DOCK DE PESOS (MOTOR CONECTOR DE LA IA) ---
    menciones_trans = df_filtrado['Abstract_Clean'].str.contains('transaction|behavio|digital|channel').sum()
    menciones_score = df_filtrado['Abstract_Clean'].str.contains('credit score|credit history|credit|risk|sbs').sum()
    menciones_demo  = df_filtrado['Abstract_Clean'].str.contains('demograph|age|gender|income|status').sum()
    
    total_menciones = menciones_trans + menciones_score + menciones_demo
    peso_trans, peso_score, peso_demo = (menciones_trans / total_menciones, menciones_score / total_menciones, menciones_demo / total_menciones) if total_menciones > 0 else (0.50, 0.30, 0.20)

    # --- ESTRUCTURACIÓN DE PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Explorador de Tendencias e Impacto", 
        "🔮 Simulador Financiero Conectado (Bancos Perú)", 
        "📚 Centro de Datos e Insights"
    ])

    # =========================================================================
    # PESTAÑA 1: REEMPLAZOS DE GRÁFICOS (DE ACADÉMICO A STARTUP)
    # =========================================================================
    with tab1:
        st.markdown("### 🧬 Universo de Datos de la Literatura Científica")
        
        # GRID INICIAL: Ránking de Métricas + Violín de Distribución
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### 🎯 Ranking de Rendimiento de Métricas")
            metricas_data = [
                {'Métrica': 'Accuracy', 'Papers': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum()},
                {'Métrica': 'F1-Score', 'Papers': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum()},
                {'Métrica': 'AUC-ROC', 'Papers': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum()}
            ]
            df_m = pd.DataFrame(metricas_data).sort_values(by="Papers", ascending=True)
            
            # Reemplazo del Radar por un Horizontal Bar Chart limpio
            fig_bar = px.bar(df_m, x="Papers", y="Métrica", orientation="h", template="plotly_dark", color="Papers", color_continuous_scale=["#FF1493", "#00CED1"])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info(f"📌 **Qué significa esto:** El gráfico jerarquiza qué indicadores matemáticos eligen los científicos de datos globales para validar modelos de fuga bancaria.")

        with col_g2:
            st.markdown("#### 📈 Distribución Temporal de Citaciones Científicas")
            # Reemplazo del Boxplot por un Violin Explorer con Scatter integrado
            fig_violin = px.violin(df_filtrado, x="Year", y="Cited by", box=True, points="all", template="plotly_dark", color_discrete_sequence=["#00CED1"])
            fig_violin.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_violin, use_container_width=True)
            
            st.info(f"🧠 **Lo que está pasando en los datos:** Cada punto representa un paper indexado. El ancho del violín te muestra las épocas con mayor concentración de publicaciones de alto impacto.")

        st.markdown("---")
        
        # SEGUNDO BLOQUE REEMPLAZO: El Scatter Bubble Intelligence Map (Reemplaza al Treemap)
        st.markdown("#### 🧩 Mapa de Relación y Relevancia de Variables Financieras (Scatter Intelligence Map)")
        
        features_plot_data = []
        for year in sorted(df_filtrado['Year'].unique()):
            df_year = df_filtrado[df_filtrado['Year'] == year]
            features_plot_data.append({'Año': year, 'Categoría': 'Transacciones e Interactividad', 'Menciones': df_year['Abstract_Clean'].str.contains('transaction|behavio|digital|channel').sum()})
            features_plot_data.append({'Año': year, 'Categoría': 'Historial Crediticio', 'Menciones': df_year['Abstract_Clean'].str.contains('credit score|credit history|credit|risk|sbs').sum()})
            features_plot_data.append({'Año': year, 'Categoría': 'Datos Demográficos', 'Menciones': df_year['Abstract_Clean'].str.contains('demograph|age|gender|income|status').sum()})
        
        df_fp = pd.DataFrame(features_plot_data)
        
        fig_scatter = px.scatter(df_fp, x="Año", y="Menciones", size="Menciones", color="Categoría", color_discrete_sequence=["#FF1493", "#00CED1", "#FFFF00"], template="plotly_dark", hover_name="Categoría")
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.success("💡 **Insight automático:** Mueve los filtros de la barra izquierda; el mapa de burbujas recalculará de forma instantánea el volumen e impacto temporal de cada variable predictora.")

    # =========================================================================
    # PESTAÑA 2: SIMULADOR FINANCIERO PERÚ (100% CONECTADO)
    # =========================================================================
    with tab2:
        st.markdown("### 🔮 Motor de Simulación de Riesgo Bancario Local")
        
        # Notificación con estilo SaaS de la interconexión de datos
        st.info(f"🔗 **Estatus del Motor:** Sincronizado de forma activa con {len(df_filtrado)} papers científicos indexados en Scopus. "
                f"Distribución de coeficientes en tiempo real: 📱 Digital: {peso_trans*100:.1f}% | 💳 Crédito: {peso_score*100:.1f}% | 👤 Perfil: {peso_demo*100:.1f}%")
        
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.markdown("#### ⚙️ Entrada del Perfil Transaccional")
            banco_seleccionado = st.selectbox("Selecciona la entidad a evaluar dentro del ecosistema nacional:", ["Banco de Crédito del Perú (BCP)", "BBVA Perú", "Interbank", "Scotiabank Perú"])
            
            # Parámetros adaptados a la infraestructura peruana
            caida_trans = st.slider("1. Contracción mensual en canales de pago móviles (Yape / Plin, transferencias interbancarias CCE) (%):", 0, 100, 30)
            score_sbs = st.slider("2. Calificación del Score Crediticio interno del usuario (Sentinel / SBS / Equifax):", 300, 850, 710)
            portabilidad_sueldo = st.radio("3. ¿Registra alertas de portabilidad de Cuenta Sueldo o retiro del fondo de CTS?", ["No", "Sí"])
            
            # Algoritmo de scoring dinámico e interconectado
            score_ponderado = 12.0
            score_ponderado += (caida_trans * (peso_trans * 1.2))
            score_ponderado += ((850 - score_sbs) * (peso_score * 0.15))
            if portabilidad_sueldo == "Sí":
                score_ponderado += (28.0 * (peso_demo + 0.4))
                
            if "BCP" in banco_seleccionado: score_ponderado += 1.5
            elif "BBVA" in banco_seleccionado: score_ponderado -= 0.5
            
            riesgo_final = min(max(score_ponderado, 0.0), 100.0)

        with col_sim2:
            st.markdown(f"#### 🎯 Diagnóstico Operativo ({banco_seleccionado})")
            color_alerta = "#00CED1" if riesgo_final < 50 else "#FF1493"
            
            st.markdown(f"""
            <div style='background-color: #161B22; padding: 25px; border-radius: 10px; border: 2px solid {color_alerta}; text-align: center;'>
                <p style='color: #E6EDF3; font-size: 1.1rem; margin-bottom: 5px; letter-spacing: 1px;'>RIESGO ESTIMADO DE ABANDONO DE CUENTA</p>
                <h1 style='color: {color_alerta} !important; font-size: 3.8rem; margin: 0;'>{riesgo_final:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Plan de Acción Táctico de Retención")
            
            if riesgo_final < 40:
                st.success("🟢 **Zona Segura:** El cliente se mantiene fidelizado. Desplegar campañas pasivas de cross-selling (Millas LATAM Pass, Puntos BBVA o Beneficios Interbank Benefit).")
            elif 40 <= riesgo_final < 70:
                st.warning("🟡 **Retención Preventiva:** Descenso inusual de actividad digital. El sistema recomienda la exoneración proactiva de la membresía anual o la habilitación de una campaña preferencial de compra de deuda externa.")
            else:
                st.error("🔴 **Intervención Inmediata / Alerta Crítica:** Fuga inminente de haberes. El protocolo bancario exige asignar el caso de forma prioritaria a un asesor Élite de telemarketing para negociar una contraoferta en su tasa de interés o beneficios en Cuenta Sueldo en menos de 12 horas.")

    # =========================================================================
    # PESTAÑA 3: TOP PAPERS ESTILO NETFLIX / CARDS VISUALES
    # =========================================================================
    with tab3:
        st.markdown("### 📚 Centro de Inteligencia y Auditoría Bibliométrica")
        st.markdown("A continuación, se exponen los papers científicos de mayor impacto que respaldan las lógicas predictivas configuradas en las secciones anteriores.")
        
        st.markdown("#### 🏆 Top 3 Papers Más Citados (Estructura de Tarjetas Avanzadas)")
        
        # Extracción dinámica de las tarjetas tipo Spotify/Netflix
        top_papers = df_filtrado.sort_values(by="Cited by", ascending=False).head(3)
        
        if len(top_papers) > 0:
            col_card1, col_card2, col_card3 = st.columns(3)
            columnas_cards = [col_card1, col_card2, col_card3]
            
            for idx, (_, row) in enumerate(top_papers.iterrows()):
                with columnas_cards[idx]:
                    st.markdown(f"""
                    <div class="paper-card">
                        <span style="color: #8B949E; font-size: 0.8rem; font-weight: bold; text-transform: uppercase;">🔥 RELEVANCIA ALTA</span>
                        <h4 style="margin: 8px 0; color: #FFF !important; font-size: 1rem; line-height: 1.4;">{row['Title'][:80]}...</h4>
                        <p style="margin: 0; color: #8B949E; font-size: 0.85rem;">Año de publicación: <b>{row['Year']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric(label="Total de Citas en Scopus", value=int(row['Cited by']))
        else:
            st.warning("⚠️ No se registran publicaciones que cumplan con los criterios mínimos de citas establecidos en los filtros.")

        st.markdown("---")
        
        # =========================================================================
        # REEMPLAZO OPTIMIZADO: MAPA DE RELACIÓN Y RELEVANCIA DE VARIABLES (INDIVIDUAL)
        # =========================================================================
        st.markdown("#### 🧩 Mapa de Relación y Relevancia de Variables Financieras (Scatter Intelligence Map)")
        st.markdown("Cada burbuja representa un **estudio científico individual**. Su posición mapea el año de publicación frente a su dimensión predictiva principal; el tamaño de la burbuja es proporcional a su impacto global (citas).")
        
        if len(df_filtrado) > 0:
            paper_features = []
            
            # Clasificación analítica individual por cada registro del Data Lake
            for _, row in df_filtrado.iterrows():
                txt = row['Abstract_Clean'] + " " + str(row['Title']).lower()
                c_trans = txt.count('transaction') + txt.count('behavio') + txt.count('digital') + txt.count('channel') + txt.count('yape') + txt.count('plin')
                c_score = txt.count('credit') + txt.count('score') + txt.count('history') + txt.count('risk') + txt.count('sbs') + txt.count('infocorp')
                c_demo = txt.count('demograph') + txt.count('age') + txt.count('gender') + txt.count('income') + txt.count('status') + txt.count('sueldo')
                
                max_val = max(c_trans, c_score, c_demo)
                if max_val == 0:
                    dominant = "⚙️ Enfoque General / Algorítmico"
                elif max_val == c_trans:
                    dominant = "📱 Transacciones e Interactividad"
                elif max_val == c_score:
                    dominant = "💳 Historial Crediticio (SBS)"
                else:
                    dominant = "👤 Datos Demográficos y Perfil"
                    
                paper_features.append({
                    'Título': row['Title'],
                    'Año': row['Year'],
                    'Citas': int(row['Cited by']),
                    'Revista': row['Source title'],
                    'Variable Dominante': dominant,
                    'Tamaño_Visual': int(row['Cited by']) + 12  # Baseline dinámico para que papers con 0 citas sean interactivos
                })
            
            df_pf = pd.DataFrame(paper_features)
            
            # Mapeo estricto de la paleta de colores Fintech (Look de Startup)
            color_map = {
                "📱 Transacciones e Interactividad": "#00CED1",       # Azul Cian
                "💳 Historial Crediticio (SBS)": "#FF1493",          # Rosa Neón
                "👤 Datos Demográficos y Perfil": "#FFFF00",         # Amarillo Neón
                "⚙️ Enfoque General / Algorítmico": "#8B949E"         # Gris Metálico
            }
            
            # Construcción de la matriz temporal de burbujas
            fig_scatter = px.scatter(
                df_pf, 
                x="Año", 
                y="Variable Dominante", 
                size="Tamaño_Visual", 
                color="Variable Dominante",
                color_discrete_map=color_map,
                template="plotly_dark", 
                hover_name="Título",
                custom_data=["Citas", "Revista"]
            )
            
            # Formateo de Hover con Inteligencia de Producto (Elimina etiquetas por defecto aburridas)
            fig_scatter.update_traces(
                hovertemplate="<b>📈 %{hovertext}</b><br><br>" +
                              "📅 <b>Año de Publicación:</b> %{x}<br>" +
                              "🔥 <b>Citas Globales (Scopus):</b> %{customdata[0]}<br>" +
                              "📚 <b>Revista Indexada:</b> %{customdata[1]}<extra></extra>"
            )
            
            # Estilización del layout para conservar la estética Dark Mode Estricta
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Línea de Tiempo de la Literatura", gridcolor="#21262D", tickmode="linear"),
                yaxis=dict(title="Dimensión Crítica del Modelo", gridcolor="#21262D", autorange="reversed"),
                showlegend=False,
                height=450
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.success("💡 **Insight Automático:** Explora el mapa interactivo pasando el cursor sobre las burbujas. Los cúmulos densos revelan hacia dónde se está moviendo la arquitectura de riesgos en la banca actual.")
        else:
            st.warning("⚠️ Filtros muy restrictivos. No hay suficientes registros científicos para trazar el Scatter Map.")