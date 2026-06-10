import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN DEL ENTORNO EMPRESARIAL ---
st.set_page_config(
    page_title="ChurnAI Horizon - Executive Analytics", 
    page_icon="🔮", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo CSS Avanzado para simular la interfaz oscura y estilizada de Geckoboard/Qlik
st.markdown("""
    <style>
        /* Fondo general oscuro y tipografía limpia */
        .reportview-container, .main { background: #0B0E14; }
        body { color: #E6EDF3; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        
        /* Títulos con acentos de color modernos */
        h1, h2, h3, h4 { color: #00CED1 !important; font-weight: 600 !important; }
        
        /* Contenedores de Tarjetas KPI (Estilo Geckoboard) */
        .kpi-container {
            background-color: #161B22;
            border: 1px solid #30363D;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            margin-bottom: 15px;
        }
        .kpi-title { font-size: 0.85rem; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; font-weight: bold; }
        .kpi-value { font-size: 2.2rem; color: #00CED1; font-weight: 700; margin: 10px 0 5px 0; }
        .kpi-sub { font-size: 0.8rem; color: #58A6FF; }

        /* Banner de Estrategia Central */
        .strategy-banner {
            background-color: #12161F; 
            padding: 22px; 
            border-radius: 10px; 
            border-left: 5px solid #FF1493; 
            margin-bottom: 30px;
            border-top: 1px solid #30363D;
            border-right: 1px solid #30363D;
            border-bottom: 1px solid #30363D;
        }

        /* Tarjetas de Papers (Pestaña 3) */
        .paper-card {
            background-color: #161B22;
            padding: 18px;
            border-radius: 8px;
            border-top: 4px solid #FF1493;
            margin-bottom: 12px;
            border-left: 1px solid #30363D;
            border-right: 1px solid #30363D;
            border-bottom: 1px solid #30363D;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE PROCESAMIENTO DE DATOS ---
@st.cache_data
def load_and_process_data():
    # Intenta cargar la base de datos de Scopus
    df = pd.read_csv('scopus_PA3.csv')
    df['Title'] = df['Title'].fillna('Untitled Paper')
    df['Source title'] = df['Source title'].fillna('Unknown Source')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    df['Abstract_Clean'] = df['Abstract'].fillna('').str.lower()
    return df

def main():
    try:
        df = load_and_process_data()
    except Exception:
        st.error("🚨 Error crítico: No se encontró la base de datos de Scopus 'scopus_PA3.csv' en la raíz de la aplicación.")
        return

    # --- BARRA LATERAL (Filtros de Control) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=50)
        st.title("Control de Datos")
        st.markdown("---")
        
        busqueda = st.text_input("🔍 Concepto Clave o Algoritmo (ej. Neural, XGBoost):", "")
        max_citas_posibles = int(df['Cited by'].max()) if len(df) > 0 else 100
        min_citas = st.slider("📈 Rigor del Filtro (Mínimo de Citas):", 0, max_citas_posibles, 0)
        
        st.markdown("---")
        st.caption("⚡ Powered by ChurnAI Engine v3.5 • Mercado Peruano 2026")

    # Filtrado Dinámico de Datos Basado en la Interacción del Usuario
    df_filtrado = df[df['Cited by'] >= min_citas].copy()
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO CORPORATIVO ---
    st.title("🔮 ChurnAI Horizon Dashboard")
    st.markdown("<p style='color:#8B949E; font-size:1.1rem; margin-top:-10px;'>Plataforma Ejecutiva de Inteligencia Analítica Aplicada al Riesgo Financiero</p>", unsafe_allow_html=True)
    
    # --- CÁLCULO DE PESOS PARA EL MOTOR CONECTOR ---
    menciones_trans = df_filtrado['Abstract_Clean'].str.contains('transaction|behavio|digital|channel').sum()
    menciones_score = df_filtrado['Abstract_Clean'].str.contains('credit score|credit history|credit|risk|sbs').sum()
    menciones_demo  = df_filtrado['Abstract_Clean'].str.contains('demograph|age|gender|income|status').sum()
    
    total_menciones = menciones_trans + menciones_score + menciones_demo
    peso_trans, peso_score, peso_demo = (menciones_trans / total_menciones, menciones_score / total_menciones, menciones_demo / total_menciones) if total_menciones > 0 else (0.50, 0.30, 0.20)

    # --- ESTRUCTURACIÓN DE PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard de Control e Impacto", 
        "🔮 Simulador Financiero Conectado (Bancos Perú)", 
        "📚 Centro de Datos e Insights"
    ])

    # =========================================================================
    # PESTAÑA 1: REDISEÑO COMPLETO ESTILO GECKBOARD / QLIK FINANCE
    # =========================================================================
    with tab1:
        # 1. FILA DE TARJETAS KPI SUPERIORES (Métricas del Documento)
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        with col_kpi1:
            total_papers = len(df_filtrado)
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Volumen de Literatura</div>
                    <div class="kpi-value">{total_papers}</div>
                    <div class="kpi-sub">Estudios Científicos Filtrados</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_kpi2:
            total_citas = df_filtrado['Cited by'].sum()
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Impacto Global</div>
                    <div class="kpi-value">{total_citas:,}</div>
                    <div class="kpi-sub">Citas Totales en Scopus</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_kpi3:
            max_citas = df_filtrado['Cited by'].max() if len(df_filtrado) > 0 else 0
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Récord de Relevancia</div>
                    <div class="kpi-value">{max_citas}</div>
                    <div class="kpi-sub font-weight-bold">Máximo de Citas en un Paper</div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_kpi4:
            promedio_citas = df_filtrado['Cited by'].mean() if len(df_filtrado) > 0 else 0
            st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Densidad Científica</div>
                    <div class="kpi-value">{promedio_citas:.1f}</div>
                    <div class="kpi-sub">Promedio de Citas por Registro</div>
                </div>
            """, unsafe_allow_html=True)

        # 2. SECCIÓN ESTRATÉGICA CENTRAL
        st.markdown(f"""
        <div class="strategy-banner">
            <h4 style='color: #FF1493; margin-top:0; margin-bottom:8px;'>📌 PREGUNTA DE INVESTIGACIÓN Y ENFOQUE ESTRATÉGICO</h4>
            <p style='color: #E6EDF3; font-size: 1.05rem; line-height: 1.5; margin:0;'>
                <b>¿Cómo optimiza el uso de machine learning la predicción de la fuga de clientes en el sector bancario?</b><br>
                La optimización se ejecuta mediante el análisis dinámico de comportamiento. Al interconectar la Big Data de Scopus con disparadores transaccionales locales (Yape, Plin, CTS y variaciones del Score SBS), la IA automatiza la toma de decisiones críticas para congelar la fuga de capitales antes de que el cliente abandone la entidad.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 3. FILA DE GRÁFICOS PRINCIPALES (Estructura en paralelo de alto rendimiento)
        col_g1, col_g2 = st.columns([1, 1])
        
        with col_g1:
            st.markdown("#### 🎯 Predominancia de Métricas Matemáticas de Validación")
            metricas_data = [
                {'Métrica': 'Accuracy', 'Papers': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum()},
                {'Métrica': 'F1-Score', 'Papers': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum()},
                {'Métrica': 'AUC-ROC', 'Papers': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum()}
            ]
            df_m = pd.DataFrame(metricas_data).sort_values(by="Papers", ascending=True)
            
            # Gráfico de Barras Horizontal Corporativo
            fig_bar = px.bar(
                df_m, x="Papers", y="Métrica", orientation="h", 
                template="plotly_dark", color="Papers", 
                color_continuous_scale=["#FF1493", "#00CED1"]
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#1F2937"), yaxis=dict(gridcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Muestra la distribución de los KPI de error/acierto seleccionados por investigadores a nivel mundial para validar modelos predictivos bancarios.")

        with col_g2:
            st.markdown("#### 📈 Distribución de Madurez Tecnológica e Impacto (Anual)")
            
            # Gráfico de Violín con visualización de densidad de datos limpia
            fig_violin = px.violin(
                df_filtrado, x="Year", y="Cited by", box=True, 
                points="all", template="plotly_dark", 
                color_discrete_sequence=["#00CED1"]
            )
            fig_violin.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Año de Publicación", gridcolor="#1F2937"),
                yaxis=dict(title="Volumen de Citaciones", gridcolor="#1F2937")
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            st.caption("Cada punto representa una investigación científica. El ancho representa la densidad de publicaciones en ese periodo.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 4. GRÁFICA DE TENDENCIAS CONTINUAS (Reemplazo del Scatter inestable)
        st.markdown("#### 📈 Evolución Histórica de Dimensiones Críticas de Entrada (Trendline Analysis)")
        
        if len(df_filtrado) > 0:
            text_comb = df_filtrado['Abstract_Clean'] + " " + df_filtrado['Title'].str.lower()
            
            c_trans = text_comb.str.contains('transaction|behavio|digital|channel|yape|plin').astype(int)
            c_score = text_comb.str.contains('credit|score|history|risk|sbs|infocorp').astype(int)
            c_demo = text_comb.str.contains('demograph|age|gender|income|status|sueldo').astype(int)
            
            df_trends = pd.DataFrame({
                'Año': df_filtrado['Year'],
                '📱 Transacciones e Interactividad': c_trans,
                '💳 Historial Crediticio (SBS)': c_score,
                '👤 Datos Demográficos y Perfil': c_demo
            })
            
            df_trends_grouped = df_trends.groupby('Año').sum().reset_index()
            df_melted = df_trends_grouped.melt(id_vars='Año', var_name='Dimensión Crítica', value_name='Cantidad de Investigaciones')
            
            color_map = {
                "📱 Transacciones e Interactividad": "#00CED1",
                "💳 Historial Crediticio (SBS)": "#FF1493",
                "👤 Datos Demográficos y Perfil": "#FFFF00"
            }
            
            fig_line = px.line(
                df_melted, x="Año", y="Cantidad de Investigaciones", color="Dimensión Crítica",
                color_discrete_map=color_map, template="plotly_dark", markers=True
            )
            
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Línea de Tiempo (Años)", gridcolor="#21262D", tickmode="linear"),
                yaxis=dict(title="Volumen de Papers Indexados", gridcolor="#21262D"),
                hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("⚠️ No hay datos suficientes con los criterios de filtrado actuales para trazar las líneas de tendencia.")

    # =========================================================================
    # PESTAÑA 2: SIMULADOR FINANCIERO (CONSERVADO COMPLETO)
    # =========================================================================
    with tab2:
        st.markdown("### 🔮 Motor de Simulación de Riesgo Bancario Local")
        
        st.info(f"🔗 **Estatus del Motor:** Sincronizado de forma activa con {len(df_filtrado)} papers científicos indexados en Scopus. "
                f"Distribución de coeficientes en tiempo real: 📱 Digital: {peso_trans*100:.1f}% | 💳 Crédito: {peso_score*100:.1f}% | 👤 Perfil: {peso_demo*100:.1f}%")
        
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.markdown("#### ⚙️ Entrada del Perfil Transaccional")
            banco_seleccionado = st.selectbox("Selecciona la entidad a evaluar dentro del ecosistema nacional:", ["Banco de Crédito del Perú (BCP)", "BBVA Perú", "Interbank", "Scotiabank Perú"])
            
            caida_trans = st.slider("1. Contracción mensual en canales de pago móviles (Yape / Plin, transferencias interbancarias CCE) (%):", 0, 100, 30)
            score_sbs = st.slider("2. Calificación del Score Crediticio interno del usuario (Sentinel / SBS / Equifax):", 300, 850, 710)
            portabilidad_sueldo = st.radio("3. ¿Registra alertas de portabilidad de Cuenta Sueldo o retiro del fondo de CTS?", ["No", "Sí"])
            
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
    # PESTAÑA 3: CENTRO DE DATOS E INSIGHTS (CONSERVADO COMPLETO)
    # =========================================================================
    with tab3:
        st.markdown("### 📚 Centro de Inteligencia y Auditoría Bibliométrica")
        st.markdown("A continuación, se exponen los papers científicos de mayor impacto que respaldan las lógicas predictivas configuradas en las secciones anteriores.")
        
        st.markdown("#### 🏆 Top 3 Papers Más Citados (Estructura de Tarjetas Advanced)")
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
        st.markdown("#### 🗂️ Data Lake Completo (Filtrado Inteligente)")
        
        df_tabla = df_filtrado[["Title", "Year", "Cited by", "Source title"]].copy()
        st.dataframe(
            df_tabla.sort_values(by="Cited by", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Título del Estudio Científico"),
                "Year": st.column_config.NumberColumn("Año", format="%d"),
                "Cited by": st.column_config.NumberColumn("Citas"),
                "Source title": st.column_config.TextColumn("Revista Científica")
            }
        )

if __name__ == "__main__":
    main()