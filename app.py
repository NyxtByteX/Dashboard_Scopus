import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURACIÓN DEL ENTORNO EMPRESARIAL ---
st.set_page_config(
    page_title="ChurnAI Horizon - Executive Analytics", 
    page_icon="🔮", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo CSS Avanzado (Estilo Geckoboard / Qlik Finance)
st.markdown("""
    <style>
        .reportview-container, .main { background: #0B0E14; }
        body { color: #E6EDF3; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        h1, h2, h3, h4 { color: #00CED1 !important; font-weight: 600 !important; }
        
        /* Tarjetas KPI */
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

        /* Banner de Estrategia */
        .strategy-banner {
            background-color: #12161F; 
            padding: 22px; 
            border-radius: 10px; 
            border-left: 5px solid #FF1493; 
            margin-bottom: 30px;
            border-top: 1px solid #30363D;
        }

        /* Tarjetas de Papers */
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
        
        /* Contenedor de Detalle de Fila Seleccionada */
        .detail-box {
            background-color: #161B22;
            border: 1px solid #00CED1;
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOTOR DE PROCESAMIENTO DE DATOS ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('scopus_PA3.csv')
    df['Title'] = df['Title'].fillna('Untitled Paper')
    df['Source title'] = df['Source title'].fillna('Unknown Source')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    df['Abstract'] = df['Abstract'].fillna('No abstract available.')
    df['Abstract_Clean'] = df['Abstract'].str.lower()
    return df

def main():
    try:
        df = load_and_process_data()
    except Exception:
        st.error("🚨 Error crítico: No se encontró la base de datos de Scopus 'scopus_PA3.csv' en la raíz.")
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

    # Filtrado Dinámico de Datos
    df_filtrado = df[df['Cited by'] >= min_citas].copy()
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO CORPORATIVO ---
    st.title("🔮 ChurnAI Horizon Dashboard")
    st.markdown("<p style='color:#8B949E; font-size:1.1rem; margin-top:-10px;'>Plataforma Ejecutiva de Inteligencia Analítica Aplicada al Riesgo Financiero</p>", unsafe_allow_html=True)
    
    # --- CÁLCULO DE PESOS CONECTORES DE LA IA ---
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
    # PESTAÑA 1: VISUALIZACIONES PRINCIPALES E INTELIGENCIA TENDENCIAL
    # =========================================================================
    with tab1:
        # 1. FILA DE TARJETAS KPI SUPERIORES
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.markdown(f'<div class="kpi-container"><div class="kpi-title">Volumen de Literatura</div><div class="kpi-value">{len(df_filtrado)}</div><div class="kpi-sub">Estudios Científicos Filtrados</div></div>', unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f'<div class="kpi-container"><div class="kpi-title">Impacto Global</div><div class="kpi-value">{df_filtrado["Cited by"].sum():,}</div><div class="kpi-sub">Citas Totales en Scopus</div></div>', unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f'<div class="kpi-container"><div class="kpi-title">Récord de Relevancia</div><div class="kpi-value">{df_filtrado["Cited by"].max() if len(df_filtrado) > 0 else 0}</div><div class="kpi-sub">Máximo de Citas en un Paper</div></div>', unsafe_allow_html=True)
        with col_kpi4:
            st.markdown(f'<div class="kpi-container"><div class="kpi-title">Densidad Científica</div><div class="kpi-value">{df_filtrado["Cited by"].mean() if len(df_filtrado) > 0 else 0:.1f}</div><div class="kpi-sub">Promedio de Citas por Registro</div></div>', unsafe_allow_html=True)

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

        # 3. FILA DE GRÁFICOS EN PARALELO
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### 🎯 Predominancia de Métricas Matemáticas de Validación")
            metricas_data = [
                {'Métrica': 'Accuracy', 'Papers': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum()},
                {'Métrica': 'F1-Score', 'Papers': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum()},
                {'Métrica': 'AUC-ROC', 'Papers': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum()}
            ]
            df_m = pd.DataFrame(metricas_data).sort_values(by="Papers", ascending=True)
            fig_bar = px.bar(df_m, x="Papers", y="Métrica", orientation="h", template="plotly_dark", color="Papers", color_continuous_scale=["#FF1493", "#00CED1"])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
            st.markdown("#### 📈 Distribución de Madurez Tecnológica e Impacto (Anual)")
            fig_violin = px.violin(df_filtrado, x="Year", y="Cited by", box=True, points="all", template="plotly_dark", color_discrete_sequence=["#00CED1"])
            fig_violin.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_violin, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 4. GRÁFICA DE LÍNEAS TEMPORAL COHESIVA
        st.markdown("#### 📈 Evolución Histórica de Dimensiones Críticas de Entrada (Trendline Analysis)")
        if len(df_filtrado) > 0:
            text_comb = df_filtrado['Abstract_Clean'] + " " + df_filtrado['Title'].str.lower()
            df_trends = pd.DataFrame({
                'Año': df_filtrado['Year'],
                '📱 Transacciones e Interactividad': text_comb.str.contains('transaction|behavio|digital|channel|yape|plin').astype(int),
                '💳 Historial Crediticio (SBS)': text_comb.str.contains('credit|score|history|risk|sbs|infocorp').astype(int),
                '👤 Datos Demográficos y Perfil': text_comb.str.contains('demograph|age|gender|income|status|sueldo').astype(int)
            })
            df_trends_grouped = df_trends.groupby('Año').sum().reset_index()
            df_melted = df_trends_grouped.melt(id_vars='Año', var_name='Dimensión Crítica', value_name='Cantidad de Investigaciones')
            
            fig_line = px.line(df_melted, x="Año", y="Cantidad de Investigaciones", color="Dimensión Crítica",
                               color_discrete_map={"📱 Transacciones e Interactividad": "#00CED1", "💳 Historial Crediticio (SBS)": "#FF1493", "👤 Datos Demográficos y Perfil": "#FFFF00"},
                               template="plotly_dark", markers=True)
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_line, use_container_width=True)

    # =========================================================================
    # PESTAÑA 2: SIMULADOR FINANCIERO PERÚ (MANTENIDO INTACTO)
    # =========================================================================
    with tab2:
        st.markdown("### 🔮 Motor de Simulación de Riesgo Bancario Local")
        st.info(f"🔗 **Estatus del Motor:** Sincronizado dinámicamente con la literatura científica actual.")
        
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            st.markdown("#### ⚙️ Entrada del Perfil Transaccional")
            banco_seleccionado = st.selectbox("Selecciona la entidad a evaluar dentro del ecosistema nacional:", ["Banco de Crédito del Perú (BCP)", "BBVA Perú", "Interbank", "Scotiabank Perú"])
            caida_trans = st.slider("1. Contracción mensual en canales de pago móviles (%):", 0, 100, 30)
            score_sbs = st.slider("2. Calificación del Score Crediticio interno:", 300, 850, 710)
            portabilidad_sueldo = st.radio("3. ¿Registra alertas de portabilidad de Cuenta Sueldo?", ["No", "Sí"])
            
            score_ponderado = 12.0 + (caida_trans * (peso_trans * 1.2)) + ((850 - score_sbs) * (peso_score * 0.15))
            if portabilidad_sueldo == "Sí": score_ponderado += (28.0 * (peso_demo + 0.4))
            riesgo_final = min(max(score_ponderado, 0.0), 100.0)

        with col_sim2:
            st.markdown(f"#### 🎯 Diagnóstico Operativo ({banco_seleccionado})")
            color_alerta = "#00CED1" if riesgo_final < 50 else "#FF1493"
            st.markdown(f"<div style='background-color: #161B22; padding: 25px; border-radius: 10px; border: 2px solid {color_alerta}; text-align: center;'><p style='color: #E6EDF3; font-size: 1.1rem; margin-bottom: 5px;'>RIESGO ESTIMADO DE ABANDONO DE CUENTA</p><h1 style='color: {color_alerta} !important; font-size: 3.8rem; margin: 0;'>{riesgo_final:.1f}%</h1></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if riesgo_final < 40: st.success("🟢 **Zona Segura:** El cliente se mantiene fidelizado.")
            elif 40 <= riesgo_final < 70: st.warning("🟡 **Retención Preventiva:** Recomienda exoneración de membresía.")
            else: st.error("🔴 **Intervención Inmediata:** Fuga inminente de haberes.")

    # =========================================================================
    # PESTAÑA 3: CENTRO DE DATOS E INSIGHTS (DATA LAKE INTERACTIVO)
    # =========================================================================
    with tab3:
        st.markdown("### 📚 Centro de Inteligencia y Auditoría Bibliométrica")
        st.markdown("#### 🗂️ Data Lake Completo (Filtrado Inteligente e Interactividad Activa)")
        st.markdown("💡 *Haz clic en cualquier celda o fila del cuadro inferior para auditar sus métricas de afinidad algorítmica, ver su abstract completo y analizar sus porcentajes.*")

        # Preparación de las columnas del Data Lake, ordenadas por Citaciones
        df_lake = df_filtrado[["Title", "Year", "Cited by", "Source title", "Abstract"]].sort_values(by="Cited by", ascending=False).reset_index(drop=True)
        
        # Cuadro de datos interactivo con evento de selección nativo de Streamlit
        seleccion = st.dataframe(
            df_lake[["Title", "Year", "Cited by", "Source title"]],
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                "Title": st.column_config.TextColumn("Título del Estudio Científico"),
                "Year": st.column_config.NumberColumn("Año", format="%d"),
                "Cited by": st.column_config.NumberColumn("Citas Scopus"),
                "Source title": st.column_config.TextColumn("Revista / Journal")
            }
        )

        # Bloque condicional: Si el usuario selecciona un paper, despliega la analítica avanzada en la base
        if len(seleccion.selection.rows) > 0:
            fila_idx = seleccion.selection.rows[0]
            paper_sel = df_lake.iloc[fila_idx]
            
            # Cálculo de porcentajes de afinidad basados en su propio Abstract
            abs_text = paper_sel['Abstract'].lower()
            m_t = abs_text.count('transaction') + abs_text.count('behavio') + abs_text.count('digital') + abs_text.count('channel')
            m_s = abs_text.count('credit') + abs_text.count('score') + abs_text.count('risk') + abs_text.count('sbs')
            m_d = abs_text.count('demograph') + abs_text.count('age') + abs_text.count('gender') + abs_text.count('income')
            
            tot = m_t + m_s + m_d
            p_t, p_s, p_d = (m_t/tot*100, m_s/tot*100, m_d/tot*100) if tot > 0 else (33.3, 33.3, 33.3)
            
            st.markdown(f"""
            <div class="detail-box">
                <span style="color: #FF1493; font-weight: bold; font-size: 0.85rem; letter-spacing: 1px;">📋 AUDITORÍA AVANZADA DEL DOCUMENTO SELECCIONADO</span>
                <h3 style="margin-top: 10px; color: #FFF !important;">{paper_sel['Title']}</h3>
                <p style="color: #8B949E; font-size: 0.9rem;"><b>Publicado en:</b> {paper_sel['Source title']} ({paper_sel['Year']})  |  <b>Impacto:</b> {paper_sel['Cited by']} citas.</p>
                <hr style="border-color: #21262D;">
                <h5 style="color: #00CED1 !important;">Resumen Científico (Abstract)</h5>
                <p style="color: #E6EDF3; font-size: 0.95rem; line-height: 1.6; text-align: justify;">{paper_sel['Abstract']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Fila de métricas porcentuales calculadas en tiempo real para ese paper
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric(label="📱 Afinidad Transaccional", value=f"{p_t:.1f}%", delta="Enfoque Digital" if p_t > 40 else None)
            with col_p2:
                st.metric(label="💳 Afinidad Crediticia (Riesgo)", value=f"{p_s:.1f}%", delta="Enfoque de Crédito" if p_s > 40 else None)
            with col_p3:
                st.metric(label="👤 Afinidad Demográfica (Perfil)", value=f"{p_d:.1f}%", delta="Enfoque de Usuario" if p_d > 40 else None)
        else:
            st.info("💡 **Tip Ejecutivo:** Para ver el Abstract, porcentajes de afinidad y análisis detallado de cualquier paper, haz un clic sobre su fila en el cuadro de arriba.")

if __name__ == "__main__":
    main() 