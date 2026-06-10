import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="ML Churn Analytics", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# --- 2. PROCESAMIENTO ESTADÍSTICO DE TEXTO ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    df['Revista Corta'] = df['Source title'].str[:35] + '...'
    df['Abstract_Clean'] = df['Abstract'].fillna('').str.lower()
    df['Keywords_Clean'] = df['Author Keywords'].fillna('').str.lower()
    return df

def count_keywords(df, keyword_dict, column='Abstract_Clean'):
    results = []
    for label, keywords in keyword_dict.items():
        # Busca cualquiera de las palabras clave en la lista
        pattern = '|'.join(keywords)
        count = df[column].str.contains(pattern, na=False).sum()
        results.append({'Categoría': label, 'Frecuencia': count})
    return pd.DataFrame(results).sort_values(by='Frecuencia', ascending=False)

def main():
    try:
        df = load_and_process_data()
    except Exception:
        st.error("🚨 Error: No se encontró la base de datos scopus_PA3.csv.")
        return

    # --- BARRA LATERAL (ESTILO MINIMALISTA) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=60)
        st.title("⚙️ ML Parámetros")
        st.markdown("---")
        
        busqueda = st.text_input("🔍 Filtro por texto (Ej. Neural, SMOTE):", "")
        min_citas = st.slider("📈 Citas mínimas del estudio:", 0, int(df['Cited by'].max()), 0)
        
        st.markdown("---")
        st.caption("Filtros aplicados en tiempo real a la extracción semántica de los abstracts.")

    # Aplicar filtros
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO ---
    st.title("🤖 Inteligencia Artificial en Banca: Análisis de Modelos Predictivos")
    st.markdown("""
    <p style='color: #555; font-size: 1.1rem;'>
    Dashboard analítico sobre el estado del arte en Machine Learning para la predicción de Churn. 
    Exploración de algoritmos, métricas de evaluación y técnicas de preprocesamiento de datos.
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    # --- MÉTRICAS GENERALES ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Datasets/Estudios Analizados", len(df_filtrado))
    
    # Análisis rápido de quién usa SMOTE u oversampling
    uso_smote = df_filtrado['Abstract_Clean'].str.contains('smote|oversampling|imbalance').sum()
    porcentaje_smote = (uso_smote / len(df_filtrado)) * 100 if len(df_filtrado) > 0 else 0
    c2.metric("⚖️ Manejo de Datos Desbalanceados", f"{porcentaje_smote:.1f}%")
    
    c3.metric("🎯 Impacto Acumulado (Citas)", df_filtrado['Cited by'].sum())
    c4.metric("🏆 Máx Citas en un Estudio", df_filtrado['Cited by'].max())

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================================================
    # PESTAÑAS TÉCNICAS (DISEÑO ORDENADO Y MODERNO)
    # =========================================================================
    tab1, tab2, tab3 = st.tabs([
        "🧠 1. Modelos y Algoritmos", 
        "📊 2. Métricas y Preprocesamiento", 
        "📈 3. Distribución Estadística (Boxplots)"
    ])

    # -------------------------------------------------------------------------
    # PESTAÑA 1: MODELOS DE MACHINE LEARNING
    # -------------------------------------------------------------------------
    with tab1:
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Arquitecturas más Utilizadas")
            algoritmos = {
                'Random Forest': ['random forest', ' rf '],
                'Redes Neuronales / Deep Learning': ['neural network', 'deep learning', 'ann', 'dnn'],
                'XGBoost / Gradient Boosting': ['xgboost', 'gradient boosting', ' gb '],
                'SVM': ['svm', 'support vector machine'],
                'Regresión Logística': ['logistic regression', ' lr '],
                'Decision Tree': ['decision tree', ' dt ']
            }
            df_alg = count_keywords(df_filtrado, algoritmos)
            
            fig_bar_alg = px.bar(
                df_alg, x='Frecuencia', y='Categoría', orientation='h',
                color='Frecuencia', color_continuous_scale='Teal',
                title="Dominancia de Algoritmos en la Literatura",
                template="plotly_white"
            )
            fig_bar_alg.update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title="")
            st.plotly_chart(fig_bar_alg, use_container_width=True)

        with colB:
            st.subheader("Modelos de Caja Negra vs Interpretabilidad")
            interpretabilidad = {
                'Mencionan Explicabilidad (SHAP, LIME)': ['shap', 'lime', 'explainabl', 'interpret'],
                'Solo Enfoque Predictivo': ['accuracy', 'predict'] # Simplificación
            }
            # Lógica simple: si menciona SHAP/LIME, va a un grupo, sino al otro.
            exp_count = df_filtrado['Abstract_Clean'].str.contains('shap|lime|explainabl|interpret').sum()
            no_exp_count = len(df_filtrado) - exp_count
            
            fig_donut = px.pie(
                names=['Con Interpretabilidad (XAI)', 'Modelos Tradicionales'], 
                values=[exp_count, no_exp_count], hole=0.5,
                color_discrete_sequence=['#1f77b4', '#d62728'],
                title="Adopción de IA Explicable (XAI)"
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_donut.update_layout(showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)

    # -------------------------------------------------------------------------
    # PESTAÑA 2: MÉTRICAS Y PREPROCESAMIENTO
    # -------------------------------------------------------------------------
    with tab2:
        colC, colD = st.columns(2)
        
        with colC:
            st.subheader("Métricas de Evaluación Reportadas")
            metricas = {
                'Accuracy (Precisión Global)': ['accuracy'],
                'F1-Score': ['f1', 'f-measure'],
                'AUC-ROC': ['auc', 'roc', 'area under'],
                'Recall / Sensitivity': ['recall', 'sensitivit'],
                'Precision': ['precision']
            }
            df_metrics = count_keywords(df_filtrado, metricas)
            
            fig_radar = px.line_polar(
                df_metrics, r='Frecuencia', theta='Categoría', line_close=True,
                title="Estándares de Medición en Churn Bancario",
                template="plotly_white"
            )
            fig_radar.update_traces(fill='toself', line_color='#ff7f0e')
            st.plotly_chart(fig_radar, use_container_width=True)

        with colD:
            st.subheader("Variables Predictoras Top (Feature Engineering)")
            features = {
                'Comportamiento de Transacciones': ['transaction', 'behavio'],
                'Historial Crediticio / Score': ['credit score', 'credit history'],
                'Datos Demográficos (Edad, Género)': ['demograph', 'age', 'gender'],
                'Saldo / Balance de Cuenta': ['balance', 'account'],
                'Productos Adquiridos (Préstamos/Tarjetas)': ['loan', 'card', 'product']
            }
            df_feat = count_keywords(df_filtrado, features)
            
            fig_tree = px.treemap(
                df_feat, path=['Categoría'], values='Frecuencia',
                color='Frecuencia', color_continuous_scale='Blues',
                title="Mapa de Variables Críticas del Cliente"
            )
            st.plotly_chart(fig_tree, use_container_width=True)

    # -------------------------------------------------------------------------
    # PESTAÑA 3: ESTADÍSTICA Y CATÁLOGO
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Distribución Estadística del Impacto (Boxplots)")
        st.markdown("Este gráfico muestra la dispersión de citas según el año de publicación. La línea central de las cajas es la mediana, y los puntos externos son *outliers* (estudios de éxito atípico).")
        
        fig_box = px.box(
            df_filtrado, x="Year", y="Cited by", color="Year",
            title="Análisis de Varianza: Citas Científicas",
            template="plotly_white", points="all" # Muestra todos los puntos encima de la caja
        )
        fig_box.update_layout(xaxis=dict(tickmode='linear', dtick=1), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.divider()
        st.subheader("🗂️ Repositorio de Investigaciones")
        st.dataframe(
            df_filtrado[['Title', 'Year', 'Cited by', 'Revista Corta', 'Link']].sort_values('Cited by', ascending=False), 
            use_container_width=True, hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Título del Estudio", width="large"),
                "Year": st.column_config.NumberColumn("Año", format="%d"),
                "Cited by": st.column_config.ProgressColumn("Autoridad (Citas)", format="%d", min_value=0, max_value=int(df['Cited by'].max())),
                "Link": st.column_config.LinkColumn("Ver en Scopus")
            }
        )

if __name__ == "__main__":
    main()