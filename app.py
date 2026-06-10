import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="ML Churn Academy", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# --- 2. PROCESAMIENTO DE DATOS ---
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
        st.error("🚨 Error: No se encontró la base de datos scopus_PA3.csv en tu GitHub.")
        return

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=60)
        st.title("⚙️ Filtros")
        
        busqueda = st.text_input("🔍 Buscar concepto:", "")
        min_citas = st.slider("📈 Citas mínimas:", 0, int(df['Cited by'].max()), 0)
        
        st.markdown("---")
        st.info("💡 **Tip:** Usa los filtros para que los gráficos se actualicen en tiempo real.")

    # Aplicar filtros
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO ---
    st.title("🤖 Predictor de Churn: IA en Banca")
    st.markdown("*Análisis bibliométrico avanzado. Interactúa con los gráficos para descubrir los secretos de la IA en la retención de clientes.*")
    st.divider()

    # --- PESTAÑAS (TABS) ---
    tab1, tab2, tab3 = st.tabs([
        "🧠 1. Algoritmos IA", 
        "📊 2. Métricas de Evaluación", 
        "🌍 3. Impacto Global"
    ])

    # =========================================================================
    # PESTAÑA 1: ALGORITMOS
    # =========================================================================
    with tab1:
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("🤖 Algoritmos de IA")
            
            # Datos enriquecidos con explicaciones para el tooltip (hover)
            algoritmos_data = [
                {'Algoritmo': 'Random Forest', 'Uso': df_filtrado['Abstract_Clean'].str.contains('random forest| rf ').sum(),
                 'Explicacion': 'Bosque de árboles de decisión que votan para un resultado.'},
                {'Algoritmo': 'Redes Neuronales', 'Uso': df_filtrado['Abstract_Clean'].str.contains('neural network|deep learning').sum(),
                 'Explicacion': 'Sistemas que imitan neuronas humanas para patrones complejos.'},
                {'Algoritmo': 'XGBoost', 'Uso': df_filtrado['Abstract_Clean'].str.contains('xgboost|gradient boosting').sum(),
                 'Explicacion': 'Algoritmo potente que corrige errores de modelos anteriores.'},
                {'Algoritmo': 'SVM', 'Uso': df_filtrado['Abstract_Clean'].str.contains('svm|support vector machine').sum(),
                 'Explicacion': 'Línea fronteriza matemática que separa clientes.'}
            ]
            df_alg = pd.DataFrame(algoritmos_data).sort_values(by='Uso')
            
            fig_bar_alg = px.bar(
                df_alg, x='Uso', y='Algoritmo', orientation='h',
                color='Uso', color_continuous_scale=['#FF1493', '#00CED1'],
                custom_data=['Explicacion'], # Se pasa la explicación oculta
                title="Pasa el mouse sobre las barras"
            )
            # Personalizar lo que dice cuando pasas el ratón
            fig_bar_alg.update_traces(hovertemplate="<b>%{y}</b><br>Menciones en papers: %{x}<br><br><i>¿Qué es?: %{customdata[0]}</i><extra></extra>")
            fig_bar_alg.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_bar_alg, use_container_width=True)
            
            with st.expander("📖 ¿Cómo leer este gráfico?"):
                st.write("**Representa:** Cuántas veces los científicos mencionan usar cada algoritmo.")
                st.write("**Uso:** El algoritmo con la barra más larga es el más probado. Un banco debería empezar por aquí.")

        with colB:
            st.subheader("🔍 Interpretabilidad vs. Caja Negra")
            exp_count = df_filtrado['Abstract_Clean'].str.contains('shap|lime|explainabl|interpret').sum()
            no_exp_count = len(df_filtrado) - exp_count
            
            df_pie = pd.DataFrame({
                'Tipo': ['IA Explicable (XAI)', 'Caja Negra'],
                'Cantidad': [exp_count, no_exp_count],
                'Explicacion': [
                    'Muestra las razones de la fuga del cliente (ej. SHAP).',
                    'Acierta en la predicción, pero no explica por qué.'
                ]
            })
            
            fig_donut = px.pie(
                df_pie, names='Tipo', values='Cantidad', hole=0.5,
                color_discrete_sequence=['#00CED1', '#FF1493'],
                custom_data=['Explicacion'],
                title="Pasa el mouse sobre el anillo"
            )
            fig_donut.update_traces(hovertemplate="<b>%{label}</b><br>Estudios: %{value}<br><br><i>%{customdata[0]}</i><extra></extra>")
            fig_donut.update_layout(showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)
            
            with st.expander("📖 ¿Cómo leer este gráfico?"):
                st.write("Muestra la proporción de estudios que usan **IA Explicable**. Los bancos prefieren modelos transparentes por temas legales y de confianza.")

    # =========================================================================
    # PESTAÑA 2: MÉTRICAS DE EVALUACIÓN
    # =========================================================================
    with tab2:
        colC, colD = st.columns(2)
        
        with colC:
            st.subheader("🎯 Métricas de Evaluación")
            
            metricas_data = [
                {'Metrica': 'Accuracy', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum(), 'Desc': 'Porcentaje total de aciertos.'},
                {'Metrica': 'F1-Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum(), 'Desc': 'Equilibrio entre precisión y recall.'},
                {'Metrica': 'AUC-ROC', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum(), 'Desc': 'Capacidad de distinguir entre clientes.'}
            ]
            df_metrics = pd.DataFrame(metricas_data)
            
            fig_radar = px.line_polar(
                df_metrics, r='Menciones', theta='Metrica', line_close=True,
                custom_data=['Desc'],
                title="Métricas de Evaluación (Pasa el mouse)",
                template="plotly_white"
            )
            fig_radar.update_traces(fill='toself', line_color='#00CED1', hovertemplate="<b>%{theta}</b><br>Menciones: %{r}<br><i>%{customdata[0]}</i><extra></extra>")
            st.plotly_chart(fig_radar, use_container_width=True)
            
            with st.expander("📖 ¿Cómo leer este gráfico?"):
                st.write("El polígono muestra la frecuencia de uso de cada métrica. **F1-Score** y **AUC-ROC** son más fiables que **Accuracy** cuando los datos están desbalanceados (poca gente se va).")

        with colD:
            st.subheader("🧩 Variables Predictoras")
            
            features_data = [
                {'Categoria': 'Transacciones', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('transaction|behavio').sum(), 'Desc': 'Uso, montos gastados.'},
                {'Categoria': 'Credit Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('credit score').sum(), 'Desc': 'Pagos, deudas.'},
                {'Categoria': 'Demografía', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('demograph|age|gender').sum(), 'Desc': 'Edad, género, ciudad.'}
            ]
            df_feat = pd.DataFrame(features_data)
            
            fig_tree = px.treemap(
                df_feat, path=['Categoria'], values='Menciones',
                custom_data=['Desc'], color='Menciones', color_continuous_scale=['#FF1493', '#00CED1'],
                title="Pasa el mouse sobre los cuadros"
            )
            fig_tree.update_traces(hovertemplate="<b>Datos de %{label}</b><br>Menciones: %{value}<br><i>%{customdata[0]}</i><extra></extra>")
            st.plotly_chart(fig_tree, use_container_width=True)
            
            with st.expander("📖 ¿Cómo leer este gráfico?"):
                st.write("El tamaño del cuadro representa la frecuencia de uso de cada tipo de dato. Las variables **Transaccionales** suelen ser las más poderosas para predecir la fuga.")

    # =========================================================================
    # PESTAÑA 3: IMPACTO GLOBAL
    # =========================================================================
    with tab3:
        st.subheader("🌍 Impacto Global")
        
        df_mostrar = df_filtrado[['Title', 'Year', 'Cited by', 'Revista Corta', 'Link']].copy()
        
        st.dataframe(
            df_mostrar.sort_values('Cited by', ascending=False), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Título"),
                "Year": st.column_config.NumberColumn("Año", format="%d"),
                "Cited by": st.column_config.ProgressColumn("Citas", format="%d", min_value=0, max_value=int(df['Cited by'].max())),
                "Link": st.column_config.LinkColumn("Leer Paper")
            }
        )

if __name__ == "__main__":
    main()