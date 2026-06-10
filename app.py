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
        st.title("⚙️ Filtros del Estudio")
        
        busqueda = st.text_input("🔍 Buscar concepto (Ej. Neural, SMOTE):", "")
        min_citas = st.slider("📈 Mínimo de citas (Autoridad):", 0, int(df['Cited by'].max()), 0)
        
        st.markdown("---")
        st.info("💡 **¿Eres nuevo?** Juega con los filtros y lee las 'Guías para Juniors' debajo de cada gráfico para entender la IA paso a paso.")

    # Aplicar filtros
    df_filtrado = df[df['Cited by'] >= min_citas]
    if busqueda:
        df_filtrado = df_filtrado[df_filtrado['Abstract_Clean'].str.contains(busqueda.lower()) | 
                                  df_filtrado['Title'].str.lower().str.contains(busqueda.lower())]

    # --- ENCABEZADO ---
    st.title("🤖 Predictor de Churn: Explorador Interactivo de IA")
    st.markdown("*Aprende qué algoritmos usan los bancos para predecir si un cliente se irá, interactuando con evidencia científica real.*")
    st.divider()

    # --- PESTAÑAS (TABS) ---
    tab1, tab2, tab3 = st.tabs([
        "🧠 1. Conoce los Algoritmos", 
        "📊 2. ¿Cómo se evalúan los modelos?", 
        "📈 3. Análisis de Éxito (Estadística)"
    ])

    # =========================================================================
    # PESTAÑA 1: ALGORITMOS
    # =========================================================================
    with tab1:
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader('🤖 ¿Qué "cerebros" matemáticos se usan más?')
            
            # Datos enriquecidos con explicaciones para el tooltip (hover)
            algoritmos_data = [
                {'Algoritmo': 'Random Forest', 'Uso': df_filtrado['Abstract_Clean'].str.contains('random forest| rf ').sum(),
                 'Explicacion': 'Un "bosque" de muchos árboles de decisión que votan para dar un resultado.'},
                {'Algoritmo': 'Redes Neuronales', 'Uso': df_filtrado['Abstract_Clean'].str.contains('neural network|deep learning').sum(),
                 'Explicacion': 'Sistemas que imitan las neuronas del cerebro humano para aprender patrones complejos.'},
                {'Algoritmo': 'XGBoost', 'Uso': df_filtrado['Abstract_Clean'].str.contains('xgboost|gradient boosting').sum(),
                 'Explicacion': 'Algoritmo muy potente que corrige los errores de modelos anteriores secuencialmente.'},
                {'Algoritmo': 'SVM', 'Uso': df_filtrado['Abstract_Clean'].str.contains('svm|support vector machine').sum(),
                 'Explicacion': 'Traza una "línea fronteriza" matemática perfecta para separar clientes que se van de los que se quedan.'}
            ]
            df_alg = pd.DataFrame(algoritmos_data).sort_values(by='Uso')
            
            fig_bar_alg = px.bar(
                df_alg, x='Uso', y='Algoritmo', orientation='h',
                color='Uso', color_continuous_scale='Blues',
                custom_data=['Explicacion'], # Se pasa la explicación oculta
                title="Pasa el mouse sobre las barras 🖱️"
            )
            # Personalizar lo que dice cuando pasas el ratón
            fig_bar_alg.update_traces(hovertemplate="<b>%{y}</b><br>Menciones en papers: %{x}<br><br><i>¿Qué es?: %{customdata[0]}</i><extra></extra>")
            fig_bar_alg.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_bar_alg, use_container_width=True)
            
            with st.expander("🎓 Guía para Juniors: ¿Cómo leer este gráfico?"):
                st.write("**Representa:** Cuántas veces los científicos mencionan usar cada algoritmo.")
                st.write("**Conclusión de Negocio:** El algoritmo con la barra más larga es el estándar actual. Si vas a armar un equipo de datos en el banco, asegúrate de que dominen esa herramienta primero.")

        with colB:
            st.subheader("🔍 ¿Cajas Negras o IA Explicable?")
            exp_count = df_filtrado['Abstract_Clean'].str.contains('shap|lime|explainabl|interpret').sum()
            no_exp_count = len(df_filtrado) - exp_count
            
            df_pie = pd.DataFrame({
                'Tipo': ['IA Explicable (XAI)', 'Modelos Tradicionales'],
                'Cantidad': [exp_count, no_exp_count],
                'Explicacion': [
                    'Usa herramientas (como SHAP) para decirle al gerente EXACTAMENTE por qué el cliente se va.',
                    'El modelo acierta, pero no puede explicar el porqué (caja negra).'
                ]
            })
            
            fig_donut = px.pie(
                df_pie, names='Tipo', values='Cantidad', hole=0.5,
                color_discrete_sequence=['#1f77b4', '#d62728'],
                custom_data=['Explicacion'],
                title="Pasa el mouse sobre el anillo 🖱️"
            )
            fig_donut.update_traces(hovertemplate="<b>%{label}</b><br>Estudios: %{value}<br><br><i>%{customdata[0]}</i><extra></extra>")
            st.plotly_chart(fig_donut, use_container_width=True)
            
            with st.expander("🎓 Guía para Juniors: ¿Por qué esto es importante?"):
                st.write("Imagina que el modelo le deniega un préstamo a un cliente y este pregunta *'¿Por qué?'*. Si usas un **Modelo Tradicional (Caja Negra)**, no sabrás qué responder. Hoy en día, los bancos prefieren **IA Explicable** por temas legales y de transparencia.")

    # =========================================================================
    # PESTAÑA 2: MÉTRICAS Y VARIABLES
    # =========================================================================
    with tab2:
        colC, colD = st.columns(2)
        
        with colC:
            st.subheader("🎯 ¿Cómo sabemos si el modelo es bueno?")
            
            metricas_data = [
                {'Metrica': 'Accuracy', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('accuracy').sum(), 'Desc': 'Mide el porcentaje total de aciertos.'},
                {'Metrica': 'F1-Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('f1|f-measure').sum(), 'Desc': 'Equilibrio perfecto cuando hay muy pocos clientes que se van frente a los que se quedan.'},
                {'Metrica': 'AUC-ROC', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('auc|roc').sum(), 'Desc': 'Mide la capacidad de distinguir entre un cliente leal y uno que desertará.'}
            ]
            df_metrics = pd.DataFrame(metricas_data)
            
            fig_radar = px.line_polar(
                df_metrics, r='Menciones', theta='Metrica', line_close=True,
                custom_data=['Desc'],
                title="Métricas de Evaluación (Pasa el mouse 🖱️)",
                template="plotly_white"
            )
            fig_radar.update_traces(fill='toself', line_color='#ff7f0e', hovertemplate="<b>%{theta}</b><br>Menciones: %{r}<br><i>%{customdata[0]}</i><extra></extra>")
            st.plotly_chart(fig_radar, use_container_width=True)
            
            with st.expander("🎓 Guía para Juniors: El problema del 'Accuracy'"):
                st.write("Si en un banco el 99% de la gente se queda y el 1% se va, un modelo que diga 'Nadie se va' tendrá un **Accuracy del 99%**, ¡pero es un modelo inútil! Por eso, los expertos miran más el **F1-Score** o el **AUC-ROC**.")

        with colD:
            st.subheader("🧩 Ingredientes del Modelo (Features)")
            st.markdown("¿Qué datos de los clientes "comen" estos algoritmos para aprender?")
            
            features_data = [
                {'Categoria': 'Transacciones', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('transaction|behavio').sum(), 'Desc': 'Frecuencia de uso, montos gastados.'},
                {'Categoria': 'Credit Score', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('credit score').sum(), 'Desc': 'Historial de pagos y deudas.'},
                {'Categoria': 'Demografía', 'Menciones': df_filtrado['Abstract_Clean'].str.contains('demograph|age|gender').sum(), 'Desc': 'Edad, género, ciudad de residencia.'}
            ]
            df_feat = pd.DataFrame(features_data)
            
            fig_tree = px.treemap(
                df_feat, path=['Categoria'], values='Menciones',
                custom_data=['Desc'], color='Menciones', color_continuous_scale='Greens',
                title="Haz clic o pasa el mouse en los cuadros 🖱️"
            )
            fig_tree.update_traces(hovertemplate="<b>Datos de %{label}</b><br>Menciones: %{value}<br><i>%{customdata[0]}</i><extra></extra>")
            st.plotly_chart(fig_tree, use_container_width=True)

    # =========================================================================
    # PESTAÑA 3: BOXPLOT Y DATOS
    # =========================================================================
    with tab3:
        st.subheader("📦 Boxplot: Entendiendo la Distribución del Éxito (Citas)")
        
        # Gráfico Boxplot
        fig_box = px.box(
            df_filtrado, x="Year", y="Cited by", color="Year",
            title="Distribución de Citas por Año",
            template="plotly_white", points="all" # points="all" muestra los puntitos individuales
        )
        fig_box.update_layout(xaxis=dict(tickmode='linear', dtick=1), showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # EXPLICACIÓN SUPER DIDÁCTICA DEL BOXPLOT
        with st.expander("🎓 Guía para Juniors: ¿Cómo diantres se lee un Boxplot (Gráfico de Cajas)?"):
            st.markdown("""
            No te asustes, es más fácil de lo que parece:
            * 🔵 **Los puntitos:** Cada punto es una investigación (un *paper*). Su altura te dice cuántas veces fue citado.
            * 📦 **La caja de color:** Representa dónde está el "pelotón principal" (el 50% de los estudios centrales).
            * ➖ **La línea que cruza la caja:** Es la **Mediana**. Significa que la mitad de los estudios están por encima de esa línea y la otra mitad por debajo. Es el valor "típico".
            * 🚀 **Los puntos muy arriba (fuera de las líneas finas):** Son los ***Outliers* o valores atípicos**. ¡Son las súper-estrellas! Estudios que lograron un éxito rotundo y anormal.
            """)

        st.divider()
        st.subheader("🗂️ Repositorio para Explorar")
        st.dataframe(
            df_filtrado[['Title', 'Year', 'Cited by', 'Link']].sort_values('Cited by', ascending=False), 
            use_container_width=True, hide_index=True,
            column_config={
                "Title": st.column_config.TextColumn("Título del Estudio"),
                "Year": st.column_config.NumberColumn("Año", format="%d"),
                "Cited by": st.column_config.ProgressColumn("Éxito (Citas)", format="%d", min_value=0, max_value=int(df['Cited by'].max())),
                "Link": st.column_config.LinkColumn("Enlace a Scopus")
            }
        )

if __name__ == "__main__":
    main()