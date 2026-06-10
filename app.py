import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import re

# 1. CONFIGURACIÓN: Modo ultra ancho y elegante
st.set_page_config(page_title="AI Churn Intelligence", page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

# 2. MOTOR DE DATOS: Carga y extracción avanzada
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('scopus_PA3.csv')
    df['Cited by'] = df['Cited by'].fillna(0).astype(int)
    df['Year'] = df['Year'].fillna(2025).astype(int)
    df['Revista Corta'] = df['Source title'].str[:35] + '...'
    
    # MAGIA: Extraer países de la columna 'Affiliations'
    def extract_countries(affil_text):
        if pd.isna(affil_text):
            return []
        # Scopus suele poner el país al final de cada afiliación separada por comas
        afiliaciones = str(affil_text).split(';')
        paises = [af.split(',')[-1].strip() for af in afiliaciones if len(af.split(',')) > 0]
        return paises

    df['Countries'] = df['Affiliations'].apply(extract_countries)
    return df

def main():
    try:
        df = load_and_process_data()
    except Exception:
        st.error("🚨 Error crítico: No se encontró la base de datos scopus_PA3.csv.")
        return

    # --- DISEÑO DEL PANEL LATERAL (MODERNO Y LIMPIO) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
        st.title("⚙️ Centro de Mando")
        st.markdown("---")
        
        # Filtros minimalistas
        busqueda = st.text_input("🔍 Búsqueda libre (Ej. Credit, Behaviour):", placeholder="Escribe aquí...")
        
        modelos = ["Cualquier Algoritmo", "Random Forest", "Neural Network", "XGBoost", "SVM", "SHAP"]
        modelo_seleccionado = st.selectbox("🤖 Tecnología de IA:", modelos)
        
        st.markdown("---")
        st.caption("💡 Tip: Usa los filtros para que los gráficos se actualicen en tiempo real.")

    # --- LÓGICA DE FILTRADO ---
    df_filtrado = df.copy()
    if busqueda:
        filtro_txt = busqueda.lower()
        df_filtrado = df_filtrado[df_filtrado['Abstract'].fillna('').str.lower().str.contains(filtro_txt) | 
                                  df_filtrado['Title'].fillna('').str.lower().str.contains(filtro_txt)]
    if modelo_seleccionado != "Cualquier Algoritmo":
        filtro_mod = modelo_seleccionado.lower()
        df_filtrado = df_filtrado[df_filtrado['Abstract'].fillna('').str.lower().str.contains(filtro_mod) | 
                                  df_filtrado['Author Keywords'].fillna('').str.lower().str.contains(filtro_mod)]

    # --- ENCABEZADO ELEGANTE ---
    st.title("🌐 Hub Global de Inteligencia: Retención de Clientes con IA")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #666;'>
    Explora el panorama científico mundial sobre modelos predictivos antifuga. Descubre de dónde provienen las soluciones y qué tecnologías dominan el sector financiero.
    </p>
    """, unsafe_allow_html=True)

    if df_filtrado.empty:
        st.warning("No hay investigaciones que coincidan con estos filtros. Intenta ser más general.")
        return

    # --- METRICAS ESTILO FINANCIERO (MÁS LIMPIAS) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Investigaciones Analizadas", len(df_filtrado))
    c2.metric("🎯 Nivel de Impacto (Citas)", df_filtrado['Cited by'].sum())
    
    # Extraer todos los países únicos del dataframe filtrado
    todos_paises = [pais for sublist in df_filtrado['Countries'] for pais in sublist]
    c3.metric("🌍 Países Involucrados", len(set(todos_paises)) if todos_paises else 0)
    c4.metric("🏢 Fuentes Únicas", df_filtrado['Source title'].nunique())
    
    st.markdown("---")

    # =====================================================================
    # SECCIÓN 1: EL MAPA DEL MUNDO (EL EFECTO "GUAU")
    # =====================================================================
    st.subheader("🌍 Ecosistema Global de Innovación")
    
    if todos_paises:
        # Contar cuántos papers por país
        conteo_paises = pd.DataFrame.from_dict(Counter(todos_paises), orient='index').reset_index()
        conteo_paises.columns = ['País', 'Investigaciones']
        
        # Crear mapa coroplético
        fig_map = px.choropleth(
            conteo_paises, 
            locations="País", 
            locationmode="country names",
            color="Investigaciones",
            hover_name="País",
            color_continuous_scale=px.colors.sequential.Blues,
            title="Calor Geográfico: ¿Dónde se está creando la tecnología?"
        )
        fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'), margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        
        with st.expander("📖 ¿Cómo interpretar este mapa?"):
            st.info("Los colores más oscuros indican los países que lideran la investigación en IA para la retención de clientes bancarios. Si el banco busca alianzas tecnológicas o consultorías de vanguardia, debe mirar hacia los hubs marcados en azul oscuro.")
    else:
        st.info("No hay datos geográficos disponibles para esta selección.")

    # =====================================================================
    # SECCIÓN 2: TECNOLOGÍA E IMPACTO (GRÁFICOS MODERNOS)
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True) # Espacio en blanco
    colA, colB = st.columns(2)

    with colA:
        st.subheader("🧬 ADN de los Algoritmos")
        if 'Author Keywords' in df_filtrado.columns:
            kws = df_filtrado['Author Keywords'].dropna().str.split(';')
            lista_kws = [k.strip().title() for sub in kws for k in sub if len(k.strip()) > 3]
            conteo_kws = pd.DataFrame(Counter(lista_kws).most_common(15), columns=['Tecnología', 'Uso'])
            
            fig_bar = px.bar(
                conteo_kws.sort_values('Uso', ascending=True), 
                x='Uso', y='Tecnología', orientation='h',
                color='Uso', color_continuous_scale='Teal',
                title="Top Tecnologías Mencionadas por Científicos"
            )
            fig_bar.update_layout(xaxis_title="Frecuencia de aparición", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            with st.expander("📖 ¿Cómo interpretar este gráfico?"):
                st.info("Muestra las palabras clave y algoritmos más utilizados por los investigadores. Las barras más largas representan el 'estándar de la industria' actual. Si 'Random Forest' o 'Machine Learning' lideran, es porque son los métodos más rentables y probados.")

    with colB:
        st.subheader("🔥 Burbujas de Autoridad Científica")
        fig_bubble = px.scatter(
            df_filtrado, x="Year", y="Cited by", size="Cited by", color="Revista Corta",
            hover_name="Title", size_max=40, 
            title="Impacto por Año (Tamaño = N° de Citas)",
            template="plotly_white"
        )
        fig_bubble.update_layout(xaxis=dict(tickmode='linear', dtick=1)) # Años enteros
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        with st.expander("📖 ¿Cómo interpretar este gráfico?"):
            st.info("Cada burbuja es una investigación. **El eje vertical y el tamaño** indican qué tan respetado (citado) es el documento. Si ves una burbuja gigante, significa que ese paper contiene un modelo o caso de éxito que el banco DEBERÍA leer obligatoriamente.")

    # =====================================================================
    # SECCIÓN 3: TABLA DE DATOS INTERACTIVA
    # =====================================================================
    st.divider()
    st.subheader("🗂️ Directorio de Soluciones Antifuga")
    st.markdown("Selecciona y ordena los papers para profundizar en sus hallazgos.")
    
    df_mostrar = df_filtrado[['Title', 'Year', 'Cited by', 'Revista Corta', 'Link']].copy()
    
    st.dataframe(
        df_mostrar.sort_values('Cited by', ascending=False), 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Title": st.column_config.TextColumn("Título de la Investigación", width="large"),
            "Year": st.column_config.NumberColumn("Año", format="%d"),
            "Cited by": st.column_config.NumberColumn("Citas (Autoridad)"),
            "Revista Corta": st.column_config.TextColumn("Publicado en"),
            "Link": st.column_config.LinkColumn("Documento Original")
        }
    )

if __name__ == "__main__":
    main()