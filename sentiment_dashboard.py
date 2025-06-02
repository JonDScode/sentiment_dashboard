import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Análisis de Sentimientos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Dashboard de Análisis de Sentimientos")
st.markdown("---")

# Cargar datos - REEMPLAZAR CON TUS DATOS REALES
@st.cache_data
def load_data():
 
    return pd.read_csv('out.csv')

# Cargar datos
df = load_data()

# Crear columna auxiliar para visualización
df['Sentimiento'] = df['label'].map({0.0: 'Negativo', 1.0: 'Positivo'})

# Sidebar para controles
st.sidebar.header("⚙️ Configuración del Dashboard")

# Selector de tipo de análisis
analysis_type = st.sidebar.selectbox(
    "Tipo de Análisis:",
    ["Exploración General", "Análisis de Variables", "Matriz de Correlación", "Distribuciones", "Comparación por Sentimiento"]
)

# Filtros generales
st.sidebar.subheader("Filtros")
sentiment_filter = st.sidebar.multiselect(
    "Filtrar por Sentimiento:",
    options=df['Sentimiento'].unique(),
    default=df['Sentimiento'].unique()
)

# Aplicar filtros
df_filtered = df[df['Sentimiento'].isin(sentiment_filter)]

# Variables numéricas para análisis (usando los nombres exactos de tu dataset)
original_vars = ['A', 'B', 'C', 'D', 'E']
standardized_vars = ['A_t', 'B_t', 'C_t', 'D_t', 'E_t']
combined_vars = ['Valor_1', 'Valor_2', 'Valor_3', 'Valor_4', 'Valor_5', 
                'Valor_6', 'Valor_7', 'Valor_8', 'Valor_9', 'Valor_10']
numeric_vars = original_vars + standardized_vars + combined_vars

if analysis_type == "Exploración General":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", len(df_filtered))
    with col2:
        st.metric("Sentimientos Positivos", len(df_filtered[df_filtered['label'] == 1.0]))
    with col3:
        st.metric("Sentimientos Negativos", len(df_filtered[df_filtered['label'] == 0.0]))
    with col4:
        st.metric("Variables Analizadas", len(numeric_vars))
    
    st.subheader("📈 Resumen Estadístico")
    
    # Tabla resumen
    summary_stats = df_filtered[numeric_vars].describe()
    st.dataframe(summary_stats.round(3))
    
    # Gráfico de barras con conteos por sentimiento
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            df_filtered.groupby('Sentimiento').size().reset_index(name='Cantidad'),
            x='Sentimiento', y='Cantidad',
            title="Distribución de Sentimientos",
            color='Sentimiento',
            color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Gráfico de torta
        fig_pie = px.pie(
            df_filtered,
            names='Sentimiento',
            title="Proporción de Sentimientos",
            color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

elif analysis_type == "Análisis de Variables":
    st.subheader("🔍 Análisis Detallado de Variables")
    
    # Selector de variables
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Variable X:", numeric_vars, index=0)
    with col2:
        y_var = st.selectbox("Variable Y:", numeric_vars, index=1)
    
    # Gráfico de dispersión interactivo
    fig_scatter = px.scatter(
        df_filtered, x=x_var, y=y_var,
        color='Sentimiento',
        title=f"Relación entre {x_var} y {y_var}",
        color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'},
        hover_data=['A', 'B', 'C', 'D', 'E']
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Histogramas comparativos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist_x = px.histogram(
            df_filtered, x=x_var, color='Sentimiento',
            title=f"Distribución de {x_var}",
            color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'},
            opacity=0.7
        )
        st.plotly_chart(fig_hist_x, use_container_width=True)
    
    with col2:
        fig_hist_y = px.histogram(
            df_filtered, x=y_var, color='Sentimiento',
            title=f"Distribución de {y_var}",
            color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'},
            opacity=0.7
        )
        st.plotly_chart(fig_hist_y, use_container_width=True)

elif analysis_type == "Matriz de Correlación":
    st.subheader("🔗 Matriz de Correlación")
    
    # Selector de variables para correlación
    selected_vars = st.multiselect(
        "Selecciona variables para la matriz de correlación:",
        numeric_vars,
        default=numeric_vars[:10]
    )
    
    if len(selected_vars) > 1:
        # Calcular matriz de correlación
        corr_matrix = df_filtered[selected_vars].corr()
        
        # Crear heatmap interactivo
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlación",
            color_continuous_scale="RdBu_r"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Mostrar correlaciones más altas
        st.subheader("🔝 Correlaciones Más Significativas")
        
        # Obtener correlaciones sin la diagonal
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                corr_pairs.append({'Variable 1': var1, 'Variable 2': var2, 'Correlación': corr_val})
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Correlación Abs'] = abs(corr_df['Correlación'])
        top_corr = corr_df.nlargest(10, 'Correlación Abs')
        
        st.dataframe(top_corr[['Variable 1', 'Variable 2', 'Correlación']].round(3))

elif analysis_type == "Distribuciones":
    st.subheader("📊 Análisis de Distribuciones")
    
    # Selector de variable
    selected_var = st.selectbox("Selecciona una variable:", numeric_vars)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma
        fig_hist = px.histogram(
            df_filtered, x=selected_var,
            color='Sentimiento',
            title=f"Histograma de {selected_var}",
            color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'},
            marginal="box"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            df_filtered, x='Sentimiento', y=selected_var,
            title=f"Box Plot de {selected_var} por Sentimiento",
            color='Sentimiento',
            color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Violin plot
    fig_violin = px.violin(
        df_filtered, x='Sentimiento', y=selected_var,
        title=f"Violin Plot de {selected_var} por Sentimiento",
        color='Sentimiento',
        color_discrete_map={'Positivo': '#2E8B57', 'Negativo': '#DC143C'},
        box=True
    )
    fig_violin.update_layout(height=400)
    st.plotly_chart(fig_violin, use_container_width=True)

elif analysis_type == "Comparación por Sentimiento":
    st.subheader("⚖️ Comparación Detallada por Sentimiento")
    
    # Métricas comparativas
    pos_data = df_filtered[df_filtered['Sentimiento'] == 'Positivo']
    neg_data = df_filtered[df_filtered['Sentimiento'] == 'Negativo']
    
    # Selector de variables para comparar
    vars_to_compare = st.multiselect(
        "Variables a comparar:",
        original_vars,  # Solo las variables originales A, B, C, D, E
        default=['A', 'B']
    )
    
    if vars_to_compare:
        # Crear tabla comparativa
        comparison_data = []
        for var in vars_to_compare:
            pos_mean = pos_data[var].mean() if len(pos_data) > 0 else 0
            neg_mean = neg_data[var].mean() if len(neg_data) > 0 else 0
            pos_std = pos_data[var].std() if len(pos_data) > 0 else 0
            neg_std = neg_data[var].std() if len(neg_data) > 0 else 0
            
            comparison_data.append({
                'Variable': var,
                'Media Positivo': pos_mean,
                'Media Negativo': neg_mean,
                'Desv. Est. Positivo': pos_std,
                'Desv. Est. Negativo': neg_std,
                'Diferencia': pos_mean - neg_mean
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.round(3))
        
        # Gráfico de barras comparativo
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            name='Sentimiento Positivo',
            x=comparison_df['Variable'],
            y=comparison_df['Media Positivo'],
            marker_color='#2E8B57'
        ))
        
        fig_comp.add_trace(go.Bar(
            name='Sentimiento Negativo',
            x=comparison_df['Variable'],
            y=comparison_df['Media Negativo'],
            marker_color='#DC143C'
        ))
        
        fig_comp.update_layout(
            title='Comparación de Medias por Sentimiento',
            xaxis_title='Variables',
            yaxis_title='Valor Promedio',
            barmode='group'
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Información del Dataset")
st.sidebar.write(f"**Registros filtrados:** {len(df_filtered):,}")
st.sidebar.write(f"**Total de variables:** {len(numeric_vars)}")
st.sidebar.write(f"**Variables originales:** {', '.join(original_vars)}")
st.sidebar.write(f"**Variables estandarizadas:** {', '.join(standardized_vars)}")
st.sidebar.write(f"**Variables combinadas:** {', '.join(combined_vars[:3])}...")
st.sidebar.write(f"**Columna objetivo:** label (0.0=Negativo, 1.0=Positivo)")

# Footer
st.markdown("---")
st.markdown("**Desarrollado para análisis de sentimientos** | Dashboard interactivo con Streamlit")