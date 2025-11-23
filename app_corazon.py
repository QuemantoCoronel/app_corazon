import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configuración de la página (Título y diseño)
st.set_page_config(page_title="CardioGuard AI", layout="wide", initial_sidebar_state="expanded")

st.title("❤️ CardioGuard AI: Análisis de Riesgo Cardíaco")
st.markdown("""
Esta aplicación permite cargar datos clínicos, visualizar estadísticas clave y recibir
**recomendaciones automáticas** para pacientes con insuficiencia cardíaca.
""")

# 1. Módulo de Carga de Datos
uploaded_file = st.file_uploader("📂 Cargar base de datos (CSV)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("¡Datos cargados exitosamente!")
    
    # Mostrar vista previa
    if st.checkbox("Ver datos crudos"):
        st.dataframe(df.head())

    # 2. Dashboard de Gráficas (Visualización)
    st.header("📊 Análisis Visual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Fallecimientos")
        fig_pie = px.pie(df, names='DEATH_EVENT', title='Proporción de Fallecimientos (0=No, 1=Sí)')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Edad vs. Fracción de Eyección")
        fig_scat = px.scatter(df, x='age', y='ejection_fraction', color='DEATH_EVENT', 
                              title="Relación Edad - Función Cardíaca")
        st.plotly_chart(fig_scat, use_container_width=True)

    # 3. Módulo de Recomendaciones Inteligentes (Lógica de Negocio)
    st.header("🩺 Recomendaciones Clínicas Automáticas")
    
    # Definir umbrales de riesgo basados en análisis estadístico previo
    # Umbrales: Creatinina > 1.4 (Cuartil superior), Eyección < 30 (Cuartil inferior)
    risk_creatinine = 1.4
    risk_ejection = 30
    
    # Filtrar pacientes de alto riesgo
    high_risk_patients = df[
        (df['serum_creatinine'] > risk_creatinine) | 
        (df['ejection_fraction'] < risk_ejection)
    ]
    
    st.warning(f"⚠️ Se han detectado **{len(high_risk_patients)} pacientes** con indicadores de alto riesgo.")
    
  # Generador de Reporte
    with st.expander(f"Ver Recomendaciones Detalladas ({len(high_risk_patients)} casos detectados)"):
        
        # 1. Tabla Resumen (Mejor para analizar muchos datos de golpe)
        st.write("### 📋 Tabla de Pacientes en Riesgo")
        st.dataframe(high_risk_patients)
        
        # 2. Lista Detallada (Uno por uno)
        st.write("### 🩺 Análisis Individual")
        for index, row in high_risk_patients.iterrows(): 
            reasons = []
            if row['serum_creatinine'] > risk_creatinine:
                reasons.append(f"Creatinina Alta ({row['serum_creatinine']} mg/dL)")
            if row['ejection_fraction'] < risk_ejection:
                reasons.append(f"Fracción de Eyección Baja ({row['ejection_fraction']}%)")
            
            # Usamos un estilo diferente si el paciente falleció (dato histórico)
            estado = "🔴 Fallecido" if row['DEATH_EVENT'] == 1 else "🟢 Vivo"
            
            st.markdown(f"**Paciente ID #{index}** (Edad: {int(row['age'])}) - {estado}")
            st.info(f"👉 Factores de riesgo: {', '.join(reasons)}")
            st.markdown("---") # Una línea separadora

else:

    st.info("Esperando archivo CSV... Por favor cargue la base de datos para iniciar.")

