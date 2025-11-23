import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configuración de página
st.set_page_config(page_title="CardioGuard AI", layout="wide")
st.title("❤️ CardioGuard AI: Sistema Clínico Inteligente")

# Carga de datos
uploaded_file = st.file_uploader("📂 Cargar expediente clínico (CSV)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Datos cargados correctamente")

    # --- SECCIÓN 1: GRÁFICAS GLOBALES (Estilo Científico) ---
    st.header("📊 Análisis Global de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Mapa de Calor (Correlaciones)")
        # Recreamos la gráfica exacta que te gustó
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

    with col2:
        st.subheader("2. Factores de Riesgo (Importancia)")
        # Entrenamos un modelo rápido para sacar la importancia real
        X = df.drop('DEATH_EVENT', axis=1)
        y = df['DEATH_EVENT']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        
        fig_imp, ax = plt.subplots(figsize=(10, 8))
        feat_importances.nlargest(10).plot(kind='barh', color='teal', ax=ax)
        ax.set_title("Top 10 Variables que predicen fallecimiento")
        st.pyplot(fig_imp)

    st.markdown("---")

    # --- SECCIÓN 2: ANÁLISIS DIVIDIDO (Vivos vs Fallecidos) ---
    st.header("👥 Gestión de Pacientes")
    
    # Crear pestañas
    tab_vivos, tab_fallecidos = st.tabs(["🟢 Pacientes Vivos (Prevención)", "🔴 Análisis de Defunciones"])

    # --- PESTAÑA 1: VIVOS ---
    with tab_vivos:
        vivos = df[df['DEATH_EVENT'] == 0]
        st.metric("Total Pacientes Vivos", len(vivos))
        
        st.subheader("🩺 Diagnóstico y Soluciones Sugeridas")
        st.info("A continuación se presentan acciones preventivas para pacientes que siguen en tratamiento.")

        for index, row in vivos.iterrows():
            # Lógica de "Doctor Virtual" para sugerir soluciones
            acciones = []
            
            # 1. Problemas Renales
            if row['serum_creatinine'] > 1.4:
                acciones.append("⚠️ **Riñones:** Nivel alto de creatinina. **Solución:** Solicitar perfil renal completo y evaluar diuréticos.")
            
            # 2. Problemas Cardíacos (Bombeo)
            if row['ejection_fraction'] < 30:
                acciones.append("⚠️ **Corazón:** Fracción de eyección crítica (<30%). **Solución:** Evaluar terapia con betabloqueantes o marcapasos.")
            
            # 3. Hipertensión
            if row['high_blood_pressure'] == 1:
                acciones.append("⚠️ **Presión:** Hipertensión detectada. **Solución:** Monitoreo diario y reducir ingesta de sodio.")

            # 4. Anemia
            if row['anaemia'] == 1:
                acciones.append("⚠️ **Sangre:** Anemia presente. **Solución:** Suplementos de hierro y eritropoyetina.")

            # Si tiene algún riesgo, lo mostramos
            if acciones:
                with st.expander(f"Paciente #{index} - Edad: {int(row['age'])} años (Riesgo Detectado)"):
                    for accion in acciones:
                        st.markdown(f"- {accion}")
            
    # --- PESTAÑA 2: FALLECIDOS ---
    with tab_fallecidos:
        fallecidos = df[df['DEATH_EVENT'] == 1]
        st.metric("Total Defunciones", len(fallecidos))
        
        st.error("Estos pacientes han fallecido. Análisis retrospectivo para mejorar protocolos futuros.")
        
        st.write("### Datos de Pacientes Fallecidos")
        st.dataframe(fallecidos.style.highlight_max(axis=0))
        
        st.write("### Distribución de Edades en Fallecidos")
        fig_hist, ax = plt.subplots()
        sns.histplot(fallecidos['age'], kde=True, color='red', ax=ax)
        st.pyplot(fig_hist)

else:
    st.info("Esperando archivo CSV...")
