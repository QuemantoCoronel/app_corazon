import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Configuración de página
st.set_page_config(page_title="CardioGuard AI", layout="wide")
st.title("❤️ CardioGuard AI: Sistema Clínico Inteligente")

# Función auxiliar para determinar la causa probable de muerte
def determinar_causa(row):
    # Jerarquía de severidad médica (Lógica clínica simplificada)
    if row['serum_creatinine'] >= 1.8:
        return "Falla Renal Severa"
    elif row['ejection_fraction'] < 30:
        return "Falla Cardíaca (Bajo Bombeo)"
    elif row['platelets'] < 150000:
        return "Trombocitopenia/Problemas Sangre"
    elif row['high_blood_pressure'] == 1:
        return "Complicación Hipertensiva"
    elif row['diabetes'] == 1:
        return "Complicación Diabética"
    else:
        return "Causas Generales / Edad"

# Carga de datos
uploaded_file = st.file_uploader("📂 Cargar expediente clínico (CSV)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Datos cargados correctamente")

    # --- SECCIÓN 1: GRÁFICAS GLOBALES ---
    st.header("📊 Análisis Global de Datos")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Mapa de Calor (Correlaciones)")
        fig_corr, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax) # Annot false para que se vea más limpio en cel
        st.pyplot(fig_corr)

    with col2:
        st.subheader("2. Factores de Riesgo (Importancia)")
        X = df.drop('DEATH_EVENT', axis=1)
        y = df['DEATH_EVENT']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        fig_imp, ax = plt.subplots(figsize=(6, 5))
        feat_importances.nlargest(7).plot(kind='barh', color='teal', ax=ax)
        st.pyplot(fig_imp)

    st.markdown("---")

    # --- SECCIÓN 2: GESTIÓN DE PACIENTES ---
    st.header("👥 Gestión de Pacientes")
    tab_vivos, tab_fallecidos = st.tabs(["🟢 Pacientes Vivos (Prevención)", "🔴 Análisis de Defunciones"])

    # --- PESTAÑA 1: VIVOS ---
    with tab_vivos:
        vivos = df[df['DEATH_EVENT'] == 0]
        st.metric("Total Pacientes Vivos", len(vivos))
        st.subheader("🩺 Diagnóstico y Soluciones Sugeridas")
        
        for index, row in vivos.iterrows():
            acciones = []
            if row['serum_creatinine'] > 1.4:
                acciones.append(f"⚠️ Riñones: Creatinina {row['serum_creatinine']} (Alta). Revisar nefrotoxicidad.")
            if row['ejection_fraction'] < 30:
                acciones.append(f"⚠️ Corazón: Eyección {row['ejection_fraction']}% (Crítica). Evaluar inotrópicos.")
            if row['high_blood_pressure'] == 1:
                acciones.append("⚠️ Presión: Hipertensión activa. Controlar sodio.")
            
            if acciones:
                with st.expander(f"Paciente #{index} (Edad: {int(row['age'])}) - {len(acciones)} Alertas"):
                    for accion in acciones:
                        st.markdown(f"- {accion}")

    # --- PESTAÑA 2: FALLECIDOS---
    with tab_fallecidos:
        fallecidos = df[df['DEATH_EVENT'] == 1].copy() # Usamos copy para evitar warnings
        
        # 1. Calcular causas probables antes de graficar
        fallecidos['Causa_Probable'] = fallecidos.apply(determinar_causa, axis=1)
        conteo_causas = fallecidos['Causa_Probable'].value_counts()

        # GRÁFICA 1: Distribución de Edades
        st.subheader("1. Distribución de Edades al Fallecer")
        fig_hist, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(fallecidos['age'], kde=True, color='darkred', bins=15, ax=ax)
        ax.set_xlabel("Edad")
        ax.set_ylabel("Cantidad de Pacientes")
        st.pyplot(fig_hist)

        st.markdown("---")

        col_pastel, col_datos = st.columns([1, 1])

        # GRÁFICA 2: Pastel de Causas Probables
        with col_pastel:
            st.subheader("2. Causas Probables de Muerte")
            st.caption("Basado en el factor clínico más crítico del paciente.")
            
            fig_pie, ax = plt.subplots()
            # Colores personalizados para el pastel
            colors = sns.color_palette('pastel')[0:len(conteo_causas)]
            ax.pie(conteo_causas, labels=conteo_causas.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  
            st.pyplot(fig_pie)

        # LISTA AGRUPADA: Agrupación por cantidades
        with col_datos:
            st.subheader("3. Detalle por Grupo")
            st.write(f"Total Fallecidos: **{len(fallecidos)}**")
            
            for causa, cantidad in conteo_causas.items():
                with st.expander(f"📂 {causa}: {cantidad} pacientes"):
                    # Filtramos quiénes son
                    pacientes_grupo = fallecidos[fallecidos['Causa_Probable'] == causa]
                    # Mostramos un resumen limpio
                    st.table(pacientes_grupo[['age', 'sex', 'diabetes', 'smoking']].head(10))
                    if len(pacientes_grupo) > 10:
                        st.caption("Mostrando los primeros 10 casos.")

else:
    st.info("Esperando archivo CSV...")
