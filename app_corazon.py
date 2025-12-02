import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Configuración de página
st.set_page_config(page_title="CardioGuard AI", layout="wide")
st.title("❤️ CardioGuard AI: Sistema Clínico Inteligente")

# --- FUNCIONES DE CLASIFICACIÓN ---
def determinar_causa(row):
    if row['serum_creatinine'] >= 1.8: return "Falla Renal Severa"
    elif row['ejection_fraction'] < 30: return "Falla Cardíaca (Bajo Bombeo)"
    elif row['platelets'] < 150000:     return "Problemas de Coagulación"
    elif row['high_blood_pressure'] == 1: return "Hipertensión Crónica"
    elif row['diabetes'] == 1:          return "Complicación Diabética"
    else:                               return "Causas Generales"

def determinar_riesgo(row):
    if row['serum_creatinine'] > 1.4:   return "Alto Riesgo Renal"
    elif row['ejection_fraction'] < 30: return "Insuficiencia Cardíaca Severa"
    elif row['high_blood_pressure'] == 1: return "Hipertensión No Controlada"
    elif row['anaemia'] == 1:           return "Anemia Persistente"
    elif row['diabetes'] == 1:          return "Diabetes"
    else:                               return "Bajo Riesgo Aparente"

# --- CARGA AUTOMÁTICA DE DATOS ---
try:
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    dataset_loaded = True
except FileNotFoundError:
    st.error("❌ Error: No se encontró el archivo CSV en el repositorio.")
    st.info("Asegúrate de subir 'heart_failure_clinical_records_dataset.csv' a GitHub junto con este código.")
    dataset_loaded = False

if dataset_loaded:
    # --- SECCIÓN 1: GRÁFICAS GLOBALES ---
    st.header("📊 Análisis Global de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Distribución de Desenlace")
        st.caption("Comparativa: Pacientes Vivos vs Fallecidos")
        fig_dist, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(x='DEATH_EVENT', data=df, palette='viridis', ax=ax)
        ax.set_xlabel("Estado (0: Vivo, 1: Fallecido)")
        ax.set_ylabel("Cantidad de Pacientes")
        st.pyplot(fig_dist)

    with col2:
        st.subheader("2. Mapa de Calor (Correlaciones)")
        st.caption("Relación numérica entre variables")
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

    st.subheader("3. Variables Críticas (Análisis de IA)")
    st.caption("Factores que más influyen en el riesgo de muerte según el modelo.")
    
    # Modelado de Predicción
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    feat_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance'])
    feat_importances = feat_importances.sort_values('importance', ascending=False)
    
    fig_imp, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=feat_importances.importance, y=feat_importances.index, palette='viridis', ax=ax)
    ax.set_xlabel("Nivel de Importancia")
    st.pyplot(fig_imp)

    st.markdown("---")

    # --- SECCIÓN 2: GESTIÓN DE PACIENTES ---
    st.header("👥 Gestión de Pacientes")
    tab_vivos, tab_fallecidos = st.tabs(["🟢 Pacientes Vivos (Prevención)", "🔴 Análisis de Defunciones"])

    # --- PESTAÑA 1: VIVOS ---
    with tab_vivos:
        vivos = df[df['DEATH_EVENT'] == 0].copy()
        vivos['Riesgo_Principal'] = vivos.apply(determinar_riesgo, axis=1)
        conteo_riesgos = vivos['Riesgo_Principal'].value_counts()
        
        # --- NUEVA GRÁFICA: DISTRIBUCIÓN DE EDADES (VIVOS) ---
        st.subheader("1. Distribución de Edades (Pacientes Vivos)")
        fig_hist_v, ax = plt.subplots(figsize=(8, 3))
        # Usamos color verde (forestgreen) para diferenciar de los fallecidos (rojo)
        sns.histplot(vivos['age'], kde=True, color='forestgreen', bins=15, ax=ax)
        ax.set_xlabel("Edad")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig_hist_v)
        
        st.markdown("---")

        col_v_graf, col_v_data = st.columns([1, 2])
        
        with col_v_graf:
            st.subheader("2. Patologías Activas")
            fig_pie_v, ax = plt.subplots(figsize=(6, 6))  # Ajustar el tamaño del gráfico para mejor visibilidad
            colors = sns.color_palette('Set3')[0:len(conteo_riesgos)]  # Cambiar a paleta Set3 para colores más suaves
            wedges, texts, autotexts = ax.pie(conteo_riesgos, 
                                            labels=conteo_riesgos.index, 
                                            autopct='%1.1f%%', 
                                            startangle=90, 
                                            colors=colors, 
                                            textprops={'color':"black", 'fontsize':12},  # Estilo de texto
                                            wedgeprops={'edgecolor': 'black'})  # Añadir borde negro para mejor definición
            ax.axis('equal')  # Hace que el gráfico sea circular
            plt.setp(autotexts, size=12, weight="bold", color="white")  # Mejorar visibilidad del porcentaje
            st.pyplot(fig_pie_v)
            
        with col_v_data:
            st.metric("Total Pacientes en Seguimiento", len(vivos))

        # Mover "Diagnóstico y Tratamiento Sugerido" debajo del gráfico
        st.subheader("3. Diagnóstico y Tratamiento Sugerido")

        # Mostrar diagnóstico y tratamiento sugerido para los primeros pacientes
        for index, row in vivos.iterrows():
            recommendations = []
            if row['serum_creatinine'] > 1.4:
                recommendations.append({
                    "area": "Riñones",
                    "diag": f"Creatinina elevada ({row['serum_creatinine']} mg/dL). Posible daño renal agudo.",
                    "sol": "Solicitar ecografía renal y ajustar dosis de medicamentos nefrotóxicos."
                })
            if row['ejection_fraction'] < 30:
                recommendations.append({
                    "area": "Corazón",
                    "diag": f"Fracción de eyección crítica ({row['ejection_fraction']}%).",
                    "sol": "Evaluar implante de dispositivo (DAI) o terapia de resincronización."
                })
            if row['high_blood_pressure'] == 1:
                recommendations.append({
                    "area": "Presión Arterial",
                    "diag": "Hipertensión arterial sistémica detectada.",
                    "sol": "Revisar adherencia al tratamiento antihipertensivo y dieta baja en sodio."
                })
            
            if recommendations:
                with st.expander(f"Paciente #{index} (Edad: {int(row['age'])}) - {row['Riesgo_Principal']}"):
                    for rec in recommendations:
                        st.markdown(f"**⚠️ Diagnóstico ({rec['area']}):** {rec['diag']}")
                        st.info(f"💡 **Solución:** {rec['sol']}")
                        st.markdown("---")

    # --- PESTAÑA 2: FALLECIDOS ---
    with tab_fallecidos:
        fallecidos = df[df['DEATH_EVENT'] == 1].copy()
        fallecidos['Causa_Probable'] = fallecidos.apply(determinar_causa, axis=1)
        conteo_causas = fallecidos['Causa_Probable'].value_counts()

        st.subheader("1. Distribución de Edades al Fallecer")
        fig_hist, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(fallecidos['age'], kde=True, color='darkred', bins=15, ax=ax)
        st.pyplot(fig_hist)

        st.markdown("---")

        col_pastel, col_datos = st.columns([1, 1])

        with col_pastel:
            st.subheader("2. Causas Probables")
            fig_pie, ax = plt.subplots()
            colors = sns.color_palette('Set2')[0:len(conteo_causas)]
            ax.pie(conteo_causas, labels=conteo_causas.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  
            st.pyplot(fig_pie)

        with col_datos:
            st.subheader("3. Detalle por Grupo")
            for causa, cantidad in conteo_causas.items():
                with st.expander(f"📂 {causa}: {cantidad} pacientes"):
                    st.table(fallecidos[fallecidos['Causa_Probable'] == causa][['age', 'sex', 'diabetes']].head(5))
