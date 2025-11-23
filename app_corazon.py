import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Configuración de página
st.set_page_config(page_title="CardioGuard AI", layout="wide")
st.title("❤️ CardioGuard AI: Sistema Clínico Inteligente")

# --- FUNCIONES DE CLASIFICACIÓN ---

# 1. Para Fallecidos (Causa probable de muerte)
def determinar_causa(row):
    if row['serum_creatinine'] >= 1.8: return "Falla Renal Severa"
    elif row['ejection_fraction'] < 30: return "Falla Cardíaca (Bajo Bombeo)"
    elif row['platelets'] < 150000:     return "Problemas de Coagulación"
    elif row['high_blood_pressure'] == 1: return "Hipertensión Crónica"
    elif row['diabetes'] == 1:          return "Complicación Diabética"
    else:                               return "Causas Generales"

# 2. Para Vivos (Riesgo Principal Latente)
def determinar_riesgo(row):
    if row['serum_creatinine'] > 1.4:   return "Alto Riesgo Renal"
    elif row['ejection_fraction'] < 30: return "Insuficiencia Cardíaca Severa"
    elif row['high_blood_pressure'] == 1: return "Hipertensión No Controlada"
    elif row['anaemia'] == 1:           return "Anemia Persistente"
    elif row['diabetes'] == 1:          return "Diabetes"
    else:                               return "Bajo Riesgo Aparente"

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
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

    with col2:
        st.subheader("2. Factores de Riesgo (IA)")
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
        vivos = df[df['DEATH_EVENT'] == 0].copy()
        
        # Clasificamos a los vivos para el gráfico
        vivos['Riesgo_Principal'] = vivos.apply(determinar_riesgo, axis=1)
        conteo_riesgos = vivos['Riesgo_Principal'].value_counts()
        
        col_v_graf, col_v_data = st.columns([1, 2])
        
        with col_v_graf:
            st.subheader("Patologías Activas")
            fig_pie_v, ax = plt.subplots()
            colors = sns.color_palette('pastel')[0:len(conteo_riesgos)]
            ax.pie(conteo_riesgos, labels=conteo_riesgos.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig_pie_v)
            
        with col_v_data:
            st.metric("Total Pacientes en Seguimiento", len(vivos))
            st.subheader("🩺 Diagnóstico y Tratamiento Sugerido")
            
            for index, row in vivos.iterrows():
                recommendations = []
                
                # Lógica detallada (Diagnóstico + Solución por separado)
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
                
                # Solo mostramos si hay recomendaciones
                if recommendations:
                    with st.expander(f"Paciente #{index} (Edad: {int(row['age'])}) - {row['Riesgo_Principal']}"):
                        for rec in recommendations:
                            st.markdown(f"**⚠️ Diagnóstico ({rec['area']}):** {rec['diag']}")
                            st.info(f"💡 **Solución:** {rec['sol']}") # El .info crea un recuadro azul/verde bonito
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

        col_pastel, col_datos = st.columns([1, 1])

        with col_pastel:
            st.subheader("2. Causas Probables")
            fig_pie, ax = plt.subplots()
            colors = sns.color_palette('Set2')[0:len(conteo_causas)] # Usamos otra paleta de colores
            ax.pie(conteo_causas, labels=conteo_causas.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  
            st.pyplot(fig_pie)

        with col_datos:
            st.subheader("3. Detalle por Grupo")
            for causa, cantidad in conteo_causas.items():
                with st.expander(f"📂 {causa}: {cantidad} pacientes"):
                    st.table(fallecidos[fallecidos['Causa_Probable'] == causa][['age', 'sex', 'diabetes']].head(5))

else:
    st.info("Esperando archivo CSV...")
