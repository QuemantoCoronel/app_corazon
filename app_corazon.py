import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Configuración de página
st.set_page_config(page_title="CardioGuard AI - Distributed", layout="wide")
st.title("❤️ CardioGuard AI: Sistema Clínico Distribuido")
st.markdown("**Arquitectura:** Maestro-Trabajador | **Consenso:** Votación Paralela")

# --- FUNCIONES DE CLASIFICACIÓN CLÍNICA (Reglas de Negocio) ---
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

# --- LÓGICA DISTRIBUIDA (Simulación Maestro-Trabajador) ---
def nodo_trabajador(model_name, model_instance, X_train, y_train, X_test, y_test):
    """Simula un nodo de procesamiento independiente"""
    model_instance.fit(X_train, y_train)
    acc = model_instance.score(X_test, y_test)
    return model_name, acc, model_instance

# --- CARGA Y PROCESAMIENTO ---
try:
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    dataset_loaded = True
except FileNotFoundError:
    st.error("❌ Error: No se encontró el archivo CSV en el repositorio.")
    dataset_loaded = False

if dataset_loaded:
    # Preparación de datos para el Cluster ML
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- PANEL DE CONTROL DE SISTEMAS (Backend Status) ---
    with st.expander("🖥️ Monitor de Sistema Distribuido (Backend)", expanded=True):
        col_sys1, col_sys2 = st.columns([1, 3])
        with col_sys1:
            st.info("Estado del Cluster: **ACTIVO**")
            st.write("Nodos disponibles: 3")
        
        with col_sys2:
            st.write("🔄 **Procesando Votación Paralela...**")
            # Ejecución Paralela (Thread Pool)
            modelos = {
                "Nodo 1 (Random Forest)": RandomForestClassifier(n_estimators=50),
                "Nodo 2 (Logistic Reg)": LogisticRegression(max_iter=1000),
                "Nodo 3 (SVM Kernel)": SVC(probability=True)
            }
            
            resultados_nodos = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(nodo_trabajador, name, m, X_train, y_train, X_test, y_test): name for name, m in modelos.items()}
                
                cols_nodos = st.columns(3)
                idx = 0
                for future in concurrent.futures.as_completed(futures):
                    nombre, acc, _ = future.result()
                    resultados_nodos[nombre] = acc
                    with cols_nodos[idx]:
                        st.metric(label=nombre, value=f"{acc:.1%}", delta="Completado")
                    idx += 1
            
            # El modelo maestro (Main) elige al mejor para mostrar las importancias
            mejor_modelo_nombre = max(resultados_nodos, key=resultados_nodos.get)
            st.success(f"✅ Consenso alcanzado. Nodo Maestro seleccionado: **{mejor_modelo_nombre}**")

    st.markdown("---")

    # --- FRONTEND CLÍNICO (Visualización) ---
    st.header("📊 Análisis Global de Datos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Distribución de Desenlace")
        st.caption("Comparativa: Pacientes Vivos vs Fallecidos")
        fig_dist, ax = plt.subplots(figsize=(6, 5))
        sns.countplot(x='DEATH_EVENT', data=df, palette='viridis', ax=ax)
        st.pyplot(fig_dist)

    with col2:
        st.subheader("2. Mapa de Calor (Correlaciones)")
        st.caption("Relación numérica entre variables")
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

    # Importancia de Variables (Usando el Random Forest del Nodo 1 para visualización)
    st.subheader("3. Variables Críticas (Votación del Nodo 1)")
    rf_model = modelos["Nodo 1 (Random Forest)"] # Recuperamos el modelo entrenado
    feat_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance'])
    feat_importances = feat_importances.sort_values('importance', ascending=False)
    
    fig_imp, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=feat_importances.importance, y=feat_importances.index, palette='viridis', ax=ax)
    st.pyplot(fig_imp)

    st.markdown("---")

    # --- GESTIÓN DE PACIENTES ---
    st.header("👥 Gestión de Pacientes")
    tab_vivos, tab_fallecidos = st.tabs(["🟢 Pacientes Vivos (Prevención)", "🔴 Análisis de Defunciones"])

    # --- PESTAÑA 1: VIVOS ---
    with tab_vivos:
        vivos = df[df['DEATH_EVENT'] == 0].copy()
        vivos['Riesgo_Principal'] = vivos.apply(determinar_riesgo, axis=1)
        conteo_riesgos = vivos['Riesgo_Principal'].value_counts()
        
        st.subheader("1. Distribución de Edades (Pacientes Vivos)")
        fig_hist_v, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(vivos['age'], kde=True, color='forestgreen', bins=15, ax=ax)
        st.pyplot(fig_hist_v)
        
        col_v_graf, col_v_data = st.columns([1, 2])
        
        with col_v_graf:
            st.subheader("2. Patologías Activas")
            fig_pie_v, ax = plt.subplots()
            colors = sns.color_palette('pastel')[0:len(conteo_riesgos)]
            ax.pie(conteo_riesgos, labels=conteo_riesgos.index, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color':"black"})
            ax.axis('equal')
            st.pyplot(fig_pie_v)
            
        with col_v_data:
            st.metric("Total Pacientes en Seguimiento", len(vivos))
            st.subheader("3. Diagnóstico y Tratamiento Sugerido")
            for index, row in vivos.iterrows():
                recommendations = []
                if row['serum_creatinine'] > 1.4:
                    recommendations.append({"area": "Riñones", "diag": f"Creatinina elevada ({row['serum_creatinine']}).", "sol": "Solicitar ecografía renal."})
                if row['ejection_fraction'] < 30:
                    recommendations.append({"area": "Corazón", "diag": f"Eyección crítica ({row['ejection_fraction']}%).", "sol": "Evaluar terapia de resincronización."})
                if row['high_blood_pressure'] == 1:
                    recommendations.append({"area": "Presión", "diag": "Hipertensión detectada.", "sol": "Revisar dieta hiposódica."})
                
                if recommendations:
                    with st.expander(f"Paciente #{index} (Edad: {int(row['age'])}) - {row['Riesgo_Principal']}"):
                        for rec in recommendations:
                            st.markdown(f"**⚠️ {rec['diag']}**")
                            st.info(f"💡 {rec['sol']}")

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
            colors = sns.color_palette('Set2')[0:len(conteo_causas)]
            ax.pie(conteo_causas, labels=conteo_causas.index, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color':"black"})
            ax.axis('equal')  
            st.pyplot(fig_pie)

        with col_datos:
            st.subheader("3. Detalle por Grupo")
            for causa, cantidad in conteo_causas.items():
                with st.expander(f"📂 {causa}: {cantidad} pacientes"):
                    st.table(fallecidos[fallecidos['Causa_Probable'] == causa][['age', 'sex', 'diabetes']].head(5))
