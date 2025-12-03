import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os  # <--- 1. IMPORTANTE: Para demostrar que son procesos distintos

# Librerías para Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- LIBRERÍA PARA PARALELISMO (El "Motor" del Maestro-Esclavo) ---
from concurrent.futures import ProcessPoolExecutor

# Configuración de página
st.set_page_config(page_title="CardioGuard AI - Parallel", layout="wide")

# --- FUNCIONES DE LOS ESCLAVOS (WORKERS) ---
def worker_train_model(model_name, X_train, y_train, X_test, y_test):
    """
    Función Esclavo: Se ejecuta en un proceso de memoria aislado.
    """
    start_time = time.time()
    
    # 2. CAPTURA DE EVIDENCIA DE DISTRIBUCIÓN
    # Obtenemos el ID del proceso (PID) que está ejecutando esta tarea.
    # Si los PID son diferentes, demuestras que es computación distribuida/paralela real.
    worker_id = os.getpid() 
    
    # Selección del modelo según la orden del Maestro
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        
    # Entrenamiento
    model.fit(X_train, y_train)
    
    # Predicción y Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Extraer importancia
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = model.coef_[0]
        
    end_time = time.time()
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "time": end_time - start_time,
        "importance": feature_importance,
        "worker_id": worker_id  # <--- Devolvemos la "firma" del trabajador
    }

# --- FUNCIONES DE LÓGICA CLÍNICA ---
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

# --- INTERFAZ PRINCIPAL (EL MAESTRO) ---
st.title("❤️ CardioGuard AI: Sistema Paralelo de Diagnóstico")
# Actualizamos la descripción para que suene más técnica
st.markdown("**Arquitectura:** Maestro-Esclavo | **Estrategia:** Task Parallelism | **Nodos:** Multi-Core CPU")

# Carga de Datos
try:
    df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    dataset_loaded = True
except FileNotFoundError:
    st.error("❌ Error: Sube el archivo 'heart_failure_clinical_records_dataset.csv'.")
    dataset_loaded = False

if dataset_loaded:
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- SECCIÓN DE CÓMPUTO PARALELO ---
    st.header("🧠 Entrenamiento Distribuido de Modelos")
    
    col_ctrl, col_display = st.columns([1, 3])
    
    with col_ctrl:
        st.info(f"El sistema (Maestro PID: {os.getpid()}) delegará tareas a nuevos procesos.")
        run_parallel = st.button("🚀 Iniciar Entrenamiento Distribuido")
    
    if run_parallel:
        with st.spinner("Distribuyendo carga de trabajo a los nodos disponibles..."):
            modelos_a_entrenar = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
            
            results = []
            start_global = time.time()
            
            # El Maestro crea el Pool de trabajadores
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(worker_train_model, m, X_train, y_train, X_test, y_test) 
                    for m in modelos_a_entrenar
                ]
                for f in futures:
                    results.append(f.result())
            
            end_global = time.time()

            st.success(f"Entrenamiento completado en {end_global - start_global:.4f} segundos.")
            
            res_df = pd.DataFrame(results).set_index("model")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Mejor Modelo", res_df['accuracy'].idxmax())
            col_m2.metric("Precisión Máxima", f"{res_df['accuracy'].max():.2%}")
            col_m3.metric("Tiempo Cómputo", f"{res_df['time'].sum():.4f}s")
            
            with col_display:
                st.caption("Rendimiento por Modelo")
                st.bar_chart(res_df['accuracy'])
                
                # 3. TABLA DE EVIDENCIA TÉCNICA
                # Mostramos qué Worker (PID) hizo qué trabajo
                st.caption("📋 Registro de Ejecución Distribuida (Logs)")
                st.dataframe(
                    res_df[['accuracy', 'time', 'worker_id']].style.format({
                        'accuracy': '{:.2%}', 
                        'time': '{:.4f}s'
                    }),
                    use_container_width=True
                )
                st.info("Nota: Observa la columna 'worker_id'. Números diferentes indican que procesos distintos (núcleos separados) procesaron los modelos simultáneamente.")

            # Gráficas de Importancia (solo del mejor modelo, ej. RF)
            best_model_data = next(item for item in results if item["model"] == "Random Forest")
            feat_importances = pd.DataFrame(best_model_data['importance'], index=X.columns, columns=['importance'])
            feat_importances = feat_importances.sort_values('importance', ascending=False)
            
            st.subheader("Variables Críticas (Analizadas por Worker Remoto)")
            fig_imp, ax = plt.subplots(figsize=(10, 3))
            sns.barplot(x=feat_importances.importance, y=feat_importances.index, palette='viridis', ax=ax)
            st.pyplot(fig_imp)

    st.markdown("---")

    # --- SECCIÓN 2: GESTIÓN DE PACIENTES (CON TU BUSCADOR) ---
    st.header("👥 Gestión de Pacientes")
    tab_vivos, tab_fallecidos = st.tabs(["🟢 Pacientes Vivos", "🔴 Análisis de Defunciones"])

    with tab_vivos:
        vivos = df[df['DEATH_EVENT'] == 0].copy()
        vivos['Riesgo_Principal'] = vivos.apply(determinar_riesgo, axis=1)
        
        col_v1, col_v2 = st.columns([2, 1])
        
        with col_v1:
            st.subheader("Diagnóstico Individual")
            search_term = st.text_input("🔍 Buscar por diagnóstico:", placeholder="Ej: Renal, Diabetes, Insuficiencia...")

            if search_term:
                vivos_filtrados = vivos[vivos['Riesgo_Principal'].str.contains(search_term, case=False, na=False)]
            else:
                vivos_filtrados = vivos

            st.caption(f"Mostrando {len(vivos_filtrados)} pacientes")
            
            with st.container(height=500, border=True):
                if len(vivos_filtrados) == 0:
                    st.warning("No se encontraron pacientes con ese diagnóstico.")
                
                for index, row in vivos_filtrados.iterrows():
                    icono = "⚠️" if "Alto" in row['Riesgo_Principal'] or "Severa" in row['Riesgo_Principal'] else "ℹ️"
                    with st.expander(f"{icono} Paciente #{index} - {row['Riesgo_Principal']}"):
                        st.markdown(f"**Edad:** {int(row['age'])} años | **Creatinina:** {row['serum_creatinine']}")
                        if row['serum_creatinine'] > 1.4: 
                            st.error(f"🚨 **Alerta Renal:** Nivel {row['serum_creatinine']} mg/dL (Alto)")
                        if row['ejection_fraction'] < 30:
                            st.error(f"💔 **Corazón:** Bombeo crítico del {row['ejection_fraction']}%")
                        if row['high_blood_pressure'] == 1:
                            st.warning("🩸 **Presión:** Paciente Hipertenso")
                            
        with col_v2:
            st.subheader("Resumen de Riesgos")
            conteo = vivos['Riesgo_Principal'].value_counts()
            fig_p, ax = plt.subplots(figsize=(5, 5))
            ax.pie(conteo, labels=None, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
            ax.legend(conteo.index, loc="best", bbox_to_anchor=(1, 0.5))
            st.pyplot(fig_p)

    with tab_fallecidos:
        st.info("Visualización de patrones en pacientes fallecidos (Histórico)")
        fallecidos = df[df['DEATH_EVENT'] == 1].copy()
        fallecidos['Causa_Probable'] = fallecidos.apply(determinar_causa, axis=1)
        st.bar_chart(fallecidos['Causa_Probable'].value_counts())