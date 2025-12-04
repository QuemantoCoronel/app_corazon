import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # IMPORTANTE: Usamos Plotly para gráficos interactivos
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

# --- FUNCIONES DE CLASIFICACIÓN CLÍNICA ---
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

# --- LÓGICA DISTRIBUIDA ---
def nodo_trabajador(model_name, model_instance, X_train, y_train, X_test, y_test):
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
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- BACKEND MONITOR ---
    with st.expander("🖥️ Monitor de Sistema Distribuido (Backend)", expanded=True):
        col_sys1, col_sys2 = st.columns([1, 3])
        with col_sys1:
            st.info("Estado del Cluster: **ACTIVO**")
            st.write("Nodos disponibles: 3")
        
        with col_sys2:
            st.write("🔄 **Procesando Votación Paralela...**")
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
            mejor_modelo_nombre = max(resultados_nodos, key=resultados_nodos.get)
            st.success(f"✅ Consenso alcanzado. Nodo Maestro seleccionado: **{mejor_modelo_nombre}**")

    st.markdown("---")

    # --- FRONTEND CLÍNICO ---
    st.header("📊 Análisis Global de Datos")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Distribución de Desenlace")
        st.caption("Comparativa: Pacientes Vivos vs Fallecidos")
        # Usamos Plotly aquí también para consistencia
        fig_dist = px.histogram(df, x="DEATH_EVENT", color="DEATH_EVENT", 
                                labels={"DEATH_EVENT": "Estado (0:Vivo, 1:Fallecido)"},
                                color_discrete_map={0: "green", 1: "red"})
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader("2. Mapa de Calor (Correlaciones)")
        st.caption("Relación numérica entre variables")
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

    st.subheader("3. Variables Críticas (Votación del Nodo 1)")
    rf_model = modelos["Nodo 1 (Random Forest)"]
    feat_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['importance'])
    feat_importances = feat_importances.sort_values('importance', ascending=False)
    
    fig_imp, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=feat_importances.importance, y=feat_importances.index, palette='viridis', ax=ax)
    st.pyplot(fig_imp)

    st.markdown("---")

    # --- GESTIÓN DE PACIENTES ---
    st.header("👥 Gestión de Pacientes")
    
    search_term = st.text_input("🔍 Buscar por enfermedad o condición (Ej: 'Renal', 'Diabetes', 'Corazón', 'Hipertensión'):", "")

    tab_vivos, tab_fallecidos = st.tabs(["🟢 Pacientes Vivos (Prevención)", "🔴 Análisis de Defunciones"])

    # --- PESTAÑA 1: VIVOS ---
    with tab_vivos:
        vivos = df[df['DEATH_EVENT'] == 0].copy()
        vivos['age'] = vivos['age'].astype(int)
        vivos['Riesgo_Principal'] = vivos.apply(determinar_riesgo, axis=1)
        
        if search_term:
            vivos = vivos[vivos['Riesgo_Principal'].str.contains(search_term, case=False, na=False)]
            st.info(f"Mostrando {len(vivos)} pacientes filtrados.")

        if not vivos.empty:
            subtab_lista, subtab_graficos = st.tabs(["📋 Lista de Diagnósticos", "📊 Ver Gráficas Interactivas"])
            
            # LISTA
            with subtab_lista:
                for index, row in vivos.iterrows():
                    recommendations = []
                    # Reglas Generales
                    if row['serum_creatinine'] > 1.4:
                        recommendations.append({"area": "Riñones", "diag": f"Creatinina elevada ({row['serum_creatinine']}).", "sol": "Solicitar ecografía renal."})
                    if row['ejection_fraction'] < 30:
                        recommendations.append({"area": "Corazón", "diag": f"Eyección crítica ({row['ejection_fraction']}%).", "sol": "Evaluar terapia de resincronización."})
                    if row['high_blood_pressure'] == 1:
                        recommendations.append({"area": "Presión", "diag": "Hipertensión detectada.", "sol": "Revisar dieta hiposódica."})
                    
                    # Reglas Adicionales (PARA QUE APAREZCAN AL BUSCAR DIABETES O ANEMIA)
                    if row['diabetes'] == 1:
                         recommendations.append({"area": "Metabólico", "diag": "Paciente Diabético.", "sol": "Control glucémico estricto y revisión de pies."})
                    if row['anaemia'] == 1:
                         recommendations.append({"area": "Sangre", "diag": "Anemia detectada.", "sol": "Evaluar ferroterapia y dieta rica en hierro."})
                    
                    # Si no tiene nada grave, ponemos mensaje genérico para que SIEMPRE aparezca
                    if not recommendations:
                        recommendations.append({"area": "General", "diag": "Sin alertas críticas inmediatas.", "sol": "Continuar monitoreo de rutina."})

                    with st.expander(f"Paciente #{index} (Edad: {row['age']}) - {row['Riesgo_Principal']}"):
                        for rec in recommendations:
                            st.markdown(f"**⚠️ {rec['diag']}**")
                            st.info(f"💡 {rec['sol']}")

            # GRÁFICAS INTERACTIVAS (PLOTLY)
            with subtab_graficos:
                col_g1, col_g2 = st.columns(2)
                conteo_riesgos = vivos['Riesgo_Principal'].value_counts().reset_index()
                conteo_riesgos.columns = ['Riesgo', 'Cantidad']

                with col_g1:
                    st.write("**Distribución de Edades**")
                    fig_hist = px.histogram(vivos, x="age", nbins=20, color_discrete_sequence=['forestgreen'])
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_g2:
                    st.write("**Patologías Activas (Interactivo)**")
                    # Gráfico de Pastel Interactivo
                    fig_pie = px.pie(conteo_riesgos, values='Cantidad', names='Riesgo', hole=0.3)
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No se encontraron pacientes.")

    # --- PESTAÑA 2: FALLECIDOS ---
    with tab_fallecidos:
        fallecidos = df[df['DEATH_EVENT'] == 1].copy()
        fallecidos['age'] = fallecidos['age'].astype(int)
        fallecidos['Causa_Probable'] = fallecidos.apply(determinar_causa, axis=1)
        
        if search_term:
            fallecidos = fallecidos[fallecidos['Causa_Probable'].str.contains(search_term, case=False, na=False)]
            st.info(f"Mostrando {len(fallecidos)} registros filtrados.")

        if not fallecidos.empty:
            subtab_lista_f, subtab_graficos_f = st.tabs(["📋 Lista Agrupada", "📊 Ver Gráficas Interactivas"])

            with subtab_lista_f:
                conteo_causas = fallecidos['Causa_Probable'].value_counts()
                for causa, cantidad in conteo_causas.items():
                    with st.expander(f"📂 {causa}: {cantidad} pacientes"):
                        st.table(fallecidos[fallecidos['Causa_Probable'] == causa][['age', 'sex', 'diabetes']].head(10))

            with subtab_graficos_f:
                col_gf1, col_gf2 = st.columns(2)
                conteo_causas_df = fallecidos['Causa_Probable'].value_counts().reset_index()
                conteo_causas_df.columns = ['Causa', 'Cantidad']

                with col_gf1:
                    st.write("**Distribución de Edades al Fallecer**")
                    fig_hist_f = px.histogram(fallecidos, x="age", nbins=20, color_discrete_sequence=['darkred'])
                    st.plotly_chart(fig_hist_f, use_container_width=True)
                
                with col_gf2:
                    st.write("**Causas Probables (Interactivo)**")
                    fig_pie_f = px.pie(conteo_causas_df, values='Cantidad', names='Causa', hole=0.3)
                    st.plotly_chart(fig_pie_f, use_container_width=True)
        else:
            st.warning("No se encontraron registros.")
