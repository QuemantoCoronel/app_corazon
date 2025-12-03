# 🫀 CardioGuard AI: Sistema Paralelo de Diagnóstico Clínico

> **Solución de Cómputo Distribuido aplicada a la Medicina**

## Información Académica
>**Profesor:** Elias Enrique Santa Cruz Damian
>**Ubicación:** Trujillo, Perú (2025)
### Autores:

- Albitres Dávila, Juan

- Angeles Pérez, Jhonny Luis

- Ballesteros Reyes, Renato

- Nolasco Castillo, Juan David 

- Rodriguez Cabrera, Marcelo


## Resumen Ejecutivo
   
CardioGuard AI es un Sistema de Soporte a la Decisión Clínica (CDSS) diseñado para optimizar la predicción de mortalidad en pacientes con insuficiencia cardíaca.

A diferencia de los sistemas tradicionales secuenciales, este proyecto implementa una Arquitectura Maestro-Esclavo utilizando técnicas de Cómputo Paralelo. Esto permite entrenar múltiples modelos de Inteligencia Artificial simultáneamente (Random Forest, Gradient Boosting y Regresión Logística), reduciendo los tiempos de procesamiento y aumentando la precisión del diagnóstico mediante votación algorítmica.

### Variables del Dataset
El conjunto de datos contiene diversas variables clínicas que ayudan a predecir la insuficiencia cardíaca:

- **Edad**: Edad del paciente.
- **Anemia**: Si el paciente presenta anemia (disminución de glóbulos rojos o hemoglobina).
- **Creatinina**: Nivel de creatinina en sangre, indicador de posibles lesiones.
- **Diabetes**: Si el paciente tiene diabetes.
- **Fracción de Eyección**: Porcentaje de sangre que sale del corazón en cada contracción.
- **Hipertensión Arterial**: Si el paciente tiene antecedentes de hipertensión.
- **Plaquetas**: Nivel de plaquetas en sangre.
- **Creatinina Sérica**: Mide la función renal.
- **Sodio Sérico**: Mide el equilibrio electrolítico en los vasos sanguíneos.
- **Sexo**: Sexo del paciente.
- **Tabaquismo**: Si el paciente fuma o no.
- **Tiempo**: Período de seguimiento (días).
- **Evento de Fallecimiento**: Si el paciente falleció durante el seguimiento.

## Arquitectura del Sistema (Solución Distribuida)

El sistema utiliza el patrón de diseño **Master-Worker** para distribuir la carga computacional, simulando un entorno distribuido:

### 1. El Maestro (Frontend - Streamlit)
* Actúa como orquestador y gestor de la interfaz de usuario.
* **No procesa** los modelos matemáticos pesados; su función es delegar tareas y visualizar resultados.

### 2. Los Esclavos (Backend - Workers)
* Implementados mediante `ProcessPoolExecutor` (Multiprocessing nativo).
* Cada modelo de IA se entrena en un **proceso independiente** con su propio espacio de memoria y *Process ID (PID)*.
* Esto permite aprovechar los múltiples núcleos (cores) de la CPU simultáneamente.

**Características Técnicas:**
* **Paralelismo de Tareas:** Entrenamiento simultáneo de 3 algoritmos.
* **Evidencia de Distribución:** Logs que muestran el ID del Worker (PID) para cada tarea.
* **Tolerancia:** Si un modelo falla, no necesariamente cae todo el sistema maestro.

## Resultados del Análisis
Gracias al procesamiento paralelo, el sistema logró identificar patrones complejos rápidamente:

1. Factores Críticos: El modelo paralelo determinó que la Creatinina Sérica y la Fracción de Eyección son los predictores más fuertes de mortalidad.
2. Eficiencia: Se logró comparar la precisión (Accuracy) de 3 modelos en el mismo tiempo que tomaría entrenar solo el más lento de ellos en modo secuencial.
3. Triaje Automático: La aplicación clasifica a los pacientes vivos en tiempo real con alertas visuales:

    🔴 Alerta Renal: Creatinina > 1.4 mg/dL

    💔 Fallo Cardíaco: Eyección < 30%

## Instalación:

1. Clona el repositorio: 

    ```
    git clone https://github.com/QuemantoCoronel/app_corazon.git
    ```
2. Instala las dependencias del proyecto:
    ```
    pip install -r requirements.txt
    ```
3. Ejecuta la aplicación Streamlit:
    ```
    streamlit run app_corazon.py
    ```
Nota: Es indispensable que el archivo heart_failure_clinical_records_dataset.csv se encuentre en la raíz del proyecto para que el sistema funcione.

## Créditos del DataSet [Heart Failure Prediction](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
- **Autores:** Davide Chicco, Giuseppe Jurman