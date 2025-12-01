# CardioGuard AI: Sistema Clínico Inteligente

**Proyecto CardioGuard AI** es una herramienta de análisis clínico basada en inteligencia artificial diseñada para ayudar a los médicos a predecir riesgos cardíacos en pacientes. Utilizando un conjunto de datos clínicos de fallas cardíacas, el sistema evalúa múltiples factores para clasificar el riesgo y las causas de muerte de los pacientes, proporcionando diagnósticos y recomendaciones de tratamiento.

## Autores:

- Albitres Dávila, Juan

- Angeles Pérez, Jhonny Luis

- Ballesteros Reyes, Renato

- Nolasco Castillo, Juan David 

- Rodriguez Cabrera, Marcelo

## Profesor:

- Elias Enrique Santa Cruz Damian
  
### TRUJILLO-PERÚ 2025

## Resumen Ejecutivo
   
El proyecto "CardioGuard AI" nace de la necesidad de mejorar la tasa de supervivencia en pacientes con insuficiencia cardíaca. Mediante el uso de algoritmos de Machine Learning (Random Forest) y una interfaz accesible vía web/móvil, se ha desarrollado una herramienta capaz de predecir la probabilidad de fallecimiento y sugerir tratamientos personalizados en tiempo real. El uso del archivo CSV proporcionado es esencial para el correcto funcionamiento de la aplicación, en el futuro se planea agregar la capacidad de usar CSVs distintas.

## Definición del Problema y Alcance

Problemática: La saturación de servicios y la revisión manual de expedientes provocan que pacientes con indicadores sutiles de riesgo pasen desapercibidos hasta que sufren un evento fatal.
Justificación: La implementación de un sistema predictivo permite priorizar recursos en los pacientes con mayor probabilidad de complicación (Triaje basado en IA).

## Metodología y Datos

Dataset: Se utilizó una base de datos clínica conteniendo 12 variables independientes (edad, anemia, creatinina, fracción de eyección, etc.) y 1 variable dependiente (evento de muerte).
Preprocesamiento: Limpieza de datos y normalización de variables numéricas.
Modelo Seleccionado: Se optó por un Random Forest Classifier debido a su alta explicabilidad (permite saber qué variables influyen más) y robustez frente a datos desbalanceados.

## Desarrollo de la Solución (CardioGuard AI)

La aplicación fue construida utilizando un enfoque modular:
Módulo de Ingesta: Permite la carga dinámica de bases de datos actualizadas.
Motor de Inferencia: Un algoritmo interno evalúa a cada paciente "Vivo" y aplica reglas médicas (ej. Si Creatinina > 1.4 mg/dL -> Alerta Renal) para generar un "Diagnóstico y Solución" automatizado.
Visualización Científica: Generación automática de mapas de calor de correlación y gráficas de distribución de patologías para la toma de decisiones gerenciales.

## Resultados del Análisis

El modelo identificó que los factores más críticos para la supervivencia son:
Nivel de Creatinina en Sangre: Indicador directo de función renal.
Fracción de Eyección: Indicador de la capacidad de bombeo del corazón.
Edad: Factor de riesgo no modificable pero crítico para la estratificación.
La aplicación demostró capacidad para clasificar correctamente a los pacientes en grupos de riesgo y visualizar las causas probables de muerte (Falla Renal vs. Falla Cardíaca) en los datos históricos.

## Conclusión

CardioGuard AI cumple con el objetivo de proveer una solución distribuida y accesible. Transforma datos clínicos crudos en inteligencia accionable, permitiendo al personal médico actuar antes de que ocurra un evento fatal, alineándose con los estándares modernos de la Medicina de Precisión.

Link del programa en WEB: https://appcorazon-dlyn2varauf5mrnw4kpn9o.streamlit.app/ 

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
