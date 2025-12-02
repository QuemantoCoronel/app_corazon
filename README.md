# CardioGuard AI: Sistema Clínico Inteligente

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
## Citation
- Davide Chicco, Giuseppe Jurman