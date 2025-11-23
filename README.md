Proyecto “CardioGuard AI”
Autores:
Albitres Dávila, Juan

Angeles Pérez, Jhonny Luis

Ballesteros Reyes, Renato

Nolasco Castillo, Juan David 

Rodriguez Cabrera, Marcelo

Profesor:

Elias Enrique Santa Cruz Damian
  
TRUJILLO-PERÚ 2025

1. Resumen Ejecutivo
El proyecto "CardioGuard AI" nace de la necesidad de mejorar la tasa de supervivencia en pacientes con insuficiencia cardíaca. Mediante el uso de algoritmos de Machine Learning (Random Forest) y una interfaz accesible vía web/móvil, se ha desarrollado una herramienta capaz de predecir la probabilidad de fallecimiento y sugerir tratamientos personalizados en tiempo real.
2. Definición del Problema y Alcance

Problemática: La saturación de servicios y la revisión manual de expedientes provocan que pacientes con indicadores sutiles de riesgo pasen desapercibidos hasta que sufren un evento fatal.
Justificación: La implementación de un sistema predictivo permite priorizar recursos en los pacientes con mayor probabilidad de complicación (Triaje basado en IA).
3. Metodología y Datos
Dataset: Se utilizó una base de datos clínica conteniendo 12 variables independientes (edad, anemia, creatinina, fracción de eyección, etc.) y 1 variable dependiente (evento de muerte).
Preprocesamiento: Limpieza de datos y normalización de variables numéricas.
Modelo Seleccionado: Se optó por un Random Forest Classifier debido a su alta explicabilidad (permite saber qué variables influyen más) y robustez frente a datos desbalanceados.
4. Desarrollo de la Solución (CardioGuard AI)
La aplicación fue construida utilizando un enfoque modular:
Módulo de Ingesta: Permite la carga dinámica de bases de datos actualizadas.
Motor de Inferencia: Un algoritmo interno evalúa a cada paciente "Vivo" y aplica reglas médicas (ej. Si Creatinina > 1.4 mg/dL -> Alerta Renal) para generar un "Diagnóstico y Solución" automatizado.
Visualización Científica: Generación automática de mapas de calor de correlación y gráficas de distribución de patologías para la toma de decisiones gerenciales.
5. Resultados del Análisis
El modelo identificó que los factores más críticos para la supervivencia son:
Nivel de Creatinina en Sangre: Indicador directo de función renal.
Fracción de Eyección: Indicador de la capacidad de bombeo del corazón.
Edad: Factor de riesgo no modificable pero crítico para la estratificación.
La aplicación demostró capacidad para clasificar correctamente a los pacientes en grupos de riesgo y visualizar las causas probables de muerte (Falla Renal vs. Falla Cardíaca) en los datos históricos.
6. Conclusión
CardioGuard AI cumple con el objetivo de proveer una solución distribuida y accesible. Transforma datos clínicos crudos en inteligencia accionable, permitiendo al personal médico actuar antes de que ocurra un evento fatal, alineándose con los estándares modernos de la Medicina de Precisión.
