# Requirements Document

## Introduction

Este documento define los requisitos para el desarrollo de un sistema de clasificación multiclase de lesiones cutáneas utilizando el dataset HAM10000. El sistema debe implementar cuatro modelos diferentes basados en redes neuronales profundas: un modelo tabular (1D), un modelo convolucional (2D), y dos estrategias de fusión (late-fusion y early-fusion) que combinen ambas modalidades de datos.

## Glossary

- **Sistema_Clasificador**: El sistema completo de clasificación de lesiones cutáneas que incluye los cuatro modelos requeridos
- **Modelo_Tabular**: Red neuronal que procesa únicamente datos tabulares (sexo, edad, localización)
- **Modelo_CNN**: Red neuronal convolucional que procesa únicamente imágenes dermatoscópicas
- **Modelo_LateFusion**: Red neuronal que combina las predicciones de los modelos tabular y CNN
- **Modelo_EarlyFusion**: Red neuronal que combina las características extraídas por los modelos tabular y CNN
- **Dataset_HAM10000**: Conjunto de 10,015 imágenes de lesiones cutáneas clasificadas en 7 categorías
- **Datos_Tabulares**: Información estructurada que incluye sexo, edad, diagnóstico, método de confirmación y localización corporal
- **Partición_Entrenamiento**: Subconjunto de datos utilizado para entrenar los modelos
- **Partición_Validación**: Subconjunto de datos utilizado para ajustar hiperparámetros durante el entrenamiento
- **Partición_Test**: Subconjunto de datos reservado exclusivamente para evaluación final
- **Embeddings**: Representaciones vectoriales de características aprendidas por las capas intermedias de las redes neuronales
- **Transfer_Learning**: Técnica que utiliza modelos preentrenados para extraer características de imágenes

## Requirements

### Requirement 1: Carga y Preprocesamiento de Datos

**User Story:** Como científico de datos, quiero cargar y preprocesar el dataset HAM10000 correctamente, para que los datos estén listos para entrenar modelos de deep learning.

#### Acceptance Criteria

1. WHEN el Sistema_Clasificador carga los datos desde Google Drive, THE Sistema_Clasificador SHALL montar el sistema de archivos y leer los ficheros hmnist_28_28_RGB.csv y HAM10000_metadata.csv
2. THE Sistema_Clasificador SHALL eliminar columnas irrelevantes que no contribuyan a la predicción
3. THE Sistema_Clasificador SHALL convertir todas las variables categóricas en representaciones numéricas utilizando codificación apropiada
4. THE Sistema_Clasificador SHALL dividir los datos en tres particiones mutuamente excluyentes: Partición_Entrenamiento, Partición_Validación y Partición_Test
5. THE Sistema_Clasificador SHALL identificar y manejar valores missing en los Datos_Tabulares aplicando estrategias de imputación o eliminación
6. THE Sistema_Clasificador SHALL normalizar los valores de píxeles de las imágenes al rango [0, 1]
7. THE Sistema_Clasificador SHALL normalizar los Datos_Tabulares utilizando estandarización o normalización min-max
8. THE Sistema_Clasificador SHALL verificar que el orden entre Datos_Tabulares, imágenes y etiquetas se mantiene consistente en todas las particiones

### Requirement 2: Modelo de Clasificación Tabular (Hito 1)

**User Story:** Como investigador médico, quiero un modelo que clasifique lesiones cutáneas usando solo datos demográficos y de localización, para entender la capacidad predictiva de la información tabular.

#### Acceptance Criteria

1. THE Modelo_Tabular SHALL procesar únicamente las variables sexo, edad y localización corporal como entrada
2. THE Modelo_Tabular SHALL implementar una arquitectura de red neuronal fully-connected con al menos una capa oculta
3. WHEN el Modelo_Tabular se entrena, THE Modelo_Tabular SHALL utilizar únicamente la Partición_Entrenamiento para actualizar pesos
4. WHILE el Modelo_Tabular se entrena, THE Modelo_Tabular SHALL evaluar el rendimiento en la Partición_Validación después de cada época
5. THE Modelo_Tabular SHALL generar visualizaciones del proceso de aprendizaje mostrando pérdida y precisión en entrenamiento y validación
6. THE Modelo_Tabular SHALL producir predicciones de probabilidad para las 7 clases de lesiones cutáneas
7. THE Modelo_Tabular SHALL alcanzar convergencia sin producir valores NaN en la función de pérdida

### Requirement 3: Modelo de Clasificación Convolucional (Hito 2)

**User Story:** Como dermatólogo, quiero un modelo CNN que clasifique lesiones cutáneas a partir de imágenes dermatoscópicas, para aprovechar la información visual de las lesiones.

#### Acceptance Criteria

1. THE Modelo_CNN SHALL procesar únicamente las imágenes de 28x28x3 píxeles como entrada
2. THE Modelo_CNN SHALL implementar al menos una capa convolucional para extracción de características
3. THE Modelo_CNN SHALL utilizar un modelo preentrenado para obtener embeddings mediante transfer learning offline
4. THE Modelo_CNN SHALL entrenar únicamente las capas de clasificación mientras mantiene congeladas las capas de extracción de características del modelo preentrenado
5. WHEN el Modelo_CNN se entrena, THE Modelo_CNN SHALL utilizar únicamente la Partición_Entrenamiento para actualizar pesos
6. WHILE el Modelo_CNN se entrena, THE Modelo_CNN SHALL evaluar el rendimiento en la Partición_Validación después de cada época
7. THE Modelo_CNN SHALL generar visualizaciones del proceso de aprendizaje mostrando pérdida y precisión
8. THE Modelo_CNN SHALL producir predicciones de probabilidad para las 7 clases de lesiones cutáneas

### Requirement 4: Modelo Late-Fusion (Hito 3)

**User Story:** Como científico de datos, quiero combinar las predicciones de los modelos tabular y CNN, para mejorar la precisión mediante la fusión de decisiones independientes.

#### Acceptance Criteria

1. THE Modelo_LateFusion SHALL recibir como entrada las predicciones de probabilidad generadas por el Modelo_Tabular y el Modelo_CNN
2. THE Modelo_LateFusion SHALL utilizar los modelos Modelo_Tabular y Modelo_CNN previamente entrenados sin reentrenarlos
3. THE Modelo_LateFusion SHALL implementar una capa de combinación aprendida que procese las predicciones concatenadas
4. THE Modelo_LateFusion SHALL entrenar únicamente los pesos de la capa de fusión
5. WHEN el Modelo_LateFusion se entrena, THE Modelo_LateFusion SHALL utilizar las mismas particiones de datos que los modelos individuales
6. THE Modelo_LateFusion SHALL producir predicciones finales de probabilidad para las 7 clases
7. THE Modelo_LateFusion SHALL generar visualizaciones del proceso de aprendizaje

### Requirement 5: Modelo Early-Fusion (Hito 4)

**User Story:** Como científico de datos, quiero combinar las características extraídas por los modelos tabular y CNN, para permitir interacciones más profundas entre modalidades antes de la clasificación.

#### Acceptance Criteria

1. THE Modelo_EarlyFusion SHALL recibir como entrada los embeddings generados por las capas intermedias del Modelo_Tabular y del Modelo_CNN
2. THE Modelo_EarlyFusion SHALL utilizar los modelos Modelo_Tabular y Modelo_CNN previamente entrenados como extractores de características sin reentrenarlos
3. THE Modelo_EarlyFusion SHALL normalizar o ajustar las dimensiones de los embeddings cuando una modalidad tenga representaciones significativamente más largas que la otra
4. THE Modelo_EarlyFusion SHALL concatenar los embeddings de ambas modalidades antes de las capas de clasificación
5. THE Modelo_EarlyFusion SHALL implementar capas fully-connected para procesar las características fusionadas
6. WHEN el Modelo_EarlyFusion se entrena, THE Modelo_EarlyFusion SHALL utilizar las mismas particiones de datos que los modelos individuales
7. THE Modelo_EarlyFusion SHALL producir predicciones finales de probabilidad para las 7 clases
8. THE Modelo_EarlyFusion SHALL generar visualizaciones del proceso de aprendizaje

### Requirement 6: Evaluación y Presentación de Resultados

**User Story:** Como evaluador del proyecto, quiero ver los resultados de todos los modelos en el conjunto de test, para comparar objetivamente su rendimiento.

#### Acceptance Criteria

1. THE Sistema_Clasificador SHALL evaluar los cuatro modelos (Modelo_Tabular, Modelo_CNN, Modelo_LateFusion, Modelo_EarlyFusion) sobre la Partición_Test
2. THE Sistema_Clasificador SHALL calcular métricas de clasificación incluyendo accuracy, precision, recall y F1-score para cada modelo
3. THE Sistema_Clasificador SHALL generar matrices de confusión para cada uno de los cuatro modelos
4. THE Sistema_Clasificador SHALL presentar una comparativa visual del rendimiento de los cuatro modelos
5. THE Sistema_Clasificador SHALL ejecutarse completamente en un Jupyter Notebook con todas las celdas ejecutadas y outputs visibles
6. THE Sistema_Clasificador SHALL incluir una sección de discusión interpretando los resultados obtenidos
7. THE Sistema_Clasificador SHALL proporcionar justificación razonada sobre el funcionamiento de cada modelo

### Requirement 7: Optimización y Buenas Prácticas

**User Story:** Como desarrollador, quiero que el código sea eficiente y siga buenas prácticas, para facilitar la ejecución en Google Colab y la comprensión del trabajo.

#### Acceptance Criteria

1. THE Sistema_Clasificador SHALL desarrollar el código de manera que pueda ejecutarse inicialmente en CPU con 1 época para verificación
2. THE Sistema_Clasificador SHALL utilizar aceleración GPU cuando esté disponible en Google Colab para entrenamiento completo
3. THE Sistema_Clasificador SHALL minimizar el uso de memoria evitando duplicación innecesaria de datos
4. THE Sistema_Clasificador SHALL incluir comentarios en español explicando las secciones principales del código
5. THE Sistema_Clasificador SHALL organizar el código en secciones claramente delimitadas correspondientes a cada hito
6. IF la función de pérdida produce valores NaN, THEN THE Sistema_Clasificador SHALL verificar la normalización de datos y ajustar la tasa de aprendizaje
7. THE Sistema_Clasificador SHALL guardar los modelos entrenados para reutilización en las estrategias de fusión
