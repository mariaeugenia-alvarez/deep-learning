# Implementation Plan - Clasificador HAM10000

- [x] 1. Setup inicial y carga de datos
  - Configurar entorno de Google Colab con imports necesarios (TensorFlow, NumPy, Pandas, Matplotlib, Scikit-learn)
  - Implementar función para montar Google Drive y verificar acceso a los archivos del dataset
  - Crear función `load_images()` que lea hmnist_28_28_RGB.csv y convierta a array numpy de shape (10015, 28, 28, 3)
  - Crear función `load_metadata()` que lea HAM10000_metadata.csv como DataFrame de Pandas
  - Verificar que los datos se cargaron correctamente mostrando shapes y primeras filas
  - _Requirements: 1.1, 1.8_

- [x] 2. Preprocesamiento de datos
  - [x] 2.1 Implementar preprocesamiento de datos tabulares
    - Crear clase `TabularPreprocessor` con métodos `fit_transform()` y `transform()`
    - Eliminar columnas irrelevantes (lesion_id, image_id, dx_type)
    - Implementar codificación de variable 'sex' usando LabelEncoder o OneHotEncoder
    - Implementar codificación de variable 'localization' usando OneHotEncoder
    - Implementar manejo de valores missing: imputar edad con mediana, categoría "unknown" para sex y localization
    - Normalizar feature 'age' usando StandardScaler
    - Verificar que no quedan valores NaN en los datos procesados
    - _Requirements: 1.2, 1.3, 1.5, 1.7_
  
  - [x] 2.2 Implementar preprocesamiento de imágenes
    - Crear clase `ImagePreprocessor` con método `normalize()`
    - Normalizar píxeles de imágenes al rango [0, 1] dividiendo por 255.0
    - Convertir dtype a float32 para optimizar memoria
    - Verificar que min=0 y max=1 después de normalización
    - _Requirements: 1.6_
  
  - [x] 2.3 Implementar división de datos en train/val/test
    - Crear función `split_data()` que divida datos en 70% train, 15% val, 15% test
    - Usar split estratificado para mantener distribución de clases en cada partición
    - Convertir labels a formato one-hot encoding con 7 clases
    - Implementar función `validate_data_consistency()` que verifique que X_tab, X_img e y tienen mismo número de muestras
    - Verificar que el orden se mantiene consistente entre datos tabulares, imágenes y labels
    - Mostrar distribución de clases en cada partición
    - _Requirements: 1.4, 1.8_

- [x] 3. Hito 1 - Modelo de clasificación tabular
  - [x] 3.1 Construir arquitectura del modelo tabular
    - Implementar función `build_tabular_model()` que cree modelo Sequential
    - Definir arquitectura: Input → Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(7, softmax)
    - Compilar modelo con optimizer Adam (lr=0.001), loss categorical_crossentropy, metrics accuracy
    - Mostrar resumen del modelo con model.summary()
    - _Requirements: 2.2_
  
  - [x] 3.2 Entrenar modelo tabular
    - Configurar callbacks: EarlyStopping (patience=10, monitor='val_loss'), NaNDetector
    - Entrenar modelo con X_train_tab e y_train por 50 epochs, batch_size=32
    - Usar X_val_tab e y_val para validación durante entrenamiento
    - Verificar que la función de pérdida no produce valores NaN
    - Guardar modelo entrenado en Google Drive
    - _Requirements: 2.3, 2.4, 2.7, 7.6_
  
  - [x] 3.3 Evaluar y visualizar resultados del modelo tabular
    - Implementar función `plot_training_history()` para visualizar loss y accuracy en train/val
    - Generar gráficas de entrenamiento del modelo tabular
    - Evaluar modelo en conjunto de validación y mostrar accuracy
    - Obtener predicciones en validación para uso posterior en late fusion
    - Guardar predicciones y embeddings del modelo tabular
    - _Requirements: 2.5, 2.6_

- [x] 4. Hito 2 - Modelo CNN con transfer learning
  - [x] 4.1 Construir arquitectura del modelo CNN
    - Implementar función `build_cnn_model()` usando MobileNetV2 preentrenado
    - Cargar MobileNetV2 con weights='imagenet', include_top=False, pooling='avg'
    - Congelar todas las capas del base_model (trainable=False)
    - Añadir capas de clasificación: Dense(128, relu) → Dropout(0.4) → Dense(7, softmax)
    - Compilar modelo con optimizer Adam (lr=0.0001), loss categorical_crossentropy, metrics accuracy
    - Mostrar resumen del modelo
    - _Requirements: 3.2, 3.3, 3.4_
  
  - [x] 4.2 Entrenar modelo CNN
    - Configurar callbacks: EarlyStopping (patience=10), NaNDetector
    - Entrenar modelo con X_train_img e y_train por 30 epochs, batch_size=64
    - Usar X_val_img e y_val para validación durante entrenamiento
    - Liberar memoria con tf.keras.backend.clear_session() después del entrenamiento
    - Guardar modelo entrenado en Google Drive
    - _Requirements: 3.5, 3.6, 7.2_
  
  - [x] 4.3 Evaluar y visualizar resultados del modelo CNN
    - Generar gráficas de entrenamiento usando plot_training_history()
    - Evaluar modelo en conjunto de validación y mostrar accuracy
    - Obtener predicciones en validación para uso posterior en late fusion
    - Crear modelo extractor de embeddings desde penúltima capa del CNN
    - Guardar predicciones y embeddings del modelo CNN
    - _Requirements: 3.7, 3.8_

- [x] 5. Hito 3 - Late fusion de predicciones
  - [x] 5.1 Construir arquitectura del modelo late fusion
    - Implementar función `build_late_fusion_model()` con dos inputs (predicciones tabular y CNN)
    - Definir arquitectura: Concatenate → Dense(32, relu) → Dropout(0.3) → Dense(7, softmax)
    - Compilar modelo con optimizer Adam (lr=0.001), loss categorical_crossentropy, metrics accuracy
    - Mostrar resumen del modelo
    - _Requirements: 4.1, 4.3_
  
  - [x] 5.2 Preparar datos y entrenar modelo late fusion
    - Cargar modelos tabular y CNN previamente entrenados
    - Obtener predicciones del modelo tabular en train, val y test: pred_tab = model_tab.predict(X_tab)
    - Obtener predicciones del modelo CNN en train, val y test: pred_cnn = model_cnn.predict(X_img)
    - Entrenar modelo late fusion con [pred_tab_train, pred_cnn_train] e y_train por 20 epochs
    - Usar [pred_tab_val, pred_cnn_val] e y_val para validación
    - _Requirements: 4.2, 4.4, 4.5_
  
  - [x] 5.3 Evaluar y visualizar resultados del modelo late fusion
    - Generar gráficas de entrenamiento
    - Evaluar modelo en conjunto de validación
    - Guardar modelo entrenado
    - _Requirements: 4.6, 4.7_

- [x] 6. Hito 4 - Early fusion de características
  - [x] 6.1 Construir arquitectura del modelo early fusion
    - Implementar función `build_early_fusion_model()` con dos inputs (embeddings tabular y CNN)
    - Implementar normalización de dimensiones si embedding_dim_cnn > embedding_dim_tab * 2
    - Definir arquitectura: Concatenate → Dense(128, relu) → Dropout(0.4) → Dense(64, relu) → Dropout(0.3) → Dense(7, softmax)
    - Compilar modelo con optimizer Adam (lr=0.001), loss categorical_crossentropy, metrics accuracy
    - Mostrar resumen del modelo
    - _Requirements: 5.1, 5.3, 5.4_
  
  - [x] 6.2 Preparar datos y entrenar modelo early fusion
    - Crear modelo extractor de embeddings tabulares desde penúltima capa del modelo tabular
    - Crear modelo extractor de embeddings CNN desde penúltima capa del modelo CNN
    - Extraer embeddings tabulares en train, val y test
    - Extraer embeddings CNN en train, val y test
    - Entrenar modelo early fusion con [emb_tab_train, emb_cnn_train] e y_train por 30 epochs
    - Usar [emb_tab_val, emb_cnn_val] e y_val para validación
    - _Requirements: 5.2, 5.5, 5.6_
  
  - [x] 6.3 Evaluar y visualizar resultados del modelo early fusion
    - Generar gráficas de entrenamiento
    - Evaluar modelo en conjunto de validación
    - Guardar modelo entrenado
    - _Requirements: 5.7, 5.8_

- [x] 7. Evaluación final en test set
  - [x] 7.1 Evaluar los 4 modelos en conjunto de test
    - Evaluar modelo tabular en X_test_tab e y_test, calcular accuracy, precision, recall, F1-score
    - Evaluar modelo CNN en X_test_img e y_test, calcular métricas
    - Evaluar modelo late fusion en [pred_tab_test, pred_cnn_test] e y_test, calcular métricas
    - Evaluar modelo early fusion en [emb_tab_test, emb_cnn_test] e y_test, calcular métricas
    - Crear diccionario con resultados de los 4 modelos
    - _Requirements: 6.1, 6.2_
  
  - [x] 7.2 Generar visualizaciones de resultados
    - Implementar función `plot_confusion_matrix()` para visualizar matriz de confusión
    - Generar matriz de confusión para cada uno de los 4 modelos
    - Implementar función `plot_model_comparison()` para comparar accuracy de los 4 modelos
    - Generar gráfica comparativa de los 4 modelos
    - Mostrar classification report detallado para cada modelo
    - _Requirements: 6.3, 6.4_
  
  - [x] 7.3 Presentar resultados en formato notebook ejecutado
    - Verificar que todas las celdas del notebook están ejecutadas con outputs visibles
    - Organizar notebook en secciones claras: Setup, Preprocesamiento, Hito 1, Hito 2, Hito 3, Hito 4, Evaluación Final
    - Añadir comentarios en español explicando cada sección del código
    - Verificar que el notebook puede ejecutarse de principio a fin sin errores
    - _Requirements: 6.5, 7.4, 7.5_

- [x] 8. Discusión y documentación de resultados
  - Crear sección de discusión en el notebook interpretando los resultados obtenidos
  - Analizar qué modelo funciona mejor y por qué (comparar accuracy, F1-score)
  - Discutir si hay diferencias de rendimiento entre clases (usando matrices de confusión)
  - Identificar posibles fuentes de error (clases confundidas, limitaciones del dataset)
  - Proponer trabajo futuro: data augmentation, fine-tuning, arquitecturas más complejas, manejo de desbalanceo
  - Justificar razonadamente las decisiones de diseño tomadas (arquitecturas, hiperparámetros, estrategias de fusión)
  - _Requirements: 6.6, 6.7_

- [ ]* 9. Optimizaciones opcionales
  - [ ]* 9.1 Implementar búsqueda de hiperparámetros
    - Experimentar con diferentes learning rates (0.0001, 0.001, 0.01)
    - Experimentar con diferentes valores de dropout (0.2, 0.3, 0.5)
    - Experimentar con diferentes batch sizes (32, 64, 128)
    - Documentar resultados de experimentos
  
  - [ ]* 9.2 Implementar manejo de desbalanceo de clases
    - Calcular class weights usando compute_class_weight de sklearn
    - Reentrenar modelos usando class_weight en fit()
    - Comparar resultados con y sin class weights
  
  - [ ]* 9.3 Implementar data augmentation para imágenes
    - Crear ImageDataGenerator con rotaciones, flips, zoom
    - Reentrenar modelo CNN con data augmentation
    - Comparar resultados con modelo sin augmentation
  
  - [ ]* 9.4 Implementar fine-tuning del modelo CNN
    - Descongelar últimas capas de MobileNetV2
    - Reentrenar con learning rate muy bajo (0.00001)
    - Comparar resultados con transfer learning offline
  
  - [ ]* 9.5 Análisis adicional de resultados
    - Analizar rendimiento por edad (crear grupos etarios)
    - Analizar rendimiento por sexo
    - Analizar rendimiento por localización corporal
    - Crear visualizaciones de estos análisis
