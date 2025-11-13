# Design Document - Clasificador HAM10000

## Overview

Este documento describe el diseño técnico del sistema de clasificación multiclase de lesiones cutáneas usando el dataset HAM10000. El sistema implementa cuatro modelos de deep learning que procesan datos tabulares e imágenes, tanto de forma independiente como combinada mediante estrategias de fusión.

El diseño sigue una arquitectura modular donde cada modelo puede ser entrenado y evaluado independientemente, y los modelos de fusión reutilizan los componentes ya entrenados para optimizar tiempo de cómputo.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HAM10000 Dataset                          │
│              (Imágenes + Datos Tabulares)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Loading & Preprocessing                    │
│  - Carga desde Google Drive                                  │
│  - Limpieza y codificación                                   │
│  - Normalización                                             │
│  - Split train/val/test                                      │
└────────┬────────────────────────────────┬───────────────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐           ┌──────────────────┐
│  Datos Tabulares │           │     Imágenes     │
│  (sexo, edad,    │           │   (28x28x3)      │
│   localización)  │           │                  │
└────┬─────────────┘           └────┬─────────────┘
     │                              │
     ▼                              ▼
┌──────────────────┐           ┌──────────────────┐
│  Modelo Tabular  │           │   Modelo CNN     │
│   (Hito 1)       │           │   (Hito 2)       │
│  - FC Layers     │           │  - Transfer      │
│  - Dropout       │           │    Learning      │
└────┬─────────────┘           └────┬─────────────┘
     │                              │
     │  Predicciones                │  Predicciones
     │  (7 clases)                  │  (7 clases)
     │                              │
     └──────┬───────────────────────┘
            │
            ▼
     ┌─────────────────┐
     │  Late Fusion    │
     │   (Hito 3)      │
     │ - Combina       │
     │   predicciones  │
     └────────┬────────┘
              │
              │
     ┌────────┴────────┐
     │                 │
     │  Embeddings     │  Embeddings
     │  Tabulares      │  CNN
     │                 │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │  Early Fusion   │
     │   (Hito 4)      │
     │ - Combina       │
     │   features      │
     └─────────────────┘
```

### Technology Stack

- **Framework**: TensorFlow 2.x + Keras
- **Entorno**: Google Colab con GPU (NVIDIA K80)
- **Lenguaje**: Python 3.7+
- **Librerías auxiliares**: NumPy, Pandas, Matplotlib, Scikit-learn

## Components and Interfaces

### 1. Data Loading Module

**Responsabilidad**: Cargar y preparar los datos desde Google Drive.

**Componentes**:
- `mount_drive()`: Monta Google Drive en Colab
- `load_images()`: Carga hmnist_28_28_RGB.csv y convierte a arrays numpy
- `load_metadata()`: Carga HAM10000_metadata.csv como DataFrame

**Outputs**:
- `X_images`: Array numpy de shape (10015, 28, 28, 3)
- `df_metadata`: DataFrame con información tabular
- `y_labels`: Array de etiquetas codificadas (0-6)

### 2. Data Preprocessing Module

**Responsabilidad**: Limpiar, transformar y normalizar los datos.

**Componentes**:

#### 2.1 Tabular Preprocessing
```python
class TabularPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def fit_transform(self, df):
        # Eliminar columnas irrelevantes
        # Codificar variables categóricas
        # Manejar missing values
        # Normalizar features numéricos
        pass
    
    def transform(self, df):
        # Aplicar transformaciones aprendidas
        pass
```

**Features a procesar**:
- `sex`: Codificación one-hot o label encoding (male/female/unknown)
- `age`: Normalización (StandardScaler)
- `localization`: Codificación one-hot (múltiples localizaciones corporales)

**Estrategia missing values**:
- `age`: Imputación con mediana
- `sex`: Categoría "unknown"
- `localization`: Categoría "unknown"

#### 2.2 Image Preprocessing
```python
class ImagePreprocessor:
    def normalize(self, images):
        # Normalizar píxeles a [0, 1]
        return images.astype('float32') / 255.0
    
    def augment(self, images):
        # Opcional: data augmentation
        pass
```

#### 2.3 Data Splitting
```python
def split_data(X_tab, X_img, y, test_size=0.15, val_size=0.15):
    # Split estratificado para mantener distribución de clases
    # Retorna: X_train, X_val, X_test para ambas modalidades
    pass
```

**Proporciones**:
- Train: 70%
- Validation: 15%
- Test: 15%

**Estrategia**: Split estratificado para mantener distribución de clases (dataset desbalanceado).

### 3. Model 1D - Tabular Classifier (Hito 1)

**Arquitectura**:
```python
def build_tabular_model(input_dim, num_classes=7):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

**Justificación del diseño**:
- Arquitectura simple fully-connected apropiada para datos tabulares
- Dropout para regularización (prevenir overfitting en dataset pequeño)
- Reducción progresiva de dimensionalidad (128 → 64 → 32)
- Softmax final para clasificación multiclase

**Hiperparámetros**:
- Optimizer: Adam (lr=0.001)
- Loss: categorical_crossentropy
- Batch size: 32
- Epochs: 50 (con early stopping)

**Output**:
- Predicciones: shape (batch_size, 7) - probabilidades por clase
- Embeddings: output de penúltima capa, shape (batch_size, 32)

### 4. Model 2D - CNN Classifier (Hito 2)

**Arquitectura con Transfer Learning**:
```python
def build_cnn_model(input_shape=(28, 28, 3), num_classes=7):
    # Base model preentrenado (MobileNetV2 o EfficientNetB0)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Congelar capas del base model
    base_model.trainable = False
    
    # Clasificador custom
    model = Sequential([
        base_model,
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    return model
```

**Justificación del diseño**:
- **Transfer Learning offline**: Usar modelo preentrenado en ImageNet
- **MobileNetV2**: Ligero, eficiente para imágenes pequeñas (28x28)
- **Pooling='avg'**: Reduce dimensionalidad espacial a vector
- **Capas congeladas**: Solo entrenar clasificador (más rápido, menos overfitting)

**Alternativa si hay problemas de memoria**:
```python
# CNN simple desde cero
def build_simple_cnn(input_shape=(28, 28, 3), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model
```

**Hiperparámetros**:
- Optimizer: Adam (lr=0.0001 para transfer learning)
- Loss: categorical_crossentropy
- Batch size: 64
- Epochs: 30 (con early stopping)

**Output**:
- Predicciones: shape (batch_size, 7)
- Embeddings: output de capa Dense(128), shape (batch_size, 128)

### 5. Late Fusion Model (Hito 3)

**Arquitectura**:
```python
def build_late_fusion_model(num_classes=7):
    # Inputs: predicciones de ambos modelos
    input_tabular_pred = Input(shape=(num_classes,), name='tabular_predictions')
    input_cnn_pred = Input(shape=(num_classes,), name='cnn_predictions')
    
    # Concatenar predicciones
    concatenated = Concatenate()([input_tabular_pred, input_cnn_pred])
    
    # Capa de fusión aprendida
    x = Dense(32, activation='relu')(concatenated)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(
        inputs=[input_tabular_pred, input_cnn_pred],
        outputs=output
    )
    
    return model
```

**Pipeline de inferencia**:
1. Obtener predicciones del Modelo Tabular: `pred_tab = model_tab.predict(X_tab)`
2. Obtener predicciones del Modelo CNN: `pred_cnn = model_cnn.predict(X_img)`
3. Concatenar: `[pred_tab, pred_cnn]`
4. Pasar por modelo de fusión: `final_pred = fusion_model.predict([pred_tab, pred_cnn])`

**Justificación**:
- Combina decisiones independientes de ambos modelos
- Permite que el modelo aprenda pesos óptimos para cada modalidad
- Arquitectura ligera (solo capa de fusión se entrena)

**Hiperparámetros**:
- Optimizer: Adam (lr=0.001)
- Loss: categorical_crossentropy
- Batch size: 32
- Epochs: 20

### 6. Early Fusion Model (Hito 4)

**Arquitectura**:
```python
def build_early_fusion_model(embedding_dim_tab, embedding_dim_cnn, num_classes=7):
    # Inputs: embeddings de ambos modelos
    input_tabular_emb = Input(shape=(embedding_dim_tab,), name='tabular_embeddings')
    input_cnn_emb = Input(shape=(embedding_dim_cnn,), name='cnn_embeddings')
    
    # Normalización de embeddings si hay desbalance de dimensiones
    if embedding_dim_cnn > embedding_dim_tab * 2:
        # Reducir dimensionalidad de CNN embeddings
        cnn_emb_normalized = Dense(64, activation='relu')(input_cnn_emb)
        tab_emb_normalized = Dense(64, activation='relu')(input_tabular_emb)
    else:
        cnn_emb_normalized = input_cnn_emb
        tab_emb_normalized = input_tabular_emb
    
    # Concatenar embeddings
    concatenated = Concatenate()([tab_emb_normalized, cnn_emb_normalized])
    
    # Capas de clasificación
    x = Dense(128, activation='relu')(concatenated)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(
        inputs=[input_tabular_emb, input_cnn_emb],
        outputs=output
    )
    
    return model
```

**Pipeline de inferencia**:
1. Crear modelos extractores de embeddings:
```python
# Extractor tabular (hasta penúltima capa)
embedding_model_tab = Model(
    inputs=model_tab.input,
    outputs=model_tab.layers[-2].output
)

# Extractor CNN (hasta penúltima capa)
embedding_model_cnn = Model(
    inputs=model_cnn.input,
    outputs=model_cnn.layers[-2].output
)
```

2. Obtener embeddings:
```python
emb_tab = embedding_model_tab.predict(X_tab)
emb_cnn = embedding_model_cnn.predict(X_img)
```

3. Pasar por modelo de fusión:
```python
final_pred = early_fusion_model.predict([emb_tab, emb_cnn])
```

**Justificación**:
- Permite interacciones más profundas entre modalidades
- Normalización de dimensiones evita que una modalidad domine
- Más capas de procesamiento conjunto que late fusion

**Hiperparámetros**:
- Optimizer: Adam (lr=0.001)
- Loss: categorical_crossentropy
- Batch size: 32
- Epochs: 30

## Data Models

### Input Data Structures

```python
# Imágenes
X_images: np.ndarray
    shape: (10015, 28, 28, 3)
    dtype: float32
    range: [0.0, 1.0]

# Datos tabulares procesados
X_tabular: np.ndarray
    shape: (10015, n_features)  # n_features ≈ 15-20 después de encoding
    dtype: float32
    features: [age_normalized, sex_encoded, localization_encoded...]

# Labels
y: np.ndarray
    shape: (10015, 7)  # one-hot encoded
    dtype: float32
    classes: [akiec, bcc, bkl, df, mel, nv, vasc]
```

### Model Outputs

```python
# Predicciones
predictions: np.ndarray
    shape: (batch_size, 7)
    dtype: float32
    range: [0.0, 1.0]  # probabilidades que suman 1

# Embeddings
embeddings_tabular: np.ndarray
    shape: (batch_size, 32)
    dtype: float32

embeddings_cnn: np.ndarray
    shape: (batch_size, 128)
    dtype: float32
```

### Metadata Structure

```python
# DataFrame original
columns: [
    'lesion_id',      # ID único de la lesión
    'image_id',       # ID de la imagen
    'dx',             # Diagnóstico (label)
    'dx_type',        # Método de confirmación
    'age',            # Edad del paciente
    'sex',            # Sexo (male/female/unknown)
    'localization'    # Localización corporal
]
```

## Error Handling

### Data Loading Errors

```python
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print(f"Error montando Google Drive: {e}")
    print("Asegúrate de estar ejecutando en Google Colab")
    raise

# Verificar existencia de archivos
import os
data_path = '/content/drive/MyDrive/HAM10000/'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"No se encuentra el directorio: {data_path}")
```

### Training Errors

```python
# Callback para detectar NaN en loss
class NaNDetector(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs and np.isnan(logs.get('loss', 0)):
            print("⚠️ NaN detectado en loss!")
            print("Verificar:")
            print("- Normalización de datos")
            print("- Learning rate muy alto")
            print("- Explosión de gradientes")
            self.model.stop_training = True

# Early stopping para prevenir overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

### Memory Management

```python
# Liberar memoria entre entrenamientos
import gc
import tensorflow as tf

def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()

# Usar después de entrenar cada modelo
clear_memory()
```

### Data Validation

```python
def validate_data_consistency(X_tab, X_img, y):
    """Verificar que las particiones mantienen consistencia"""
    assert len(X_tab) == len(X_img) == len(y), \
        "Inconsistencia en número de muestras"
    
    assert not np.isnan(X_tab).any(), \
        "NaN encontrado en datos tabulares"
    
    assert not np.isnan(X_img).any(), \
        "NaN encontrado en imágenes"
    
    assert X_img.min() >= 0 and X_img.max() <= 1, \
        "Imágenes no normalizadas correctamente"
    
    print("✓ Validación de datos exitosa")
```

## Testing Strategy

### Unit Testing

Dado que el enfoque es desarrollo rápido en notebook, los tests serán verificaciones inline:

```python
# Test 1: Verificar shapes después de preprocesamiento
print(f"Shape imágenes train: {X_train_img.shape}")
print(f"Shape tabular train: {X_train_tab.shape}")
print(f"Shape labels train: {y_train.shape}")
assert X_train_img.shape[0] == X_train_tab.shape[0] == y_train.shape[0]

# Test 2: Verificar normalización
assert X_train_img.min() >= 0 and X_train_img.max() <= 1
print("✓ Imágenes normalizadas correctamente")

# Test 3: Verificar distribución de clases
class_distribution = y_train.argmax(axis=1)
print("Distribución de clases en train:")
print(np.bincount(class_distribution))

# Test 4: Verificar output del modelo
sample_pred = model.predict(X_train_img[:5])
assert sample_pred.shape == (5, 7)
assert np.allclose(sample_pred.sum(axis=1), 1.0)
print("✓ Predicciones con formato correcto")
```

### Integration Testing

```python
# Test end-to-end de pipeline completo
def test_full_pipeline():
    # 1. Cargar datos
    X_img, X_tab, y = load_and_preprocess_data()
    
    # 2. Split
    splits = split_data(X_tab, X_img, y)
    
    # 3. Entrenar modelo tabular (1 época)
    model_tab = build_tabular_model(X_tab.shape[1])
    model_tab.compile(optimizer='adam', loss='categorical_crossentropy')
    model_tab.fit(splits['X_train_tab'], splits['y_train'], epochs=1, verbose=0)
    
    # 4. Entrenar modelo CNN (1 época)
    model_cnn = build_cnn_model()
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy')
    model_cnn.fit(splits['X_train_img'], splits['y_train'], epochs=1, verbose=0)
    
    # 5. Late fusion
    pred_tab = model_tab.predict(splits['X_val_tab'])
    pred_cnn = model_cnn.predict(splits['X_val_img'])
    model_late = build_late_fusion_model()
    model_late.compile(optimizer='adam', loss='categorical_crossentropy')
    model_late.fit([pred_tab, pred_cnn], splits['y_val'], epochs=1, verbose=0)
    
    print("✓ Pipeline completo funcional")

# Ejecutar test con 1 época antes de entrenamiento completo
test_full_pipeline()
```

### Performance Testing

```python
# Verificar que el modelo puede ejecutarse en tiempo razonable
import time

start = time.time()
predictions = model.predict(X_test)
end = time.time()

print(f"Tiempo de inferencia: {end - start:.2f}s")
print(f"Muestras por segundo: {len(X_test) / (end - start):.0f}")
```

## Visualization Strategy

### Training Monitoring

```python
def plot_training_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Acc')
    axes[1].plot(history.history['val_accuracy'], label='Val Acc')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
```

### Evaluation Visualizations

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()

def plot_model_comparison(results_dict):
    """Comparar accuracy de los 4 modelos"""
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('Accuracy')
    plt.title('Comparación de Modelos en Test Set')
    plt.ylim([0, 1])
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

## Implementation Notes

### Development Workflow

1. **Fase de desarrollo (CPU)**:
   - Desarrollar código con 1 época
   - Verificar que no hay errores
   - Validar shapes y flujo de datos

2. **Fase de experimentación (GPU)**:
   - Conectar a GPU en Colab
   - Entrenar con epochs completos
   - Ajustar hiperparámetros

3. **Fase de evaluación**:
   - Evaluar en test set
   - Generar visualizaciones
   - Documentar resultados

### Code Organization

```
# Estructura del notebook

## 1. Setup y Configuración
- Imports
- Montaje de Google Drive
- Configuración de GPU

## 2. Carga y Preprocesamiento
- Carga de datos
- EDA básico
- Preprocesamiento
- Split train/val/test

## 3. Hito 1: Modelo Tabular
- Construcción del modelo
- Entrenamiento
- Evaluación en validación
- Visualizaciones

## 4. Hito 2: Modelo CNN
- Construcción del modelo
- Transfer learning
- Entrenamiento
- Evaluación en validación
- Visualizaciones

## 5. Hito 3: Late Fusion
- Obtención de predicciones
- Construcción del modelo de fusión
- Entrenamiento
- Evaluación en validación
- Visualizaciones

## 6. Hito 4: Early Fusion
- Extracción de embeddings
- Construcción del modelo de fusión
- Entrenamiento
- Evaluación en validación
- Visualizaciones

## 7. Evaluación Final
- Evaluación de los 4 modelos en test
- Matrices de confusión
- Comparación de resultados
- Métricas detalladas

## 8. Discusión y Conclusiones
- Interpretación de resultados
- Análisis de errores
- Trabajo futuro
```

### Hyperparameter Tuning Strategy

Dado el tiempo limitado, se recomienda:

1. **Prioridad 1**: Ajustar learning rate (probar: 0.0001, 0.001, 0.01)
2. **Prioridad 2**: Ajustar dropout (probar: 0.2, 0.3, 0.5)
3. **Prioridad 3**: Ajustar batch size (probar: 32, 64)
4. **Opcional**: Ajustar arquitectura (número de capas, neuronas)

### Saving Models

```python
# Guardar modelos entrenados
model_tab.save('/content/drive/MyDrive/HAM10000/models/model_tabular.h5')
model_cnn.save('/content/drive/MyDrive/HAM10000/models/model_cnn.h5')

# Cargar modelos
from tensorflow.keras.models import load_model
model_tab = load_model('/content/drive/MyDrive/HAM10000/models/model_tabular.h5')
```

## Performance Considerations

### Memory Optimization

- Usar `batch_size` apropiado (32-64) para no saturar memoria
- Liberar memoria con `tf.keras.backend.clear_session()` entre modelos
- No duplicar datos innecesariamente

### Training Time Estimates

Con GPU K80 en Colab:
- Modelo Tabular: ~2-3 minutos (50 epochs)
- Modelo CNN (transfer learning): ~5-10 minutos (30 epochs)
- Late Fusion: ~1-2 minutos (20 epochs)
- Early Fusion: ~3-5 minutos (30 epochs)

**Total estimado**: 15-25 minutos de entrenamiento

### Class Imbalance Handling

El dataset HAM10000 está desbalanceado (clase 'nv' domina). Estrategias:

1. **Stratified split**: Mantener proporciones en train/val/test
2. **Class weights**: Opcional, calcular pesos inversamente proporcionales
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train.argmax(axis=1)),
    y=y_train.argmax(axis=1)
)
class_weight_dict = dict(enumerate(class_weights))

# Usar en fit
model.fit(X_train, y_train, class_weight=class_weight_dict, ...)
```

3. **Métricas balanceadas**: Usar F1-score macro además de accuracy
