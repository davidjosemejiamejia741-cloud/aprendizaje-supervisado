# Resumen de Cambios: Implementación de VGG16 (Keras)

Hemos completado la estructuración e implementación del nuevo pipeline de **Deep Learning** para tu proyecto usando **TensorFlow/Keras**, enfocándonos en el uso de **VGG16** (con transfer learning) e incorporando estrategias sólidas contra el sobreajuste.

---

## Archivos Creados y Modificados

### [NEW] [requirements.txt](file:///C:/Users/ESTUDIANTE/Downloads/aprendizaje_supervisado/requirements.txt)
* Contiene las dependencias necesarias: `tensorflow` y `tqdm`.

### [NEW] [models_keras.py](file:///C:/Users/ESTUDIANTE/Downloads/aprendizaje_supervisado/src/models_keras.py)
* **Cargador Optimizado (RGB):** Lee imágenes de `data/raw/asl_alphabet_train` usando `image_dataset_from_directory` para evitar problemas de memoria (RAM) al procesar en lotes. Mantiene el rango de píxeles `[0, 255]` requerido por VGG16.
* **Data Augmentation:** Capas secuenciales de Keras que aplican rotación, zoom y traslación aleatoria durante el entrenamiento para evitar que el modelo memorice el fondo o el tono de piel del sujeto (resolviendo la duda sobre la falta de varianza). Se omitió el volteo horizontal para no invalidar señas izquierda/derecha.
* **Modelo VGG16 (Transfer Learning):** Carga `VGG16` preentrenado con ImageNet, congela sus pesos extractores y añade un clasificador lineal denso (`Dense`) con regularización `Dropout`.
* **CNN Propia:** Mantiene la opción alternativa de entrenar una red convolucional propia desde cero si decides probarla.

### [NEW] [main_deep.py](file:///C:/Users/ESTUDIANTE/Downloads/aprendizaje_supervisado/src/main_deep.py)
* Script de ejecución independiente para el modelo de Deep Learning. Por defecto entrena **VGG16** por 5 épocas sobre CPU.
* Genera y guarda las curvas de aprendizaje (exactitud y pérdida) en `reports/learning_curves_vgg16.png`.
* Guarda el reporte final en `reports/evaluation_report_deep_vgg16.txt` y el modelo entrenado en `models/keras_vgg16_model.h5`.

### [MODIFY] [predict.py](file:///C:/Users/ESTUDIANTE/Downloads/aprendizaje_supervisado/src/predict.py)
* Actualizado para soportar inferencia híbrida. Si encuentra el modelo de Keras (`models/keras_vgg16_model.h5`), realiza la predicción de imágenes de prueba usando VGG16 en formato RGB; si no está, recurre automáticamente al modelo clásico (PCA + Regresión Logística).

---

## Cómo Ejecutar el Entrenamiento y la Predicción

1. **Instalar Dependencias:**
   Ejecuta en tu terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Entrenar VGG16:**
   Ejecuta:
   ```bash
   python src/main_deep.py
   ```

3. **Ejecutar Predicción de Prueba:**
   Ejecuta:
   ```bash
   python src/predict.py
   ```
