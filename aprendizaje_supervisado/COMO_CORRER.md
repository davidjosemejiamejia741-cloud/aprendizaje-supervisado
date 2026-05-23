# Cómo correr el proyecto

## Requisitos previos

Tener Python instalado. Luego instalar las dependencias:

```bash
pip install -r requirements.txt
```

Dependencias: `tensorflow`, `scikit-learn`, `opencv-python`, `matplotlib`, `seaborn`, `tqdm`

---

## Estructura del proyecto

```
aprendizaje_supervisado/
├── src/
│   ├── main.py               # Pipeline clásico (scikit-learn)
│   ├── main_deep.py          # Pipeline deep learning (VGG16, ResNet50, MobileNetV2)
│   ├── predict.py            # Predicción sobre imágenes nuevas
│   ├── preprocessing.py      # Limpieza y preprocesamiento de imágenes
│   ├── feature_engineering.py# Vectorización, filtro de varianza y PCA
│   ├── visualization.py      # Generación de todos los gráficos
│   └── models_keras.py       # Arquitecturas de redes neuronales
├── data/
│   ├── train/                # Imágenes de entrenamiento (29 subcarpetas A-Z, del, nothing, space)
│   ├── test/                 # Imágenes de prueba
│   ├── processed/            # Generado automáticamente: X_train.npy, y_train.npy, etc.
│   └── reduced/              # Generado automáticamente: datos reducidos con PCA
├── models/                   # Modelos entrenados guardados
├── reports/                  # Gráficos y reportes generados
├── embed_supervisado.py      # Embebe imágenes del notebook en base64
├── zip_project.py            # Empaqueta el proyecto en .zip
└── requirements.txt
```

---

## Paso 1 — Pipeline clásico (recomendado)

Entrena y compara 4 modelos: Regresión Logística, Árbol de Decisión, Random Forest y MLP.

```bash
python src/main.py
```

**Qué hace internamente:**
1. Carga imágenes preprocesadas desde `data/processed/`
2. Aplica submuestreo estratificado (máx. 1400 imágenes por clase)
3. Divide en Train / Validación / Test (~1000 / 200 / 200 por clase)
4. Vectoriza imágenes: (N, 128×128) → (N, 16,384 píxeles)
5. Elimina características cuasi-constantes (~13,000 quedan)
6. Reduce con PCA a 950 componentes (99.5% varianza explicada)
7. Entrena los 4 modelos con GridSearchCV (cv=3)
8. Evalúa en validación y test, selecciona el mejor

**Archivos generados en `reports/`:**
- `muestras_asl.png` — ejemplos del dataset
- `distribucion_clases.png` — balance de clases
- `distribucion_particion.png` — distribución train/val/test
- `varianza_pca.png` — varianza acumulada del PCA
- `proyeccion_2d_pca.png` — proyección 2D del dataset
- `confusion_matrix_*.png` — matriz de confusión de cada modelo
- `evaluation_report.txt` — tabla comparativa con accuracy e hiperparámetros

**Archivos generados en `models/`:**
- `modelo_sign_language.pkl` — mejor modelo clásico
- `pca_transform.pkl` — transformador PCA
- `variance_selector.pkl` — selector de características
- `label_encoder.pkl` — codificador de clases

**Resultados esperados:**

| Modelo              | Accuracy (Test) |
|---------------------|-----------------|
| Regresión Logística | ~100%           |
| Red Neuronal MLP    | ~100%           |
| Random Forest       | ~99.85%         |
| Árbol de Decisión   | ~98.95%         |

---

## Paso 2 — Pipeline deep learning (opcional, lento en CPU)

Entrena una red neuronal con Transfer Learning usando VGG16, ResNet50 o MobileNetV2.

```bash
python src/main_deep.py
```

Se puede elegir el modelo con el argumento `--model`:

```bash
python src/main_deep.py --model transfer_learning  # MobileNetV2 (más rápido en CPU)
python src/main_deep.py --model vgg16              # VGG16
python src/main_deep.py --model resnet50           # ResNet50 (mejor accuracy)
python src/main_deep.py --model custom_cnn         # CNN propia desde cero
```

**Qué hace internamente:**
1. Carga imágenes en RGB directamente desde `data/train/`
2. Aplica data augmentation (rotación, zoom, traslación)
3. Usa Transfer Learning con pesos preentrenados de ImageNet
4. Entrena durante 8 épocas
5. Evalúa en el conjunto de prueba

**Archivos generados:**
- `reports/learning_curves_*.png` — curvas de accuracy y loss
- `reports/confusion_matrix_deep_*.png` — matriz de confusión
- `reports/evaluation_report_deep_*.txt` — reporte de clasificación
- `models/keras_*_model.keras` y `.h5` — modelo entrenado

> **Nota:** sin GPU este paso puede tardar horas. Si solo tienes CPU, usa `--model transfer_learning` (MobileNetV2 es el más ligero).

---

## Paso 3 — Predicción sobre imágenes nuevas

```bash
python src/predict.py
```

Detecta automáticamente qué modelo usar:
- Si existe `models/keras_vgg16_model.keras` → usa VGG16 (imagen en RGB)
- Si no → usa el modelo clásico (imagen en escala de grises + PCA)

Muestra las **Top 3 predicciones** con su porcentaje de confianza.

---

## Paso 4 — Generar notebook con imágenes embebidas (opcional)

Para tener un notebook que funcione sin depender de la carpeta `reports/`:

```bash
python embed_supervisado.py
```

Genera `proyecto_final_final.ipynb` con todas las imágenes incrustadas en base64.

---

## Paso 5 — Empaquetar para entregar (opcional)

```bash
python zip_project.py
```

Genera `proyecto_codigo.zip` con el código fuente, notebooks y reportes.
No incluye los datos ni los modelos (archivos demasiado grandes).

---

## Orden recomendado de ejecución

```
1. pip install -r requirements.txt
2. python src/main.py
3. python src/predict.py           (opcional)
4. python embed_supervisado.py     (opcional)
5. python zip_project.py           (opcional)
```

---

## Notas importantes

- Los datos ya están preprocesados en `data/processed/`. Si quieres reprocesar desde cero, cambia `REPROCESS_DATA = True` en `src/main.py`.
- Si los gráficos no aparecen en VS Code después de ejecutar, haz click derecho en la carpeta `reports/` y selecciona **Refresh**.
- El notebook principal del proyecto es `proyecto_final_final.ipynb`.
