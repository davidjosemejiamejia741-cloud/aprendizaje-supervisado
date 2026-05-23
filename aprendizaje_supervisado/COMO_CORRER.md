# Cómo correr el proyecto

## Requisitos previos

Tener Python instalado. Luego instalar las dependencias ejecutando en la terminal:

```bash
pip install -r requirements.txt
```

---

## Estructura del proyecto

```
aprendizaje_supervisado/
├── src/
│   ├── main.py                  # Pipeline clásico (scikit-learn)
│   ├── main_deep.py             # Pipeline deep learning (VGG16)
│   ├── predict.py               # Predicción sobre imágenes nuevas
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   └── models_keras.py
├── data/
│   ├── train/                   # Imágenes de entrenamiento por clase
│   ├── test/                    # Imágenes de prueba
│   ├── processed/               # Datos preprocesados (.npy)
│   └── reduced/                 # Datos reducidos con PCA (.npy)
├── models/                      # Modelos entrenados guardados
├── reports/                     # Gráficos y reportes generados
├── embed_supervisado.py         # Embebe imágenes en el notebook
├── zip_project.py               # Empaqueta el proyecto en .zip
└── requirements.txt
```

---

## Paso 1 — Pipeline clásico (recomendado, rápido)

Entrena y compara 4 modelos clásicos: Regresión Logística, Árbol de Decisión, Random Forest y MLP.

```bash
python src/main.py
```

**Qué hace:**
- Carga las imágenes preprocesadas desde `data/processed/`
- Aplica filtro de varianza y reducción PCA (950 componentes)
- Entrena los 4 modelos con GridSearchCV (búsqueda de hiperparámetros)
- Genera gráficos en `reports/`
- Guarda el mejor modelo en `models/modelo_sign_language.pkl`

**Resultados esperados:**
- Regresión Logística: ~100% accuracy
- Random Forest: ~99.85% accuracy
- MLP: ~100% accuracy
- Árbol de Decisión: ~98.95% accuracy

---

## Paso 2 — Pipeline deep learning con VGG16 (opcional, lento en CPU)

```bash
python src/main_deep.py
```

**Qué hace:**
- Carga imágenes en RGB desde `data/train/`
- Usa transfer learning con VGG16 preentrenado en ImageNet
- Genera curvas de aprendizaje en `reports/learning_curves_vgg16.png`
- Guarda el modelo en `models/keras_vgg16_model.h5`

> **Advertencia:** este script puede tardar varios minutos u horas dependiendo de si tienes GPU. Si solo tienes CPU, usa el Paso 1.

---

## Paso 3 — Predicción sobre imágenes nuevas

```bash
python src/predict.py
```

Usa el modelo VGG16 si ya fue entrenado; si no, usa automáticamente el modelo clásico.

---

## Paso 4 — Generar notebook con imágenes embebidas

Si quieres un notebook que funcione sin depender de la carpeta `reports/`:

```bash
python embed_supervisado.py
```

Genera `proyecto_final_final.ipynb` con todas las imágenes incrustadas en base64.

---

## Paso 5 — Empaquetar el proyecto en ZIP

Para subir el proyecto a Google Drive o Kaggle:

```bash
python zip_project.py
```

Genera `proyecto_codigo.zip` con el código fuente, notebooks y reportes.

---

## Orden recomendado de ejecución

```
1. pip install -r requirements.txt
2. python src/main.py
3. python embed_supervisado.py   (opcional)
4. python zip_project.py         (opcional, para entregar)
```

---

## Notas importantes

- Los datos ya están preprocesados en `data/processed/`. Si quieres reprocesar desde las imágenes crudas, cambia `REPROCESS_DATA = True` en `src/main.py`.
- Todos los gráficos se guardan automáticamente en `reports/`. Si no aparecen en VS Code, haz click derecho sobre la carpeta y selecciona **Refresh**.
- El notebook principal para revisar el proyecto es `proyecto_final_final.ipynb`.
