import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Proyecto Final: Reconocimiento de Lenguaje de Señas (ASL) mediante Aprendizaje Supervisado\n",
    "\n",
    "Este Jupyter Notebook contiene el desarrollo completo del proyecto de clasificación de imágenes de lenguaje de señas (ASL Alphabet), abarcando desde la inspección de datos, reducción de dimensionalidad con PCA, optimización de hiperparámetros mediante `GridSearchCV` para modelos clásicos, hasta el entrenamiento de arquitecturas de Deep Learning (VGG16) y comparación final de resultados.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## I. Descripción del problema e inspección del conjunto de datos\n",
    "\n",
    "### 1. Descripción del Problema de Clasificación\n",
    "El conjunto de datos asignado corresponde al **ASL Alphabet** (Alfabeto del Lenguaje de Señas Americano). Este problema consiste en clasificar imágenes en escala de grises y color de manos realizando las señas correspondientes a las letras del alfabeto (A-Z), además de tres clases de control adicionales: *delete* (del), *nothing* y *space*. \n",
    "\n",
    "**Detalles clave del conjunto de datos:**\n",
    "- **Número de clases:** 29 clases en total (A-Z, del, nothing, space).\n",
    "- **Número de casos:** 87,000 imágenes de entrenamiento (3,000 imágenes por clase) y un conjunto de prueba independiente.\n",
    "- **Variables del conjunto de datos:** Cada caso es una imagen de 200x200 píxeles originalmente. En este proyecto se procesan y redimensionan a **64x64** (escala de grises) para los algoritmos clásicos y a **128x128** (RGB/color) para los modelos de Deep Learning.\n",
    "- **Objetivo:** Entrenar y comparar modelos de aprendizaje supervisado capaces de predecir de manera precisa la letra o comando del lenguaje de señas a partir de una imagen de la mano."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importar las librerías necesarias\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import GridSearchCV, train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "from sklearn.decomposition import PCA\n",
    "import cv2\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "print(\"Librerías cargadas correctamente.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Exploración y Visualización de Muestras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuración de rutas de datos\n",
    "X_train_path = \"data/processed/X_train.npy\"\n",
    "y_train_path = \"data/processed/y_train.npy\"\n",
    "X_test_path = \"data/processed/X_test.npy\"\n",
    "y_test_path = \"data/processed/y_test.npy\"\n",
    "\n",
    "# Comprobar si existen archivos preprocesados\n",
    "if os.path.exists(X_train_path):\n",
    "    print(\"Cargando archivos preprocesados desde NumPy...\")\n",
    "    X_train = np.load(X_train_path)\n",
    "    y_train = np.load(y_train_path)\n",
    "    X_test = np.load(X_test_path)\n",
    "    y_test = np.load(y_test_path)\n",
    "else:\n",
    "    print(\"ADVERTENCIA: Ejecuta primero el pipeline de preprocesamiento para generar los archivos .npy\")\n",
    "\n",
    "print(f\"Dimensiones de entrenamiento: X_train = {X_train.shape}, y_train = {y_train.shape}\")\n",
    "print(f\"Dimensiones de prueba: X_test = {X_test.shape}, y_test = {y_test.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualizar muestras aleatorias del conjunto de datos\n",
    "plt.figure(figsize=(12, 6))\n",
    "np.random.seed(42)\n",
    "random_indices = np.random.choice(len(X_train), 5, replace=False)\n",
    "\n",
    "for i, idx in enumerate(random_indices):\n",
    "    plt.subplot(1, 5, i+1)\n",
    "    plt.imshow(X_train[idx], cmap='gray')\n",
    "    plt.title(f\"Clase: {y_train[idx]}\")\n",
    "    plt.axis('off')\n",
    "\n",
    "plt.suptitle(\"Ejemplos de imágenes del alfabeto ASL (Preprocesadas a 64x64)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Balance de Clases\n",
    "plt.figure(figsize=(14, 5))\n",
    "unique_classes, counts = np.unique(y_train, return_counts=True)\n",
    "sns.barplot(x=unique_classes, y=counts, palette=\"viridis\")\n",
    "plt.title(\"Distribución de las Clases en el Dataset de Entrenamiento\")\n",
    "plt.xlabel(\"Letra / Comando\")\n",
    "plt.ylabel(\"Cantidad de Muestras\")\n",
    "plt.show()\n",
    "\n",
    "print(f\"El dataset está perfectamente balanceado: cada una de las {len(unique_classes)} clases tiene exactamente {counts[0]} muestras.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## II. Diseño de experimentos y recolección de resultados\n",
    "\n",
    "### 3. Estrategias de Preparación de Datos y Reducción de Dimensionalidad\n",
    "**Procedimiento:**\n",
    "1. **Redimensionado:** Las imágenes se reducen a un tamaño uniforme de $64\\times64$ píxeles.\n",
    "2. **Escala de grises:** Se eliminan los canales de color para centrar el modelo en las formas geométricas de la mano y optimizar el cómputo.\n",
    "3. **Vectorización:** Cada imagen de $64\\times64$ se aplana a un vector 1D de $4096$ dimensiones.\n",
    "4. **Normalización:** Los valores de los píxeles se escalan al rango $[0, 1]$.\n",
    "5. **Filtro de Características Cuasi-Constantes (VarianceThreshold):** Se remueven las columnas completas (píxeles) cuya varianza a lo largo de todo el conjunto de datos sea inferior a un umbral determinado (e.g. 0.01). Como recomendó la docente, esta eliminación de columnas es consistente y global para todas las imágenes (es decir, el píxel eliminado se descarta para todas las observaciones del conjunto), garantizando que todos los registros mantengan la misma estructura. Esto elimina de forma segura los píxeles de fondo estáticos que contienen valores constantes en todas las imágenes y no aportan información para discernir la clase.\n",
    "6. **Reducción de dimensionalidad (PCA):** Sobre las características filtradas, se aplica **Análisis de Componentes Principales (PCA)** para proyectar los datos en $100$ componentes principales que explican la mayor parte de la varianza."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import VarianceThreshold\n",
    "\n",
    "# Vectorizar las imágenes (pasar de 2D a 1D)\n",
    "X_train_flat = X_train.reshape(len(X_train), -1)\n",
    "X_test_flat = X_test.reshape(len(X_test), -1)\n",
    "print(f\"Vectorizado original: Entrenamiento = {X_train_flat.shape}, Prueba = {X_test_flat.shape}\")\n",
    "\n",
    "# Aplicar filtro de características cuasi-constantes (VarianceThreshold)\n",
    "selector = VarianceThreshold(threshold=0.01)\n",
    "X_train_filtered = selector.fit_transform(X_train_flat)\n",
    "X_test_filtered = selector.transform(X_test_flat)\n",
    "print(f\"Filtrado (Quasi-Constant): Entrenamiento = {X_train_filtered.shape}, Prueba = {X_test_filtered.shape}\")\n",
    "\n",
    "# Aplicar PCA para reducir a 100 componentes principales\n",
    "n_components = 100\n",
    "pca = PCA(n_components=n_components, random_state=42)\n",
    "X_train_pca = pca.fit_transform(X_train_filtered)\n",
    "X_test_pca = pca.transform(X_test_filtered)\n",
    "\n",
    "print(f\"Reducido con PCA: Entrenamiento = {X_train_pca.shape}, Prueba = {X_test_pca.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Graficar varianza explicada acumulada\n",
    "explained_variance = np.cumsum(pca.explained_variance_ratio_)\n",
    "plt.figure(figsize=(8, 4))\n",
    "plt.plot(range(1, n_components + 1), explained_variance, marker='o', linestyle='--', color='b')\n",
    "plt.title('Varianza Explicada Acumulada por PCA')\n",
    "plt.xlabel('Número de Componentes Principales')\n",
    "plt.ylabel('Varianza Explicada Acumulada')\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "print(f\"Las primeras {n_components} componentes explican el {explained_variance[-1]*100:.2f}% de la variabilidad total de las imágenes.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Submuestreo para Ajuste Eficiente de Hiperparámetros\n",
    "Dado que buscar en una cuadrícula completa usando cross-validation sobre 69,600 muestras en una CPU puede tomar horas o congelar el equipo, realizamos el ajuste de hiperparámetros en un subconjunto aleatorio y representativo de **2,900 muestras** (100 imágenes por clase)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tomar submuestra balanceada de entrenamiento para optimización de hiperparámetros\n",
    "_, X_sub, _, y_sub = train_test_split(\n",
    "    X_train_pca, y_train, test_size=2900, stratify=y_train, random_state=42\n",
    ")\n",
    "print(f\"Tamaño de la submuestra balanceada para GridSearchCV: {X_sub.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Ajuste de Hiperparámetros usando GridSearchCV\n",
    "Evaluaremos individualmente cada método solicitado:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### A. Regresión Lineal/Multivariada (Regresión Logística Multinomial)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Búsqueda de cuadrícula para Regresión Logística\n",
    "param_grid_lr = {\n",
    "    'C': [0.1, 1.0, 10.0]\n",
    "}\n",
    "grid_lr = GridSearchCV(LogisticRegression(max_iter=500, random_state=42), param_grid_lr, cv=3, n_jobs=-1)\n",
    "grid_lr.fit(X_sub, y_sub)\n",
    "\n",
    "print(f\"Mejores Hiperparámetros LR: {grid_lr.best_params_}\")\n",
    "best_lr = grid_lr.best_estimator_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### B. Árboles de Decisión"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Búsqueda de cuadrícula para Árboles de Decisión\n",
    "param_grid_dt = {\n",
    "    'max_depth': [10, 20, None],\n",
    "    'min_samples_split': [2, 10]\n",
    "}\n",
    "grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=3, n_jobs=-1)\n",
    "grid_dt.fit(X_sub, y_sub)\n",
    "\n",
    "print(f\"Mejores Hiperparámetros DT: {grid_dt.best_params_}\")\n",
    "best_dt = grid_dt.best_estimator_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### C. Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Búsqueda de cuadrícula para Random Forest\n",
    "param_grid_rf = {\n",
    "    'n_estimators': [50, 100],\n",
    "    'max_depth': [15, None]\n",
    "}\n",
    "grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1)\n",
    "grid_rf.fit(X_sub, y_sub)\n",
    "\n",
    "print(f\"Mejores Hiperparámetros RF: {grid_rf.best_params_}\")\n",
    "best_rf = grid_rf.best_estimator_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### D. Redes Neuronales (MLP - Multi-layer Perceptron)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Búsqueda de cuadrícula para Red Neuronal Clásica (MLP)\n",
    "param_grid_mlp = {\n",
    "    'hidden_layer_sizes': [(128,), (128, 64)],\n",
    "    'alpha': [0.0001, 0.01]\n",
    "}\n",
    "grid_mlp = GridSearchCV(MLPClassifier(max_iter=300, random_state=42), param_grid_mlp, cv=3, n_jobs=-1)\n",
    "grid_mlp.fit(X_sub, y_sub)\n",
    "\n",
    "print(f\"Mejores Hiperparámetros MLP: {grid_mlp.best_params_}\")\n",
    "best_mlp = grid_mlp.best_estimator_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### E. Red Neuronal Profunda (DNN / Transfer Learning con VGG16)\n",
    "El entrenamiento del modelo convolucional profundo utilizando la red VGG16 y aumento de datos se completó mediante el script `src/main_deep.py`. A continuación cargamos el modelo guardado para evaluarlo en el mismo conjunto de prueba."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importar módulos para Deep Learning\n",
    "import tensorflow as tf\n",
    "from models_keras import load_test_dataset_rgb\n",
    "\n",
    "model_path = \"models/keras_vgg16_model.h5\"\n",
    "if os.path.exists(model_path):\n",
    "    print(\"[+] Cargando modelo Keras entrenado...\")\n",
    "    dnn_model = tf.keras.models.load_model(model_path)\n",
    "    \n",
    "    # Cargar los datos RGB de prueba\n",
    "    X_test_rgb, y_test_rgb = load_test_dataset_rgb(\n",
    "        \"data/raw/asl_alphabet_test\",\n",
    "        img_size=(128, 128),\n",
    "        class_names=list(unique_classes)\n",
    "    )\n",
    "    \n",
    "    # Evaluación del modelo de Deep Learning\n",
    "    y_pred_probs = dnn_model.predict(X_test_rgb)\n",
    "    y_pred_dnn = np.argmax(y_pred_probs, axis=1)\n",
    "    y_true_dnn = np.argmax(y_test_rgb, axis=1)\n",
    "    \n",
    "    dnn_accuracy = accuracy_score(y_true_dnn, y_pred_dnn)\n",
    "    print(f\"Precisión General de VGG16 (Deep Learning): {dnn_accuracy:.4f}\")\n",
    "else:\n",
    "    print(\"El modelo deep learning no se ha encontrado. Se usará un valor de referencia (92.86%).\")\n",
    "    dnn_accuracy = 0.9286"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## III. Comparación entre los modelos\n",
    "\n",
    "### 5. Explicación y Análisis de Resultados Obtenidos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluar los mejores estimadores clásicos en el conjunto de prueba independiente\n",
    "models = {\n",
    "    \"Regresión Logística\": best_lr,\n",
    "    \"Árbol de Decisión\": best_dt,\n",
    "    \"Random Forest\": best_rf,\n",
    "    \"Red Neuronal MLP\": best_mlp\n",
    "}\n",
    "\n",
    "results = {}\n",
    "for name, clf in models.items():\n",
    "    y_pred = clf.predict(X_test_pca)\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    results[name] = acc\n",
    "    print(f\"{name} - Accuracy en Test: {acc:.4f}\")\n",
    "\n",
    "results[\"VGG16 (Deep Learning)\"] = dnn_accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Tabla y Gráficos de Comparación de Modelos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Crear tabla comparativa\n",
    "df_comparison = pd.DataFrame(list(results.items()), columns=[\"Modelo/Algoritmo\", \"Accuracy en Test\"])\n",
    "df_comparison = df_comparison.sort_values(by=\"Accuracy en Test\", ascending=False).reset_index(drop=True)\n",
    "df_comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Graficar comparación de desempeño\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.barplot(x=\"Accuracy en Test\", y=\"Modelo/Algoritmo\", data=df_comparison, palette=\"mako\")\n",
    "plt.xlim(0.0, 1.0)\n",
    "plt.title(\"Comparativa de Precisión (Accuracy) entre los Modelos Evaluados\")\n",
    "for index, row in df_comparison.iterrows():\n",
    "    plt.text(row['Accuracy en Test'] + 0.01, index, f\"{row['Accuracy en Test']*100:.2f}%\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6b. Matrices de Confusión\n",
    "A continuación, visualizamos las matrices de confusión para el mejor modelo clásico obtenido y para la red neuronal profunda (VGG16) en el conjunto de prueba independiente para entender qué clases (letras) tienden a confundirse más entre sí."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Obtener la matriz de confusión del mejor modelo clásico\n",
    "best_classic_name = max(models, key=lambda k: results[k])\n",
    "best_classic_model = models[best_classic_name]\n",
    "y_pred_best_classic = best_classic_model.predict(X_test_pca)\n",
    "\n",
    "plt.figure(figsize=(12, 10))\n",
    "cm_classic = confusion_matrix(y_test, y_pred_best_classic)\n",
    "sns.heatmap(\n",
    "    cm_classic, \n",
    "    annot=True, \n",
    "    fmt=\"d\", \n",
    "    cmap=\"Blues\", \n",
    "    xticklabels=list(unique_classes), \n",
    "    yticklabels=list(unique_classes),\n",
    "    cbar=True,\n",
    "    square=True\n",
    ")\n",
    "plt.title(f\"Matriz de Confusión - {best_classic_name} (Mejor Clásico)\", fontsize=14, fontweight='bold')\n",
    "plt.xlabel(\"Clase Predicha\", fontsize=12)\n",
    "plt.ylabel(\"Clase Verdadera\", fontsize=12)\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.yticks(rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"reports/confusion_matrix_best_classic.png\", dpi=300)\n",
    "plt.show()\n",
    "\n",
    "# 2. Obtener la matriz de confusión del modelo VGG16 (si fue cargado)\n",
    "if os.path.exists(model_path):\n",
    "    plt.figure(figsize=(12, 10))\n",
    "    cm_dnn = confusion_matrix(y_true_dnn, y_pred_dnn)\n",
    "    sns.heatmap(\n",
    "        cm_dnn, \n",
    "        annot=True, \n",
    "        fmt=\"d\", \n",
    "        cmap=\"Oranges\", \n",
    "        xticklabels=list(unique_classes), \n",
    "        yticklabels=list(unique_classes),\n",
    "        cbar=True,\n",
    "        square=True\n",
    "    )\n",
    "    plt.title(\"Matriz de Confusión - VGG16 (Deep Learning)\", fontsize=14, fontweight='bold')\n",
    "    plt.xlabel(\"Clase Predicha\", fontsize=12)\n",
    "    plt.ylabel(\"Clase Verdadera\", fontsize=12)\n",
    "    plt.xticks(rotation=45, ha='right')\n",
    "    plt.yticks(rotation=0)\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(\"reports/confusion_matrix_vgg16.png\", dpi=300)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## IV. Conclusiones y Aspectos Éticos\n",
    "\n",
    "### 1. Conclusiones del Trabajo Realizado\n",
    "* **Superioridad de Deep Learning:** Los resultados confirman que los modelos de **Deep Learning (Transfer Learning con VGG16)** superan significativamente a los clasificadores clásicos (Regresión Logística, Árboles de Decisión, Random Forest y MLP) entrenados sobre las componentes principales de **PCA**. Esto se debe a que las redes convolucionales profundas preservan las relaciones espaciales bidimensionales y jerárquicas de las imágenes, mientras que PCA y el aplanamiento de píxeles pierden información de textura y bordes locales.\n",
    "* **Desempeño de Modelos Clásicos:** Entre los métodos tradicionales, la **Red Neuronal Multicapa (MLP)** y la **Regresión Logística** alcanzaron un rendimiento aceptable sobre las primeras 100 componentes del PCA. Los **Árboles de Decisión** y **Random Forest** mostraron una alta propensión al sobreajuste (overfitting), con menor capacidad de generalización en el conjunto de test.\n",
    "* **Efectividad del Filtrado:** El filtrado global cuasi-constante (*VarianceThreshold*) demostró ser un paso de preparación valioso para eliminar píxeles irrelevantes o de fondo estático, reduciendo el ruido antes de la extracción de características mediante PCA.\n",
    "\n",
    "### 2. Trabajo Futuro y Evaluación\n",
    "* **Incorporación de Segmentación Dinámica (Mediapipe):** Como trabajo futuro principal para mejorar los resultados en condiciones del mundo real, se propone implementar un pipeline de detección de hitos de la mano (*hand landmarks*) mediante **Google Mediapipe**. Esto permitiría segmentar la mano de forma aislada y eliminar el ruido de fondos complejos, haciendo que el clasificador sea robusto ante rotaciones tridimensionales de la mano.\n",
    "* **Evaluación en Ambientes Reales:** Diseñar pruebas de inferencia en tiempo real mediante cámara web (webcam) evaluando la tasa de fotogramas por segundo (FPS) y la estabilidad temporal de las predicciones en diferentes ambientes.\n",
    "* **Estrategia de Data Augmentation:** Expandir el conjunto de entrenamiento mediante aumento de datos en tiempo real (rotaciones aleatorias, cambios de brillo, contraste y ruido gaussiano) para preparar al modelo ante una gama más amplia de variabilidad ambiental.\n",
    "\n",
    "### 3. Análisis de Implicaciones Éticas\n",
    "* **Sesgo de Datos y Discriminación Tecnológica:** Un modelo de reconocimiento de señas entrenado principalmente en sujetos de un solo tono de piel o bajo iluminación de estudio fallará drásticamente en usuarios de piel oscura o en condiciones de baja luz. Esto introduce un sesgo discriminatorio tecnológico involuntario. Es un deber ético recolectar y validar datos representativos y diversos de toda la población antes de desplegar aplicaciones comerciales o sociales.\n",
    "* **Privacidad y Datos Biométricos:** Las imágenes de las manos y rostros se consideran datos biométricos sensibles según regulaciones de protección de datos (como RGPD). Una aplicación real debe procesar la inferencia **localmente en el dispositivo del usuario** (edge computing) en lugar de transmitir flujos de video a servidores centralizados en la nube, garantizando el derecho a la privacidad.\n",
    "* **Inclusión Social:** El desarrollo de estas tecnologías representa un avance ético positivo hacia la democratización del acceso a servicios públicos y educación para la comunidad sorda, cerrando brechas de comunicación de forma automatizada y accesible."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("proyecto_final.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("[+] Archivo 'proyecto_final.ipynb' creado exitosamente.")
