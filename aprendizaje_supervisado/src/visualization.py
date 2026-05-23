#aprendizaje_supervisado\src\visualization.py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar bloqueos y guardar imágenes directamente
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Rectangle

# Asegurar que el directorio de reportes exista
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configurar el estilo de seaborn para que los gráficos se vean profesionales
sns.set_theme(style="whitegrid")

# ==============================
# 1. VISUALIZAR IMÁGENES
# ==============================
def show_samples(X, y, num_samples=5, save_path=None):
    """
    Muestra una muestra aleatoria de imágenes en escala de grises con sus etiquetas
    y opcionalmente guarda el gráfico en disco.
    """
    if len(X) == 0:
        print("No hay imágenes para mostrar.")
        return

    # Elegir índices aleatorios para que varíe en cada ejecución
    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)
    
    cols = min(num_samples, len(X))
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.4, 2.8), squeeze=False)
    axes = axes.ravel()
    for i, idx in enumerate(indices):
        axes[i].imshow(X[idx], cmap='gray', aspect='equal')
        axes[i].set_title(f"Clase: {y[idx]}")
        axes[i].set_box_aspect(1)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlim(-0.5, X[idx].shape[1] - 0.5)
        axes[i].set_ylim(X[idx].shape[0] - 0.5, -0.5)
        axes[i].add_patch(
            Rectangle(
                (-0.5, -0.5),
                X[idx].shape[1],
                X[idx].shape[0],
                fill=False,
                edgecolor="#111827",
                linewidth=1.4,
            )
        )
        for spine in axes[i].spines.values():
            spine.set_visible(False)

    fig.suptitle("Ejemplos del Dataset ASL", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico guardado en: {full_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"No se pudo mostrar la ventana del gráfico (entorno sin GUI): {e}")
    finally:
        plt.close()

# ==============================
# 2. DISTRIBUCIÓN DE CLASES
# ==============================
def plot_class_distribution(y, save_path=None):
    """
    Grafica la distribución de frecuencia de las clases (balance de dataset)
    y opcionalmente guarda el gráfico en disco.
    """
    if len(y) == 0:
        print("No hay etiquetas para graficar distribución.")
        return

    plt.figure(figsize=(12, 6))
    
    # Ordenar las clases alfabéticamente para una visualización ordenada
    unique_classes, counts = np.unique(y, return_counts=True)
    
    sns.barplot(x=unique_classes, y=counts, palette="viridis")
    
    plt.title("Distribución de Clases en el Dataset ASL", fontsize=14, fontweight='bold')
    plt.xlabel("Clase / Letra", fontsize=12)
    plt.ylabel("Cantidad de Imágenes", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de distribución guardado en: {full_path}")
        
    plt.close()

# ==============================
# 2.5. DISTRIBUCIÓN DE PARTICIÓN DE DATOS (TRAIN/VAL/TEST)
# ==============================
# ==============================
# 2. DISTRIBUCIÓN DE CLASES
# ==============================
def plot_class_distribution(y, save_path=None, title="Distribución de imágenes por clase en el Dataset ASL"):
    """
    Grafica la distribución de frecuencia de las clases.
    Muestra la cantidad exacta de imágenes por cada letra.
    """

    if len(y) == 0:
        print("No hay etiquetas para graficar distribución.")
        return

    plt.figure(figsize=(14, 6))

    # Contar imágenes por clase
    unique_classes, counts = np.unique(y, return_counts=True)

    # Crear gráfico de barras
    ax = sns.barplot(
        x=unique_classes,
        y=counts,
        palette="viridis"
    )

    # Título y etiquetas
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Clase / Letra", fontsize=12)
    plt.ylabel("Cantidad de imágenes", fontsize=12)
    plt.xticks(rotation=45)

    # Mostrar el número exacto encima de cada barra
    for i, count in enumerate(counts):
        ax.text(
            i,
            count + 2,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    # Ajustar límite del eje Y para que se vean los números
    plt.ylim(0, max(counts) + 30)

    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de distribución guardado en: {full_path}")

    plt.close()


def plot_data_split_distribution(y_train, y_val, y_test, save_path=None):
    """
    Grafica la cantidad de imágenes por clase en las particiones
    de entrenamiento, validación y prueba.
    """
    labels = sorted(np.unique(np.concatenate([y_train, y_val, y_test])))

    def count_by_label(y):
        unique, counts = np.unique(y, return_counts=True)
        counter = dict(zip(unique, counts))
        return [counter.get(label, 0) for label in labels]

    train_counts = count_by_label(y_train)
    val_counts = count_by_label(y_val)
    test_counts = count_by_label(y_test)

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width, train_counts, width, label="Entrenamiento", color="#4C78A8")
    ax.bar(x, val_counts, width, label="Validación", color="#F58518")
    ax.bar(x + width, test_counts, width, label="Prueba", color="#54A24B")

    ax.set_title("Distribución de clases por partición", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clase / Letra", fontsize=12)
    ax.set_ylabel("Cantidad de imágenes", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de partición guardado en: {full_path}")

    plt.close()

# ==============================
# 3. VARIANZA PCA
# ==============================
def plot_pca_variance(pca, total_initial_features=None, save_path=None):
    """
    Grafica la varianza acumulada explicada por el análisis de componentes principales (PCA)
    y opcionalmente la guarda en disco, mostrando el total de componentes utilizados.
    """
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_pca = len(cumsum)
    
    # Intentar obtener el número de características que entraron a PCA
    n_filtered = getattr(pca, "n_features_in_", None)
    if n_filtered is None:
        n_filtered = getattr(pca, "n_features_", None)  # Compatibilidad con versiones anteriores de sklearn
        
    # Si no se especificó total_initial_features, tomar n_filtered o n_pca
    if total_initial_features is None:
        total_initial_features = n_filtered if n_filtered is not None else n_pca

    # Configuración de colores moderna e HSL adaptada para un acabado premium
    line_color = "#6366F1"      # Indigo 500
    accent_color = "#EC4899"    # Pink 500
    text_box_bg = "#F9FAFB"     # Gray 50
    text_box_border = "#E5E7EB" # Gray 200
    
    plt.figure(figsize=(10.5, 6.5))
    
    # Trazar la varianza acumulada
    plt.plot(
        range(1, n_pca + 1), 
        cumsum, 
        marker='o', 
        markersize=5, 
        linestyle='-', 
        linewidth=2, 
        color=line_color,
        label='Varianza Explicada Acumulada'
    )
    
    # Línea de referencia del 90%
    plt.axhline(y=0.90, color=accent_color, linestyle='--', linewidth=1.5, label='Umbral 90% de Varianza')
    
    # Línea vertical indicando la cantidad de componentes seleccionados
    plt.axvline(x=n_pca, color='#9CA3AF', linestyle=':', linewidth=1.5, label=f'Componentes Seleccionados ({n_pca})')
    
    # Resaltar y etiquetar el punto final
    final_variance = cumsum[-1]
    plt.plot(n_pca, final_variance, marker='s', color=accent_color, markersize=8)
    plt.annotate(
        f"{final_variance*100:.2f}%", 
        xy=(n_pca, final_variance), 
        xytext=(n_pca - 12, final_variance - 0.05),
        arrowprops=dict(facecolor='#1F2937', shrink=0.05, width=1, headwidth=6, headlength=6),
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#D97706", lw=1)
    )

    # Etiquetas de ejes
    plt.xlabel("Número de Componentes Principales Seleccionados (PCA)", fontsize=12, fontweight='semibold')
    plt.ylabel("Varianza Explicada Acumulada", fontsize=12, fontweight='semibold')
    
    # Títulos
    plt.title("Análisis de Varianza Explicada Acumulada por PCA", fontsize=14, fontweight='bold', pad=15)
    
    # Subtítulo descriptivo con el desglose de componentes
    subtitle = (
        f"Flujo de características: {total_initial_features} Originales (Píxeles 128x128) → "
        f"{n_filtered} Post-filtro Cuasi-Constante → {n_pca} PCA (Modelado)"
    )
    plt.figtext(0.5, 0.91, subtitle, wrap=True, horizontalalignment='center', fontsize=10, style='italic', color='#4B5563')
    
    # Cuadro informativo flotante sobre los componentes que utilizábamos
    info_text = (
        f"RESUMEN DE COMPONENTES:\n"
        f"- Pixeles totales (Originales): {total_initial_features}\n"
        f"- Componentes Activos (Post-Filtro): {n_filtered}\n"
        f"- Componentes Reducidos (PCA): {n_pca}\n"
        f"- Varianza Explicada Total: {final_variance*100:.2f}%"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor=text_box_bg, edgecolor=text_box_border, alpha=0.95)
    plt.text(
        0.05, 0.15, 
        info_text, 
        transform=plt.gca().transAxes, 
        fontsize=9.5,
        verticalalignment='bottom', 
        bbox=props,
        fontfamily='monospace'
    )
    
    plt.legend(loc="center right", frameon=True, facecolor='white', edgecolor='#E5E7EB')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Límites para mejorar visualización y espacio de etiquetas
    plt.ylim(0, 1.05)
    plt.xlim(0, n_pca + 5)
    
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de varianza PCA guardado en: {full_path}")
        
    plt.close()

# ==============================
# 4. VISUALIZACIÓN 2D
# ==============================
def plot_2d(X, y, title="PCA 2D Projection", save_path=None):
    """
    Grafica la proyección 2D de los datos usando las primeras dos dimensiones
    y opcionalmente la guarda en disco.
    """
    plt.figure(figsize=(10, 8))
    
    # Codificar etiquetas a números para el mapa de colores
    unique_labels = sorted(list(set(y)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    y_ids = np.array([label_to_id[label] for label in y])

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_ids, cmap='tab20', s=15, alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Componente Principal 1", fontsize=12)
    plt.ylabel("Componente Principal 2", fontsize=12)
    
    # Configurar leyenda interactiva
    cbar = plt.colorbar(scatter)
    cbar.set_label("ID de Clase")
    
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de proyección 2D guardado en: {full_path}")
        
    try:
        plt.show()
    except Exception as e:
        print(f"No se pudo mostrar la ventana del gráfico: {e}")
    finally:
        plt.close()

# ==============================
# 5. MATRIZ DE CONFUSIÓN
# ==============================
def plot_confusion_matrix(y_true, y_pred, class_names, title="Matriz de Confusión", save_path=None):
    """
    Grafica la matriz de confusión como un heatmap de Seaborn
    y opcionalmente la guarda en disco.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Clase Predicha", fontsize=12)
    plt.ylabel("Clase Verdadera", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de matriz de confusión guardado en: {full_path}")
        
    try:
        plt.show()
    except Exception as e:
        print(f"No se pudo mostrar la ventana del gráfico: {e}")
    finally:
        plt.close()


# ==============================
# 6. MATRICES DE CORRELACIÓN
# ==============================
def plot_image_feature_correlation(X, save_path=None):
    """
    Calcula y grafica una matriz de correlación usando variables resumen
    extraídas de las imágenes preprocesadas.
    """
    import pandas as pd

    image_features = pd.DataFrame({
        "media_intensidad": X.mean(axis=(1, 2)),
        "desviacion_intensidad": X.std(axis=(1, 2)),
        "min_intensidad": X.min(axis=(1, 2)),
        "max_intensidad": X.max(axis=(1, 2)),
    })

    corr = image_features.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Matriz de correlación de características resumen de imágenes", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Matriz de correlación de imágenes guardada en: {full_path}")

    plt.close()

    return corr


def plot_experiment_correlation(results_df, save_path=None):
    """
    Calcula y grafica la correlación entre hiperparámetros y métricas
    de los experimentos realizados.
    """
    columns = [
        "Épocas",
        "Batch size",
        "Learning rate",
        "Test accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
    ]
    corr = results_df[columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        square=True,
    )
    plt.title("Matriz de correlación entre hiperparámetros y métricas", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Matriz de correlación de experimentos guardada en: {full_path}")

    plt.close()

    return corr


# ==============================
# 7. EXACTITUD DE PRUEBA POR EXPERIMENTO
# ==============================
def plot_test_accuracy_by_experiment(results_df, experiment_col="Experimento", save_path=None):
    """
    Grafica la exactitud en prueba de cada experimento como barras verticales.
    Espera columnas: experiment_col, 'Test accuracy'.
    """
    df = results_df.copy()
    if experiment_col not in df.columns:
        df[experiment_col] = [f"Exp {i+1}" for i in range(len(df))]

    plt.figure(figsize=(max(10, len(df) * 0.8), 5))
    ax = sns.barplot(
        data=df,
        x=experiment_col,
        y="Test accuracy",
        palette="mako",
        hue=experiment_col,
        legend=False,
    )

    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["Test accuracy"] + 0.003,
            f"{row['Test accuracy']*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.ylim(0, 1.08)
    plt.title("Exactitud en Prueba por Experimento", fontsize=14, fontweight="bold")
    plt.xlabel("Experimento", fontsize=12)
    plt.ylabel("Accuracy en Test", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de exactitud por experimento guardado en: {full_path}")

    plt.close()


# ==============================
# 8. COMPARACIÓN DE MÉTRICAS MACRO POR EXPERIMENTO
# ==============================
def plot_macro_metrics_by_experiment(results_df, experiment_col="Experimento", save_path=None):
    """
    Grafica barras agrupadas comparando Precision macro, Recall macro y F1 macro
    por experimento.
    Espera columnas: experiment_col, 'Macro precision', 'Macro recall', 'Macro F1'.
    """
    df = results_df.copy()
    if experiment_col not in df.columns:
        df[experiment_col] = [f"Exp {i+1}" for i in range(len(df))]

    df_melted = df[[experiment_col, "Macro precision", "Macro recall", "Macro F1"]].melt(
        id_vars=experiment_col,
        var_name="Métrica",
        value_name="Valor",
    )

    plt.figure(figsize=(max(12, len(df) * 1.2), 6))
    sns.barplot(
        data=df_melted,
        x=experiment_col,
        y="Valor",
        hue="Métrica",
        palette=["#4C78A8", "#F58518", "#54A24B"],
    )

    plt.ylim(0, 1.08)
    plt.title("Comparación de Métricas Macro por Experimento", fontsize=14, fontweight="bold")
    plt.xlabel("Experimento", fontsize=12)
    plt.ylabel("Valor de la Métrica", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Métrica", loc="lower right")
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de métricas macro guardado en: {full_path}")

    plt.close()


# ==============================
# 9. EFECTO DEL BATCH SIZE SOBRE EL ACCURACY
# ==============================
def plot_batch_size_effect(results_df, save_path=None):
    """
    Grafica el efecto del batch size sobre el accuracy en prueba.
    Agrupa por batch size y muestra promedio con barras de error.
    Espera columnas: 'Batch size', 'Test accuracy'.
    """
    df = results_df.groupby("Batch size")["Test accuracy"].agg(["mean", "std"]).reset_index()
    df.columns = ["Batch size", "Media", "Std"]
    df["Std"] = df["Std"].fillna(0)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df,
        x="Batch size",
        y="Media",
        palette="viridis",
        hue="Batch size",
        legend=False,
    )

    for i, row in df.iterrows():
        ax.errorbar(i, row["Media"], yerr=row["Std"], fmt="none", color="black", capsize=4, linewidth=1.5)
        ax.text(
            i,
            row["Media"] + row["Std"] + 0.004,
            f"{row['Media']*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.ylim(0, 1.1)
    plt.title("Efecto del Batch Size sobre el Accuracy en Prueba", fontsize=13, fontweight="bold")
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Accuracy promedio en Test", fontsize=12)
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de efecto batch size guardado en: {full_path}")

    plt.close()


# ==============================
# 10. EFECTO DE LAS ÉPOCAS SOBRE EL ACCURACY
# ==============================
def plot_epochs_effect(results_df, save_path=None):
    """
    Grafica el efecto del número de épocas sobre el accuracy en prueba.
    Agrupa por épocas y muestra promedio con banda de desviación estándar.
    Espera columnas: 'Épocas', 'Test accuracy'.
    """
    df = results_df.groupby("Épocas")["Test accuracy"].agg(["mean", "std"]).reset_index()
    df.columns = ["Épocas", "Media", "Std"]
    df["Std"] = df["Std"].fillna(0)

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(len(df)),
        df["Media"],
        marker="o",
        linewidth=2,
        color="#6366F1",
        markersize=8,
        label="Accuracy promedio",
    )
    plt.fill_between(
        range(len(df)),
        df["Media"] - df["Std"],
        df["Media"] + df["Std"],
        alpha=0.2,
        color="#6366F1",
        label="± Desv. estándar",
    )

    for i, row in df.iterrows():
        plt.text(
            i,
            row["Media"] + row["Std"] + 0.004,
            f"{row['Media']*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.ylim(0, 1.1)
    plt.title("Efecto del Número de Épocas sobre el Accuracy en Prueba", fontsize=13, fontweight="bold")
    plt.xlabel("Épocas", fontsize=12)
    plt.ylabel("Accuracy en Test", fontsize=12)
    plt.xticks(range(len(df)), df["Épocas"].astype(str))
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de efecto épocas guardado en: {full_path}")

    plt.close()


# ==============================
# 11. EFECTO DEL LEARNING RATE SOBRE EL ACCURACY
# ==============================
def plot_learning_rate_effect(results_df, save_path=None):
    """
    Grafica el efecto del learning rate sobre el accuracy en prueba en escala log.
    Agrupa por learning rate y muestra promedio con banda de desviación estándar.
    Espera columnas: 'Learning rate', 'Test accuracy'.
    """
    df = results_df.groupby("Learning rate")["Test accuracy"].agg(["mean", "std"]).reset_index()
    df.columns = ["Learning rate", "Media", "Std"]
    df["Std"] = df["Std"].fillna(0)
    df = df.sort_values("Learning rate")

    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["Learning rate"],
        df["Media"],
        marker="s",
        linewidth=2,
        color="#EC4899",
        markersize=8,
        label="Accuracy promedio",
    )
    ax.fill_between(
        df["Learning rate"],
        df["Media"] - df["Std"],
        df["Media"] + df["Std"],
        alpha=0.2,
        color="#EC4899",
        label="± Desv. estándar",
    )

    for _, row in df.iterrows():
        ax.text(
            row["Learning rate"],
            row["Media"] + row["Std"] + 0.004,
            f"{row['Media']*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.1)
    ax.set_title("Efecto del Learning Rate sobre el Accuracy en Prueba", fontsize=13, fontweight="bold")
    ax.set_xlabel("Learning Rate (escala log)", fontsize=12)
    ax.set_ylabel("Accuracy en Test", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de efecto learning rate guardado en: {full_path}")

    plt.close()


# ==============================
# 12. RANKING DE EXPERIMENTOS POR ACCURACY
# ==============================
def plot_experiment_ranking(results_df, experiment_col="Experimento", save_path=None):
    """
    Grafica un ranking horizontal de experimentos ordenados de mayor a menor
    accuracy en el conjunto de prueba.
    Espera columnas: experiment_col, 'Test accuracy'.
    """
    df = results_df.copy()
    if experiment_col not in df.columns:
        df[experiment_col] = [f"Exp {i+1}" for i in range(len(df))]

    df = df.sort_values("Test accuracy", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(10, max(5, len(df) * 0.5)))
    ax = sns.barplot(
        data=df,
        x="Test accuracy",
        y=experiment_col,
        palette="rocket",
        hue=experiment_col,
        legend=False,
    )

    for i, row in df.reset_index(drop=True).iterrows():
        ax.text(
            row["Test accuracy"] + 0.002,
            i,
            f"{row['Test accuracy']*100:.2f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.xlim(0, 1.08)
    plt.title("Ranking de Experimentos por Accuracy en Prueba", fontsize=14, fontweight="bold")
    plt.xlabel("Accuracy en Test", fontsize=12)
    plt.ylabel("Experimento", fontsize=12)
    plt.tight_layout()

    if save_path:
        full_path = os.path.join(REPORTS_DIR, save_path)
        plt.savefig(full_path, dpi=300)
        print(f"Gráfico de ranking guardado en: {full_path}")

    plt.close()
