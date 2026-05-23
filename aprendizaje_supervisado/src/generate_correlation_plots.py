#aprendizaje_supervisado\src\generate_correlation_plots.py
import os

import numpy as np
import pandas as pd

from visualization import plot_experiment_correlation, plot_image_feature_correlation


RESULTS_PATH = "resultados_deep.csv"
X_TRAIN_PATH = "data/processed/X_train.npy"


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)

    print("[~] Cargando imágenes preprocesadas para matriz de correlación...")
    X_train = np.load(X_TRAIN_PATH)
    plot_image_feature_correlation(
        X_train,
        save_path="correlacion_caracteristicas_imagenes.png",
    )

    print("[~] Cargando resultados de experimentos para matriz de correlación...")
    results_df = pd.read_csv(RESULTS_PATH, sep=";")
    plot_experiment_correlation(
        results_df,
        save_path="correlacion_hiperparametros_metricas.png",
    )

    print("[+] Gráficas de correlación generadas correctamente.")
