#aprendizaje_supervisado\src\feature_engineering.py
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from visualization import show_samples, plot_class_distribution, plot_pca_variance

# ==============================
# FILTRO DE CARACTERÍSTICAS CUASI-CONSTANTES
# ==============================
def remove_quasi_constant_features(X_train, X_test, threshold=0.01):
    """
    Elimina características (píxeles) cuasi-constantes que no aportan variabilidad.
    Esto implementa el filtro 'quasi constant' sugerido por el docente.
    """
    import gc
    print(f"Aplicando filtro VarianceThreshold (umbral de varianza = {threshold})...")
    selector = VarianceThreshold(threshold=threshold)
    X_train_filtered = selector.fit_transform(X_train)
    X_test_filtered = selector.transform(X_test)
    print(f"Características antes: {X_train.shape[1]} | Características después: {X_train_filtered.shape[1]}")
    gc.collect()
    return X_train_filtered, X_test_filtered, selector


# ==============================
# ETAPA 3: VECTORIZACIÓN
# ==============================
def vectorize_images(X):
    """
    Aplastará las imágenes de 2D (N, H, W) a 1D (N, H * W).
    """
    N = X.shape[0]
    return X.reshape(N, -1)

# ==============================
# ETAPA 4: PCA (REDUCCIÓN DE DIMENSIONES)
# ==============================
def apply_pca(X_train, X_test, n_components=100):
    """
    Entrena PCA en el conjunto de entrenamiento y transforma ambos conjuntos.
    Retorna los datos transformados y el objeto PCA entrenado.
    """
    import gc
    if n_components is None:
        print("Entrenando PCA con todos los componentes posibles (máximo)...")
        pca = PCA(n_components=None)
    else:
        print(f"Entrenando PCA con {n_components} componentes...")
        pca = PCA(n_components=n_components)
    
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Varianza explicada acumulada: {explained_variance:.4f}")
    gc.collect()
    return X_train_pca, X_test_pca, pca

# ==============================
# EJECUCIÓN DEL MÓDULO (STANDALONE)
# ==============================
if __name__ == "__main__":
    print("--- INICIANDO INGENIERÍA DE CARACTERÍSTICAS ---")
    
    # Cargar datos preprocesados
    try:
        print("Cargando datos preprocesados...")
        X_train = np.load("data/processed/X_train.npy")
        X_test = np.load("data/processed/X_test.npy")
        y_train = np.load("data/processed/y_train.npy")
    except FileNotFoundError as e:
        print(f"Error al cargar archivos preprocesados: {e}")
        print("Por favor, ejecuta primero 'src/preprocessing.py' para generar estos archivos.")
        exit(1)

    ENABLE_VISUALS = True

    # Visualizaciones iniciales
    if ENABLE_VISUALS:
        print("Mostrando muestras y distribución de clases...")
        show_samples(X_train, y_train)
        plot_class_distribution(y_train)

    # Etapa 3: Vectorización
    print("Vectorizando imágenes...")
    X_train_flat = vectorize_images(X_train)
    X_test_flat = vectorize_images(X_test)
    
    print(f"Train plano: {X_train_flat.shape} | Test plano: {X_test_flat.shape}")

    # Guardar datos vectorizados
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train_flat.npy", X_train_flat)
    np.save("data/processed/X_test_flat.npy", X_test_flat)

    # Etapa 3.5: Filtro Cuasi-Constante
    print("Aplicando filtro de características cuasi-constantes...")
    total_initial_features = X_train_flat.shape[1]
    X_train_filtered, X_test_filtered, selector = remove_quasi_constant_features(X_train_flat, X_test_flat, threshold=0.01)

    # Etapa 4: PCA
    print("Aplicando PCA para reducción de dimensionalidad...")
    X_train_pca, X_test_pca, pca = apply_pca(X_train_filtered, X_test_filtered, n_components=100)

    print(f"Train PCA: {X_train_pca.shape} | Test PCA: {X_test_pca.shape}")

    # Visualización de varianza explicada por PCA
    if ENABLE_VISUALS:
        plot_pca_variance(pca, total_initial_features=total_initial_features)

    # Guardar resultados reducidos
    os.makedirs("data/reduced", exist_ok=True)
    np.save("data/reduced/X_train_pca.npy", X_train_pca)
    np.save("data/reduced/X_test_pca.npy", X_test_pca)

    print("¡Ingeniería de características completada y datos guardados!")