#aprendizaje_supervisado\src\main.py
import numpy as np
import os
import joblib
import time
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Importaciones de nuestros módulos propios
from preprocessing import load_dataset, load_test_dataset
from feature_engineering import vectorize_images, apply_pca, remove_quasi_constant_features
from visualization import show_samples, plot_class_distribution, plot_data_split_distribution, plot_pca_variance, plot_2d, plot_confusion_matrix

# ==============================
# CONFIGURACIÓN
# ==============================
IMG_SIZE = (128, 128)
ENABLE_VISUALS = True      # Si es True, mostrará y guardará gráficos en la carpeta 'reports/'
REPROCESS_DATA = False      # Si es True, fuerza el preprocesamiento de imágenes crudas
N_COMPONENTS =      950   # Número de componentes para PCA (valor óptimo sugerido)
SAMPLES_PER_CLASS = 1400    # Submuestreo estratificado (1000 train + 200 val + 200 test por clase)

# ==============================
# ORQUESTACIÓN PRINCIPAL (MAIN)
# ==============================
if __name__ == "__main__":
    # Crear directorios del proyecto si no existen
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/reduced", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("==================================================")
    print("      INICIANDO PIPELINE DE APRENDIZAJE ASL       ")
    print("==================================================\n")

    # ----------------------------------------------
    # 1. CARGA Y PREPROCESAMIENTO DEL DATASET
    # ----------------------------------------------
    X_train_path = "data/processed/X_train.npy"
    y_train_path = "data/processed/y_train.npy"
    X_test_path = "data/processed/X_test.npy"
    y_test_path = "data/processed/y_test.npy"

    preprocessed_exists = all(os.path.exists(p) for p in [X_train_path, y_train_path, X_test_path, y_test_path])

    if REPROCESS_DATA or not preprocessed_exists:
        print("[!] No se encontraron archivos procesados o se forzó el procesamiento.")
        print("Iniciando procesamiento de imágenes crudas...")

        # Cargar y preprocesar (escala de grises + mismo tamaño)
        X_train, y_train = load_dataset("data/train", img_size=IMG_SIZE)
        X_test, y_test = load_test_dataset("data/test", img_size=IMG_SIZE)

        # Guardar en disco
        print("\nGuardando imágenes preprocesadas en NumPy...")
        np.save(X_train_path, X_train)
        np.save(y_train_path, y_train)
        np.save(X_test_path, X_test)
        np.save(y_test_path, y_test)
    else:
        print("[+] Cargando imágenes preprocesadas existentes desde 'data/processed/'...")
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

    print(f"Dimensiones de entrenamiento original: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Dimensiones de prueba original: X_test={X_test.shape}, y_test={y_test.shape}\n")
    y_total_original = np.concatenate([y_train, y_test])

    # ----------------------------------------------
    # 1.5. SUBMUESTREO ESTRATIFICADO (PARA EVITAR ERRORES DE MEMORIA Y COMPENSAR RENDIMIENTO)
    # ----------------------------------------------
    if SAMPLES_PER_CLASS is not None:
        print(f"[~] Aplicando submuestreo estratificado (máx. {SAMPLES_PER_CLASS} muestras por clase)...")
        unique_classes = np.unique(y_train)
        indices_to_keep = []

        for cls in unique_classes:
            cls_indices = np.where(y_train == cls)[0]
            if len(cls_indices) > SAMPLES_PER_CLASS:
                rng = np.random.default_rng(42)
                sampled_indices = rng.choice(cls_indices, size=SAMPLES_PER_CLASS, replace=False)
            else:
                sampled_indices = cls_indices
            indices_to_keep.extend(sampled_indices)

        indices_to_keep = sorted(indices_to_keep)
        X_train = X_train[indices_to_keep]
        y_train = y_train[indices_to_keep]
        print(f"[+] Submuestreo completado. Nuevo tamaño de entrenamiento: {X_train.shape}")
        gc.collect()

    # ----------------------------------------------
    # 1.7. PARTICIÓN DE DATOS (TRAIN / VALIDATION / TEST)
    # ----------------------------------------------
    print("[~] Realizando partición de datos en entrenamiento, validación y prueba de forma estratificada...")
    # Aunque el dataset incluye imágenes de prueba independientes, para los modelos clásicos
    # se realiza una partición estratificada del conjunto principal para garantizar una
    # evaluación más balanceada por clase.
    # Primera división: separar el 14.2857% (200 de 1400) para prueba (Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_train, y_train, test_size=(1/7), stratify=y_train, random_state=42
    )
    # Segunda división: del 85.7143% restante, separar un 16.6667% (200 de 1200) para validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(1/6), stratify=y_train_val, random_state=42
    )
    print(f"[+] Partición completada (~1000 train / ~200 val / ~200 test por clase):")
    print(f"    Train: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"    Validation: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"    Test: X_test={X_test.shape}, y_test={y_test.shape}\n")

    # ----------------------------------------------
    # 2. ANÁLISIS DE BALANCE DE CLASES Y MUESTRAS
    # ----------------------------------------------
    if ENABLE_VISUALS:
        print("[~] Generando visualizaciones iniciales y reporte de balance...")
        show_samples(X_train, y_train, num_samples=5, save_path="muestras_asl.png")
        plot_class_distribution(
            y_total_original,
            save_path="distribucion_clases.png",
            title="Distribución total de imágenes por clase en el Dataset ASL"
        )
        plot_data_split_distribution(y_train, y_val, y_test, save_path="distribucion_particion.png")

    # ----------------------------------------------
    # 3. VECTORIZACIÓN DE IMÁGENES
    # ----------------------------------------------
    print("[~] Vectorizando imágenes (aplanado de 2D a 1D)...")
    X_train_flat = vectorize_images(X_train)
    X_val_flat = vectorize_images(X_val)
    X_test_flat = vectorize_images(X_test)

    print(f"Dimensiones vectorizadas: Train={X_train_flat.shape}, Validation={X_val_flat.shape}, Test={X_test_flat.shape}\n")

    # Registrar el total de características/componentes iniciales antes de los filtros
    total_initial_features = X_train_flat.shape[1]

    # ----------------------------------------------
    # 3.5. FILTRO DE CARACTERÍSTICAS CUASI-CONSTANTES
    # ----------------------------------------------
    print("[~] Aplicando filtro de características cuasi-constantes...")
    X_train_filtered, X_test_filtered, selector = remove_quasi_constant_features(
        X_train_flat,
        X_test_flat,
        threshold=0.01
    )
    X_val_filtered = selector.transform(X_val_flat)

    # Liberar memoria de datos vectorizados y crudos que ya no se necesitan
    del X_train, X_val, X_test, X_train_flat, X_val_flat, X_test_flat
    gc.collect()

    # ----------------------------------------------
    # 4. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
    # ----------------------------------------------
    print("[~] Iniciando PCA...")
    X_train_pca, X_test_pca, pca = apply_pca(
        X_train_filtered,
        X_test_filtered,
        n_components=N_COMPONENTS
    )
    X_val_pca = pca.transform(X_val_filtered)

    print(f"Dimensiones reducidas por PCA: Train={X_train_pca.shape}, Validation={X_val_pca.shape}, Test={X_test_pca.shape}\n")

    del X_train_filtered, X_val_filtered, X_test_filtered
    gc.collect()

    if ENABLE_VISUALS:
        plot_pca_variance(pca, total_initial_features=total_initial_features, save_path="varianza_pca.png")
        plot_2d(X_train_pca[:, :2], y_train, title="Proyección 2D del Dataset ASL (PCA)", save_path="proyeccion_2d_pca.png")

    # Guardar los datos reducidos
    np.save("data/reduced/X_train_pca.npy", X_train_pca)
    np.save("data/reduced/X_val_pca.npy", X_val_pca)
    np.save("data/reduced/X_test_pca.npy", X_test_pca)

    # ----------------------------------------------
    # 5. ENTRENAMIENTO DE LOS MODELOS CLASIFICADORES (GridSearchCV)
    # ----------------------------------------------
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    classifiers = {
        "Regresión Logística": LogisticRegression(random_state=42),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
        "Red Neuronal MLP": MLPClassifier(early_stopping=True, random_state=42)
    }

    param_grids = {
        "Regresión Logística": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"],
            "max_iter": [1000]
        },
        "Árbol de Decisión": {
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [10, 15, None],
            "min_samples_split": [2, 5]
        },
        "Red Neuronal MLP": {
            "hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.001, 0.01]
        }
    }

    results = {}
    best_acc = 0.0
    best_model_name = ""
    best_model = None

    print("--- INICIANDO BÚSQUEDA DE HIPERPARÁMETROS (GridSearchCV, cv=3) ---")
    for name, clf in classifiers.items():
        print(f"\n[~] Buscando mejores hiperparámetros para {name}...")
        start_time = time.time()

        grid = GridSearchCV(
            estimator=clf,
            param_grid=param_grids[name],
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train_pca, y_train_encoded)
        train_time = time.time() - start_time

        best_clf = grid.best_estimator_
        print(f"[+] {name} - GridSearchCV completado en {train_time:.2f} segundos.")
        print(f"    Mejores hiperparámetros: {grid.best_params_}")
        print(f"    Mejor accuracy (CV, cv=3): {grid.best_score_:.4f}")

        # Evaluar en Validación
        y_val_pred_encoded = best_clf.predict(X_val_pca)
        y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)
        val_acc = accuracy_score(y_val, y_val_pred)

        # Evaluar en Test
        y_pred_encoded = best_clf.predict(X_test_pca)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        test_acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "val_accuracy": val_acc,
            "accuracy": test_acc,
            "train_time": train_time,
            "predictions": y_pred,
            "clf": best_clf,
            "best_params": grid.best_params_,
            "cv_score": grid.best_score_
        }
        print(f"    Precisión (Accuracy) en Validación: {val_acc:.4f}")
        print(f"    Precisión (Accuracy) en Test: {test_acc:.4f}")

        # Se selecciona el mejor modelo según su precisión en validación
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_name = name
            best_model = best_clf

    print("\n==================================================")
    print("        RESULTADOS COMPARATIVOS CLÁSICOS          ")
    print("==================================================")
    print(f"{'Modelo/Algoritmo':<25} | {'CV Score':<10} | {'Exactitud (Val)':<18} | {'Exactitud (Test)':<18} | {'Tiempo Train (s)':<16}")
    print("-" * 97)
    for name, info in results.items():
        print(f"{name:<25} | {info['cv_score']*100:<9.2f}% | {info['val_accuracy']*100:<17.2f}% | {info['accuracy']*100:<17.2f}% | {info['train_time']:<16.2f}")
    print("==================================================\n")

    # ----------------------------------------------
    # 6. EVALUACIÓN Y MATRICES DE CONFUSIÓN
    # ----------------------------------------------
    print(f"[~] Evaluando a detalle el mejor modelo clásico ({best_model_name})...")
    y_pred_best = results[best_model_name]["predictions"]
    conf_matrix = confusion_matrix(y_test, y_pred_best)
    class_report = classification_report(y_test, y_pred_best)

    if ENABLE_VISUALS:
        class_names = sorted(list(set(y_test)))
        confusion_file_names = {
            "Regresión Logística": "confusion_matrix_regresion_logistica.png",
            "Árbol de Decisión": "confusion_matrix_arbol_decision.png",
            "Random Forest": "confusion_matrix_random_forest.png",
            "Red Neuronal MLP": "confusion_matrix_mlp.png",
        }

        for name, info in results.items():
            plot_confusion_matrix(
                y_test,
                info["predictions"],
                class_names=class_names,
                title=f"Matriz de Confusión - {name} (PCA)",
                save_path=confusion_file_names[name]
            )

        # Se mantiene una copia con el nombre usado previamente para el mejor modelo clásico.
        plot_confusion_matrix(
            y_test,
            y_pred_best,
            class_names=class_names,
            title=f"Matriz de Confusión - {best_model_name} (PCA)",
            save_path="confusion_matrix_classic.png"
        )

    # Guardar reporte de clasificación en reports para todos los modelos
    with open("reports/evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("      REPORTE COMPARATIVO DE MODELOS CLÁSICOS     \n")
        f.write("==================================================\n\n")
        f.write(f"{'Modelo':<25} | {'CV Score':<10} | {'Exactitud (Val)':<18} | {'Exactitud (Test)':<18} | {'Tiempo (s)':<10}\n")
        f.write("-" * 90 + "\n")
        for name, info in results.items():
            f.write(f"{name:<25} | {info['cv_score']*100:<9.2f}% | {info['val_accuracy']*100:<17.2f}% | {info['accuracy']*100:<17.2f}% | {info['train_time']:<10.2f}\n")
        f.write("\nMejores hiperparámetros por modelo:\n")
        for name, info in results.items():
            f.write(f"  {name}: {info['best_params']}\n")
        f.write("\n" + "="*70 + "\n\n")
        f.write(f"Detalle de clasificación para el Mejor Modelo Clásico ({best_model_name}):\n\n")
        f.write("Reporte de Clasificación (Evaluado en Test):\n")
        f.write(class_report)
        f.write("\nMatriz de Confusión:\n")
        f.write(np.array2string(conf_matrix))

    print("[+] Reporte de evaluación guardado en 'reports/evaluation_report.txt'")

    # ----------------------------------------------
    # 7. GUARDAR EL MEJOR MODELO, SELECTOR Y PCA TRANSFORMER
    # ----------------------------------------------
    print(f"\n[~] Guardando el mejor modelo clásico ({best_model_name}), selector, PCA y encoder...")
    joblib.dump(best_model, "models/modelo_sign_language.pkl")
    joblib.dump(pca, "models/pca_transform.pkl")
    joblib.dump(selector, "models/variance_selector.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    print("[+] Archivos guardados correctamente en 'models/'")
    print("\n==================================================")
    print("             PIPELINE FINALIZADO                  ")
    print("==================================================")
