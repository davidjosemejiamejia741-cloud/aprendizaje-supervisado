#aprendizaje_supervisado\src\main_deep.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf

from models_keras import (
    load_training_datasets,
    load_test_dataset_rgb,
    create_custom_cnn,
    create_resnet50_transfer_learning_model,
    create_vgg16_transfer_learning_model,
    create_transfer_learning_model
)

# ==============================
# CONFIGURACIÓN
# ==============================
# Opciones de MODEL_TYPE: 'transfer_learning' (MobileNetV2), 'vgg16', 'resnet50' o 'custom_cnn'
DEFAULT_MODEL_TYPE = 'resnet50'

IMG_SIZE = (128, 128)      # 128x128 es ideal para CPU (224x224 puede ralentizar mucho el proceso)
BATCH_SIZE = 32

EPOCHS = 8           # Cambiar a 10 o 15 para mayor precisión si se dispone de tiempo
ENABLE_VISUALS = True

# Configuración automática del dataset según el entorno (Colab, Kaggle o Local)
if os.path.exists("/content/data/asl_alphabet_train/asl_alphabet_train"):
    # Google Colab (almacenamiento local temporal rápido)
    TRAIN_DIR = "/content/data/asl_alphabet_train/asl_alphabet_train"
    TEST_DIR = "/content/data/asl_alphabet_test/asl_alphabet_test" if os.path.exists("/content/data/asl_alphabet_test/asl_alphabet_test") else "/content/data/asl_alphabet_test"
    print("[INFO] Ejecutando en Google Colab con dataset local temporal.")
elif os.path.exists("/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"):
    # Kaggle Kernels
    TRAIN_DIR = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
    TEST_DIR = "/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test" if os.path.exists("/kaggle/input/asl-alphabet/asl_alphabet_test/asl_alphabet_test") else "/kaggle/input/asl-alphabet/asl_alphabet_test"
    print("[INFO] Ejecutando en Kaggle con dataset montado de Kaggle.")
else:
    # Ejecución local en PC
    TRAIN_DIR = "data/train"
    TEST_DIR = "data/test"
    print("[INFO] Ejecutando en entorno local.")

# ==============================
# ORQUESTACIÓN PRINCIPAL (MAIN DEEP)
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo deep learning para ASL.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_TYPE,
        choices=["transfer_learning", "vgg16", "resnet50", "custom_cnn"],
        help="Modelo a entrenar: transfer_learning, vgg16, resnet50 o custom_cnn."
    )
    args = parser.parse_args()
    MODEL_TYPE = args.model

    # Crear directorios si no existen
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("==================================================")
    print(f"   INICIANDO DEEP LEARNING ({MODEL_TYPE.upper()})    ")
    print("==================================================\n")

    # 1. Cargar conjuntos de datos
    train_ds, val_ds, class_names = load_training_datasets(
        TRAIN_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Obtener nombres de las clases desde el dataset cargado
    num_classes = len(class_names)
    print(f"[+] Clases encontradas ({num_classes}): {class_names}")

    # Guardar las clases entrenadas para la inferencia futura
    import json
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)
    print("[+] Archivo 'models/class_names.json' guardado.")

    # Cargar conjunto de prueba independiente
    X_test, y_test = load_test_dataset_rgb(
        TEST_DIR,
        img_size=IMG_SIZE,
        class_names=class_names
    )
    print(f"[+] Dataset de prueba cargado. Tamaño: {X_test.shape}")

    # 2. Inicializar el Modelo según la selección
    if MODEL_TYPE == 'transfer_learning':
        print("[~] Cargando modelo preentrenado MobileNetV2 para Transfer Learning (Rápido/CPU)...")
        model = create_transfer_learning_model(img_size=IMG_SIZE, num_classes=num_classes)
    elif MODEL_TYPE == 'vgg16':
        print("[~] Cargando modelo preentrenado VGG16 para Transfer Learning (Pesado)...")
        model = create_vgg16_transfer_learning_model(img_size=IMG_SIZE, num_classes=num_classes)
    elif MODEL_TYPE == 'resnet50':
        print("[~] Cargando modelo preentrenado ResNet50 para Transfer Learning...")
        model = create_resnet50_transfer_learning_model(img_size=IMG_SIZE, num_classes=num_classes)
    elif MODEL_TYPE == 'custom_cnn':
        print("[~] Inicializando arquitectura CNN propia...")
        model = create_custom_cnn(img_size=IMG_SIZE, num_classes=num_classes)
    else:
        raise ValueError(f"MODEL_TYPE '{MODEL_TYPE}' no reconocido. Usa 'transfer_learning', 'vgg16', 'resnet50' o 'custom_cnn'.")

    # 3. Compilación del modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 4. Entrenamiento
    print(f"\n[~] Entrenando el modelo por {EPOCHS} épocas en CPU...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    print("[+] ¡Entrenamiento finalizado exitosamente!\n")

    # 5. Guardar el modelo entrenado
    # Formato nativo recomendado por Keras. Tambien se deja una copia .h5
    # para compatibilidad con ejecuciones/reportes anteriores del proyecto.
    model_save_path = f"models/keras_{MODEL_TYPE}_model.keras"
    legacy_model_save_path = f"models/keras_{MODEL_TYPE}_model.h5"
    model.save(model_save_path)
    model.save(legacy_model_save_path)
    print(f"[+] Modelo guardado en '{model_save_path}'")
    print(f"[+] Copia compatible guardada en '{legacy_model_save_path}'")

    # 6. Evaluación en el conjunto de prueba
    print("[~] Evaluando modelo en el conjunto de prueba...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    print(f"\nPrecisión General en Test (Accuracy): {accuracy:.4f}")
    print("\nReporte de Clasificación en Test:")
    print(class_report)

    # Guardar reporte de evaluación
    report_path = f"reports/evaluation_report_deep8_{MODEL_TYPE}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {MODEL_TYPE.upper()}\n")
        f.write(f"Precisión General en Test (Accuracy): {accuracy:.4f}\n\n")
        f.write("Reporte de Clasificación:\n")
        f.write(class_report)
        f.write("\nMatriz de Confusión:\n")
        f.write(np.array2string(conf_matrix))
    print(f"[+] Reporte guardado en '{report_path}'")

    if ENABLE_VISUALS:
        print("[~] Generando matriz de confusión del modelo deep...")
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"Matriz de Confusión - {MODEL_TYPE.upper()}", fontweight="bold")
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase verdadera")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        threshold = conf_matrix.max() / 2
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                value = conf_matrix[i, j]
                if value > 0:
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        color="white" if value > threshold else "black",
                        fontsize=6,
                    )

        fig.tight_layout()
        confusion_plot_path = f"reports/confusion_matrix_deep_{MODEL_TYPE}.png"
        fig.savefig(confusion_plot_path, dpi=300)
        plt.close(fig)
        print(f"[+] Matriz de confusión guardada en '{confusion_plot_path}'")

    # 7. Graficar curvas de aprendizaje
    if ENABLE_VISUALS:
        print("[~] Generando curvas de aprendizaje...")
        plt.figure(figsize=(12, 4))

        # Gráfico de Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Exactitud (Accuracy)')
        plt.xlabel('Época')
        plt.ylabel('Exactitud')
        plt.legend()
        plt.grid(True)

        # Gráfico de Pérdida
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Pérdida (Loss)')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = f"reports/learning_curves_8{MODEL_TYPE}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"[+] Gráfico guardado en '{plot_path}'")
