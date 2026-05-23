#aprendizaje_supervisado\src\predict.py
import os
import sys
import cv2
import numpy as np
import joblib

# Añadir la carpeta src al path de python para permitir importaciones robustas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import process_single_image

# Lista ordenada de las clases ASL correspondientes al orden del entrenamiento
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]
for p in ["models/class_names.json", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "class_names.json")]:
    if os.path.exists(p):
        try:
            import json
            with open(p, "r", encoding="utf-8") as f:
                CLASS_NAMES = json.load(f)
            break
        except Exception as e:
            pass


# ==============================
# CARGAR MODELOS CON MANEJO DE ERRORES
# ==============================
def load_classic_model():
    model_path = "models/modelo_sign_language.pkl"
    pca_path = "models/pca_transform.pkl"
    selector_path = "models/variance_selector.pkl"
    encoder_path = "models/label_encoder.pkl"

    if not os.path.exists(model_path) or not os.path.exists(pca_path):
        return None, None, None, None
    try:
        model = joblib.load(model_path)
        pca = joblib.load(pca_path)
        selector = joblib.load(selector_path) if os.path.exists(selector_path) else None
        encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
        return model, pca, selector, encoder
    except Exception as e:
        print(f"[!] Error al cargar modelo clásico: {e}")
        return None, None, None, None

def load_keras_model(model_type='vgg16'):
    native_model_path = f"models/keras_{model_type}_model.keras"
    legacy_model_path = f"models/keras_{model_type}_model.h5"
    try:
        import tensorflow as tf

        if os.path.exists(native_model_path):
            return tf.keras.models.load_model(native_model_path)

        if not os.path.exists(legacy_model_path):
            return None

        # Los modelos .h5 antiguos pueden fallar al cargar por capas internas
        # serializadas como GetItem. En ese caso reconstruimos la arquitectura
        # y cargamos los pesos desde el mismo archivo .h5.
        from models_keras import (
            create_custom_cnn,
            create_transfer_learning_model,
            create_vgg16_transfer_learning_model,
        )

        num_classes = len(CLASS_NAMES)
        if model_type == 'vgg16':
            model = create_vgg16_transfer_learning_model(img_size=(128, 128), num_classes=num_classes)
        elif model_type == 'transfer_learning':
            model = create_transfer_learning_model(img_size=(128, 128), num_classes=num_classes)
        elif model_type == 'custom_cnn':
            model = create_custom_cnn(img_size=(128, 128), num_classes=num_classes)
        else:
            raise ValueError(f"Tipo de modelo Keras no soportado: {model_type}")

        model.load_weights(legacy_model_path)
        return model
    except Exception as e:
        print(f"[!] Error al cargar modelo Keras ({model_type}): {e}")
        return None

# ==============================
# PREDICCIÓN CON MODELO CLÁSICO (PCA + LR)
# ==============================
def predict_classic(img_path, model, pca, selector=None, encoder=None):
    # Carga imagen en escala de grises y rango [0, 1]
    img = process_single_image(img_path)
    if img is None:
        print(f"[!] Error: No se pudo cargar o procesar la imagen clásica en '{img_path}'.")
        return

    # Aplanar y agregar dimensión de lote
    img_flat = img.reshape(1, -1)
    
    # Aplicar filtro de varianza si está disponible
    if selector is not None:
        img_flat = selector.transform(img_flat)
        
    img_pca = pca.transform(img_flat)

    # Predicción y Probabilidades
    prediction = model.predict(img_pca)[0]
    
    # Decodificar predicción si hay encoder
    if encoder is not None:
        prediction_str = encoder.inverse_transform([prediction])[0]
    else:
        prediction_str = prediction
    
    print("\n========================================")
    print(f"  Inferencia Clásica (PCA + LR): {os.path.basename(img_path)}")
    print("----------------------------------------")
    print(f"Clase Predicha: {prediction_str}")
    
    try:
        probabilities = model.predict_proba(img_pca)[0]
        if encoder is not None:
            classes = list(encoder.inverse_transform(model.classes_))
        else:
            classes = list(model.classes_)
        class_idx = classes.index(prediction_str)
        confidence = probabilities[class_idx]
        print(f"Confianza:      {confidence * 100:.2f}%")
        
        top_indices = np.argsort(probabilities)[::-1][:3]
        print("\nTop 3 predicciones más probables:")
        for idx in top_indices:
            print(f"  - {classes[idx]}: {probabilities[idx]*100:.2f}%")
    except AttributeError:
        pass
    print("========================================\n")

# ==============================
# PREDICCIÓN CON MODELO KERAS (DEEP LEARNING / VGG16 / CNN)
# ==============================
def predict_keras(img_path, model, img_size=(128, 128)):
    # VGG16 y la CNN propia esperan imágenes RGB en rango [0, 255]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Error: No se pudo cargar la imagen para Keras en '{img_path}'.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    
    # Agregar dimensión de lote (1, H, W, C)
    img_batch = np.expand_dims(img, axis=0).astype(np.float32)

    # Inferencia
    probabilities = model.predict(img_batch)[0]
    pred_idx = np.argmax(probabilities)
    prediction = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx]

    print("\n========================================")
    print(f"  Inferencia Keras (Deep Learning): {os.path.basename(img_path)}")
    print("----------------------------------------")
    print(f"Clase Predicha: {prediction}")
    print(f"Confianza:      {confidence * 100:.2f}%")
    
    top_indices = np.argsort(probabilities)[::-1][:3]
    print("\nTop 3 predicciones más probables:")
    for idx in top_indices:
        print(f"  - {CLASS_NAMES[idx]}: {probabilities[idx]*100:.2f}%")
    print("========================================\n")

# ==============================
# EJECUCIÓN PRINCIPAL
# ==============================
if __name__ == "__main__":
    # Buscar imagen de muestra
    sample_test_dir = "data/test"
    img_path = None
    if os.path.exists(sample_test_dir) and len(os.listdir(sample_test_dir)) > 0:
        first_img = [f for f in os.listdir(sample_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0]
        img_path = os.path.join(sample_test_dir, first_img)
    else:
        print("[!] No se encontró la carpeta 'data/test'. Inferencia no ejecutada.")
        sys.exit(0)

    print(f"Imagen para inferencia de prueba seleccionada: {img_path}\n")

    # Intentar cargar y ejecutar modelo de Keras (VGG16)
    keras_model = load_keras_model(model_type='vgg16')
    if keras_model is not None:
        predict_keras(img_path, keras_model)
    else:
        print("[~] Modelo Keras (vgg16) no encontrado en 'models/'. Buscando modelo clásico...")
        
        # Si no hay Keras, intentar cargar clásico
        model, pca, selector, encoder = load_classic_model()
        if model is not None and pca is not None:
            predict_classic(img_path, model, pca, selector, encoder)
        else:
            print("[!] ERROR: No hay ningún modelo entrenado disponible en 'models/'.")
            print("Ejecuta 'python src/main.py' o 'python src/main_deep.py' primero.")
