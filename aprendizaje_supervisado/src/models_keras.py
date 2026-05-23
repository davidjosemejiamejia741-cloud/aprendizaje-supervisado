#aprendizaje_supervisado\src\models_keras.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# CARGA DE DATASETS (RGB Y OPTIMIZADO)
# ==============================

def load_training_datasets(path, img_size=(128, 128), batch_size=32, seed=42):
    """
    Carga el dataset de entrenamiento dividiéndolo automáticamente en 
    entrenamiento (80%) y validación (20%) de forma altamente eficiente 
    sin sobrecargar la memoria RAM usando tf.data.
    Retorna datasets con píxeles en rango [0, 255] como float32.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"La ruta del dataset de entrenamiento no existe: {path}")

    print(f"[~] Cargando imágenes desde '{path}'...")
    all_possible_classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space'
    ]
    valid_classes = sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d in all_possible_classes
    ])
    if not valid_classes:
        valid_classes = sorted(all_possible_classes)

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="both",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=valid_classes
    )

    class_names = train_ds.class_names

    # Pre-cargar datos para maximizar la velocidad de lectura en CPU (SIN cachear en RAM para evitar OOM)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def load_test_dataset_rgb(path, img_size=(128, 128), class_names=None):
    """
    Carga las imágenes de prueba en color (RGB), en rango [0, 255]
    y codifica las etiquetas a one-hot en base a la lista ordenada de clases.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"La ruta del dataset de prueba no existe: {path}")
    
    if class_names is None:
        raise ValueError("Es necesario suministrar la lista de class_names para la codificación.")

    data = []
    labels = []
    
    # Mapear nombre de clase a índice según el orden del entrenamiento
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    for img_name in sorted(os.listdir(path)):
        img_path = os.path.join(path, img_name)
        
        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Extraer etiqueta del nombre del archivo (ej. 'A_test.jpg' -> 'A')
        label = img_name.split("_")[0]
        if label not in class_to_idx:
            # Si la clase no está en el conjunto de entrenamiento, ignoramos
            continue

        # Leer la imagen en color (BGR)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Cambiar de BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar al tamaño del modelo
        img = cv2.resize(img, img_size)
        
        # Convertir a float32 pero mantener rango [0, 255]
        img = img.astype(np.float32)
        
        data.append(img)
        
        # Codificación One-Hot
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[class_to_idx[label]] = 1.0
        labels.append(one_hot)

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

# ==============================
# DEFINICIÓN DE MODELOS
# ==============================

def get_data_augmentation_layer():
    """
    Retorna una capa secuencial de aumento de datos para evitar sobreajuste.
    Omitimos volteo horizontal porque en el alfabeto ASL la lateralidad importa.
    """
    return models.Sequential([
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
    ])

def create_custom_cnn(img_size=(128, 128), num_classes=29):
    """
    Crea una red convolucional propia (CNN) con regularización Dropout.
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    
    # Capa de Data Augmentation
    x = get_data_augmentation_layer()(inputs)
    
    # Normalización de entrada [0, 255] -> [0, 1]
    x = layers.Rescaling(1.0 / 255.0)(x)
    
    # Bloque 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bloque 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bloque 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Clasificador
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="custom_cnn")
    return model

def create_vgg16_transfer_learning_model(img_size=(128, 128), num_classes=29):
    """
    Crea un modelo de Transfer Learning usando la arquitectura VGG16.
    """
    base_model = tf.keras.applications.VGG16(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar pesos de la base preentrenada
    base_model.trainable = False
    
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = get_data_augmentation_layer()(inputs)
    
    # Preprocesamiento específico para VGG16 (espera rango [0, 255] en BGR/RGB centrado)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="vgg16_transfer")
    return model

def create_resnet50_transfer_learning_model(img_size=(128, 128), num_classes=29):
    """
    Crea un modelo de Transfer Learning usando la arquitectura ResNet50.
    """
    base_model = tf.keras.applications.ResNet50(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )

    # Congelar pesos de la base preentrenada
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = get_data_augmentation_layer()(inputs)

    # Preprocesamiento específico para ResNet50
    x = tf.keras.applications.resnet50.preprocess_input(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="resnet50_transfer")
    return model

def create_transfer_learning_model(img_size=(128, 128), num_classes=29):
    """
    Crea un modelo de Transfer Learning usando la arquitectura MobileNetV2 (Más ligera y eficiente para CPU).
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar pesos de la base preentrenada
    base_model.trainable = False
    
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = get_data_augmentation_layer()(inputs)
    
    # Preprocesamiento específico para MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer")
    return model
