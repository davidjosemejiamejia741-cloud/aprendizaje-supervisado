#aprendizaje_supervisado\src\preprocessing.py
import cv2
import os
import numpy as np

IMG_SIZE = (128, 128)

# Lista de clases válidas para el dataset ASL
VALID_LABELS = set([chr(i) for i in range(ord('A'), ord('Z')+1)] + ['del', 'nothing', 'space'])

# ==============================
# PROCESAMIENTO DE IMÁGENES
# ==============================
def remove_magenta_border(img):
    """
    Quita el marco magenta/fucsia si existe y pinta de blanco cualquier
    resto magenta antes de redimensionar.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    magenta_mask = cv2.inRange(
        hsv,
        np.array([125, 40, 40], dtype=np.uint8),
        np.array([175, 255, 255], dtype=np.uint8),
    )

    cleaned = img.copy()
    cleaned[magenta_mask > 0] = (255, 255, 255)

    h, w = magenta_mask.shape
    row_ratio = (magenta_mask > 0).mean(axis=1)
    col_ratio = (magenta_mask > 0).mean(axis=0)
    min_border_ratio = 0.45

    top = 0
    while top < h and row_ratio[top] >= min_border_ratio:
        top += 1

    bottom = h
    while bottom > top and row_ratio[bottom - 1] >= min_border_ratio:
        bottom -= 1

    left = 0
    while left < w and col_ratio[left] >= min_border_ratio:
        left += 1

    right = w
    while right > left and col_ratio[right - 1] >= min_border_ratio:
        right -= 1

    if top >= bottom or left >= right:
        return cleaned

    return cleaned[top:bottom, left:right]


def remove_dark_frame(img, threshold=245, max_trim_ratio=0.08):
    """
    Recorta lÃ­neas de marco oscuras o grises pegadas a los bordes.
    No usa la caja completa de pÃ­xeles oscuros porque la mano tambiÃ©n cuenta
    como contenido; solo elimina filas/columnas externas que parecen marco.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_mask = gray < threshold
    h, w = gray.shape
    max_trim_y = max(1, int(h * max_trim_ratio))
    max_trim_x = max(1, int(w * max_trim_ratio))

    top = 0
    while top < max_trim_y and dark_mask[top].mean() > 0.70:
        top += 1

    bottom = h
    while bottom > h - max_trim_y and dark_mask[bottom - 1].mean() > 0.70:
        bottom -= 1

    left = 0
    while left < max_trim_x and dark_mask[:, left].mean() > 0.70:
        left += 1

    right = w
    while right > w - max_trim_x and dark_mask[:, right - 1].mean() > 0.70:
        right -= 1

    if top >= bottom or left >= right:
        return img

    return img[top:bottom, left:right]


def clear_outer_edges(img, margin_ratio=0.025, fill_color=(255, 255, 255)):
    """
    Limpia una franja pequeÃ±a del borde original para quitar lÃ­neas de marco.
    """
    cleaned = img.copy()
    h, w = cleaned.shape[:2]
    margin_y = max(1, int(h * margin_ratio))
    margin_x = max(1, int(w * margin_ratio))

    cleaned[:margin_y, :] = fill_color
    cleaned[-margin_y:, :] = fill_color
    cleaned[:, :margin_x] = fill_color
    cleaned[:, -margin_x:] = fill_color
    return cleaned


def resize_to_fixed_size(img, img_size=IMG_SIZE):
    """
    Redimensiona cualquier imagen al mismo tamaÃ±o final (ancho, alto).
    Usa una interpolaciÃ³n adecuada segÃºn si se reduce o se amplÃ­a.
    """
    target_w, target_h = img_size
    h, w = img.shape[:2]
    interpolation = cv2.INTER_AREA if w > target_w or h > target_h else cv2.INTER_LINEAR
    return cv2.resize(img, (target_w, target_h), interpolation=interpolation)


def pad_to_square(img, fill_color=(255, 255, 255)):
    """
    Coloca la imagen en un lienzo cuadrado para conservar la proporciÃ³n.
    Esto evita que letras horizontales como P se vean aplastadas al redimensionar.
    """
    h, w = img.shape[:2]
    size = max(h, w)

    if img.ndim == 2:
        square = np.full((size, size), fill_color[0], dtype=img.dtype)
    else:
        square = np.full((size, size, img.shape[2]), fill_color, dtype=img.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return square


def process_single_image(img_path, img_size=IMG_SIZE):
    """
    Carga, redimensiona y convierte una imagen a escala de grises.
    Retorna la imagen normalizada en rango [0, 1] como float32.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = remove_magenta_border(img)
    img = pad_to_square(img)
    
    # Redimensionado uniforme: todas quedan exactamente con el mismo tamaÃ±o
    # sin deformar la proporciÃ³n original de la mano.
    img = resize_to_fixed_size(img, img_size)
    
    # Conversión a escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalización a float32 para optimizar memoria
    img = (img / 255.0).astype(np.float32)
    return img

# ==============================
# CARGA DEL DATASET DE ENTRENAMIENTO
# ==============================
def load_dataset(path, img_size=IMG_SIZE):
    """
    Recorre las carpetas del dataset de entrenamiento, filtra clases válidas,
    procesa cada imagen y retorna los datos y etiquetas como arrays de numpy.
    """
    data = []
    labels = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"La ruta del dataset de entrenamiento no existe: {path}")

    # Obtener y ordenar las carpetas para procesar en orden alfabético
    labels_folders = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    for label in labels_folders:
        # Filtrar solo carpetas correspondientes a clases válidas
        if label not in VALID_LABELS:
            print(f"Ignorando carpeta no válida para clases: '{label}'")
            continue

        folder_path = os.path.join(path, label)
        print(f"Procesando clase '{label}'...")

        # Listar y procesar imágenes dentro de la carpeta
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            
            # Solo procesar archivos con extensiones de imagen comunes
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img = process_single_image(img_path, img_size)
            if img is not None:
                data.append(img)
                labels.append(label)

    return np.array(data, dtype=np.float32), np.array(labels)

# ==============================
# CARGA DEL DATASET DE TEST
# ==============================
def load_test_dataset(path, img_size=IMG_SIZE):
    """
    Carga y procesa las imágenes de prueba individuales que tienen el
    nombre con formato 'LABEL_test.jpg'.
    """
    data = []
    labels = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"La ruta del dataset de prueba no existe: {path}")

    # Procesar las imágenes de prueba
    for img_name in sorted(os.listdir(path)):
        img_path = os.path.join(path, img_name)
        
        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Extraer etiqueta del nombre del archivo (ej. 'A_test.jpg' -> 'A')
        label = img_name.split("_")[0]
        
        img = process_single_image(img_path, img_size)
        if img is not None:
            data.append(img)
            labels.append(label)

    return np.array(data, dtype=np.float32), np.array(labels)

# ==============================
# EJECUCIÓN DEL MÓDULO
# ==============================
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("--- INICIANDO PROCESAMIENTO DEL DATASET ---")
    
    # Rutas por defecto del proyecto
    train_raw_path = "data/train"
    test_raw_path = "data/test"

    print(f"Cargando y procesando TRAIN desde '{train_raw_path}'...")
    X_train, y_train = load_dataset(train_raw_path)

    print(f"\nCargando y procesando TEST desde '{test_raw_path}'...")
    X_test, y_test = load_test_dataset(test_raw_path)

    print("\n--- RESUMEN DE DIMENSIONES ---")
    print(f"X_train shape: {X_train.shape} | Tipo: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}   | Tipo: {X_test.dtype}")
    print(f"y_test shape: {y_test.shape}")

    # Guardar en formato numpy comprimido para ahorrar espacio/tiempo de I/O
    print("\nGuardando datos procesados en 'data/processed/'...")
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)

    print("¡Preprocesamiento completado con éxito!")
