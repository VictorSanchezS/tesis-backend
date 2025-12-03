import cv2
import numpy as np

IMG_SIZE = 224

def aplicar_clahe(imagen):
    """
    Aplica CLAHE en el canal Y del espacio YCrCb.
    Mismo preprocesamiento usado durante el entrenamiento en Colab.
    """
    ycrcb = cv2.cvtColor(imagen, cv2.COLOR_RGB2YCrCb)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Aplicar solo a canal Y
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def preprocess_image(image_bytes: bytes):
    """
    Pipeline EXACTO del entrenamiento:
    - cv2.imdecode (bytes → BGR)
    - BGR → RGB
    - resize 224x224
    - aplicar CLAHE
    - normalizar [0,1]
    """
    # Leer imagen desde bytes usando OpenCV
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Error al decodificar la imagen")

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionar como en entrenamiento
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Aplicar CLAHE
    img = aplicar_clahe(img)

    # Normalizar a [0, 1]
    img = img.astype("float32") / 255.0

    # Expandir batch
    img = np.expand_dims(img, axis=0)

    return img


# ===========================
# NUEVO: para crops de uñas
# ===========================

def preprocess_nail_crop_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocesa un recorte de uña en BGR (de cv2) usando
    EXACTAMENTE el mismo pipeline que el entrenamiento:

    - BGR → RGB
    - resize 224x224
    - CLAHE en YCrCb
    - normalizar [0,1]

    Devuelve un array (224, 224, 3) float32.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Crop vacío o inválido")

    # BGR → RGB
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionar
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # CLAHE
    img_rgb = aplicar_clahe(img_rgb)

    # Normalizar [0, 1]
    img_rgb = img_rgb.astype("float32") / 255.0

    return img_rgb
