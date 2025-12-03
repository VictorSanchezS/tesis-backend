import cv2
import numpy as np

TARGET_SIZE = (224, 224)

# Crear CLAHE una sola vez y reutilizar
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def aplicar_clahe(imagen_rgb: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE solo en el canal de luminancia (Y) para mejorar contraste.
    """
    ycrcb = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0]
    y = _CLAHE.apply(y)
    ycrcb[:, :, 0] = y
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def preprocess_nail_crop_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocesa el recorte BGR:
    - Convierte a RGB
    - Aplica CLAHE
    - Redimensiona a 224x224
    - Normaliza a [0,1] float32
    """
    # BGR -> RGB
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # CLAHE para mejorar contraste
    rgb = aplicar_clahe(rgb)

    # Redimensionar
    resized = cv2.resize(rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Normalizar a [0,1] en float32 (ligero)
    processed = resized.astype(np.float32) / 255.0

    return processed
