import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import base64


# =====================================================================
# 1. Definiciones de backbone (MobileNetV2) y última capa conv
# =====================================================================

# Estos índices son los mismos que usaste en el Colab
BACKBONE_START = 1
BACKBONE_END = 153  # hasta Conv_1 + BN + ReLU


def get_backbone_layers(model: keras.Model):
    """Devuelve las capas del backbone (MobileNetV2 ya plano)."""
    return model.layers[BACKBONE_START : BACKBONE_END + 1]


def get_last_conv_layer_name(model: keras.Model) -> str:
    """Encuentra el nombre de la última Conv2D del backbone."""
    backbone = get_backbone_layers(model)

    for layer in reversed(backbone):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name

    raise RuntimeError("No existe ninguna capa Conv2D en el backbone.")


# =====================================================================
# 2. Grad-CAM++ (versión ligera para backend)
# =====================================================================

def grad_cam_pp(model: keras.Model, img_array: np.ndarray) -> np.ndarray:
    """
    Calcula el mapa Grad-CAM++ para una sola imagen.
    - model: modelo Keras cargado (.keras)
    - img_array: array (1, 224, 224, 3) normalizado [0,1]
    Devuelve: heatmap (H, W) en float32 normalizado [0,1].
    """
    # A tensor y aseguramos float32
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Localizar la última conv real del backbone
    last_conv_name = get_last_conv_layer_name(model)
    target_layer = model.get_layer(last_conv_name)

    # Modelo auxiliar: entrada -> (feature map de última conv, salida del modelo)
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )

    # Gradientes respecto a la clase positiva (preds[:, 0])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        tape.watch(conv_out)
        # preds shape (1, 1) o (1,) → tomar la columna 0
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        # Fallback: heatmap neutro
        h = int(img_tensor.shape[1] or 224)
        w = int(img_tensor.shape[2] or 224)
        return np.zeros((h, w), dtype=np.float32)

    # Pasar a numpy (CPU) — mapas pequeñitos, no hay problema de RAM
    conv = conv_out[0].numpy()  # (H, W, C)
    g = grads[0].numpy()        # (H, W, C)

    # Fórmula Grad-CAM++
    g2 = g ** 2
    g3 = g ** 3

    # Evitar división por 0
    denom = 2.0 * g2 + np.sum(g3 * conv, axis=(0, 1), keepdims=True)
    denom = np.where(denom != 0.0, denom, 1e-10)

    alpha = g2 / denom
    weights = np.sum(alpha * np.maximum(g, 0.0), axis=(0, 1))  # (C,)

    # Combinar pesos con feature maps: (H,W,C) · (C,) → (H,W)
    heatmap = np.tensordot(conv, weights, axes=([2], [0]))
    heatmap = np.maximum(heatmap, 0.0)

    # Normalizar a [0,1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    return heatmap.astype(np.float32)


# =====================================================================
# 3. Superposición (igual que antes)
# =====================================================================

def superimpose_heatmap(img_uint8: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    img_uint8: imagen RGB uint8 (H, W, 3) en [0,255]
    heatmap: mapa (H, W) en [0,1]
    """
    h, w = img_uint8.shape[:2]

    # Redimensionar heatmap al tamaño de la imagen
    heat_resized = cv2.resize(heatmap, (w, h))
    heat_uint8 = np.uint8(255 * heat_resized)

    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_uint8, 0.55, heat_color, 0.45, 0)
    return overlay


# =====================================================================
# 4. Helpers opcionales (por si luego quieres seguir usando base64)
# =====================================================================

def generate_gradcampp(model: keras.Model, img_tensor: np.ndarray) -> str:
    """
    Genera una imagen Grad-CAM++ superpuesta y la devuelve en base64 (JPG).
    img_tensor: (1, 224, 224, 3) en [0,1]
    """
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # Recuperar imagen como uint8
    img_np = img_tensor[0].numpy()
    img_uint8 = np.uint8(np.clip(img_np * 255.0, 0, 255))

    heatmap = grad_cam_pp(model, img_tensor.numpy())
    overlay = superimpose_heatmap(img_uint8, heatmap)

    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode("utf-8")


def generate_gradcam_pp_for_batch(model: keras.Model, images: np.ndarray):
    """
    Aplica Grad-CAM++ para un batch de imágenes (N, 224, 224, 3).
    Devuelve lista de strings base64.
    """
    result = []
    for img in images:
        img_tensor = np.expand_dims(img, axis=0)
        result.append(generate_gradcampp(model, img_tensor))
    return result
