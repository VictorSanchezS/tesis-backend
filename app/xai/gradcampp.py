import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import base64


# ============================================================
# 1. UBICAR LA ÚLTIMA CAPA CONVOLUCIONAL
# ============================================================

def get_last_conv_layer(model: keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
    raise ValueError("No se encontró una capa Conv2D en el modelo.")


# ============================================================
# 2. GRAD-CAM++
# ============================================================

def grad_cam_pp(model, img_array):
    # ----------------------------------------
    # Paso 1: Predicción y conversión segura
    # ----------------------------------------
    preds = model.predict(img_array)
    preds = np.array(preds, dtype=np.float32)  # <<< FIX DEL ERROR
    pred_class = preds[0]

    # ----------------------------------------
    # Paso 2: Obtener la última capa conv
    # ----------------------------------------
    last_conv_layer = get_last_conv_layer(model)

    grad_model = keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    # ----------------------------------------
    # Paso 3: Calcular Gradientes
    # ----------------------------------------
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Clase positiva (binaria)

    grads = tape.gradient(loss, conv_outputs)

    # ----------------------------------------
    # Paso 4: Grad-CAM++
    # ----------------------------------------
    grads = tf.cast(grads, tf.float32)
    conv_outputs = tf.cast(conv_outputs, tf.float32)

    # 1) Grads positivos
    grad_2 = tf.square(grads)
    grad_3 = grads * grad_2

    # 2) Alpha
    numerator = grad_2
    denominator = 2 * grad_2 + conv_outputs * grad_3
    denominator = tf.where(denominator != 0, denominator, tf.ones_like(denominator))

    alpha = numerator / denominator

    # 3) Pesos
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))

    # 4) Heatmap
    cam = tf.reduce_sum(weights[..., tf.newaxis, tf.newaxis] * conv_outputs, axis=-1)
    heatmap = tf.nn.relu(cam)

    # Normalizar
    heatmap = heatmap[0].numpy()
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


# ============================================================
# 3. SUPERPOSICIÓN
# ============================================================

def superimpose_heatmap(img_uint8: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w = img_uint8.shape[:2]

    heat_resized = cv2.resize(heatmap, (w, h))
    heat_uint8 = np.uint8(255 * heat_resized)

    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_uint8, 0.55, heat_color, 0.45, 0)
    return overlay


# ============================================================
# 4. UNA SOLA IMAGEN → BASE64 (por si lo usas algún día)
# ============================================================

def generate_gradcampp(model: keras.Model, img_tensor: np.ndarray) -> str:
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    img_np = img_tensor[0].numpy()
    img_uint8 = np.uint8(np.clip(img_np * 255, 0, 255))

    heatmap = grad_cam_pp(model, img_tensor)
    overlay = superimpose_heatmap(img_uint8, heatmap)

    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode("utf-8")


def generate_gradcam_pp_for_batch(model, images):
    result = []
    for img in images:
        img_tensor = np.expand_dims(img, axis=0)
        result.append(generate_gradcampp(model, img_tensor))
    return result
