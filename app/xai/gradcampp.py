import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import base64


# ============================================================
# 1. ENCONTRAR LA ÚLTIMA CAPA CONVOLUCIONAL DEL MODELO
# ============================================================

def get_last_conv_layer(model: keras.Model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
    raise ValueError("No se encontró ninguna capa Conv2D en el modelo.")


# ============================================================
# 2. GRAD-CAM++
# ============================================================

def grad_cam_pp(model: keras.Model, img_tensor: tf.Tensor) -> np.ndarray:
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    last_conv_layer = get_last_conv_layer(model)

    grad_model = keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        tape.watch(conv_out)

        # ⚠️ CLAVE: convertir a tensor
        preds = tf.convert_to_tensor(preds)

        loss = preds[:, 0]  # binario

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        h = img_tensor.shape[1]
        w = img_tensor.shape[2]
        return np.zeros((h, w), dtype=np.float32)

    conv = conv_out[0].numpy()
    g = grads[0].numpy()

    g2 = g ** 2
    g3 = g ** 3

    denom = 2 * g2 + np.sum(g3 * conv, axis=(0, 1), keepdims=True)
    denom = np.where(denom != 0, denom, 1e-10)

    alpha = g2 / denom
    weights = np.sum(alpha * np.maximum(g, 0), axis=(0, 1))

    heatmap = np.tensordot(conv, weights, axes=([2], [0]))
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap.astype(np.float32)


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
# 4. UNA SOLA IMAGEN → BASE64
# ============================================================

def generate_gradcampp(model: keras.Model, img_tensor: np.ndarray) -> str:
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    img_np = img_tensor[0].numpy()
    img_uint8 = np.uint8(np.clip(img_np * 255, 0, 255))

    heatmap = grad_cam_pp(model, img_tensor)
    overlay = superimpose_heatmap(img_uint8, heatmap)

    _, buffer = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buffer).decode("utf-8")


# ============================================================
# 5. BATCH (lista de imágenes 224x224x3)
# ============================================================

def generate_gradcam_pp_for_batch(model, images):
    result = []
    for img in images:
        img_tensor = np.expand_dims(img, axis=0)
        result.append(generate_gradcampp(model, img_tensor))
    return result
