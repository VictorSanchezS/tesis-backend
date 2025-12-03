import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import base64
import os
import tempfile
from dotenv import load_dotenv

# === Cargar variables de entorno ===
load_dotenv()

# === Roboflow SDK ===
from inference_sdk import InferenceHTTPClient

# === Cargar tu modelo ===
from app.model.load_model import load_model
from app.utils.preprocess import preprocess_nail_crop_bgr
from app.xai.gradcampp import grad_cam_pp, superimpose_heatmap

# =======================================================
# Inicialización
# =======================================================

app = FastAPI(title="Detección de Anemia - FastAPI")

model = load_model()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise ValueError("Falta la variable de entorno ROBOFLOW_API_KEY")

WORKSPACE = "victor-873iw"
WORKFLOW_ID = "custom-workflow"

rf_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# -------------------------------------------------------
# UTILIDADES
# -------------------------------------------------------

def crop_polygon_exact(image, points):
    """
    Recorta usando la máscara poligonal EXACTA.
    Devuelve: recorte_bgr, bounding_box, mask_recorte
    """
    parsed = []

    for p in points:
        if isinstance(p, dict) and "x" in p and "y" in p:
            parsed.append([int(p["x"]), int(p["y"])])
        elif isinstance(p, dict) and "point" in p:
            xy = p["point"]
            parsed.append([int(xy["x"]), int(xy["y"])])
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            parsed.append([int(p[0]), int(p[1])])
        else:
            print("⚠ Punto no reconocido:", p)

    if len(parsed) < 3:
        return None, None, None

    pts = np.array(parsed, dtype=np.int32)

    # --- máscara completa en la imagen original ---
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None, None, None

    x1, y1 = np.min(xs), np.min(ys)
    x2, y2 = np.max(xs), np.max(ys)

    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    # Aplicamos máscara exacta
    crop_exact = cv2.bitwise_and(crop, crop, mask=crop_mask)

    bbox = [int(x1), int(y1), int(x2), int(y2)]
    return crop_exact, bbox, crop_mask


def compute_sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_color_score(rgb):
    R, G, B = np.mean(rgb, axis=(0, 1))
    score = 0
    if R > 100: score += 0.4
    if abs(R - G) < 80: score += 0.3
    if B > 60: score += 0.3
    return min(score, 1.0)


def is_valid_nail(area, ratio, color_score):
    return area >= 400 and (0.3 <= ratio <= 2.8) and color_score >= 0.25


def encode_img(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


# =======================================================
# ENDPOINT PRINCIPAL
# =======================================================

@app.post("/predict-hand", response_model=dict)
async def predict_hand(
    file: UploadFile = File(...),
    xai: bool = Query(False)
):
    contents = await file.read()

    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    np_img = np.array(pil_img)
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Archivo temporal para Roboflow
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Ejecutar workflow
    try:
        result = rf_client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={"image": tmp_path},
            use_cache=False
        )
    except Exception as e:
        os.unlink(tmp_path)
        return {"error": f"Error llamando al Workflow: {e}"}

    os.unlink(tmp_path)

    # Extraer detecciones
    detections = []
    if isinstance(result, list):
        if "predictions" in result[0] and "predictions" in result[0]["predictions"]:
            detections = result[0]["predictions"]["predictions"]

    if not detections:
        return {"error": "No se detectaron uñas", "workflow_raw": result}

    # Evaluación de candidatos
    candidates = []
    for det in detections:
        crop, bbox, mask_crop = crop_polygon_exact(cv_img, det["points"])
        if crop is None:
            continue

        h, w = crop.shape[:2]
        area = h * w
        ratio = h / w

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sharpness = compute_sharpness(gray)
        color_score = compute_color_score(rgb)
        sharp_norm = sharpness / (sharpness + 5000)

        score = det["confidence"] * 0.5 + sharp_norm * 0.3 + color_score * 0.2
        valid = is_valid_nail(area, ratio, color_score)

        candidates.append({
            "crop": crop,
            "mask": mask_crop,
            "bbox": bbox,
            "area": area,
            "ratio": ratio,
            "sharpness": sharpness,
            "color_score": color_score,
            "score": score,
            "prob": det["confidence"],
            "valid": valid
        })

    # Selección final
    valid = [c for c in candidates if c["valid"]]
    best = max(valid, key=lambda x: x["score"]) if valid else max(candidates, key=lambda x: x["prob"])

    crop = best["crop"]
    mask_crop = best["mask"]
    crop = np.ascontiguousarray(crop)

    # PREPROCESAMIENTO EXACTO
    processed = preprocess_nail_crop_bgr(crop)

    # PREDICCIÓN
    pred = float(model.predict(np.expand_dims(processed, axis=0), verbose=0)[0][0])
    predicted_class = int(pred >= 0.5)

    # ===============================
    # XAI EXACTO SOLO DENTRO DE LA UÑA
    # ===============================
    xai_img = None
    if xai:
        # 1. Heatmap
        heatmap = grad_cam_pp(model, np.expand_dims(processed, axis=0))

        # 2. Superposición
        overlay = superimpose_heatmap(
            np.uint8(processed * 255),
            heatmap
        )

        # 3. Ajustar máscara al tamaño 224×224
        mask_resized = cv2.resize(mask_crop, (224, 224), interpolation=cv2.INTER_NEAREST)

        # 4. Aplicar máscara → SOLO uña
        overlay_masked = cv2.bitwise_and(overlay, overlay, mask=mask_resized)

        xai_img = encode_img(overlay_masked)

    # =======================================================
    # RESPUESTA FINAL
    # =======================================================

    return {
        "probability": pred,
        "class": predicted_class,
        "selected_nail": {
            "bbox": best["bbox"],
            "area": best["area"],
            "aspect_ratio": best["ratio"],
            "sharpness": best["sharpness"],
            "color_score": best["color_score"],
            "score": best["score"],
            "crop_image_base64": encode_img(crop)
        },
        "selected_nail_xai": xai_img,
        "workflow_raw": result
    }


@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}
