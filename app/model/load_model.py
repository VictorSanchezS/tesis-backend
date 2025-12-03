import os
import requests
import keras  # 👈 usamos Keras directamente


# URL del asset en GitHub Releases (v1.0)
DEFAULT_MODEL_URL = (
    "https://github.com/VictorSanchezS/tesis-backend/releases/download/"
    "v1.0/modelo_mobilenetv2_anemia_final_XAI.keras"
)

# Permito override por variables de entorno si quieres
MODEL_URL = os.getenv("MODEL_URL", DEFAULT_MODEL_URL)
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_mobilenetv2_anemia_final_XAI.keras")


def download_model_if_needed():
    """
    Descarga el modelo desde GitHub Releases si:
    - No existe el archivo, o
    - Es sospechosamente pequeño (< 1 MB, típico de HTML roto)
    """
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        if size > 1_000_000:  # ~1 MB
            print(f"✔ Modelo ya existe en {MODEL_PATH} ({size} bytes).")
            return
        else:
            print(f"⚠ Archivo existente muy pequeño ({size} bytes). Se volverá a descargar.")

    print(f"⬇ Descargando modelo desde GitHub: {MODEL_URL}")

    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "Mozilla/5.0",
    }

    # stream=True evita cargar todo en memoria y reduce riesgo de corrupción
    with requests.get(MODEL_URL, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    size = os.path.getsize(MODEL_PATH)
    print(f"✔ Descarga completada. Tamaño final: {size} bytes.")


def load_model():
    """Asegura que el modelo exista y lo carga."""
    download_model_if_needed()

    print(f"Cargando modelo desde: {MODEL_PATH} ...")
    # 👇 Usamos Keras (formato .keras moderno)
    model = keras.models.load_model(MODEL_PATH)
    print("✔ Modelo cargado correctamente.")

    return model
