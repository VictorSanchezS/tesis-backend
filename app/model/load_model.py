import tensorflow as tf
import os

def load_model():
    """Carga y retorna el modelo entrenado."""
    model_path = os.getenv("MODEL_PATH", "modelo_mobilenetv2_anemia_final_XAI.keras")

    print(f"Cargando modelo desde: {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado correctamente.")

    return model
