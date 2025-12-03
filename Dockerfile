# Imagen base
FROM python:3.10-slim

# Evitar prompts
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar el backend
COPY . .

# === DESCARGAR MODELO DESDE GITHUB RELEASES ===
RUN wget -O modelo_mobilenetv2_anemia_final_XAI.keras \
    https://github.com/VictorSanchezS/tesis-backend/releases/download/v1.0/modelo_mobilenetv2_anemia_final_XAI.keras

# Verificar archivo (importante)
RUN ls -lh modelo_mobilenetv2_anemia_final_XAI.keras

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
