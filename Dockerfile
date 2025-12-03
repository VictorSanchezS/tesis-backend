# Imagen base mínima con Python
FROM python:3.10-slim

# Evitar prompts
ENV DEBIAN_FRONTEND=noninteractive

# Instalar Git + Git LFS + dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar TODO el proyecto al contenedor
COPY . .

# ↓↓↓ AQUI VIENE LA PARTE IMPORTANTE ↓↓↓
# Descargar desde Git LFS los archivos reales del modelo
RUN git lfs pull

# Mostrar el tamaño REAL del archivo .keras dentro del contenedor
RUN ls -lh modelo_mobilenetv2_anemia_final_XAI.keras

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 8000

# Comando para correr FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
