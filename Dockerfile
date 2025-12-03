# Imagen base
FROM python:3.10-slim

# Evitar prompts en la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema + Git + Git LFS
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

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Verificar que el archivo LFS se descargó correctamente
RUN ls -lh modelo_mobilenetv2_anemia_final_XAI.keras || echo "Modelo NO encontrado"

# Exponer el puerto
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
