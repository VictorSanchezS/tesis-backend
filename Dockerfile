# Imagen base con TensorFlow ya instalado (CPU only)
FROM tensorflow/tensorflow:2.15.0

# Evitar que Python bufee stdout/stderr
ENV PYTHONUNBUFFERED=1

# Limitar RAM usada por TensorFlow (menos hilos)
ENV TF_ENABLE_ONEDNN_OPTS=0 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

WORKDIR /app

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

EXPOSE 8000

# Un solo proceso uvicorn (un solo modelo en RAM)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
