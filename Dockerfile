# Imagen base: TensorFlow 2.15 con CPU
FROM tensorflow/tensorflow:2.15.0

# Evita que TF use optimizaciones que consumen RAM extra
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_NUM_INTRAOP_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1

# Instalar dependencias necesarias para OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Crear carpeta de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto que Render necesita
EXPOSE 10000

# Comando para iniciar FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
