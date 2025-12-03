# 📘 Guía de instalación -- Proyecto tesis-app

Este documento explica cómo clonar el proyecto y ejecutar el backend
(FastAPI) en cualquier dispositivo.

## ✅ 1. Clonar el repositorio

Abre la terminal (CMD, PowerShell o Git Bash) y ejecuta:

`git clone https://github.com/VictorSanchezS/tesis-backend.git`


## ✅ 3. Preparar el backend 

3.1 Crear el entorno virtual (Windows) `python -m venv venv`

3.3 Activar el entorno virtual 

Copiar y pega en consola:

`.\venv\Scripts\Activate.ps1`

O

venv`\Scripts`{=tex}`\activate`{=tex}

Si la terminal muestra (venv) significa que está activado.

3.4 Instalar dependencias `pip install -r requirements.txt`

3.5 Ejecutar FastAPI `uvicorn app.main:app --reload`

El backend estará disponible en:

👉 http://127.0.0.1:8000

Y la documentación interactiva:

👉 http://127.0.0.1:8000/docs
