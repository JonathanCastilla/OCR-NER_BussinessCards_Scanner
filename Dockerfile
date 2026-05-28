# 1. Usar una computadora virtual ligera con Python 3.9
FROM python:3.11-slim

# 2. Instalar el motor Tesseract OCR en el sistema operativo Linux
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# 3. Crear una carpeta llamada /app en la nube y entrar en ella
WORKDIR /app

# 4. Copiar tu lista de compras y descargar las librerías de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar todo el resto de tu código (main.py, carpetas static, templates, etc.)
COPY . .

# 6. Abrir el puerto 7860 (es el puerto obligatorio que pide Hugging Face)
EXPOSE 7860

# 7. Encender la aplicación usando Gunicorn (un servidor web profesional)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]