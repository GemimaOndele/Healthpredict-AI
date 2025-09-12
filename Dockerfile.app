# Dockerfile.app
FROM python:3.11-slim

# Déps système minimales (OCR/Poppler seulement si tu en as besoin dans le conteneur)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HP_AUTO_DOWNLOAD=1 \
    HP_DB=/app/data/app.db \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["streamlit", "run", "app/healthpredict_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
