FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS deps (OCR + PDF + build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-fra poppler-utils libgl1 libglib2.0-0 \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Torch CPU (optionnel mais recommandé si CamemBERT)
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# spaCy modèles optionnels (si tu utilises spaCy)
# RUN python -m spacy download fr_core_news_sm && python -m spacy download en_core_web_sm

COPY . .

# Activer CamemBERT/Traduction/auto-download assets au runtime si tu veux
ENV HP_USE_CAMEMBERT=1 \
    HP_USE_SPACY=1 \
    HP_AUTO_DOWNLOAD=1 \
    HP_MAX_ROWS_UI=150000

# Streamlit config
ENV PORT=8501
EXPOSE 8501
CMD ["streamlit", "run", "app/healthpredict_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# === End of Dockerfile ===