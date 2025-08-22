# === Build image ===
FROM python:3.11-slim

# Pré-requis système (build wheels usuels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie requirements en premier (cache Docker)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code
COPY . .

# Assure que start.sh est exécutable (si tu es sur Windows, CRLF -> LF recommandé)
RUN chmod +x ./start.sh

# Variables par défaut (tu peux les surcharger sur Render)
ENV HP_ASSETS_DIR=assets
# HP_ASSETS_SPECS et HF_TOKEN seront injectés via l’UI Render/env group

# Streamlit (sans prompts d’update)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Port Render
EXPOSE 10000

# CMD: télécharge puis lance l’app
CMD ["./start.sh"]
# === End of Dockerfile ===