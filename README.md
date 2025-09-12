````markdown
# 📄 🤖 HealthPredict AI — Maintenance prédictive des équipements médicaux

[![CI](https://github.com/GemimaOndele/Healthpredict-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/GemimaOndele/Healthpredict-AI/actions)  
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://hub.docker.com/)  
[![Hugging Face](https://img.shields.io/badge/Datasets-HuggingFace-yellow?logo=huggingface)](https://huggingface.co/datasets/Gkop/healthpredict-assets)

---

## 📖 Présentation

**HealthPredict AI** est une application de **maintenance prédictive des équipements médicaux**.  
Elle permet d’**analyser des rapports d’incidents** (issus d’OpenFDA et de documents hospitaliers anonymisés) et de **prédire la criticité** grâce à des modèles IA :

- 🔹 **Baseline TF-IDF + Logistic Regression**  
- 🔹 **CamemBERT + classifieur** (optionnel, plus lourd)

⚡ Objectif : **améliorer la disponibilité, la sécurité et la traçabilité** des équipements médicaux en s’appuyant sur l’IA et l’automatisation.

---

## 🎯 Objectifs du projet

- Automatiser le **chargement et nettoyage de données** (OpenFDA, rapports hospitaliers).  
- Développer et entraîner des **modèles IA de classification** (TF-IDF et CamemBERT).  
- Offrir une **application interactive (Streamlit)** pour les prédictions et l’exploration de données.  
- Fournir une **API REST (FastAPI)** pour intégration externe.  
- Mettre en place un **CI/CD (GitHub Actions + Docker)** et un **monitoring santé**.  

---

## ✨ Fonctionnalités

- 📂 Chargement dataset (processed/raw), fallback auto.  
- 📝 Prédiction texte libre & documents (PDF, DOCX, images → OCR).  
- 🔑 Extraction de mots-clés TF-IDF et recherche de cas similaires (cosinus).  
- 🌍 Traduction EN→FR (Transformers, optionnel).  
- 🗄️ Historique des prédictions (SQLite).  
- 📊 Graphiques de tendance (Altair) et prévision simple.  
- 🔌 API REST (FastAPI) sécurisée par clé API.  
- ⚙️ Monitoring santé (`scripts/monitor.py`).  
- 🚢 CI/CD avec GitHub Actions + Docker.  

---

## 🚀 Installation rapide

### 🖥️ Windows (PowerShell)
```powershell
python -m venv .henv
.\.henv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
$env:HP_AUTO_DOWNLOAD="1"
$env:HP_DB="$pwd\data\app.db"
````

### 🐧 Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export HP_AUTO_DOWNLOAD=1
export HP_DB="$PWD/data/app.db"
```

### 📥 Téléchargement des assets & préparation

```bash
python scripts/download_assets.py
python scripts/build_processed_csv.py
python scripts/validate_dataset.py   # → "Validation dataset réussie."
```

### 🔄 Entraînement des modèles

```bash
python scripts/train_minimal_tfidf.py
python scripts/train_camembert_baseline.py   # optionnel
```

### ▶️ Lancer l’application Streamlit

```bash
streamlit run app/healthpredict_app.py
# → http://localhost:8501
```

---

## 🔌 API REST (FastAPI)

### Lancer l’API

```powershell
$env:HP_API_KEY="changeme"
uvicorn api.main:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

### Endpoints

* `GET /health` → statut API
* `GET /version` → info version & modèles
* `POST /predict_text` → prédiction texte
* `POST /predict_file` → prédiction fichier (OCR/parse côté Streamlit)

### Exemple PowerShell

```powershell
$body = @{ text = "scanner error overheating pump"; model = "tfidf"; return_keywords = $true } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict_text" -Method POST -ContentType "application/json" -Headers @{ "X-API-Key" = "changeme" } -Body $body
```

---

## 🧪 Tests

```bash
pytest -q
```

➡️ Vérifie que ton `pytest.ini` contient :

```ini
[pytest]
pythonpath = .
```

---

## 🛠️ Dépannage rapide

### 🗂️ Base SQLite non initialisée

```bash
python -c "import hpdb,os; hpdb.init_db(os.environ.get('HP_DB','data/app.db'))"
```

### ⚡ Erreurs courantes

* **Erreur NumPy**

  ```bash
  pip install --force-reinstall "numpy<3,>=1.26"
  ```

* **OCR inactif**
  Installer **Tesseract** (et Poppler pour PDF).
  Renseigner le chemin dans l’onglet Documents de l’app.

---

### 🧹 Nettoyage Docker (libérer de la mémoire)

Si Docker Desktop consomme trop d’espace, exécuter ces commandes dans PowerShell :

```powershell
# Supprimer les conteneurs arrêtés
docker container prune -f

# Supprimer les images inutilisées
docker image prune -a -f

# Supprimer les volumes inutilisés
docker volume prune -f

# Supprimer les réseaux inutilisés
docker network prune -f

# Tout nettoyer (⚠️ supprime images, conteneurs, volumes, réseaux)
docker system prune -a --volumes -f
```

---

## 📂 Données & artefacts

* OpenFDA (Device Event) → `assets/data/raw/...`
* Processed CSV → `assets/data/processed/medical_imaging_text_labeled.csv`
* Évaluation (ROC/PR/CM) → `assets/eval/`
* Modèles → `assets/models/*.joblib`

Datasets & modèles lourds hébergés sur Hugging Face :
👉 [Hugging Face - HealthPredict Assets](https://huggingface.co/datasets/Gkop/healthpredict-assets)

---

## 📦 CI / CD

* **CI** : GitHub Actions (tests + lint + API smoke test)
* **CD** : Docker (images `healthpredict-app` et `healthpredict-api`)

### Lancer Docker localement

```bash
docker build -f Dockerfile.app -t healthpredict-app .
docker run --rm -p 8501:8501 -v %cd%/data:/app/data healthpredict-app
# → http://localhost:8501
```

---

## 🔎 Monitoring (optionnel)

`scripts/monitor.py` effectue un **health check** sur `/health` et enregistre les résultats dans :

```
logs/healthcheck.log
```

---

## 🔒 Avertissement

⚠️ **Projet éducatif/démo** — ne pas utiliser pour décision médicale sans validation clinique.
✅ Données anonymisées — aucune **PII** (donnée personnelle identifiable) n’est stockée.

---

## 🔗 Liens utiles

* 🐙 GitHub : [HealthPredict-AI](https://github.com/GemimaOndele/Healthpredict-AI)
* 🤗 Hugging Face : [HealthPredict Assets](https://huggingface.co/datasets/Gkop/healthpredict-assets)
* 💼 LinkedIn : [Gémima Ondele Pourou](https://www.linkedin.com/in/g%C3%A9mima-ondele-pourou-1515251a7)

---

## 👤 Auteur

**Gémima Keren ONDELE POUROU**
📧 [gemimakerenondelepourou@gmail.com](mailto:gemimakerenondelepourou@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/g%C3%A9mima-ondele-pourou-1515251a7)

© 2025 — HealthPredict AI — Licence MIT


