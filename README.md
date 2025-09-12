````markdown
# 🏥 HealthPredict AI — Maintenance prédictive des équipements médicaux

**HealthPredict AI** est une application **Streamlit** et une **API REST (FastAPI)** qui analysent des rapports d’incidents (OpenFDA + documents) pour **prédire la criticité** des événements (⚠️ critique / ✅ non critique) à l’aide de modèles IA : **TF-IDF + régression logistique** (baseline) et **CamemBERT + classif** (optionnel).  
Le projet intègre : **OCR** (PDF/Images), **traduction EN→FR** (option), **similarité historique**, **export CSV/XLSX**, **historique SQLite**, **tests CI**, **Docker** et un **monitoring santé** basique.

---

## 🧭 Présentation rapide

- **But** : aider à prioriser les incidents/maintenances sur équipements médicaux (IRM, scanner, radio, etc.) à partir de **textes**.
- **Objet** : transformer des rapports bruts en **indicateurs actionnables** (score de risque + explications).
- **Public** : équipes techniques/qualité (et pédagogie IA).
- **Périmètre** : démonstrateur **éducatif** (pas d’accès production hospitalière, données anonymisées).

> 🔒 **Confidentialité & PII** : aucune **donnée personnelle identifiable** n’est versionnée. Les jeux de test réels sont **anonymisés** et non poussés sur GitHub.  

---

## ✨ Fonctionnalités principales

- 📦 Chargement **dataset** (processed/raw) avec **fallback** automatique.  
- 📝 Prédiction **texte libre** & **documents** (PDF, DOCX, images → **OCR**).  
- 🧩 **Mots-clés TF-IDF** + recherche de **cas similaires** (cosinus).  
- 🌐 **Traduction EN→FR** (option Transformers).  
- 🗃️ **Historique des prédictions** (SQLite) + courbes de tendance (Altair).  
- 🧪 **Évaluation** (accuracy/F1/ROC-AUC) et figures d’analyse.  
- 🔌 **API REST** (FastAPI) pour intégrations tierces.  
- 🐳 **Docker** (app & API) et **CI** (pytest + lint).  
- 🩺 **Monitoring santé** : ping périodique `/health` → `logs/healthcheck.log`.

---

## 🚀 Installation & démarrage

### ▶️ Windows (PowerShell)
```powershell
# 1) Environnement
python -m venv .henv
.\.henv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Variables (optionnel)
Copy-Item .env.example .env
$env:HP_AUTO_DOWNLOAD="1"
$env:HP_DB="$pwd\data\app.db"

# 3) Assets & données
python scripts/download_assets.py
python scripts/build_processed_csv.py
python scripts/validate_dataset.py   # → "Validation dataset réussie."

# 4) (Option) Entraînement modèles
python scripts/train_minimal_tfidf.py
python scripts/train_camembert_baseline.py

# 5) Lancer l’application
streamlit run app/healthpredict_app.py
# → http://localhost:8501
````

### ▶️ Linux / macOS (bash)

```bash
# 1) Environnement
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Variables (optionnel)
cp .env.example .env
export HP_AUTO_DOWNLOAD=1
export HP_DB="$PWD/data/app.db"

# 3) Assets & données
python scripts/download_assets.py
python scripts/build_processed_csv.py
python scripts/validate_dataset.py   # → "Validation dataset réussie."

# 4) (Option) Entraînement modèles
python scripts/train_minimal_tfidf.py
python scripts/train_camembert_baseline.py

# 5) Lancer l’application
streamlit run app/healthpredict_app.py
# → http://localhost:8501
```

---

## 🔌 API REST (FastAPI)

**Démarrer l’API :**

```powershell
# Windows PowerShell (clé API optionnelle)
$env:HP_API_KEY="changeme"
uvicorn api.main:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs
```

**Endpoints :**

* `GET /health` → statut
* `GET /version` → versions & modèles
* `POST /predict_text` → `{"text":"...", "model":"tfidf|camembert", "return_keywords":true}`
* `POST /predict_file` → upload fichier (OCR/parse avancé côté app)

**Exemples d’appel :**

*Bash (curl)*

```bash
curl -X POST http://localhost:8000/predict_text \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{"text":"scanner error overheating pump","model":"tfidf","return_keywords":true}'
```

*PowerShell (Invoke-RestMethod)*

```powershell
$body = @{ text = "scanner error overheating pump"; model = "tfidf"; return_keywords = $true } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/predict_text" `
  -Method POST -ContentType "application/json" `
  -Headers @{ "X-API-Key" = "changeme" } -Body $body
```

---

## 🧪 Tests & qualité

```bash
# Important : rendre 'api' importable
# pytest.ini doit contenir :
# [pytest]
# pythonpath = .

pytest -q
```

> 💡 Si vos tests attendent une clé API (ex: `test-key`), définir avant `pytest` :
> **Windows** `setx HP_API_KEY "test-key"` puis ré-ouvrir le shell.
> **bash** `export HP_API_KEY="test-key"`.

---

## 🛠️ Dépannage rapide

**Historique désactivé / DB non initialisée**

```bash
python -c "import hpdb,os; hpdb.init_db(os.environ.get('HP_DB','data/app.db'))"
```

**Erreur NumPy**

```bash
pip install --force-reinstall "numpy<3,>=1.26"
```

**OCR inactif**

* Installer **Tesseract** (et **Poppler** pour PDF).
* Renseigner son **chemin** dans l’onglet *📎 Documents* de l’app.

---

## 📂 Données & artefacts

* **OpenFDA (Device Event)** → `assets/data/raw/...`
* **Processed CSV** → `assets/data/processed/medical_imaging_text_labeled.csv`
* **Évaluation (ROC/PR/CM)** → `assets/eval/`
* **Modèles** → `assets/models/*.joblib`
* **Datasets & modèles lourds** : hébergés sur **Hugging Face**
  👉 [https://huggingface.co/datasets/Gkop/healthpredict-assets](https://huggingface.co/datasets/Gkop/healthpredict-assets)

---

## 🐳 Docker (local)

> ⚠️ Si un build échoue sur une dépendance exotique (`hf-xet`), **retirer tout pin strict non supporté** de `requirements.txt`
> ou remplacer par `huggingface_hub[hf_xet]>=0.21`.

**App Streamlit**

```bash
docker build -f Dockerfile.app -t healthpredict-app .
docker run --rm -p 8501:8501 -v %cd%/data:/app/data healthpredict-app
# → http://localhost:8501
```

**API FastAPI**

```bash
docker build -f Dockerfile.api -t healthpredict-api .
docker run --rm -e HP_API_KEY=changeme -p 8000:8000 healthpredict-api
# → http://localhost:8000/docs
```

---

## 🩺 Monitoring (optionnel mais valorisant)

Un mini-script ping périodiquement l’API et logge le statut.

`scripts/monitor.py`

```python
import os, time, logging, requests, pathlib
LOG_DIR = pathlib.Path("logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_DIR/"healthcheck.log",
                    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
URL = os.getenv("HP_HEALTH_URL", "http://localhost:8000/health")
KEY = os.getenv("HP_API_KEY", "")
HDR = {"X-API-Key": KEY} if KEY else {}
while True:
    try:
        r = requests.get(URL, headers=HDR, timeout=5)
        logging.info("status=%s body=%s", r.status_code, r.text.strip()[:200])
    except Exception as e:
        logging.error("error=%s", e)
    time.sleep(int(os.getenv("HP_HEALTH_INTERVAL", "60")))
```

**Lancer :**

```bash
python scripts/monitor.py
# Log → logs/healthcheck.log
```

---

## 📦 CI / CD

* **CI GitHub Actions** : tests `pytest` + lint `ruff` (badge README).
* **CD optionnel** : build & push d’images `healthpredict-app` et `healthpredict-api` (Docker Hub / GHCR).

*Badge CI (remplacer par votre org & repo)*

```markdown
![CI](https://github.com/GemimaOndele/Healthpredict-AI/actions/workflows/ci.yml/badge.svg)
```

---

## 🔒 Avertissements

* Projet **éducatif/démo** — ne pas utiliser pour **décision clinique** sans validation réglementaire.
* **Pas de PII** dans le dépôt : jeux de test **anonymisés** et/ou hébergés hors repo.

---

## 🔗 Liens utiles

* 🧑‍💻 GitHub (code) : [https://github.com/GemimaOndele/Healthpredict-AI](https://github.com/GemimaOndele/Healthpredict-AI)
* 💾 Hugging Face (assets) : [https://huggingface.co/datasets/Gkop/healthpredict-assets](https://huggingface.co/datasets/Gkop/healthpredict-assets)
* 🔗 LinkedIn (auteure) : [https://www.linkedin.com/in/g%C3%A9mima-ondele-pourou-1515251a7](https://www.linkedin.com/in/g%C3%A9mima-ondele-pourou-1515251a7)

---

## 👤 Contact

**Gémima Keren ONDELE POUROU** — [gemimakerenondelepourou@gmail.com](mailto:gemimakerenondelepourou@gmail.com)

© 2025 — HealthPredict AI — Licence MIT

```
```
