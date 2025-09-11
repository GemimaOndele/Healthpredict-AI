# HealthPredict AI — Maintenance prédictive des équipements médicaux

Application Streamlit pour analyser des rapports d’incidents (OpenFDA & documents) et prédire la criticité via IA (TF-IDF/LogReg ou CamemBERT + classif). OCR, traduction EN→FR optionnelle, similarité historique, export CSV/XLSX, historique SQLite.

## ✨ Fonctionnalités
- Chargement dataset (processed/raw), fallback auto.
- Prédiction texte libre & documents (PDF, DOCX, images → OCR).
- Mots-clés TF-IDF, cas similaires (cosinus).
- Traduction EN→FR (Transformers, optionnel).
- Historique des prédictions (SQLite).
- Graphiques de tendance (Altair).

## 🚀 Démarrage rapide (local)
```bash #terminal git bash
python -m venv .venv && source .env/bin/activate   
python -m venv .henv && source .henv/bin/activate  

# (Windows: .henv\Scripts\activate)
.henv\Scripts\activate

pip install --no-cache-dir -r requirements.txt  #ou
pip install -r requirements.txt
# (optionnel) torch CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# (optionnel) spaCy modèles:
python -m spacy download fr_core_news_sm && python -m spacy download en_core_web_sm

# Variables (optionnel)
cp .env.example .env

#Installer hugging face(plateforme publique) qui contient la données utilisé et les modèles d'IA que j'ai entrainés ainsi que d'autres modèles d'IA disponible pré entrainés 
pip install huggingface_hub

#installer quelques dépendances 
pip install joblib 
pip install scikit-learn
pip install install-scripts
pip install db
pip install scripts
pip install scripts.build_processed_csv
pip install --no-cache-dir python-dotenv
pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
pip install --no-cache-dir joblib scikit-learn huggingface_hub

#lancer le téléchargement des modèles IA entrainées publié sur hugging face  
python scripts/download_assets.py

#lancer le génération du csv pour entrainer le modèle TFIDF par la suite
python scripts/build_processed_csv.py

#lancer l'entrainement du modèle TFIDF
python scripts/train_minimal_tfidf.py 

#lancer l'entrainement du modèle BERT (camemBERT)
python scripts/train_camembert_baseline.py

# (option) activer CamemBERT pour l'éval
$env:HP_USE_CAMEMBERT="1"

#lancer l'évaluation des prédictions 
python notebooks/eval_healthpredict.py


# Dans PowerShell, à la racine du projet
$env:HP_AUTO_DOWNLOAD="1"
$env:HP_DOWNLOAD_CAMEMBERT="1"   # optionnel
$env:HP_DB="C:\Healthpredict-AI-clean\data\app.db"  #(facultatif, sinon data/app.db par défaut)

# Lancer l'application :

#1ère possibilité
streamlit run app/healthpredict_app.py  

#2ème possibilité
  .\start.ps1

## Ouvre le lien pour voir l'application IA
http://localhost:8501

```




## 🛠️ Maintenance & Dépannage

### 🔄 Réentraînement des modèles

* **TF-IDF** :

  ```bash
  python scripts/train_minimal_tfidf.py
  ```
* **CamemBERT** (plus lourd, nécessite torch/transformers) :

  ```bash
  python scripts/train_camembert_baseline.py
  ```

### 🗂️ Base SQLite

* Si l’historique des prédictions ne fonctionne pas (`no such table: predictions`), exécuter :

  ```bash
  python -c "import hpdb; hpdb.init_db('data/app.db')"
  ```
* Par défaut, la base est créée dans `data/app.db`.
* Supprimer ce fichier permet de repartir avec une base vide.

### ⚡ Problèmes fréquents

* **Erreur NumPy** (`Numpy is not available`) :
  Vérifier que vous avez `numpy<2` installé :

  ```bash
  pip install --force-reinstall "numpy<2,>=1.26"
  ```
* **OCR inactif** :

  * Vérifier que Tesseract est installé et accessible.
  * Modifier son chemin dans l’interface Streamlit si besoin.

### 🚀 Mise à jour des assets (modèles & données)

* Si un modèle ou dataset manque :

  ```bash
  python scripts/download_assets.py
  ```
* Les modèles sont aussi disponibles sur [Hugging Face](https://huggingface.co/?activityType=update-dataset&feedType=user).


# Données — HealthPredict AI

## 1) Sources
- **OpenFDA – Device Event** (publique) : incidents de dispositifs médicaux.
  - Export initial via `scripts/01_download_openfda.py` (requêtes paginées).
  - Fichier brut : `assets/data/raw/raw_openfda_imaging_reports.csv` (et `.jsonl`).
- **Rapports de test** (usage académique) : documents hospitaliers anonymisés (ex. Joigny) pour valider l’OCR et la prédiction.
  - Traités localement via l’onglet **📎 Documents** de l’app.

## 2) Processus de préparation
- Normalisation & étiquetage dans `scripts/build_processed_csv.py`.
- Résultat : `assets/data/processed/medical_imaging_text_labeled.csv` avec colonnes clés :
  - `event_text` (texte), `event_type` (Death/Injury/… si présent), `label` (0/1), `date_received` (date si présente).

## 3) Confidentialité
- Les fichiers de test contenant des éléments réels sont **anonymisés** et **non versionnés** publiquement.
- Ne **push** jamais de données sensibles sur GitHub. Utiliser `.gitignore` pour tout dépôt de documents réels.

## 4) Versionnement & traçabilité
- Les artefacts lourds (datasets / modèles) sont publiés sur **Hugging Face** pour limiter la taille du repo :
  - Modèles : `assets/models/*.joblib`
  - Datasets : `assets/data/raw/*`, `assets/data/processed/*` (si nécessaire)
- Variables & chemins dans `config/config.yaml` (voir README pour les commandes).

## 5) Contrôles qualité
- Lancer `scripts/validate_dataset.py` (voir README) pour vérifier :
  - Présence des colonnes obligatoires
  - Encodage UTF-8
  - Duplicats & vides
  - Statistiques de base


## 📦 Données & Gouvernance

- **Sources & API** : voir `docs/api_data_sources.md`.
- **Dictionnaire de données** : voir `docs/data_dictionary.md`.
- **Traçabilité / confidentialité** : voir `data/README_data.md`.
- **Pipeline** : voir `docs/architecture_data_pipeline.md`.

### ✅ Contrôle qualité (dataset)

# 1) Configurer les chemins (si besoin)
#   -> config/config.yaml (paths.processed_csv / paths.raw_csv / ...)

# 2) Valider le CSV "processed"
python scripts/validate_dataset.py
# -> code 0 si OK, sinon détail des erreurs (colonnes manquantes, encodage, etc.)


### 🔁 Préparation / entraînement / évaluation

# Générer le processed CSV (si absent)
python scripts/build_processed_csv.py

# Entraîner TF-IDF
python scripts/train_minimal_tfidf.py

# (optionnel) Entraîner CamemBERT
python scripts/train_camembert_baseline.py

# Évaluer et générer figures
python notebooks/eval_healthpredict.py
markdown


# 2) Ce qu’il faut faire (et seulement ça)

1) **Créer les fichiers** ci-dessus aux emplacements indiqués.  
2) **Ajouter** la section “📦 Données & Gouvernance” dans le `README.md`.  
3) **Exécuter** rapidement :
   - `python scripts/build_processed_csv.py` (si besoin),
   - `python scripts/validate_dataset.py` (doit afficher `Validation dataset réussie.`).



✅ À METTRE À JOUR (README.md)

Ajoute une petite section « API REST » :

## 🔌 API REST (FastAPI)

L’API expose des endpoints pour la prédiction (texte & fichier) — utile pour intégration tierce (applications internes, scripts).

**Lancer l’API**

# depuis la racine du projet
set HP_API_KEY=changeme  # Windows PowerShell: $env:HP_API_KEY="changeme"
uvicorn api.main:app --host 0.0.0.0 --port 8000


  Endpoints
  
  GET /health → statut
  
  GET /version → info version & modèles
  
  POST /predict_text → {"text": "...", "model": "tfidf|camembert", "return_keywords": true}
  
  POST /predict_file → upload fichier (texte brut côté API ; OCR/parse avancé via l’app Streamlit)
  
  Sécurité
  
  Header requis si HP_API_KEY défini : X-API-Key: <clé>
  
  Exemple cURL
  
  curl -X POST http://localhost:8000/predict_text \
    -H "Content-Type: application/json" \
    -H "X-API-Key: changeme" \
    -d "{\"text\":\"radiologie en panne et erreurs\",\"model\":\"tfidf\",\"return_keywords\":true}"


## CI : Intégration continue
![CI](https://github.com/<TON_ORG>/<TON_REPO>/actions/workflows/ci.yml/badge.svg)


## CD : 🚢 Déploiement Docker

### Lancer localement (sans API séparée)
```bash
docker build -f Dockerfile.app -t healthpredict-app .
docker run --rm -p 8501:8501 -v %cd%/data:/app/data healthpredict-app
# puis http://localhost:8501
