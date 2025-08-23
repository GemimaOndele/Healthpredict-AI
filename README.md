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