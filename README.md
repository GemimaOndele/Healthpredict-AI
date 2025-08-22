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
```bash
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# (optionnel) torch CPU:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# (optionnel) spaCy modèles:
# python -m spacy download fr_core_news_sm && python -m spacy download en_core_web_sm

# Variables (optionnel)
cp .env.example .env

# Lancer l'app
streamlit run app/healthpredict_app.py
