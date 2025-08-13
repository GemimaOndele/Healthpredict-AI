
# Healthpredict-AI
Maintenance prédictive des équipements médicaux (NLP + IA) — analyse de rapports, détection de mots-clés, similarité historique, OCR, traduction, et estimation du prochain incident.
# HealthPredict AI Dashboard

Projet de maintenance prédictive des équipements médicaux.
## Créer une environnement virtuel pour enregistrer les librairies à installer dedans au lieu l'espace de l'ordinateur

**entrer dans le dossier cd HealthPredict_ai_dashboard

**puis créer l'environnement
python -m venv .henv

## Activer l'environnement vrituel 

.henv\Scripts\activate  ##Sur Windows

source .henv/bin/activate  ##Sur MACOSX

## Optionnel  pour le désactiver : deactivate 

## Lancer l'application

```bash
streamlit run healthpredict_app.py

Dépendances :

-Streamlit
-Pandas
-Numpy
-Matplotlib

---

## 🖥️ Lancer le projet

Dans ton terminal :
```bash
streamlit run app.py






# HealthPredict AI (texte → maintenance prédictive)

## Setup
```bash
python -m venv .henv
.henv/Scripts/activate            # Windows
pip install -r requirements.txt




Pipeline données → modèles
Collecte OpenFDA

bash
Copier
Modifier
python scripts/01_download_openfda.py
→ data/raw/raw_openfda_imaging_reports.csv

Préparation + labellisation heuristique

bash
Copier
Modifier
python scripts/02_prepare_and_label.py
→ data/processed/medical_imaging_text_labeled.csv

Entraînement modèles texte

bash
Copier
Modifier
python scripts/03_train_text_models.py     # TF-IDF + (LogReg ou RF)
python scripts/train_camembert.py          # CamemBERT + LogReg
→ models/healthpredict_model.joblib + models/healthpredict_camembert_model.joblib

Lancer l’app
bash
Copier
Modifier
streamlit run app/healthpredict_app.py
Notes
Les labels initiaux sont heuristiques (mots-clés). Pour un usage réel, prévoir une annotation humaine.

Respecter RGPD / HIPAA / anonymisation.

markdown
Copier
Modifier

---

## Ce que ça corrige par rapport à mes erreurs
- **OpenFDA 404**: on utilise `date_received` (pas `receivedate`) et `mdr_text.text`; pagination gérée, 404 traité proprement.
- **Fichiers manquants**: les scripts écrivent exactement **aux chemins** lus par les étapes suivantes et par l’app (via `config.yaml`).
- **Stratify**: activé seulement si chaque classe a assez d’échantillons → fini l’erreur “test_size = 1 …”.
- **SHAP**: on passe par **`maskers.Text()`** + une fonction proba qui **nettoie** les textes avant de les donner au modèle (évite les `zero-size array`), et on encadre d’un `try/except` propre.
- **CamemBERT**: sauvegarde `(tokenizer, camembert, clf)` et fonction d’inférence dédi
