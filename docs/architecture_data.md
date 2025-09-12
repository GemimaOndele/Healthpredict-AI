# Pipeline Données (vue rapide)

1. **Collecte**  
   `scripts/01_download_openfda.py` → `assets/data/raw/*.csv|.jsonl`

2. **Préparation & étiquetage**  
   `scripts/build_processed_csv.py` → `assets/data/processed/medical_imaging_text_labeled.csv`

3. **Validation**  
   `scripts/validate_dataset.py` (colonnes, encodage, duplicats, stats)

4. **Entraînement**  
   - `scripts/train_minimal_tfidf.py` → `assets/models/healthpredict_model.joblib`
   - `scripts/train_camembert_baseline.py` → `assets/models/healthpredict_camembert_model.joblib`

5. **Évaluation**  
   `notebooks/eval_healthpredict.py` → `assets/eval/*`

6. **Application**  
   `app/healthpredict_app.py` (Streamlit) + SQLite (`data/app.db`)

# En résumé : 
1. Collecte OpenFDA (`scripts/01_download_openfda.py`)
2. Préparation / labellisation (`scripts/build_processed_csv.py`)
3. Validation (`scripts/validate_dataset.py`)
4. Entraînement (`scripts/train_minimal_tfidf.py`, `scripts/train_camembert_baseline.py`)
5. Évaluation (notebook) → `assets/eval/*`
6. App & API consomment les artefacts (`assets/models/*.joblib`)