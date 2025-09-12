# Sources & APIs

## OpenFDA - Device Event
- Endpoint : `https://api.fda.gov/device/event.json`
- Paramètres typiques :
  - `search`: mots clés (ex. radiology, ultrasound…)
  - `date_received`: filtre temporel (ex. `date_received:[20180101+TO+20250101]`)
  - `limit`: max 100 par requête
  - `skip`: pagination
- Script de collecte : `scripts/01_download_openfda.py` (paginé, sauvegarde CSV/JSONL).

## Hugging Face Hub
- Artefacts volumineux (modèles / datasets) publiés sur un repo dataset : `Gkop/healthpredict-assets`.
- Téléchargement côté app via `scripts/download_assets.py` (variable `HP_DOWNLOAD_CAMEMBERT=1` pour inclure le modèle CamemBERT).
