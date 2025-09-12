# Security Policy – HealthPredict AI

## Supported Versions
- Main branch (latest) – security fixes only via PR.

## Reporting a Vulnerability
- Contact: gemimakerenondelepourou@gmail.com
- Ne publiez aucun POC exploitant des données réelles. Donnez un scénario minimal reproductible.

## Surface d’attaque
- App Streamlit (localhost:8501)
- API FastAPI (localhost:8000) protégée par `X-API-Key`.

## Bonnes pratiques
- Ne jamais uploader de données sensibles sur GitHub.
- Variables secrètes via `.env` (jamais commit).
- Limiter l’upload de fichiers à 200 MB (déjà dans l’app).
- Désactiver `USE_CAMEMBERT` si manque de ressources pour éviter OOM.

## Correctifs rapides
- Révoquer clés (`HF_TOKEN`, `HP_API_KEY`) en cas de fuite.
- Regénérer `app.db` si compromission.
