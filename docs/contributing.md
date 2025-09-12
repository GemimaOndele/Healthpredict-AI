# Contribuer à HealthPredict AI

## Pré-requis
- Python 3.11/3.12
- `pip install -r requirements.txt`
- Tests: `pytest -q` (doit passer)

## Style & lint
- `pip install ruff mypy`
- `ruff check .`
- `mypy api/ app/ scripts/ hpdb.py --ignore-missing-imports`

## Branches & PR
- Feature branch → PR → CI verte (tests + lint).
- Décrire: scope, data impact, sécurité.

## Données
- Ne commitez jamais de données personnelles.
- Les artefacts lourds → Hugging Face (voir `scripts/upload_assets.py`).
