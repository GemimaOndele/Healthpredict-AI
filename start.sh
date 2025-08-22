#!/usr/bin/env bash
set -euo pipefail

# 1) Télécharge les assets (ne fait rien si déjà présents)
python -m scripts.download_assets || true

# 2) Exécute l'évaluation (optionnel : active via HP_RUN_EVAL=1)
if [[ "${HP_RUN_EVAL:-0}" == "1" ]]; then
  python notebooks/eval_healthpredict.py || true
fi

# 3) Lance Streamlit
exec streamlit run app/healthpredict_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}"
