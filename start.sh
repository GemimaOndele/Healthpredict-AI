#!/usr/bin/env bash
set -e

# 1) Télécharge les assets (HF_TOKEN et HP_ASSETS_SPECS viennent des env)
python scripts/download_assets.py

# 2) Lance Streamlit
exec streamlit run app/healthpredict_app.py --server.address=0.0.0.0 --server.port=${PORT:-10000}
