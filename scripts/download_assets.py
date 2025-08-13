# scripts/download_assets.py
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Gkop/healthpredict-assets"
REPO_TYPE = "dataset"  # IMPORTANT
TOKEN = os.environ.get("HF_TOKEN")

# 1ère valeur = chemin du fichier DANS le repo HF (relative path)
# 2ème valeur = chemin local où tu veux l'écrire dans ton projet
FILES = [
    ("healthpredict_model.joblib",            "assets/models/healthpredict_model.joblib"),
    ("healthpredict_camembert_model.joblib",  "assets/models/healthpredict_camembert_model.joblib"),
    ("raw_openfda_imaging_reports.csv",      "assets/data/raw/raw_openfda_imaging_reports.csv"),
]

def main():
    for filename_in_repo, dest_path in FILES:
        # Sécurité: empêcher une URL par erreur en destination
        if "://" in dest_path:
            raise ValueError(f"dest_path must be a local path, not a URL: {dest_path}")

        dst = Path(dest_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[hf] {REPO_ID} [{REPO_TYPE}] : {filename_in_repo} -> {dst}")

        try:
            fp = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename_in_repo,   # nom/chemin RELATIF dans le repo HF
                repo_type=REPO_TYPE,
                revision="main",
                token=TOKEN,                 # requis si le repo est privé
            )
            Path(fp).replace(dst)
            print(f"[ok] {dst} téléchargé depuis Hugging Face.")
        except Exception as e:
            print("[ERROR] download_assets:", e)
            raise

if __name__ == "__main__":
    main()
