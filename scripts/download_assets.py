# scripts/download_assets.py
# Télécharge les assets (modèles + données) depuis Hugging Face.
# - Ajoute l’option HP_DOWNLOAD_CAMEMBERT=1 pour récupérer le modèle CamemBERT.
# - Tolère les environnements proxy/SSL (messages d’aide).
# - Fallback possible: entraînement local si le download du CamemBERT échoue
#   en mettant HP_FALLBACK_TRAIN_CAMEMBERT=1.

from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("❌ huggingface_hub non installé. Faites: pip install huggingface_hub")
    raise

# --- Réglages généraux
ROOT       = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DATA_RAW   = ASSETS_DIR / "data" / "raw"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW.mkdir(parents=True, exist_ok=True)

REPO_ID   = os.environ.get("HP_ASSETS_REPO", "Gkop/healthpredict-assets")
REPO_TYPE = "dataset"
REVISION  = os.environ.get("HP_ASSETS_REV", "main")
TOKEN     = os.environ.get("HF_TOKEN")  # recommandé si repo privé

# Téléchargements optionnels
DOWNLOAD_CAMEMBERT = os.getenv("HP_DOWNLOAD_CAMEMBERT", "0") == "1"
FALLBACK_TRAIN     = os.getenv("HP_FALLBACK_TRAIN_CAMEMBERT", "0") == "1"

# Liste de base
FILES: list[tuple[str, Path]] = [
    ("healthpredict_model.joblib",      MODELS_DIR / "healthpredict_model.joblib"),
    ("raw_openfda_imaging_reports.csv", DATA_RAW   / "raw_openfda_imaging_reports.csv"),
]

# Ajout du CamemBERT si demandé
if DOWNLOAD_CAMEMBERT:
    FILES.append((
        "healthpredict_camembert_model.joblib",
        MODELS_DIR / "healthpredict_camembert_model.joblib"
    ))

def _hint_proxy_ssl():
    print(
        "\nℹ️ Si vous êtes derrière un proxy/SSL d’entreprise :\n"
        "   - Définissez éventuellement HTTPS_PROXY / HTTP_PROXY\n"
        "   - Spécifiez le certificat: REQUESTS_CA_BUNDLE=C:\\chemin\\monCA.pem\n"
        "   - Vérifiez aussi que votre HF_TOKEN est bien défini si le repo est privé\n"
    )

def _download(remote_name: str, dst: Path):
    print(f"[hf] {REPO_ID} [{REPO_TYPE}] : {remote_name} -> {dst}")
    try:
        # Télécharge dans le cache, on déplace ensuite
        fp = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=remote_name,
            revision=REVISION,
            token=TOKEN,
        )
        Path(fp).replace(dst)
        print("[ok]", dst, "téléchargé depuis Hugging Face.")
    except Exception as e:
        print(f"[err] Échec de téléchargement pour '{remote_name}': {e}")
        _hint_proxy_ssl()
        # Fallback: si c’est le modèle CamemBERT et que l’utilisateur l’autorise, on entraîne localement
        if remote_name == "healthpredict_camembert_model.joblib" and FALLBACK_TRAIN:
            print("↪️ Fallback local: entraînement CamemBERT…")
            try:
                sys.path.append(str(ROOT))
                from scripts.train_camembert_baseline import main as train_camembert
                train_camembert()
                if dst.exists():
                    print("[ok] Modèle CamemBERT créé localement :", dst)
                else:
                    print("[err] Entraînement terminé mais le fichier attendu n’existe pas :", dst)
            except Exception as e2:
                print("[err] Fallback entraînement CamemBERT a échoué :", e2)
                traceback.print_exc()
                raise
        else:
            raise

def ensure_assets():
    for remote_name, dst in FILES:
        if dst.exists():
            print("[skip] déjà présent:", dst)
            continue
        _download(remote_name, dst)

if __name__ == "__main__":
    ensure_assets()
