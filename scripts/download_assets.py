import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID   = "Gkop/healthpredict-assets"
REPO_TYPE = "dataset"
TOKEN     = os.environ.get("HF_TOKEN")

ASSETS_DIR = Path("assets")
(ASSETS_DIR / "models").mkdir(parents=True, exist_ok=True)
(ASSETS_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)

DOWNLOAD_CAMEMBERT = os.getenv("HP_DOWNLOAD_CAMEMBERT", "0") == "1"

FILES = [
    ("healthpredict_model.joblib",      ASSETS_DIR / "models" / "healthpredict_model.joblib"),
    ("raw_openfda_imaging_reports.csv", ASSETS_DIR / "data" / "raw" / "raw_openfda_imaging_reports.csv"),
]
if DOWNLOAD_CAMEMBERT:
    FILES.append(("healthpredict_camembert_model.joblib", ASSETS_DIR / "models" / "healthpredict_camembert_model.joblib"))

def _download(remote_name: str, dst: Path):
    print(f"[hf] {REPO_ID} [{REPO_TYPE}] : {remote_name} -> {dst}")
    fp = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote_name,
        revision="main",
        token=TOKEN,
    )
    Path(fp).replace(dst)
    print("[ok]", dst, "téléchargé depuis Hugging Face.")

def ensure_assets():
    for remote_name, dst in FILES:
        if dst.exists():
            print("[skip] déjà présent:", dst)
            continue
        _download(remote_name, dst)

if __name__ == "__main__":
    ensure_assets()
