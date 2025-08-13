# upload_assets.py
# -*- coding: utf-8 -*-
"""
Uploader local -> Hugging Face (repo "dataset").

Ce script envoie tes fichiers locaux vers le dépôt HF "Gkop/healthpredict-assets".
Il fait 1 seul commit avec les 3 fichiers suivants :
  - models/healthpredict_model.joblib                 -> healthpredict_model.joblib
  - models/healthpredict_camembert_model.joblib       -> healthpredict_camembert_model.joblib
  - data/raw/raw_openfda_imaging_reports.csv          -> raw_openfda_imaging_reports.csv

Prérequis (déjà OK chez toi) :
  pip install -U "huggingface_hub[cli]" hf_transfer
  hf auth login   # (ou définir HF_TOKEN dans l'environnement)

Exécution :
  python upload_assets.py
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, CommitOperationAdd

# --- Config dépôt HF ---
REPO_ID = "Gkop/healthpredict-assets"
REPO_TYPE = "dataset"     # IMPORTANT: ton repo est de type "dataset"
PRIVATE = False           # mets True si tu veux le passer en privé

# --- Fichiers à envoyer : (chemin_local -> chemin_dans_le_repo) ---
FILES: List[Tuple[str, str]] = [
    ("models/healthpredict_model.joblib",            "healthpredict_model.joblib"),
    ("models/healthpredict_camembert_model.joblib",  "healthpredict_camembert_model.joblib"),
    ("data/raw/raw_openfda_imaging_reports.csv",     "raw_openfda_imaging_reports.csv"),
]

def main():
    # Accélère les gros uploads si hf_transfer est installé
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # On se base sur l’emplacement de ce script pour résoudre les chemins
    root = Path(__file__).resolve().parent
    api = HfApi()

    # 1) Créer le repo s’il n’existe pas (sinon ne fait rien)
    print(f"[info] Préparation du dépôt: {REPO_ID} ({REPO_TYPE})")
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=PRIVATE, exist_ok=True)

    # 2) Préparer les opérations d’upload
    ops: List[CommitOperationAdd] = []
    total_bytes = 0

    for local, dest in FILES:
        p = (root / local).resolve()
        if not p.exists():
            print(f"[warn] Fichier introuvable en local: {p} — ignoré")
            continue
        size_mb = p.stat().st_size / (1024 * 1024)
        total_bytes += p.stat().st_size
        print(f"[add] {p}  ->  {dest}  ({size_mb:.1f} MB)")
        ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)))

    if not ops:
        print("[error] Aucun fichier trouvé. Vérifie FILES et les chemins locaux.")
        return

    # 3) Commit unique avec tous les fichiers
    total_mb = total_bytes / (1024 * 1024)
    print(f"[commit] Envoi de {len(ops)} fichier(s), total ≈ {total_mb:.1f} MB …")
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=ops,
        commit_message="Upload assets (models + raw CSV) via upload_assets.py",
    )

    print("[ok] Upload terminé.")
    print(f"[url] {commit.commit_url}")

if __name__ == "__main__":
    main()
