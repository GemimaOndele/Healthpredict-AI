# scripts/download_assets.py
# -*- coding: utf-8 -*-
"""
Télécharger les modèles/données au runtime dans un dossier non versionné (ex: ./assets/).

Variables d'environnement principales :
- HP_ASSETS_DIR   : dossier racine des assets (défaut: "assets")
- HP_ASSETS_SPECS : specs JSON OU liste courte séparée par ';'
- HF_TOKEN / HP_HF_TOKEN : token Hugging Face si repo privé
- HP_REDOWNLOAD   : "1" pour re-télécharger même si le fichier existe

Formats acceptés pour HP_ASSETS_SPECS :
1) JSON (liste d'objets) :
   [
     {"type":"http","url":"https://.../a.joblib","dest":"assets/models/a.joblib"},
     {"type":"hf","repo":"owner/repo","file":"a.joblib","dest":"assets/models/a.joblib","revision":"main","repo_type":"dataset"},
     {"type":"gdrive","id":"FILEID","dest":"assets/data/raw.csv"}
   ]

2) Format court "x|y;..." :
   - HTTP(S)      : https://.../a.joblib|assets/models/a.joblib
   - HF (4 ou 5 champs) :
       hf://owner/repo|filename|dest|revision
       hf://owner/repo|filename|dest|revision|repo_type   # repo_type par défaut = "dataset"
   - Google Drive : gdrive://FILEID|assets/data/raw.csv
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import pathlib
from typing import List, Dict, Any, Optional

# (optionnel) .env en local
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# "requests" et "gdown" sont optionnels (HTTP et Google Drive)
try:
    import requests  # pour HTTP direct
except Exception:
    requests = None  # type: ignore

ASSETS_DIR = pathlib.Path(os.environ.get("HP_ASSETS_DIR", "assets")).resolve()


# ---------------------------
# Helpers fichiers / dossiers
# ---------------------------
def _ensure_parent(p: pathlib.Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _exists_nonempty(p: pathlib.Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _is_within_assets(dest: pathlib.Path) -> bool:
    """Empêche une destination d'échapper à ASSETS_DIR (sécurité basique, robuste)."""
    try:
        dest_resolved = dest.resolve()
        ASSETS_DIR.resolve()  # s'assure que ça ne lève pas
        # Utilise relative_to pour éviter les faux positifs (ex: assets2)
        dest_resolved.relative_to(ASSETS_DIR)
        return True
    except Exception:
        return False


# ---------------------------
# Téléchargements
# ---------------------------
def _download_http(url: str, dest: pathlib.Path) -> None:
    if requests is None:
        raise RuntimeError("Le package 'requests' est requis pour HTTP. Ajoute-le à requirements.txt.")
    _ensure_parent(dest)
    if _exists_nonempty(dest) and os.environ.get("HP_REDOWNLOAD", "0") != "1":
        print(f"[skip] {dest} existe déjà")
        return

    print(f"[http] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(4 * 1024 * 1024):  # 4 MB
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    print(f"\r... {dest.name}: {pct}% ({done}/{total} bytes)", end="")
    print()


def _download_hf(
    repo_id: str,
    filename: str,
    dest: pathlib.Path,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    repo_type: str = "dataset",
) -> None:
    from huggingface_hub import hf_hub_download

    # normalise repo_type
    repo_type = (repo_type or "dataset").lower().strip()
    if repo_type not in {"dataset", "model", "space"}:
        repo_type = "dataset"

    _ensure_parent(dest)
    if _exists_nonempty(dest) and os.environ.get("HP_REDOWNLOAD", "0") != "1":
        print(f"[skip] {dest} existe déjà")
        return

    print(f"[hf] {repo_id} [{repo_type}] : {filename} -> {dest}")
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
        repo_type=repo_type,
    )
    shutil.copy2(cached, dest)
    print(f"[ok] {dest} téléchargé depuis Hugging Face.")


def _download_gdrive(file_id: str, dest: pathlib.Path) -> None:
    try:
        import gdown  # type: ignore
    except Exception:
        raise RuntimeError("Le package 'gdown' est requis pour Google Drive. Ajoute-le à requirements.txt.")

    _ensure_parent(dest)
    if _exists_nonempty(dest) and os.environ.get("HP_REDOWNLOAD", "0") != "1":
        print(f"[skip] {dest} existe déjà")
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[gdrive] {file_id} -> {dest}")
    gdown.download(url, str(dest), quiet=False)


# ---------------------------
# Parsing HP_ASSETS_SPECS
# ---------------------------
def parse_specs(raw: str) -> List[Dict[str, Any]]:
    """
    Retourne une liste de dicts normalisés:
    { "type": "http"|"hf"|"gdrive", "...", "dest": "<path>", ... }
    """
    if not raw:
        return []

    raw = raw.strip()
    if raw.startswith("{") or raw.startswith("["):
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]

    specs: List[Dict[str, Any]] = []
    for item in [s.strip() for s in raw.split(";") if s.strip()]:
        parts = [p.strip() for p in item.split("|")]
        if item.startswith("hf://"):
            # hf://owner/repo|filename|dest(|revision)(|repo_type)
            if len(parts) < 3:
                print(f"[warn] Spéc '{item}' invalide (attendu au moins 3 champs). Ignoré.")
                continue
            repo = item.split("hf://", 1)[1].split("|")[0].strip()
            filename = parts[1]
            dest = parts[2]
            rev = parts[3] if len(parts) > 3 and parts[3] else None
            rtype = parts[4] if len(parts) > 4 and parts[4] else "dataset"
            specs.append(
                {
                    "type": "hf",
                    "repo": repo,
                    "file": filename,
                    "dest": dest,
                    "revision": rev,
                    "repo_type": rtype,
                }
            )
        elif item.startswith("gdrive://"):
            # gdrive://FILEID|dest
            if len(parts) < 2:
                print(f"[warn] Spéc '{item}' invalide (gdrive nécessite 2 champs). Ignoré.")
                continue
            fid = item.split("gdrive://", 1)[1].split("|")[0].strip()
            dest = parts[1]
            specs.append({"type": "gdrive", "id": fid, "dest": dest})
        else:
            # http(s) url|dest
            if len(parts) < 2:
                print(f"[warn] Spéc '{item}' invalide (http nécessite 2 champs). Ignoré.")
                continue
            url = parts[0]
            dest = parts[1]
            specs.append({"type": "http", "url": url, "dest": dest})
    return specs


# ---------------------------
# Orchestrateur
# ---------------------------
def ensure_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    raw = os.environ.get("HP_ASSETS_SPECS", "")
    if not raw:
        print("[info] HP_ASSETS_SPECS non défini. Rien à télécharger.")
        return

    specs = parse_specs(raw)
    # Token HF : accepte HF_TOKEN ou HP_HF_TOKEN
    token = os.environ.get("HF_TOKEN") or os.environ.get("HP_HF_TOKEN")

    print(f"[cfg] ASSETS_DIR = {ASSETS_DIR}")
    print(f"[cfg] Nb. specs = {len(specs)}")

    for s in specs:
        # normalise la destination
        dest = pathlib.Path(s["dest"])
        if not dest.is_absolute():
            dest = (pathlib.Path(".") / dest).resolve()

        # sécurité basique : rester sous ASSETS_DIR
        if not _is_within_assets(dest):
            print(f"[warn] Destination hors de {ASSETS_DIR}: {dest}. Je continue quand même.")
            _ensure_parent(dest)

        try:
            if s["type"] == "http":
                _download_http(s["url"], dest)
            elif s["type"] == "hf":
                _download_hf(
                    s["repo"],
                    s["file"],
                    dest,
                    revision=s.get("revision"),
                    token=token,
                    repo_type=(s.get("repo_type") or "dataset"),
                )
            elif s["type"] == "gdrive":
                _download_gdrive(s["id"], dest)
            else:
                print(f"[warn] Type inconnu: {s}")
        except Exception as e:
            print(f"[ERROR] Échec pour {s.get('dest','?')}: {e}", file=sys.stderr)

    print("[ok] Assets prêts.")


# ---------------------------
# Entrée CLI
# ---------------------------
if __name__ == "__main__":
    try:
        ensure_assets()
    except Exception as e:
        print(f"[FATAL] download_assets: {e}", file=sys.stderr)
        sys.exit(1)
    commit_message="Upload assets",
    print("[info] download_assets.py terminé.")