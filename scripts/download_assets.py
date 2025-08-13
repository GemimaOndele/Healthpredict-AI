# scripts/download_assets.py
import os, sys, json, shutil, pathlib
from typing import List, Dict

try:
    import requests
except Exception:
    requests = None

ASSETS_DIR = pathlib.Path(os.environ.get("HP_ASSETS_DIR", "assets"))

def _ensure_parent(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _exists_nonempty(p: pathlib.Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def _download_http(url: str, dest: pathlib.Path):
    if requests is None:
        raise RuntimeError("Le package 'requests' est requis (ajoute-le dans requirements.txt).")
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
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk); done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    print(f"\r... {dest.name}: {pct}% ({done}/{total} bytes)", end="")
    print()

def _download_hf(repo_id: str, filename: str, dest: pathlib.Path, revision: str | None = None, token: str | None = None):
    from huggingface_hub import hf_hub_download
    _ensure_parent(dest)
    if _exists_nonempty(dest) and os.environ.get("HP_REDOWNLOAD", "0") != "1":
        print(f"[skip] {dest} existe déjà")
        return
    print(f"[hf] {repo_id}:{filename} -> {dest}")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)
    shutil.copy2(cached, dest)

def _download_gdrive(file_id: str, dest: pathlib.Path):
    import gdown
    _ensure_parent(dest)
    if _exists_nonempty(dest) and os.environ.get("HP_REDOWNLOAD", "0") != "1":
        print(f"[skip] {dest} existe déjà")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[gdrive] {file_id} -> {dest}")
    gdown.download(url, str(dest), quiet=False)

def parse_specs(raw: str) -> List[Dict]:
    """
    HP_ASSETS_SPECS = JSON ou liste séparée par ';'
    Formats acceptés (utiliser des chemins relatifs pour 'dest'):
      1) URL direct:
         {"type":"http","url":"https://.../model.joblib","dest":"assets/models/model.joblib"}
      2) Hugging Face:
         {"type":"hf","repo":"owner/repo","file":"model.joblib","dest":"assets/models/model.joblib","revision":"main"}
      3) Google Drive:
         {"type":"gdrive","id":"<FILE_ID>","dest":"assets/data/raw.csv"}
    Variante «liste courte» séparée par ';' :
      "https://.../a.joblib|assets/models/a.joblib;hf://owner/repo|model.joblib|assets/models/a.joblib;gdrive://FILEID|assets/data/raw.csv"
    """
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("{") or raw.startswith("["):
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    # parse format court "x|y;..."
    specs = []
    for item in [s.strip() for s in raw.split(";") if s.strip()]:
        parts = item.split("|")
        if item.startswith("hf://"):
            # hf://owner/repo|filename|dest(|revision)
            repo = item.split("hf://",1)[1].split("|")[0]
            filename = parts[1]
            dest = parts[2]
            rev = parts[3] if len(parts) > 3 else None
            specs.append({"type":"hf","repo":repo,"file":filename,"dest":dest,"revision":rev})
        elif item.startswith("gdrive://"):
            fid = item.split("gdrive://",1)[1].split("|")[0]
            dest = parts[1]
            specs.append({"type":"gdrive","id":fid,"dest":dest})
        else:
            # http(s) url|dest
            url = parts[0]; dest = parts[1]
            specs.append({"type":"http","url":url,"dest":dest})
    return specs

def ensure_assets():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    raw = os.environ.get("HP_ASSETS_SPECS","")
    if not raw:
        print("[info] HP_ASSETS_SPECS non défini. Rien à télécharger.")
        return
    specs = parse_specs(raw)
    token = os.environ.get("HF_TOKEN")
    for s in specs:
        dest = pathlib.Path(s["dest"])
        if s["type"] == "http":
            _download_http(s["url"], dest)
        elif s["type"] == "hf":
            _download_hf(s["repo"], s["file"], dest, revision=s.get("revision"), token=token)
        elif s["type"] == "gdrive":
            _download_gdrive(s["id"], dest)
        else:
            print(f"[warn] Type inconnu: {s}")
    print("[ok] Assets prêts.")

if __name__ == "__main__":
    try:
        ensure_assets()
    except Exception as e:
        print(f"[ERROR] download_assets: {e}", file=sys.stderr)
        sys.exit(1)
