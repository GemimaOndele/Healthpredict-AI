# scripts/train_camembert_baseline.py
from pathlib import Path
import os, json, yaml, re
import pandas as pd
import numpy as np
import joblib

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "config.yaml"
ASSETS = ROOT / "assets"
MODELS_DIR = ASSETS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def ensure_str(x):
    if isinstance(x, str): return x
    if x is None: return ""
    try:
        return str(x)
    except Exception:
        return ""

def pick_text_column(df: pd.DataFrame) -> str:
    for c in ["event_text", "Texte", "Description", "Alerte", "Notes", "Commentaire"]:
        if c in df.columns:
            return c
    obj_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    for c in obj_cols:
        if df[c].astype(str).str.len().gt(0).any():
            return c
    return df.columns[0] if len(df.columns) else "event_text"

def load_config():
    if CFG.exists():
        with open(CFG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"paths": {}}
    # normaliser chemins
    paths = cfg.get("paths", {})
    for k, v in list(paths.items()):
        if not os.path.isabs(v):
            paths[k] = str((ROOT / v).resolve())
    cfg["paths"] = paths
    return cfg

def load_dataset(cfg):
    p_proc = cfg["paths"].get("processed_csv", str(ASSETS / "data" / "processed" / "medical_imaging_text_labeled.csv"))
    p_raw  = cfg["paths"].get("raw_csv", str(ASSETS / "data" / "raw" / "raw_openfda_imaging_reports.csv"))

    if Path(p_proc).exists():
        df = pd.read_csv(p_proc)
    elif Path(p_raw).exists():
        df = pd.read_csv(p_raw)
    else:
        raise FileNotFoundError("Aucun CSV trouvé (processed ni raw).")

    # colonne de texte
    tcol = pick_text_column(df)
    df[tcol] = df[tcol].astype(str).fillna("")

    # label binaire : Mort/Blessure => 1 (Critique), sinon 0
    if "event_type" in df.columns:
        et = df["event_type"].astype(str).str.lower()
        y = et.isin({"death","injury"}).astype(int)
    else:
        # fallback simple si pas d'event_type: mots-clés dans le texte
        txt = df[tcol].str.lower()
        y = (txt.str.contains(r"\b(death|injur|hemorrhag|cardiac|respiratory|severe)\b")).astype(int)

    # on retire lignes vides
    mask = df[tcol].str.len() > 0
    df = df[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    return df, y, tcol

@torch.no_grad()
def embed_texts(texts, tok, model, batch_size=16, device="cpu", max_length=256):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state[:,0,:]  # [CLS]
        vecs.append(out.detach().cpu().numpy())
    return np.vstack(vecs)

def main():
    cfg = load_config()
    df, y, tcol = load_dataset(cfg)

    # échantillonnage pour aller vite la 1re fois
    NMAX = int(os.environ.get("HP_CAMEMBERT_TRAIN_MAX", "4000"))
    if len(df) > NMAX:
        df = df.sample(NMAX, random_state=42).reset_index(drop=True)
        y = y.loc[df.index].reset_index(drop=True)

    model_name = "camembert-base"
    tok = AutoTokenizer.from_pretrained(model_name)
    cam = AutoModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam = cam.to(device).eval()

    texts = df[tcol].astype(str).fillna("").tolist()
    X = embed_texts(texts, tok, cam, batch_size=16, device=device)

    Xtr, Xte, ytr, yte = train_test_split(X, y.values, test_size=0.2, random_state=42, stratify=y.values)
    clf = LogisticRegression(max_iter=2000, n_jobs=None).fit(Xtr, ytr)

    # petite éval console
    proba = clf.predict_proba(Xte)[:,1]
    pred  = (proba >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    f1  = f1_score(yte, pred)
    try:
        auc = roc_auc_score(yte, proba)
    except Exception:
        auc = float("nan")
    print(f"[Eval] acc={acc:.3f} f1={f1:.3f} auc={auc:.3f} (n={len(yte)})")

    out_path = MODELS_DIR / "healthpredict_camembert_model.joblib"
    joblib.dump((tok, cam, clf), out_path)
    print(f"✅ Modèle CamemBERT enregistré → {out_path}")

if __name__ == "__main__":
    main()
