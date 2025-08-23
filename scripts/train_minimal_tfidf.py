# scripts/train_minimal_tfidf.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, sys, joblib, yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # pour "import scripts.build_processed_csv" peu importe d'où on lance

CFG_PATH = ROOT / "config" / "config.yaml"
DEFAULTS = {
    "paths": {
        "processed_csv": str(ROOT / "assets" / "data" / "processed" / "medical_imaging_text_labeled.csv"),
        "tfidf_model":   str(ROOT / "assets" / "models" / "healthpredict_model.joblib"),
    },
    "training": {
        "test_size": 0.2,
        "random_state": 42
    }
}

def load_cfg():
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    # merge simples
    for k,v in DEFAULTS.items():
        cfg.setdefault(k, v)
        if isinstance(v, dict):
            for kk,vv in v.items():
                cfg[k].setdefault(kk, vv)
    # to abs paths
    for k,v in list(cfg["paths"].items()):
        cfg["paths"][k] = str(Path(v)) if os.path.isabs(v) else str((ROOT / v).resolve())
    return cfg

def clean_text(s: str) -> str:
    import unicodedata
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("utf-8","ignore")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_processed_csv(path_csv: str) -> str:
    p = Path(path_csv)
    if p.exists():
        return str(p)
    # build si absent
    from scripts.build_processed_csv import build_processed
    return build_processed(force=True)

def main():
    cfg = load_cfg()
    processed_csv = ensure_processed_csv(cfg["paths"]["processed_csv"])
    model_path = cfg["paths"]["tfidf_model"]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_csv)
    if not {"event_text","label"} <= set(df.columns):
        raise SystemExit("Le processed CSV doit contenir 'event_text' et 'label'.")

    X = df["event_text"].fillna("").astype(str).map(clean_text).tolist()
    y = df["label"].astype(int).values

    # split (stratify si possible)
    strat = y if len(set(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None))
    ])
    pipe.fit(Xtr, ytr)

    joblib.dump(pipe, model_path)
    print(f"[OK] Modèle sauvé → {model_path}")

if __name__ == "__main__":
    main()
