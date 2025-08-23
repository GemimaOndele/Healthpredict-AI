# scripts/train_minimal_tfidf.py
# Entraînement minimal TF-IDF + LogReg et sauvegarde du pipeline joblib.

from __future__ import annotations
import os, re, unicodedata
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "assets" / "data" / "processed" / "medical_imaging_text_labeled.csv"
RAW = ROOT / "assets" / "data" / "raw" / "raw_openfda_imaging_reports.csv"
OUT_MODEL = ROOT / "assets" / "models" / "healthpredict_model.joblib"

def ensure_processed_csv():
    if PROCESSED.exists():
        return
    # tente de générer via notre builder local (aucune dépendance externe)
    from scripts.build_processed_csv import build_processed
    print("[info] Processed CSV manquant — génération…")
    build_processed()
    if not PROCESSED.exists():
        raise FileNotFoundError(f"Echec génération processed CSV: {PROCESSED}")

def clean_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8", "ignore")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_dataset():
    ensure_processed_csv()
    df = pd.read_csv(PROCESSED)
    # attend colonnes: Texte, Alerte
    if not {"Texte", "Alerte"}.issubset(df.columns):
        raise ValueError(f"Colonnes attendues manquantes dans {PROCESSED}")
    df = df.dropna(subset=["Texte", "Alerte"])
    # équilibrage simple (optionnel)
    if {"Critique", "Pas critique"}.issubset(set(df["Alerte"].unique())):
        # on limite pour éviter un gros déséquilibre
        m = min((df["Alerte"] == "Critique").sum(), (df["Alerte"] == "Pas critique").sum())
        if m > 0:
            df_pos = df[df["Alerte"] == "Critique"].sample(n=m, random_state=42)
            df_neg = df[df["Alerte"] == "Pas critique"].sample(n=m, random_state=42)
            df = pd.concat([df_pos, df_neg], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def build_pipeline():
    vec = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95
    )
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline([("tfidf", vec), ("clf", clf)])
    return pipe

if __name__ == "__main__":
    df = load_dataset()
    X = df["Texte"].astype(str).tolist()
    y = (df["Alerte"] == "Critique").astype(int).values  # 1=Critique, 0=Pas critique

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)

    print(classification_report(yte, yp, target_names=["Pas critique","Critique"]))

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, OUT_MODEL)
    print(f"[ok] Modèle TF-IDF sauvegardé → {OUT_MODEL}")
