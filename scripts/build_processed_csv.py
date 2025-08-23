# scripts/build_processed_csv.py
# Génère un CSV "processed" minimal à partir du CSV brut OpenFDA.
# Sortie : assets/data/processed/medical_imaging_text_labeled.csv

from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
import unicodedata, re

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "assets" / "data" / "raw" / "raw_openfda_imaging_reports.csv"
OUT_DIR = ROOT / "assets" / "data" / "processed"
OUT_CSV = OUT_DIR / "medical_imaging_text_labeled.csv"

def ensure_str(x):
    if isinstance(x, str): return x
    if x is None: return ""
    try:
        return str(x)
    except Exception:
        return ""

def clean_text(text: str) -> str:
    text = ensure_str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pick_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "event_description", "event_text", "mdr_text", "narrative", "description",
        "summary", "text", "report_text", "notes", "comment", "comments"
    ]
    for c in candidates:
        if c in df.columns:
            if df[c].astype(str).str.len().gt(0).any():
                return c
    # fallback: concat de colonnes texte
    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        df["_concat_"] = df[obj_cols].astype(str).fillna("").agg(" ".join, axis=1)
        return "_concat_"
    # dernier recours
    return df.columns[0]

def build_processed():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"CSV brut introuvable: {RAW_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, low_memory=False)
    if df.empty:
        raise RuntimeError("Le CSV brut est vide.")

    tcol = pick_text_column(df)
    df["_txt_"] = df[tcol].astype(str).fillna("").str.strip()

    # Mapping binaire "Alerte" depuis event_type (si dispo)
    if "event_type" in df.columns:
        crit = df["event_type"].astype(str).str.lower().isin({"death", "injury"})
        df["_alerte_"] = crit.map({True: "Critique", False: "Pas critique"})
    else:
        # Sans colonne event_type -> tout en "Pas critique" (entraînement possible mais moins pertinent)
        df["_alerte_"] = "Pas critique"

    # Garde quelques colonnes utiles
    keep = ["_txt_", "_alerte_"]
    if "event_type" in df.columns: keep.append("event_type")
    if "date_received" in df.columns: keep.append("date_received")
    out = df[keep].rename(columns={"_txt_": "Texte", "_alerte_": "Alerte"}).copy()

    # Nettoyage basique + filtrage
    out["Texte"] = out["Texte"].astype(str).fillna("").map(lambda s: s.strip())
    out = out[out["Texte"].str.len() >= 5].drop_duplicates(subset=["Texte"])

    # Sauvegarde
    out.to_csv(OUT_CSV, index=False)
    print(f"[ok] Processed CSV écrit: {OUT_CSV} (lignes={len(out)})")

if __name__ == "__main__":
    build_processed()
