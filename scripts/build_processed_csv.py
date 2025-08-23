# scripts/build_processed_csv.py
# -*- coding: utf-8 -*-
"""
Génère assets/data/processed/medical_imaging_text_labeled.csv
à partir de assets/data/raw/raw_openfda_imaging_reports.csv

Colonnes garanties:
- event_text (str)
- label (int: 1=critique, 0=pas critique)
- event_type (original si présent)
- date_received (si présent)

Règle de labellisation simple:
  label=1 si event_type ∈ {"Death","Injury"} ; sinon 0
"""

from __future__ import annotations
import os, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "assets" / "data" / "raw" / "raw_openfda_imaging_reports.csv"
PROCESSED_DIR = ROOT / "assets" / "data" / "processed"
PROCESSED = PROCESSED_DIR / "medical_imaging_text_labeled.csv"

TEXT_CANDIDATES = [
    "event_text", "event_description_text", "text", "narrative",
    "manufacturer_narrative", "description", "report_text", "summary",
    "detail", "long_text", "short_text"
]

def _ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def _first_non_empty(row) -> str:
    for c in TEXT_CANDIDATES:
        if c in row and isinstance(row[c], str):
            s = row[c].strip()
            if s:
                return s
    # fallback: concat de toutes les colonnes objet non vides (sécurité)
    parts = []
    for c in row.index:
        v = row[c]
        if isinstance(v, str):
            v = v.strip()
            if v:
                parts.append(v)
    return " ".join(parts)[:20000] if parts else ""

def build_processed(force: bool = True) -> str:
    if not RAW.exists():
        raise FileNotFoundError(f"RAW introuvable: {RAW}")

    df = pd.read_csv(RAW, low_memory=False)
    if df.empty:
        raise RuntimeError("RAW vide.")

    # Normalise quelques noms fréquents
    # (aucun prérequis strict, on essaye juste d'être robustes)
    cols_lower = {c: c.lower() for c in df.columns}
    df.rename(columns=cols_lower, inplace=True)

    # event_type (si existe)
    if "event_type" not in df.columns:
        # On crée event_type vide pour la suite
        df["event_type"] = ""

    # date_received (si existe)
    if "date_received" not in df.columns:
        # On essaie quelques variantes courantes
        for alt in ["date", "received_date", "report_date", "date_dt"]:
            if alt in df.columns:
                df["date_received"] = df[alt]
                break
        else:
            df["date_received"] = ""

    # event_text
    df["event_text"] = df.apply(_first_non_empty, axis=1)

    # label : 1 si Death/Injury, sinon 0
    def _label(et: str) -> int:
        et = (et or "").strip().lower()
        return 1 if et in {"death", "injury"} else 0

    df["label"] = df["event_type"].astype(str).map(_label).fillna(0).astype(int)

    # Filtre minimum: garder lignes avec event_text non vide
    df2 = df.loc[df["event_text"].astype(str).str.strip().ne("")].copy()

    # (Option) élagage colonnes pour rester léger
    keep = ["event_text", "label", "event_type", "date_received"]
    for c in keep:
        if c not in df2.columns:
            df2[c] = ""
    df2 = df2[keep]

    _ensure_dirs()
    if PROCESSED.exists() and not force:
        return str(PROCESSED)

    df2.to_csv(PROCESSED, index=False, encoding="utf-8")
    print(f"[OK] Processed CSV créé: {PROCESSED}  (lignes: {len(df2)})")
    return str(PROCESSED)

if __name__ == "__main__":
    build_processed(force=True)
