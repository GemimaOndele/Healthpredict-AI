# scripts/build_processed_csv.py
import os, re, json, yaml
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG  = ROOT / "config" / "config.yaml"

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _ensure_str(x):
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def _pick_text_col(df: pd.DataFrame) -> str:
    # colonnes fréquentes OpenFDA / tes données
    candidates = [
        "event_text","mdr_text","narrative","description_of_event",
        "event_description","summary","Texte","Description","Alerte","Notes","Commentaire"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # sinon, on concatène toutes les colonnes texte
    obj = [c for c in df.columns if df[c].dtype == "object"]
    if obj:
        df["_concat_text"] = df[obj].astype(str).fillna("").agg(" ".join, axis=1)
        return "_concat_text"
    return df.columns[0]

def _heuristic_label(df: pd.DataFrame, text_col: str) -> pd.Series:
    # 1) si “label” existe déjà
    if "label" in df.columns:
        s = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        return s.clip(0,1)

    # 2) si “Alerte” existe (Critique / Pas critique)
    if "Alerte" in df.columns:
        return df["Alerte"].astype(str).str.lower().str.startswith("critique").astype(int)

    # 3) si “event_type” existe (Death/Injury/Malfunction => 1, sinon 0)
    if "event_type" in df.columns:
        crit = {"death","injury","malfunction"}
        return df["event_type"].astype(str).str.lower().isin(crit).astype(int)

    # 4) fallback mots-clés
    text = df[text_col].astype(str).str.lower().fillna("")
    kw_pos = [
        "death","fatal","injury","serious","critical","fire","smoke","burn",
        "electric shock","overheat","leak","failure","broken","hazard","risk"
    ]
    pat = r"(" + "|".join(map(re.escape, kw_pos)) + r")"
    return text.str.contains(pat, regex=True, na=False).astype(int)

def build_processed():
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_csv = cfg["paths"].get("raw_csv", str(ROOT / "assets" / "data" / "raw" / "raw_openfda_imaging_reports.csv"))
    out_csv = cfg["paths"].get("processed_csv", str(ROOT / "assets" / "data" / "processed" / "medical_imaging_text_labeled.csv"))
    raw_csv = str((ROOT / raw_csv).resolve()) if not os.path.isabs(raw_csv) else raw_csv
    out_csv = str((ROOT / out_csv).resolve()) if not os.path.isabs(out_csv) else out_csv

    if not os.path.exists(raw_csv):
        raise SystemExit(f"RAW CSV introuvable: {raw_csv}")

    df = pd.read_csv(raw_csv)
    if df.empty:
        raise SystemExit("Le RAW CSV est vide.")

    # colonne texte
    text_col = _pick_text_col(df)
    df["event_text"] = df[text_col].astype(str).fillna("")

    # label
    df["label"] = _heuristic_label(df, "event_text").astype(int)
    df["Alerte"] = np.where(df["label"] == 1, "Critique", "Pas critique")

    # colonnes utiles si dispo
    keep = ["event_text","label","Alerte","event_type","date_received","brand_name","generic_name"]
    keep = [c for c in keep if c in df.columns] + ["event_text","label","Alerte"]
    keep = list(dict.fromkeys(keep))  # unique & order

    out = df[keep].copy()

    out_path = Path(out_csv)
    _ensure_dir(out_path)
    out.to_csv(out_path, index=False, encoding="utf-8")
    crit = int(out["label"].sum())
    print(f"[OK] Processed sauvegardé → {out_path}")
    print(f"     Lignes: {len(out)} | Critiques: {crit} | Non-critiques: {len(out)-crit}")

if __name__ == "__main__":
    build_processed()
