# scripts/validate_dataset.py
import os, sys, json, re
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG  = ROOT / "config" / "config.yaml"
REQ_COLS = {"event_text", "label"}

def load_cfg():
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # chemins absolus
    for k,v in list(cfg.get("paths", {}).items()):
        if not os.path.isabs(v):
            cfg["paths"][k] = str((ROOT / v).resolve())
    return cfg

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def ok(msg):
    print(f"[OK] {msg}")

def main():
    if not CFG.exists():
        fail(f"Config introuvable: {CFG}")
    cfg = load_cfg()
    csv_path = cfg["paths"].get("processed_csv") or str(ROOT/"assets/data/processed/medical_imaging_text_labeled.csv")
    p = Path(csv_path)
    if not p.exists():
        fail(f"CSV introuvable: {p}")

    try:
        df = pd.read_csv(p)
    except UnicodeDecodeError:
        fail("Encodage non UTF-8 détecté. Re-sauvegarder en UTF-8 (sans BOM).")

    # Colonnes obligatoires
    if not REQ_COLS.issubset(df.columns):
        fail(f"Colonnes manquantes. Requis: {REQ_COLS}, trouvées: {set(df.columns)}")

    # Vides / NaN
    n_nan_text = df["event_text"].isna().sum()
    if n_nan_text > 0:
        fail(f"`event_text` contient {n_nan_text} valeurs vides/NaN.")

    # Label binaire
    bad_label = (~df["label"].isin([0,1])).sum()
    if bad_label > 0:
        fail(f"`label` contient {bad_label} valeurs non binaires (attendu 0/1).")

    # Duplicats
    n_dup = df.duplicated(subset=["event_text"]).sum()
    print(f"[INFO] Duplicats (event_text): {n_dup}")

    # Stats simples
    n = len(df)
    lens = df["event_text"].astype(str).str.len()
    print(json.dumps({
        "rows": int(n),
        "len_min": int(lens.min()),
        "len_p50": int(lens.quantile(0.5)),
        "len_p95": int(lens.quantile(0.95)),
        "len_max": int(lens.max()),
        "label_ratio": float(df["label"].mean()),
    }, indent=2))

    ok("Validation dataset réussie.")
    sys.exit(0)

if __name__ == "__main__":
    main()
