# -*- coding: utf-8 -*-
import os, re, unicodedata, yaml, pandas as pd
from pathlib import Path

CONFIG = "config/config.yaml"

CRIT_PATTERNS = [
    r"\b(fire|smoke|burn|shock|overheat|short\s?circuit|injur(y|ies)|death|serious)\b",
    r"\b(failure|shutdown|stopp?ed|crash|alarm)\b",
    r"\b(hazard|risk|exposure)\b",
    # FR
    r"\b(incendie|fumée|brûl\w+|choc|surchauffe|court\s?circuit|blessure|décès|grave)\b",
    r"\b(panne|arr[êe]t|stop|crash|alarme)\b",
    r"\b(danger|risque|exposition)\b",
]

def load_cfg():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("utf-8","ignore")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_critical(cleaned: str) -> int:
    for pat in CRIT_PATTERNS:
        if re.search(pat, cleaned):
            return 1
    return 0

def main():
    cfg = load_cfg()
    raw_csv = cfg["paths"]["raw_csv"]
    out_csv = cfg["paths"]["processed_csv"]
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"{raw_csv} introuvable. Lance d'abord 01_download_openfda.py")

    df = pd.read_csv(raw_csv)
    # texte source
    df["text"] = df["event_text"].fillna("").astype(str)
    # nettoyage
    df["text_clean"] = df["text"].apply(clean_text)
    # label
    df["label_critique"] = df["text_clean"].apply(is_critical)

    # garder colonnes utiles
    keep = ["date_received","event_type","brand_name","generic_name","text","text_clean","label_critique"]
    df[keep].to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] {len(df)} lignes préparées -> {out_csv}")

if __name__ == "__main__":
    main()
