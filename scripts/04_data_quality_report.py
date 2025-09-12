# scripts/04_data_quality_report.py
import pandas as pd, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV  = ROOT / "assets/data/processed/medical_imaging_text_labeled.csv"
OUTD = ROOT / "assets/eval"; OUTD.mkdir(parents=True, exist_ok=True)
OUTJ = OUTD / "data_quality.json"
OUTM = OUTD / "data_quality.md"

def md5(path): return hashlib.md5(path.read_bytes()).hexdigest()

df = pd.read_csv(CSV)
rep = {
  "rows": len(df),
  "cols": df.shape[1],
  "nulls_by_col": df.isna().sum().to_dict(),
  "dup_rows": int(df.duplicated().sum()),
  "class_balance": df["label"].value_counts(dropna=False).to_dict() if "label" in df else {},
  "top_event_type": df.get("event_type", pd.Series()).value_counts().head(10).to_dict(),
  "file_md5": md5(CSV)
}
OUTJ.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

# Version Markdown courte
OUTM.write_text(
  f"""# Data Quality — processed CSV
- Lignes: **{rep['rows']}** — Colonnes: **{rep['cols']}**
- Doublons: **{rep['dup_rows']}**
- Répartition label: **{rep.get('class_balance',{})}**
- Top event_type: **{rep.get('top_event_type',{})}**
- Nulls par colonne: `{rep['nulls_by_col']}`
- MD5: `{rep['file_md5']}`
""", encoding="utf-8")
print("[OK] Rapport qualité généré →", OUTM)
