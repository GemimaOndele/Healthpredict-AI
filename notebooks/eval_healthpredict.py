# notebooks/eval_healthpredict.py
import os, json, yaml, joblib, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CFG  = ROOT / "config" / "config.yaml"
EVAL_DIR = ROOT / "assets" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(s: str) -> str:
    import unicodedata
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("utf-8","ignore")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

with open(CFG,"r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# chemins normalisés
for k, v in list(cfg.get("paths", {}).items()):
    if not os.path.isabs(v):
        cfg["paths"][k] = str((ROOT / v).resolve())

processed_csv = cfg["paths"].get("processed_csv", str(ROOT / "assets" / "data" / "processed" / "medical_imaging_text_labeled.csv"))
model_path    = cfg["paths"].get("tfidf_model", str(ROOT / "assets" / "models" / "healthpredict_model.joblib"))

processed_csv = str(Path(processed_csv))
model_path    = str(Path(model_path))

if not os.path.exists(processed_csv):
    print(f"CSV introuvable: {processed_csv}")
    # essaie de le générer
    try:
        from scripts.build_processed_csv import build_processed
        build_processed()
    except Exception as e:
        raise SystemExit(f"Impossible de générer le processed CSV ({e})")

if not os.path.exists(processed_csv):
    raise SystemExit(f"Toujours introuvable: {processed_csv}")

df = pd.read_csv(processed_csv)
if not {"event_text","label"} <= set(df.columns):
    raise SystemExit("Le processed CSV doit contenir 'event_text' et 'label'.")

X = df["event_text"].fillna("").astype(str).map(clean_text).tolist()
y = df["label"].astype(int).values

# split stable pour évaluer
test_size    = cfg.get("training",{}).get("test_size", 0.2)
random_state = cfg.get("training",{}).get("random_state", 42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None)

if not os.path.exists(model_path):
    raise SystemExit(f"Modèle introuvable: {model_path}. Lance d'abord scripts/train_minimal_tfidf.py")

pipe = joblib.load(model_path)
proba = pipe.predict_proba(Xte)[:,1]
pred  = (proba >= 0.5).astype(int)

acc = accuracy_score(yte, pred)
f1  = f1_score(yte, pred, zero_division=0)
try:
    auc = roc_auc_score(yte, proba)
except Exception:
    auc = float("nan")

print(f"Accuracy: {acc:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
cm = confusion_matrix(yte, pred)
print("Confusion matrix:\n", cm)

# === Figures ===
plt.figure()
RocCurveDisplay.from_predictions(yte, proba)
plt.title("ROC - HealthPredict TF-IDF")
plt.savefig(EVAL_DIR / "roc.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
PrecisionRecallDisplay.from_predictions(yte, proba)
plt.title("Precision-Recall - HealthPredict TF-IDF")
plt.savefig(EVAL_DIR / "pr.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
import seaborn as sns  # juste pour la heatmap
sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues")
plt.xlabel("Prédit"); plt.ylabel("Réel"); plt.title("Matrice de confusion")
plt.savefig(EVAL_DIR / "cm.png", dpi=150, bbox_inches="tight")
plt.close()

# Répartition (bar/pie) pour l’interprétation
if "event_type" in df.columns:
    cnt = df["event_type"].astype(str).value_counts().head(15)
    plt.figure(figsize=(7,4))
    cnt.plot(kind="bar")
    plt.title("Répartition event_type (top 15)")
    plt.ylabel("Nombre")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "event_type_bar.png", dpi=150)
    plt.close()

if "TypeDetecte" in df.columns:
    cnt2 = df["TypeDetecte"].astype(str).value_counts()
    plt.figure(figsize=(5,5))
    cnt2.plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel("")
    plt.title("Répartition TypeDetecte")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "typedetecte_pie.png", dpi=150)
    plt.close()

# === metrics.json ===
with open(EVAL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(auc),
        "n_samples": int(len(yte))
    }, f, ensure_ascii=False, indent=2)

print(f"[OK] Évaluation sauvegardée dans {EVAL_DIR}")
