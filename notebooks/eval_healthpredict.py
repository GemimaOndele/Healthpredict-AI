A
import os, yaml, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "config.yaml"

def clean_text(s):
    import re, unicodedata
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("utf-8","ignore")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

with open(CFG,"r",encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

data_csv = cfg["paths"]["processed_csv"]
model_path = cfg["paths"]["tfidf_model"]

df = pd.read_csv(data_csv)
# Colonnes candidates pour le texte
for col in ["event_text","Texte","Description","Alerte","Notes","Commentaire"]:
    if col in df.columns:
        text_col = col
        break
else:
    text_col = df.columns[0]

# Label binaire (adapter si besoin)
if "label" in df.columns:
    y = df["label"].astype(int).values
else:
    # essai via colonne "Alerte" si présente ("Critique"/"Pas critique")
    if "Alerte" in df.columns:
        y = df["Alerte"].astype(str).str.startswith("Critique").astype(int).values
    else:
        raise SystemExit("Pas de label trouvé. Ajoute 'label' ou 'Alerte' dans le CSV.")

X_text = df[text_col].fillna("").astype(str).map(clean_text).tolist()

pipe = joblib.load(model_path)
proba = pipe.predict_proba(X_text)[:,1]
pred = (proba >= 0.5).astype(int)

acc = accuracy_score(y, pred)
f1 = f1_score(y, pred)
try:
    auc = roc_auc_score(y, proba)
except Exception:
    auc = float("nan")

print(f"Accuracy: {acc:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
cm = confusion_matrix(y, pred)
print("Confusion matrix:\n", cm)

# ROC
RocCurveDisplay.from_predictions(y, proba)
plt.title("ROC - HealthPredict TF-IDF")
plt.show()
# Export résultats
results = pd.DataFrame({    
    "text": X_text,
    "proba": proba,
    "pred": pred,
    "label": y
})