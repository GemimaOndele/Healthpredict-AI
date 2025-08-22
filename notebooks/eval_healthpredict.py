# notebooks/eval_healthpredict.py
import os, sys, yaml, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

# ----------- chemins de base -----------
ROOT = Path(__file__).resolve().parents[1]   # racine projet
CFG  = ROOT / "config" / "config.yaml"

def abs_path(p: str | os.PathLike) -> Path:
    p = str(p)
    return Path(p) if os.path.isabs(p) else (ROOT / p).resolve()

def clean_text(s):
    import re, unicodedata
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("utf-8","ignore")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------- charge config -----------
if not CFG.exists():
    sys.exit(f"config introuvable: {CFG}")

with open(CFG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ----------- résout chemins dataset -----------
c_paths = cfg.get("paths", {})
cand_csv = []
if c_paths.get("processed_csv"):
    cand_csv.append(abs_path(c_paths["processed_csv"]))
# fallback processed dans assets/
cand_csv.append(ROOT / "assets" / "data" / "processed" / "medical_imaging_text_labeled.csv")
# raw de config
if c_paths.get("raw_csv"):
    cand_csv.append(abs_path(c_paths["raw_csv"]))
# fallback raw dans assets/
cand_csv.append(ROOT / "assets" / "data" / "raw" / "raw_openfda_imaging_reports.csv")

data_csv = next((p for p in cand_csv if Path(p).exists()), None)

# si rien trouvé: essayer de télécharger via script
if data_csv is None:
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.download_assets import ensure_assets  # type: ignore
        print("Assets absents → téléchargement…")
        ensure_assets()
    except Exception as e:
        print(f"(info) Téléchargement auto impossible : {e}")
    # re-scan
    data_csv = next((p for p in cand_csv if Path(p).exists()), None)

if data_csv is None:
    print("❌ Aucune donnée trouvée.")
    print("   Lance d'abord:  python scripts/download_assets.py")
    print("   ou vérifie config.yaml > paths.processed_csv / paths.raw_csv")
    sys.exit(1)

data_csv = Path(data_csv)
print(f"✔ Dataset: {data_csv}")

# ----------- modèle TF-IDF -----------
cand_model = []
if c_paths.get("tfidf_model"):
    cand_model.append(abs_path(c_paths["tfidf_model"]))
cand_model.append(ROOT / "assets" / "models" / "healthpredict_model.joblib")

model_path = next((p for p in cand_model if Path(p).exists()), None)
if model_path is None:
    print("❌ Modèle TF-IDF introuvable. Lance: python scripts/download_assets.py")
    sys.exit(1)
model_path = Path(model_path)
print(f"✔ Modèle TF-IDF: {model_path}")

# ----------- charge dataset -----------
df = pd.read_csv(data_csv)
# colonne texte
for col in ["event_text","Texte","Description","Alerte","Notes","Commentaire"]:
    if col in df.columns:
        text_col = col
        break
else:
    text_col = df.columns[0]

# colonne label (1=Critique, 0=Pas critique)
if "label" in df.columns:
    y = df["label"].astype(int).values
elif "Alerte" in df.columns:
    y = df["Alerte"].astype(str).str.startswith("Critique").astype(int).values
else:
    sys.exit("❌ Pas de label détecté. Ajoute 'label' (0/1) ou 'Alerte' ('Critique' vs autre).")

# limiter RAM si gros fichier
max_rows = int(os.getenv("HP_EVAL_MAX_ROWS", "50000"))
if len(df) > max_rows:
    print(f"(info) Dataset tronqué à {max_rows} lignes sur {len(df)} pour l'évaluation.")
    df = df.head(max_rows).copy()
    y  = y[:max_rows]

X_text = df[text_col].fillna("").astype(str).map(clean_text).tolist()

# ----------- évalue TF-IDF -----------
pipe = joblib.load(model_path)
proba = pipe.predict_proba(X_text)[:, 1]
pred  = (proba >= 0.5).astype(int)

acc = accuracy_score(y, pred)
f1  = f1_score(y, pred)
try:
    auc = roc_auc_score(y, proba)
except Exception:
    auc = float("nan")

print(f"\n== TF-IDF ==")
print(f"Accuracy: {acc:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
cm = confusion_matrix(y, pred)
print("Confusion matrix:\n", cm)

# ROC → PNG
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.figure()
RocCurveDisplay.from_predictions(y, proba)
plt.title("ROC - HealthPredict TF-IDF")
roc_png = OUT_DIR / "roc_tfidf.png"
plt.savefig(roc_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"ROC sauvegardée: {roc_png}")

# résultats détaillés
res_df = pd.DataFrame({"text": X_text, "proba": proba, "pred": pred, "label": y})
res_csv = OUT_DIR / "eval_results_tfidf.csv"
res_df.to_csv(res_csv, index=False)
print(f"Résultats: {res_csv}")

# ----------- (optionnel) CamemBERT -----------
use_cam = os.getenv("HP_USE_CAMEMBERT", "0") == "1"
if use_cam:
    try:
        cand_cam = []
        if c_paths.get("camembert_model"):
            cand_cam.append(abs_path(c_paths["camembert_model"]))
        cand_cam.append(ROOT / "assets" / "models" / "healthpredict_camembert_model.joblib")
        cam_path = next((p for p in cand_cam if Path(p).exists()), None)
        if cam_path is None:
            print("(info) Modèle CamemBERT indisponible — skipping.")
        else:
            print(f"\n== CamemBERT ==")
            import torch  # noqa
            tok, cam, clf = joblib.load(cam_path)  # (tokenizer, camembert_model, clf)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cam = cam.to(device).eval()
            batch_size = int(os.getenv("HP_CAM_BATCH", "16"))
            probs = []
            for i in range(0, len(X_text), batch_size):
                seg = X_text[i:i+batch_size]
                enc = tok(seg, padding=True, truncation=True, return_tensors="pt")
                if device == "cuda":
                    enc = {k: v.cuda() for k,v in enc.items()}
                with torch.no_grad():
                    out = cam(**enc).last_hidden_state[:,0,:].detach().cpu().numpy()
                p = clf.predict_proba(out)[:,1]
                probs.append(p)
            proba_cam = np.concatenate(probs)
            pred_cam  = (proba_cam >= 0.5).astype(int)
            acc_c = accuracy_score(y, pred_cam); f1_c = f1_score(y, pred_cam)
            try: auc_c = roc_auc_score(y, proba_cam)
            except: auc_c = float("nan")
            print(f"Accuracy: {acc_c:.3f}  F1: {f1_c:.3f}  ROC-AUC: {auc_c:.3f}")
            plt.figure()
            RocCurveDisplay.from_predictions(y, proba_cam)
            plt.title("ROC - HealthPredict CamemBERT")
            roc2 = OUT_DIR / "roc_camembert.png"
            plt.savefig(roc2, dpi=120, bbox_inches="tight"); plt.close()
            print(f"ROC CamemBERT sauvegardée: {roc2}")
            res2 = OUT_DIR / "eval_results_camembert.csv"
            pd.DataFrame({"text": X_text, "proba": proba_cam, "pred": pred_cam, "label": y}).to_csv(res2, index=False)
            print(f"Résultats CamemBERT: {res2}")
    except Exception as e:
        print(f"(info) Évaluation CamemBERT ignorée: {e}")
