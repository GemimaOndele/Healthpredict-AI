# scripts/train_minimal_tfidf.py
import os, yaml, joblib, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
CFG  = ROOT / "config" / "config.yaml"

def clean_text(s: str) -> str:
    import unicodedata
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("utf-8","ignore")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ensure_processed_csv(cfg):
    from importlib import import_module
    out_csv = cfg["paths"]["processed_csv"]
    out_path = (ROOT / out_csv).resolve() if not os.path.isabs(out_csv) else Path(out_csv)
    if out_path.exists():
        return str(out_path)
    print("[info] Processed CSV manquant — génération…")
    bp = import_module("scripts.build_processed_csv")
    bp.build_processed()
    if not out_path.exists():
        raise SystemExit("Impossible de générer le processed CSV.")
    return str(out_path)

if __name__ == "__main__":
    with open(CFG,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # chemins
    for k, v in list(cfg.get("paths", {}).items()):
        if not os.path.isabs(v):
            cfg["paths"][k] = str((ROOT / v).resolve())

    csv_path = ensure_processed_csv(cfg)
    model_out = cfg["paths"].get("tfidf_model", str(ROOT / "assets" / "models" / "healthpredict_model.joblib"))
    model_out = str((ROOT / model_out).resolve()) if not os.path.isabs(model_out) else model_out
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if not {"event_text","label"} <= set(df.columns):
        raise SystemExit("Le processed CSV doit contenir 'event_text' et 'label'.")

    X = df["event_text"].fillna("").astype(str).map(clean_text).tolist()
    y = df["label"].astype(int).values

    test_size   = cfg.get("training",{}).get("test_size", 0.2)
    random_state= cfg.get("training",{}).get("random_state", 42)
    min_df      = cfg.get("training",{}).get("tfidf",{}).get("min_df", 2)
    ngram_range = tuple(cfg.get("training",{}).get("tfidf",{}).get("ngram_range",[1,2]))
    max_features= cfg.get("training",{}).get("tfidf",{}).get("max_features", 20000)
    max_iter    = cfg.get("training",{}).get("logreg",{}).get("max_iter", 2000)
    C           = cfg.get("training",{}).get("logreg",{}).get("C", 1.0)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, max_features=max_features)),
        ("clf",   LogisticRegression(max_iter=max_iter, C=C, n_jobs=None, solver="lbfgs")),
    ])

    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    pred  = (proba >= 0.5).astype(int)

    acc = accuracy_score(yte, pred)
    f1  = f1_score(yte, pred, zero_division=0)
    try:
        auc = roc_auc_score(yte, proba)
    except Exception:
        auc = float("nan")

    print(f"[train] Accuracy={acc:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
    print(classification_report(yte, pred, digits=3))

    joblib.dump(pipe, model_out)
    print(f"[OK] Modèle sauvegardé → {model_out}")
