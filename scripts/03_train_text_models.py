# -*- coding: utf-8 -*-
import os, yaml, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

CONFIG = "config/config.yaml"

def load_cfg():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    data_csv = cfg["paths"]["processed_csv"]
    model_path = cfg["paths"]["tfidf_model"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"{data_csv} introuvable. Lance 02_prepare_and_label.py")

    df = pd.read_csv(data_csv)
    X = df["text_clean"].fillna("").astype(str)
    y = df["label_critique"].astype(int)

    test_size = float(cfg["training"]["test_size"])
    rs = int(cfg["training"]["random_state"])
    min_per_class = int(cfg["training"]["use_stratify_min_per_class"])

    # activer stratify si assez d'exemples dans chaque classe
    counts = y.value_counts()
    use_strat = counts.min() >= min_per_class and len(counts) >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y if use_strat else None
    )

    # Vectorizer
    tv = cfg["training"]["tfidf"]
    vectorizer = TfidfVectorizer(
        min_df=int(tv["min_df"]),
        ngram_range=tuple(tv["ngram_range"]),
        max_features=int(tv["max_features"])
    )

    # Classifier
    which = cfg["training"]["clf"].lower()
    if which == "rf":
        params = cfg["training"]["rf"]
        clf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            random_state=rs,
            n_jobs=-1
        )
    else:
        params = cfg["training"]["logreg"]
        clf = LogisticRegression(
            max_iter=int(params["max_iter"]),
            C=float(params["C"]),
            n_jobs=-1 if "n_jobs" in params else None
        )

    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(pipe, model_path)
    print(f"[OK] Modèle sauvegardé -> {model_path}")

if __name__ == "__main__":
    main()
