import yaml, os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

cfg = yaml.safe_load(open("config/config.yaml","r",encoding="utf-8"))
df = pd.read_csv(cfg["paths"]["processed_csv"])
X = df["text_clean"].astype(str); y = df["label_critique"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg["training"]["test_size"], random_state=cfg["training"]["random_state"])
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=tuple(cfg["training"]["tfidf"]["ngram_range"]), max_features=cfg["training"]["tfidf"]["max_features"])),
    ("clf", RandomForestClassifier(n_estimators=cfg["training"]["rf"]["n_estimators"], random_state=cfg["training"]["random_state"], n_jobs=-1))
])
pipe.fit(Xtr, ytr)
os.makedirs(os.path.dirname(cfg["paths"]["tfidf_model"]), exist_ok=True)
joblib.dump(pipe, cfg["paths"]["tfidf_model"])
print("[OK] TF-IDF+RF re-généré")
