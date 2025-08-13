# Runner simple alternatif si tu veux juste (re)générer le modèle TF-IDF rapidement
import yaml, os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

cfg = yaml.safe_load(open("config/config.yaml","r",encoding="utf-8"))
df = pd.read_csv(cfg["paths"]["processed_csv"])
X = df["text_clean"].astype(str)
y = df["label_ritique"] if "label_ritique" in df.columns else df["label_critique"]  # robustesse typo
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg["training"]["test_size"], random_state=cfg["training"]["random_state"])
pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=tuple(cfg["training"]["tfidf"]["ngram_range"]))), ("clf", LogisticRegression(max_iter=2000))])
pipe.fit(Xtr, ytr)
os.makedirs(os.path.dirname(cfg["paths"]["tfidf_model"]), exist_ok=True)
joblib.dump(pipe, cfg["paths"]["tfidf_model"])
print("[OK] TF-IDF+LR re-généré")
