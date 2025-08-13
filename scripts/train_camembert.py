# scripts/train_camembert.py
# CamemBERT -> embeddings CLS -> LogisticRegression
# Robuste: détection colonnes, batching, faible RAM/CPU, compatible app Streamlit

import os, gc, joblib, numpy as np, pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel

# ---------- Paramètres ----------
DATA = os.getenv("DATA", "data/processed/medical_imaging_text_labeled.csv")
OUT  = os.getenv("OUT",  "models/healthpredict_camembert_model.joblib")

MODEL_NAME = os.getenv("HF_MODEL", "camembert-base")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))   # 128/256 selon ressources
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))     # 8-16 CPU; 16-32 GPU
NUM_WORKERS = 0

TEXT_CANDIDATES  = ["text_clean","text","event_text","narrative","summary","description","Alerte_clean","Alerte"]
LABEL_CANDIDATES = ["label_critique","label","Critique","target","y"]

# ---------- Dataset ----------
class TextDataset(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return str(self.texts[idx])

def collate_fn(batch_texts, tokenizer, max_length, device):
    enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

@torch.no_grad()
def embed_texts(texts, tokenizer, model, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    ds = TextDataset(texts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                    collate_fn=lambda bt: collate_fn(bt, tokenizer, max_length, device))
    out_chunks = []
    for batch in tqdm(dl, desc="Embedding", unit="batch"):
        outputs = model(**batch)
        cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        out_chunks.append(cls)
        del outputs, cls, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return np.vstack(out_chunks) if out_chunks else np.zeros((0, model.config.hidden_size), dtype=np.float32)

def pick_column(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Colonnes {kind} introuvables. Cherché {candidates}. "
        f"Colonnes dispo: {list(df.columns)}"
    )

def main():
    # Chargement
    df = pd.read_csv(DATA)
    print(f"[INFO] Fichier chargé: {DATA} avec colonnes: {list(df.columns)}")

    # Sélection colonnes
    text_col  = pick_column(df, TEXT_CANDIDATES, "texte")
    label_col = pick_column(df, LABEL_CANDIDATES, "label")

    # Prépare X/y
    texts  = df[text_col].astype(str).fillna("").tolist()
    labels = df[label_col].astype(int).to_numpy()

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # Device + modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval().to(device)

    # Optionnel: demi-précision sur GPU pour gagner RAM
    if device.type == "cuda":
        try:
            model.half()
            print("[INFO] FP16 activé (GPU).")
        except Exception:
            print("[WARN] FP16 non activé, on reste en FP32.")

    # Embeddings
    train_emb = embed_texts(X_train, tokenizer, model, device)
    test_emb  = embed_texts(X_test,  tokenizer, model, device)

    if train_emb.shape[0] != len(y_train) or test_emb.shape[0] != len(y_test):
        raise RuntimeError("Mismatch taille embeddings/labels.")

    # Classifier
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(train_emb, y_train)

    # Éval
    y_pred = clf.predict(test_emb)
    print(classification_report(y_test, y_pred))

    # Sauvegarde (compatible app Streamlit)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump((tokenizer, model, clf), OUT)
    print(f"[OK] Modèle CamemBERT sauvegardé -> {OUT}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Nettoyage mémoire sans crash si objets non définis
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("[INFO] Fin du script.")
# End of script