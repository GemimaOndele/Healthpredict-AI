# api/main.py
from __future__ import annotations

import base64
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, List, Tuple

import joblib
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ========= Sécurité simple par clé API (optionnelle) =========

def require_api_key(x_api_key: Optional[str] = Header(None)):
    required = os.getenv("HP_API_KEY", "").strip()  # lu dynamiquement
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return True

# ========= Localisation des modèles =========
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TFIDF = os.getenv(
    "HP_TFIDF_MODEL",
    str(ROOT / "assets" / "models" / "healthpredict_model.joblib"),
)
DEFAULT_CAMEMBERT = os.getenv(
    "HP_CAMEMBERT_MODEL",
    str(ROOT / "assets" / "models" / "healthpredict_camembert_model.joblib"),
)

# ========= Chargements paresseux =========
_pipe_tfidf = None
_camembert_tuple = None  # (tokenizer, camembert_model, clf)

def load_tfidf():
    global _pipe_tfidf
    if _pipe_tfidf is None:
        p = Path(DEFAULT_TFIDF)
        if not p.exists():
            raise RuntimeError(
                f"Modèle TF-IDF introuvable ({p}). Entraînez-le ou lancez scripts/download_assets.py."
            )
        _pipe_tfidf = joblib.load(p)
    return _pipe_tfidf

def load_camembert():
    global _camembert_tuple
    if _camembert_tuple is None:
        p = Path(DEFAULT_CAMEMBERT)
        if not p.exists():
            raise RuntimeError(
                f"Modèle CamemBERT introuvable ({p}). Il est optionnel, mais requis si vous l'utilisez."
            )
        _camembert_tuple = joblib.load(p)  # (tokenizer, model, clf)
    return _camembert_tuple

# ========= Utilitaires texte =========
def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8", "ignore")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_vectorizer_from_pipeline(pipe):
    if hasattr(pipe, "named_steps"):
        for _, step in pipe.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                return step
    return None

def top_keywords_tfidf_from_pipe(pipe, cleaned_text: str, topk: int = 10):
    from sklearn.exceptions import NotFittedError
    vec = get_vectorizer_from_pipeline(pipe)
    if vec is None or not hasattr(vec, "vocabulary_"):
        return []
    try:
        X = vec.transform([cleaned_text])
    except NotFittedError:
        return []
    feats = vec.get_feature_names_out()
    idx = X.nonzero()[1]
    scores = X.data
    pairs = [(feats[i], float(scores[j])) for j, i in enumerate(idx)]
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:topk]
    s = sum(p for _, p in pairs) or 1.0
    return [(w, round(100.0 * p / s, 1)) for w, p in pairs]

# ========= Schémas Pydantic =========
class PredictRequest(BaseModel):
    text: str
    model: str = "tfidf"               # "tfidf" ou "camembert"
    return_keywords: bool = True       # mots-clés TF-IDF

class PredictResponse(BaseModel):
    label: str
    proba: float
    model: str
    keywords: Optional[List[Tuple[str, float]]] = None

# ========= App =========
app = FastAPI(
    title="HealthPredict AI - API",
    version="1.0.0",
    description="API REST de prédiction (texte ou fichier) pour la criticité des incidents.",
)

# CORS (facultatif mais pratique pour tests locaux)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "version": "1.0.0",
        "models": {
            "tfidf": Path(DEFAULT_TFIDF).name,
            "camembert": Path(DEFAULT_CAMEMBERT).name,
        },
    }

@app.post("/predict_text", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict_text(body: PredictRequest):
    cleaned = clean_text(body.text or "")

    if body.model.lower().startswith("cam"):
        # CamemBERT
        try:
            import torch  # import local pour éviter la dépendance si non utilisé
            tok, cam, clf = load_camembert()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cam = cam.to(device).eval()
            with torch.no_grad():
                batch = tok([cleaned], padding=True, truncation=True, return_tensors="pt").to(device)
                out = cam(**batch).last_hidden_state[:, 0, :].detach().cpu().numpy()
            proba = float(clf.predict_proba(out)[:, 1][0])
            label = "Critique" if proba >= 0.5 else "Pas critique"
            return PredictResponse(label=label, proba=proba, model="camembert", keywords=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CamemBERT indisponible: {e}")
    else:
        # TF-IDF baseline
        try:
            pipe = load_tfidf()
            proba = float(pipe.predict_proba([cleaned])[0][1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TF-IDF indisponible: {e}")
        label = "Critique" if proba >= 0.5 else "Pas critique"
        kws = top_keywords_tfidf_from_pipe(pipe, cleaned, topk=10) if body.return_keywords else None
        return PredictResponse(label=label, proba=proba, model="tfidf", keywords=kws)

@app.post("/predict_file", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict_file(
    file: UploadFile = File(...),
    model: str = "tfidf",
    return_keywords: bool = True,
):
    # Lecture naïve du texte (OCR/parse avancé = côté app Streamlit)
    raw = await file.read()
    try:
        txt = raw.decode("utf-8", errors="ignore")
    except Exception:
        txt = ""
    if not txt.strip():
        # fallback: encodage base64 des premiers octets pour tracer l'appel
        txt = "(binaire) " + base64.b64encode(raw[:64]).decode("ascii") + "…"

    req = PredictRequest(text=txt, model=model, return_keywords=return_keywords)
    return predict_text(req)

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
