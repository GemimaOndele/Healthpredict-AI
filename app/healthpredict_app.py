# ===================== app/healthpredict_app.py =====================
# -*- coding: utf-8 -*-
# HealthPredict AI — Dashboard + Prédiction + 📎 Documents + 📚 Historique & Prévisions + 📑 Évaluation
# OCR images & PDF (Tesseract + Poppler, optionnel), Similarité historique, Traduction FR (optionnel),
# Exports CSV/Excel, Historique SQLite (optionnel)

import os, io, sys, json, joblib, yaml, warnings, unicodedata, re
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import datetime as dt  # noqa: F401
from typing import Optional, Tuple, List
from sklearn.exceptions import NotFittedError

# ========= Fix chemins: ajouter la racine du projet au PYTHONPATH =========
APP_DIR = Path(__file__).resolve().parent            # .../app
ROOT_DIR = APP_DIR.parent                            # projet/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ✅ DOIT ÊTRE AVANT TOUT AUTRE st.* :
st.set_page_config(page_title="HealthPredict AI", layout="wide")

# =========================
# Options lourdes (désactivables via env)
# =========================
USE_CAMEMBERT = os.getenv("HP_USE_CAMEMBERT", "0") == "1"
USE_SPACY     = os.getenv("HP_USE_SPACY", "0") == "1"

# ==== ML / NLP (imports légers, chargements lourds en lazy) ====
try:
    import shap  # optionnel
except Exception:
    shap = None

torch = None
AutoTokenizer = AutoModelForSeq2SeqLM = None
if USE_CAMEMBERT:
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
    except Exception as e:
        USE_CAMEMBERT = False
        st.warning(f"CamemBERT/Transformers indisponible ({e}). Utilisation du modèle TF-IDF uniquement.")

# ----- Téléchargement des assets (optionnel) -----
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

def _safe_ensure_assets():
    """Télécharge les assets si possible, sinon silencieux (1 seule fois)."""
    if os.environ.get("HP_AUTO_DOWNLOAD", "1") != "1":
        return
    if st.session_state.get("_assets_checked_", False):
        return
    st.session_state["_assets_checked_"] = True

    try:
        # 1) Essai import standard
        try:
            from scripts.download_assets import ensure_assets
        except ModuleNotFoundError:
            # 2) Fallback: import direct par chemin
            import importlib.util
            cand = ROOT_DIR / "scripts" / "download_assets.py"
            if not cand.exists():
                st.caption("ℹ️ Téléchargement des assets ignoré (module absent).")
                return
            spec = importlib.util.spec_from_file_location("download_assets", str(cand))
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            ensure_assets = mod.ensure_assets  # type: ignore

        # 3) Lancer le téléchargement si nécessaire
        ensure_assets()
    except Exception as e:
        st.warning(f"Téléchargement des assets échoué: {e}")


# ----- Base SQLite (optionnel) -----
DB_ENABLED = True
try:
    import hpdb as db  # notre module local hpdb.py (ROOT_DIR déjà dans sys.path)
    DB_PATH = os.environ.get("HP_DB", str(ROOT_DIR / "data" / "app.db"))
    db.init_db(DB_PATH)
    st.caption(f"DB utilisée : {DB_PATH}")
except Exception as e:
    DB_ENABLED = False
    st.caption(f"Historique SQLite désactivé (DB non initialisée). Détail: {e}")

# ----- parsers optionnels -----
try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx as docxlib
except Exception:
    docxlib = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

warnings.filterwarnings("ignore")

# --- spaCy optionnel (pour extras UI) ---
SPACY_READY = False
if USE_SPACY:
    try:
        import spacy  # type: ignore
        try:
            NLP_FR = spacy.load("fr_core_news_sm")
        except Exception:
            NLP_FR = None
        try:
            NLP_EN = spacy.load("en_core_web_sm")
        except Exception:
            NLP_EN = None
        SPACY_READY = bool(NLP_FR or NLP_EN)
    except Exception as e:
        st.info(f"spaCy indisponible ({e}). Continuité sans spaCy.")

# =========================
# Config + Helpers généraux
# =========================
CFG = str(ROOT_DIR / "config" / "config.yaml")  # ← chemin absolu

def load_cfg():
    if os.path.exists(CFG):
        with open(CFG, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        st.warning("⚠️ Fichier config.yaml introuvable — chemins par défaut utilisés.")
        cfg = {
            "paths": {
                "processed_csv": "assets/data/processed/medical_imaging_text_labeled.csv",
                "tfidf_model": "assets/models/healthpredict_model.joblib",
                "camembert_model": "assets/models/healthpredict_camembert_model.joblib",
                "raw_csv": "assets/data/raw/raw_openfda_imaging_reports.csv",
            },
            "openfda": {
                "base_url": "https://api.fda.gov/device/event.json",
                "start_date": "2018-01-01",
                "end_date": "2025-01-01",
                "keywords": ["radiology", "ultrasound", "infusion", "ventilator"],
                "limit": 100,
                "max_records": 5000,
                "sleep_sec": 0.2,
            },
        }
    # normaliser en chemins absolus
    for k, v in list(cfg.get("paths", {}).items()):
        if not os.path.isabs(v):
            cfg["paths"][k] = str((ROOT_DIR / v).resolve())
    return cfg

def ensure_str(x):
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    try:
        import numpy as _np
        if isinstance(x, float) and _np.isnan(x):
            return ""
    except Exception:
        pass
    try:
        return str(x)
    except Exception:
        return ""

def clean_text(text: str) -> str:
    text = ensure_str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_data
def load_csv_safe(path_csv: str) -> pd.DataFrame:
    if os.path.exists(path_csv):
        try:
            return pd.read_csv(path_csv)
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data
def load_main_dataset(cfg) -> pd.DataFrame:
    paths = cfg.get("paths", {})
    # 1) processed_csv
    cand_processed = []
    if paths.get("processed_csv"):
        cand_processed.append(paths["processed_csv"])
    cand_processed.append(str(ROOT_DIR / "assets" / "data" / "processed" / "medical_imaging_text_labeled.csv"))
    for p in cand_processed:
        df = load_csv_safe(p)
        if not df.empty:
            return df
    # 2) raw_csv
    cand_raw = []
    if paths.get("raw_csv"):
        cand_raw.append(paths["raw_csv"])
    cand_raw.append(str(ROOT_DIR / "assets" / "data" / "raw" / "raw_openfda_imaging_reports.csv"))
    for p in cand_raw:
        df = load_csv_safe(p)
        if not df.empty:
            return df
    return pd.DataFrame()

# ====== reco pour Recommandation ======
def reco(row):
    alerte = ensure_str(row.get("Alerte", ""))
    if alerte.startswith("Critique"):
        return "⚠️ Maintenance immédiate requise"
    if alerte.startswith("Modérée"):
        return "🔍 Surveillance accrue"
    return "✅ Fonctionnement normal"

# ================ Détection langue ================
FR_STOPS = {"le","la","les","des","un","une","et","ou","de","du","au","aux","est","avec","sans","pour","sur","dans","pas","ne","à","été","entre","proche"}
EN_STOPS = {"the","and","or","of","to","with","without","for","on","in","is","are","was","were","near","soon","due","fault","error","failure","leak","overheating"}

def detect_lang_simple(txt: str) -> str:
    t = ensure_str(txt).lower()
    fr = sum(1 for w in FR_STOPS if f" {w} " in f" {t} ")
    en = sum(1 for w in EN_STOPS if f" {w} " in f" {t} ")
    return "en" if en > fr else "fr"

# ============== Traduction (Transformers optionnel) ==============
@st.cache_resource(show_spinner=False)
def get_translator(src: str, tgt: str):
    if not USE_CAMEMBERT or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        return None, None, "cpu"
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        model = model.to(device)
        return tok, model, device
    except Exception as e:
        st.info(f"Traduction désactivée ({e}).")
        return None, None, "cpu"

def translate_text(texts, src="en", tgt="fr", max_len=512, batch_size=16):
    if src == tgt or not texts:
        return texts
    tok, model, device = get_translator(src, tgt)
    if tok is None or model is None:
        return texts
    out = []
    for i in range(0, len(texts), batch_size):
        seg = texts[i:i+batch_size]
        batch = tok(seg, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        if device == "cuda" and torch is not None:
            batch = batch.to(device)
        gen = model.generate(**batch, max_length=max_len)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out

def translate_to_fr_if_needed(raw_txt: str, lang_choice: str, force_to_fr: bool = True):
    raw_txt = ensure_str(raw_txt)
    if not raw_txt.strip():
        return raw_txt, "fr", False
    if lang_choice == "Français":
        src = "fr"
    elif lang_choice == "Anglais":
        src = "en"
    else:
        src = detect_lang_simple(raw_txt)
    if force_to_fr and src == "en":
        try:
            t = translate_text([raw_txt], src="en", tgt="fr")[0]
            if t != raw_txt:
                return t, "en", True
            return raw_txt, src, False
        except Exception as e:
            st.info(f"Traduction EN→FR indisponible ({e}). Utilisation du texte original.")
            return raw_txt, src, False
    return raw_txt, src, False

# ======================= Modèles de prédiction =======================
@st.cache_resource
def load_tfidf_model(path):
    return joblib.load(path)

@st.cache_resource
def load_camembert(path):
    return joblib.load(path)  # (tokenizer, camembert_model, clf)

def _vectorizer_is_fitted(vec) -> bool:
    try:
        return hasattr(vec, "vocabulary_") and hasattr(vec, "_tfidf") and hasattr(vec._tfidf, "idf_")
    except Exception:
        return False

def camembert_proba_fn(texts, tokenizer, camembert, clf):
    if torch is None:
        raise RuntimeError("Torch indisponible — CamemBERT désactivé.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    camembert = camembert.to(device)
    camembert.eval()
    with torch.no_grad():
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if device == "cuda":
            batch = batch.to(device)
        out = camembert(**batch)
        emb = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return clf.predict_proba(emb)[:, 1]

# ================== Mots-clés TF-IDF ==================
def get_vectorizer_from_pipeline(pipe):
    if hasattr(pipe, "named_steps"):
        for _, step in pipe.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                return step
    return None

def top_keywords_tfidf_from_pipe(pipe, cleaned_text: str, topk: int = 15):
    vec = get_vectorizer_from_pipeline(pipe)
    if vec is None or not _vectorizer_is_fitted(vec):
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

# ======================= OCR helpers (optionnels) =======================
def ocr_is_ready():
    if pytesseract is None or Image is None:
        return False, "Modules non importés (pytesseract/Pillow)."
    try:
        ver = pytesseract.get_tesseract_version()
        return True, str(ver)
    except Exception as e:
        return False, str(e)

def set_tesseract_cmd_if_needed(path_str: str):
    if path_str:
        try:
            pytesseract.pytesseract.tesseract_cmd = path_str
        except Exception:
            pass

def ocr_image_pil(img: "Image.Image", lang="eng+fra") -> str:
    try:
        img = img.convert("L")
        img = img.point(lambda p: 255 if p > 180 else 0)
        return pytesseract.image_to_string(img, lang=lang)
    except Exception:
        return ""

def extract_text_from_image(fobj, lang="eng+fra") -> str:
    if pytesseract is None or Image is None:
        return ""
    try:
        img = Image.open(fobj)
        txt = pytesseract.image_to_string(img, lang=lang) or ""
        if txt.strip():
            return txt
        fobj.seek(0)
        img = Image.open(fobj)
        return ocr_image_pil(img, lang=lang)
    except Exception:
        return ""

def extract_text_from_pdf_with_ocr(pdf_bytes: bytes, lang="eng+fra", poppler_path=None,
                                   dpi: int = 300, max_pages: int = 10) -> str:
    if convert_from_bytes is None or pytesseract is None or Image is None:
        return ""
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi,
                                  first_page=1, last_page=max_pages,
                                  poppler_path=poppler_path or None)
        texts = []
        for im in imgs:
            t = pytesseract.image_to_string(im, lang=lang) or ""
            if not t.strip():
                t = ocr_image_pil(im, lang=lang)
            texts.append(t or "")
        return "\n".join(texts).strip()
    except Exception:
        return ""

# ======================= Extracteurs multi-formats =======================
def _read_bytesio(uploaded_file):
    data = uploaded_file.read()
    return io.BytesIO(data)

def extract_text_from_pdf(fobj) -> str:
    if PyPDF2 is None:
        return ""
    try:
        reader = PyPDF2.PdfReader(fobj)
        pages = []
        for p in reader.pages[:50]:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx(fobj) -> str:
    if docxlib is None:
        return ""
    try:
        doc = docxlib.Document(fobj)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def extract_text_from_html(fobj) -> str:
    if BeautifulSoup is None:
        return ""
    try:
        html = fobj.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator=" ")
    except Exception:
        return ""

def extract_text_from_excel(fobj) -> str:
    try:
        df = pd.read_excel(fobj)
        return "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist())
    except Exception:
        return ""

def extract_text_from_csv(fobj) -> str:
    try:
        df = pd.read_csv(fobj)
        return "\n".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist())
    except Exception:
        return ""

def extract_text_from_json(fobj) -> str:
    try:
        obj = json.load(fobj)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return ""

def extract_text_from_any(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    bio = _read_bytesio(uploaded_file)

    if name.endswith(".pdf"):
        return extract_text_from_pdf(bio)
    if name.endswith(".docx"):
        return extract_text_from_docx(bio)
    if name.endswith(".html") or name.endswith(".htm"):
        return extract_text_from_html(bio)
    if name.endswith(".xlsx"):
        return extract_text_from_excel(bio)
    if name.endswith(".csv"):
        return extract_text_from_csv(bio)
    if name.endswith(".json"):
        return extract_text_from_json(bio)
    if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
        return ""  # OCR côté UI

    bio.seek(0)
    try:
        return bio.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ======================= Utils Dataset (type, dates, similarité) =======================
DEVICE_MAP = [
    ("mri|magnetic resonance", "IRM"),
    ("ct|scanner|computed tomography", "Scanner"),
    ("x[- ]?ray|radiograph|radiology", "Radiologie"),
    ("ultrasound|echograph|sonograph", "Échographe"),
    ("ventilator|respirator", "Ventilateur"),
    ("anesthesia|anaesthesia", "Anesthésie"),
    ("infusion|pump", "Pompe à perfusion"),
    ("defibrillator|aed", "Défibrillateur"),
    ("pacemaker", "Pacemaker"),
    ("dialysis|hemodialysis", "Dialyse"),
    ("endoscope|endoscopy", "Endoscope"),
    ("monitor|ecg|ekg|spo2|oximeter", "Moniteur patient"),
]

def detect_device_type_text(txt) -> Optional[str]:
    t = ensure_str(txt).lower()
    for pat, lab in DEVICE_MAP:
        if re.search(rf"\b({pat})\b", t):
            return lab
    return None

def pick_text_column(df: pd.DataFrame) -> str:
    for c in ["event_text", "Texte", "Description", "Alerte", "Notes", "Commentaire"]:
        if c in df.columns:
            return c
    obj_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    for c in obj_cols:
        if df[c].astype(str).str.len().gt(0).any():
            return c
    return df.columns[0] if len(df.columns) else "event_text"

def add_detected_type(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df2 = df.copy()
    tcol = pick_text_column(df2)

    s_generic = df2["generic_name"].astype(str) if "generic_name" in df2.columns else pd.Series("", index=df2.index)
    s_brand   = df2["brand_name"].astype(str)   if "brand_name"   in df2.columns else pd.Series("", index=df2.index)
    s_text    = df2[tcol].astype(str)

    s_all = (s_generic + " " + s_brand + " " + s_text).str.lower()
    out = pd.Series("Autre", index=df2.index, dtype="object")

    for pat, lab in DEVICE_MAP:
        mask = s_all.str.contains(rf"\b({pat})\b", regex=True, na=False)
        out[mask] = lab

    df2["TypeDetecte"] = out
    return df2

def to_date_safe(s):
    try:
        if pd.isna(s):
            return pd.NaT
        s = ensure_str(s)
        if re.fullmatch(r"\d{8}", s):
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

def add_parsed_date(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if "date_received" in df2.columns:
        df2["date_dt"] = df2["date_received"].apply(to_date_safe)
    elif "Date" in df2.columns:
        df2["date_dt"] = df2["Date"].apply(to_date_safe)
    else:
        df2["date_dt"] = pd.NaT
    return df2

@st.cache_resource
def build_hist_matrix(df_hist: pd.DataFrame, pipe_tfidf_path: str, text_col: str):
    pipe = load_tfidf_model(pipe_tfidf_path)
    vec = get_vectorizer_from_pipeline(pipe)
    if vec is None or not _vectorizer_is_fitted(vec):
        return None, None, None
    corpus = df_hist[text_col].fillna("").astype(str).tolist()
    cleaned_corpus = [clean_text(x) for x in corpus]
    try:
        X = vec.transform(cleaned_corpus)
    except NotFittedError:
        return None, None, None
    try:
        row_norms = np.sqrt(X.power(2).sum(axis=1)).A1
    except Exception:
        row_norms = np.array([np.linalg.norm(r.toarray()) for r in X])
    row_norms[row_norms == 0] = 1.0
    return pipe, X, row_norms

def top_similar(text_query: str, pipe, X_hist, row_norms, df_hist: pd.DataFrame, text_col: str, k=10):
    if pipe is None or X_hist is None:
        return []
    vec = get_vectorizer_from_pipeline(pipe)
    if vec is None or not _vectorizer_is_fitted(vec):
        return []
    try:
        vq = vec.transform([clean_text(text_query)])
    except NotFittedError:
        return []
    try:
        qn = float(np.sqrt(vq.multiply(vq).sum()))
    except Exception:
        qn = float(np.linalg.norm(vq.toarray()))
    if qn == 0.0:
        return []
    sims = (X_hist @ vq.T).toarray().ravel()
    denom = (row_norms * qn)
    denom[denom == 0] = 1.0
    sims = sims / denom
    if not np.any(np.isfinite(sims)):
        return []
    top_idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), df_hist.iloc[int(i)]) for i in top_idx]

# ======================= UI helpers =======================
def confidence_bar(proba: float):
    pct = float(round(100*proba, 1))
    dfp = pd.DataFrame({"label": ["Confiance"], "value": [pct]})
    bg  = pd.DataFrame({"label": ["Confiance"], "value": [100]})
    bar_bg = alt.Chart(bg).mark_bar(color="#eeeeee").encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0,100])), y=alt.Y("label:N")
    )
    bar_fg = alt.Chart(dfp).mark_bar().encode(
        x="value:Q", y="label:N",
        tooltip=[alt.Tooltip("value:Q", title="Confiance (%)", format=".1f")]
    )
    text = alt.Chart(dfp).mark_text(dx=6, color="black").encode(
        x="value:Q", y="label:N", text=alt.Text("value:Q", format=".1f")
    )
    return (bar_bg + bar_fg + text).properties(height=48)

# ======================= Chargement & UI =======================
st.title("🔧 HealthPredict AI - Maintenance Prédictive des Équipements Médicaux")
st.markdown("**Optimisez la disponibilité, la sécurité et les coûts grâce à l’IA.**")

# Charger la config puis le dataset
cfg = load_cfg()
df_main = load_main_dataset(cfg)
if df_main.empty:
    st.error("Aucune donnée chargée (processed/raw). Vérifiez vos chemins de données dans config.yaml.")
    st.stop()

# ✅ Limitation RAM
MAX_ROWS_UI = int(os.getenv("HP_MAX_ROWS_UI", "200000"))
if len(df_main) > MAX_ROWS_UI:
    st.info(f"Aperçu limité aux {MAX_ROWS_UI} lignes sur {len(df_main)} (pour préserver la mémoire).")
    df_main = df_main.head(MAX_ROWS_UI).copy()

text_col_main = pick_text_column(df_main)
if "Recommandation" not in df_main.columns and "Alerte" in df_main.columns:
    df_main["Recommandation"] = df_main.apply(reco, axis=1)

df_main = add_detected_type(df_main)
df_main = add_parsed_date(df_main)

# ---- Onglets ----
tab_dash, tab_pred, tab_docs, tab_hist, tab_eval = st.tabs(
    ["📊 Tableau de bord", "🤖 Prédiction", "📎 Documents", "📚 Historique & Prévisions", "📑 Évaluation"]
)

# ---- TAB Dashboard ----
with tab_dash:
    st.sidebar.header("🎧 Filtres")
    types_uniques = sorted(df_main["TypeDetecte"].dropna().unique().tolist()) if "TypeDetecte" in df_main.columns else []
    equipement = st.sidebar.selectbox("Type d'équipement", ["Tous"] + types_uniques, index=0, key="type_equipement")

    df_view = df_main if equipement == "Tous" else df_main[df_main["TypeDetecte"] == equipement]
    if "event_type" in df_view.columns:
        sev_all = sorted([s for s in df_view["event_type"].dropna().unique()])
        sev_sel = st.sidebar.multiselect("Gravité (event_type)", sev_all, default=sev_all, key="sev_filter_evt")
        if sev_sel:
            df_view = df_view[df_view["event_type"].isin(sev_sel)]
    elif "Alerte" in df_view.columns:
        al_all = sorted([s for s in df_view["Alerte"].dropna().unique()])
        al_sel = st.sidebar.multiselect("Gravité (Alerte)", al_all, default=al_all, key="sev_filter_alert")
        if al_sel:
            df_view = df_view[df_view["Alerte"].isin(al_sel)]

    max_rows = st.sidebar.slider("Nombre max. de lignes à afficher", 100, 5000, 1000, step=100, key="slider_max_rows")
    df_display = df_view.head(max_rows)
    if len(df_view) > max_rows:
        st.info(f"Affichage limité aux {max_rows} premières lignes sur {len(df_view)} pour préserver la mémoire.")

    c1, c2 = st.columns(2)
    c1.metric("🔧 Total rapports", int(df_view.shape[0]))
    if "date_dt" in df_view.columns:
        last_date = df_view["date_dt"].dropna().max()
        c2.metric("🗓️ Dernier incident", last_date.date().isoformat() if pd.notna(last_date) else "—")

    st.subheader("📋 Détails (aperçu)")
    st.dataframe(df_display, use_container_width=True)

    st.subheader("📊 Fréquence mensuelle (tous types ou type sélectionné)")
    if "date_dt" in df_view.columns and df_view["date_dt"].notna().any():
        m = df_view.dropna(subset=["date_dt"]).copy()
        m["mois"] = m["date_dt"].dt.to_period("M").dt.to_timestamp()
        gr = m.groupby("mois").size().reset_index(name="Incidents")
        chart = alt.Chart(gr).mark_line(point=True).encode(x="mois:T", y="Incidents:Q").properties(height=280)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Pas de dates exploitables pour la tendance.")

    st.subheader("📊 Répartition par type d'événement")
    if "event_type" in df_view.columns and df_view["event_type"].notna().any():
        cnt = df_view["event_type"].astype(str).value_counts().reset_index()
        cnt.columns = ["event_type","n"]
        bar = alt.Chart(cnt.head(12)).mark_bar().encode(
            x="n:Q", y=alt.Y("event_type:N", sort="-x"), tooltip=["event_type","n"]
        ).properties(height=280)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.caption("Pas de colonne event_type.")

    st.subheader("🧩 Répartition par type détecté")
    if "TypeDetecte" in df_view.columns and df_view["TypeDetecte"].notna().any():
        cnt2 = df_view["TypeDetecte"].astype(str).value_counts().reset_index()
        cnt2.columns = ["Type","n"]
        pie = alt.Chart(cnt2).mark_arc(innerRadius=50).encode(
            theta="n:Q", color="Type:N", tooltip=["Type","n"]
        ).properties(height=300)
        st.altair_chart(pie, use_container_width=True)
    else:
        st.caption("Pas de colonne TypeDetecte.")

# ---- TAB Prédiction (texte saisi) ----
@st.cache_resource
def _pipe_tfidf(path):
    p = path
    if not os.path.exists(p):
        fb = str(ROOT_DIR / "assets" / "models" / "healthpredict_model.joblib")
        if os.path.exists(fb):
            p = fb
    if not os.path.exists(p):
        st.error("Modèle TF-IDF introuvable. Activez HP_AUTO_DOWNLOAD=1 (scripts/download_assets.py) "
                 "ou placez assets/models/healthpredict_model.joblib. "
                 "Sinon, exécutez: python scripts/train_minimal_tfidf.py")
        raise FileNotFoundError("healthpredict_model.joblib not found")
    return load_tfidf_model(p)

@st.cache_resource
def _camembert(path):
    p = path
    if not os.path.exists(p):
        fb = str(ROOT_DIR / "assets" / "models" / "healthpredict_camembert_model.joblib")
        if os.path.exists(fb):
            p = fb
    if not os.path.exists(p):
        return None
    try:
        return load_camembert(p)  # (tokenizer, camembert_model, clf)
    except Exception:
        return None

def _predict_text(txt_for_model, which: str) -> Tuple[str, float, Optional[List[Tuple[str, float]]]]:
    cleaned = clean_text(txt_for_model)
    if which == "CamemBERT + IA" and USE_CAMEMBERT:
        cm = _camembert(cfg["paths"]["camembert_model"])
        if cm is not None:
            try:
                tok, cam, clf = cm
                proba = float(camembert_proba_fn([cleaned], tok, cam, clf)[0])
                label = "Critique" if proba >= 0.5 else "Pas critique"
                return label, proba, None
            except Exception as e:
                st.info(f"CamemBERT indisponible ({e}). Bascule TF-IDF.")
        else:
            st.caption("Modèle CamemBERT manquant — utilisation du TF-IDF.")
    # TF-IDF
    pipe = _pipe_tfidf(cfg["paths"]["tfidf_model"])
    try:
        proba = float(pipe.predict_proba([cleaned])[0][1])
    except NotFittedError:
        st.error("Le modèle TF-IDF chargé n'est pas ajusté (NotFittedError). "
                 "Exécutez: python scripts/train_minimal_tfidf.py")
        raise
    label = "Critique" if proba >= 0.5 else "Pas critique"
    keywords = top_keywords_tfidf_from_pipe(pipe, cleaned, topk=15)
    return label, proba, keywords

with tab_pred:
    st.subheader("🤖 Prédiction IA sur les alertes")
    lang_choice = st.selectbox("Langue du texte :", ["Auto", "Français", "Anglais"], key="sel_lang_pred")
    force_fr = st.toggle("Traduire vers le français pour la prédiction (recommandé)", value=True, key="toggle_force_fr_pred")
    model_type = st.selectbox("Choisir le modèle IA :", ["Random Forest / Logistic Regression", "CamemBERT + IA"], key="sel_model_pred")
    texte_alerte = st.text_input("Entrer une alerte à analyser", key="txt_alert_pred")
    use_sim = st.toggle("Chercher des cas similaires dans l'historique", value=False, key="toggle_sim_pred")

    if texte_alerte:
        txt_for_model, src_lang, translated = translate_to_fr_if_needed(texte_alerte, lang_choice, force_to_fr=force_fr)
        if translated:
            st.caption("Texte en **anglais** → traduit en **français** pour la prédiction.")
            with st.expander("Voir le texte traduit"):
                st.write(txt_for_model)

        try:
            label, proba, kw = _predict_text(txt_for_model, model_type)
        except NotFittedError:
            st.stop()

        st.markdown("🔴 **ALERTE CRITIQUE** (p=%.2f)" % proba if label == "Critique" else "🟢 **Pas critique** (p=%.2f)" % proba)
        st.altair_chart(confidence_bar(proba), use_container_width=True)

        # 💾 Enregistrer dans SQLite (si activée)
        if DB_ENABLED:
            try:
                model_name = "CamemBERT" if (model_type.startswith("CamemBERT") and USE_CAMEMBERT) else "TFIDF"
                type_pred = detect_device_type_text(txt_for_model) or "Autre"
                db.insert_prediction(
                    source="input",
                    file_name=None,
                    input_text=txt_for_model,
                    cleaned_text=clean_text(txt_for_model),
                    model_type=model_name,
                    label=label,
                    proba=proba,
                    detected_type=type_pred,
                    src_lang=("en" if translated else "fr") if lang_choice == "Auto"
                             else ("fr" if lang_choice == "Français" else "en"),
                    translated=bool(translated),
                    top_keywords=(kw or []),
                    db_path=DB_PATH,  # on force la même DB que celle initialisée
                )
                st.caption("💾 Enregistré dans l’historique (SQLite).")
            except Exception as e:
                st.warning(f"Historique non enregistré ({e}).")


        st.subheader("🧠 Mots-clés (pondération TF-IDF)")
        if kw:
            df_kw = pd.DataFrame(kw, columns=["Mot-clé", "Contribution %"])
            st.dataframe(df_kw, use_container_width=True)
            st.download_button(
                "⬇️ Exporter (CSV)",
                df_kw.to_csv(index=False).encode("utf-8"),
                "mots_cles_prediction.csv",
                "text/csv",
                key="dl_kw_pred",
            )
        else:
            st.caption("Les mots-clés détaillés sont disponibles avec le modèle TF-IDF.")

        if use_sim:
            st.subheader("🔎 Cas historiques similaires (TF-IDF cosinus)")
            pipe_hist, X_hist, norms_hist = build_hist_matrix(df_main, cfg["paths"]["tfidf_model"], text_col_main)
            if pipe_hist is None:
                st.info("Vectorizer indisponible ou non ajusté (modèle TF-IDF). Exécutez d'abord le training minimal.")
            else:
                sims = top_similar(txt_for_model, pipe_hist, X_hist, norms_hist, df_main, text_col_main, k=10)
                if sims:
                    rows = []
                    for s, r in sims:
                        rows.append({
                            "Similarité": round(100*s,1),
                            "Date": ensure_str(r.get("date_received", r.get("date_dt", ""))),
                            "Type détecté": ensure_str(r.get("TypeDetecte", "")),
                            "Extrait": (
                                ensure_str(r.get(text_col_main, ""))[:200] + "…"
                                if len(ensure_str(r.get(text_col_main, ""))) > 200
                                else ensure_str(r.get(text_col_main, ""))
                            ),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.caption("Aucun résultat similaire trouvé.")

# ---- TAB Documents ----
with tab_docs:
    st.subheader("📎 Analyse de documents (PDF, DOCX, TXT, CSV, XLSX, JSON, HTML, images…)")
    st.caption("Déposez vos fichiers. Pour les scans (images/PDF), activez l’OCR.")
    lang_docs = st.selectbox("Langue des documents :", ["Auto", "Français", "Anglais"], index=0, key="sel_lang_docs")
    force_fr_docs = st.toggle("Traduire vers le français pour la prédiction (recommandé)", value=True, key="toggle_force_fr_docs")
    model_docs = st.selectbox("Modèle pour documents :", ["Random Forest / Logistic Regression", "CamemBERT + IA"], key="sel_model_docs")
    max_chars = st.slider("Taille max analysée par document (caractères)", 2000, 40000, 12000, 1000, key="slider_max_chars_docs")

    st.markdown("**Paramètres OCR (optionnels)**")
    use_ocr = st.toggle("Activer l’OCR pour images", value=False, key="toggle_use_ocr")
    use_pdf_ocr = st.toggle("Activer l’OCR pour PDF scannés", value=False, key="toggle_use_pdf_ocr")
    tesseract_path = st.text_input("Chemin Tesseract (Windows)", value=r"C:\Program Files\Tesseract-OCR\tesseract.exe", key="inp_tess_path")
    poppler_path = st.text_input("Chemin Poppler (bin) pour PDF (Windows)", value=r"C:\Program Files\poppler-24.08.0\Library\bin", key="inp_poppler_path")
    ocr_langs = st.text_input("Langues OCR (codes tesseract)", value="eng+fra", key="inp_tess_langs")
    pdf_dpi = st.slider("DPI OCR PDF", 150, 400, 300, 50, key="slider_pdf_dpi")
    max_pdf_ocr_pages = st.slider("Pages max OCR PDF", 1, 20, 5, 1, key="slider_max_pdf_pages")

    if (use_ocr or use_pdf_ocr) and tesseract_path:
        set_tesseract_cmd_if_needed(tesseract_path)
    ready_tess, ver_or_msg = ocr_is_ready()
    st.caption(("✅ OCR actif — Tesseract " + ver_or_msg) if ready_tess else ("❌ OCR indisponible — " + ver_or_msg))

    use_sim_docs = st.toggle("Chercher des cas similaires (historique)", value=False, key="toggle_sim_docs")

    ups = st.file_uploader("Déposer des fichiers", type=None, accept_multiple_files=True, key="uploader_docs")
    if ups:
        pipe_cached = _pipe_tfidf(cfg["paths"]["tfidf_model"])
        pipe_hist, X_hist, norms_hist = (None, None, None)
        if use_sim_docs:
            pipe_hist, X_hist, norms_hist = build_hist_matrix(df_main, cfg["paths"]["tfidf_model"], text_col_main)

        all_kw_rows = []
        for up in ups:
            with st.container(border=True):
                st.markdown(f"**Fichier :** `{up.name}`")
                name = up.name.lower()
                text = extract_text_from_any(up) or ""

                if use_ocr and name.endswith((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
                    up.seek(0)
                    if ready_tess:
                        text = extract_text_from_image(up, lang=ocr_langs) or text
                    else:
                        st.warning("Image détectée mais OCR indisponible.")

                if use_pdf_ocr and name.endswith(".pdf") and not text.strip() and convert_from_bytes is not None:
                    up.seek(0)
                    pdf_bytes = up.read()
                    if ready_tess:
                        text = extract_text_from_pdf_with_ocr(pdf_bytes, lang=ocr_langs, poppler_path=poppler_path, dpi=pdf_dpi, max_pages=max_pdf_ocr_pages) or ""
                    else:
                        st.warning("PDF détecté mais OCR indisponible.")

                text = ensure_str(text).replace("\x00", "").strip()
                if not text:
                    st.warning("Impossible d’extraire du texte de ce fichier.")
                    continue

                if len(text) > max_chars:
                    st.info(f"Analyse tronquée aux {max_chars} premiers caractères (taille réelle = {len(text)}).")
                    text = text[:max_chars]

                txt_for_model, src_lang, translated = translate_to_fr_if_needed(text, lang_docs, force_to_fr=force_fr_docs)
                if translated:
                    st.caption("Document EN → FR (automatique).")

                try:
                    label, proba, _ = _predict_text(txt_for_model, model_docs)
                except NotFittedError:
                    st.stop()

                type_doc = detect_device_type_text(text) or "Autre"
                st.markdown(
                    f"**Résultat :** {'🔴 Critique' if label=='Critique' else '🟢 Pas critique'} — "
                    f"**Confiance : {proba:.2f}** — **Type détecté : {type_doc}**"
                )
                st.altair_chart(confidence_bar(proba), use_container_width=True)

                cleaned = clean_text(txt_for_model)
                kw = top_keywords_tfidf_from_pipe(pipe_cached, cleaned, topk=15)
                if kw:
                    df_kw = pd.DataFrame(kw, columns=["Mot-clé", "Contribution %"])
                    st.dataframe(df_kw, use_container_width=True)
                    for mot, pct in kw:
                        all_kw_rows.append({"Fichier": up.name, "Mot-clé": mot, "Contribution %": pct})

                # 💾 Enregistrer dans SQLite
                if DB_ENABLED:
                    try:
                        model_name = "CamemBERT" if model_docs.startswith("CamemBERT") and USE_CAMEMBERT else "TFIDF"
                        db.insert_prediction(
                            source="doc",
                            file_name=up.name,
                            input_text=txt_for_model,
                            cleaned_text=cleaned,
                            model_type=model_name,
                            label=label,
                            proba=proba,
                            detected_type=type_doc,
                            src_lang=("en" if translated else "fr") if lang_docs == "Auto" else ("fr" if lang_docs=="Français" else "en"),
                            translated=bool(translated),
                            top_keywords=(kw or []),
                            db_path=DB_PATH,  # ✅ même DB
                        )
                        st.caption("💾 Enregistré dans l’historique (SQLite).")
                    except Exception as e:
                        st.warning(f"Historique non enregistré ({e}).")

                with st.expander("Aperçu du texte extrait"):
                    st.write(txt_for_model[:2000] + ("…" if len(txt_for_model) > 2000 else ""))

        if all_kw_rows:
            df_all = pd.DataFrame(all_kw_rows)
            st.download_button(
                "⬇️ Exporter mots-clés (CSV)",
                df_all.to_csv(index=False).encode("utf-8"),
                "mots_cles_documents.csv",
                "text/csv",
                key="dl_kw_docs_csv",
            )
            buf_xlsx = io.BytesIO()
            with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
                df_all.to_excel(writer, index=False, sheet_name="MotsCles")
            st.download_button(
                "⬇️ Exporter mots-clés (Excel)",
                data=buf_xlsx.getvalue(),
                file_name="mots_cles_documents.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_kw_docs_xlsx",
            )

# ---- TAB Historique & Prévisions ----
with tab_hist:
    st.subheader("📚 Historique & Prévisions (basé sur le dataset)")
    cols_top = st.columns(3)
    cols_top[0].metric("Total rapports", int(df_main.shape[0]))
    if "date_dt" in df_main.columns:
        last_all = df_main["date_dt"].dropna().max()
        cols_top[1].metric("Dernier rapport", last_all.date().isoformat() if pd.notna(last_all) else "—")
    types = ["Tous"] + sorted(df_main["TypeDetecte"].dropna().unique().tolist())
    tsel = cols_top[2].selectbox("Type d’équipement", types, key="hist_type_sel")

    dft = df_main if tsel == "Tous" else df_main[df_main["TypeDetecte"] == tsel]

    st.markdown("**Tendance mensuelle**")
    if "date_dt" in dft.columns and dft["date_dt"].notna().any():
        m = dft.dropna(subset=["date_dt"]).copy()
        m["mois"] = m["date_dt"].dt.to_period("M").dt.to_timestamp()
        gr = m.groupby("mois").size().reset_index(name="Incidents").sort_values("mois")
        chart = alt.Chart(gr).mark_area(opacity=0.4).encode(x="mois:T", y="Incidents:Q").properties(height=260)
        st.altair_chart(chart, use_container_width=True)
        # 🔮 Prévision simple (3 mois)
        if len(gr) >= 6:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(gr)).reshape(-1,1)
            y = gr["Incidents"].values
            reg = LinearRegression().fit(X, y)
            horizon = 3
            fut_idx = np.arange(len(gr), len(gr)+horizon).reshape(-1,1)
            pred = reg.predict(fut_idx).clip(min=0)
            fut_dates = pd.date_range(gr["mois"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
            df_fut = pd.DataFrame({"mois": fut_dates, "Incidents": pred})
            ch_past = alt.Chart(gr).mark_line(point=True).encode(x="mois:T", y="Incidents:Q", tooltip=["mois","Incidents"])
            ch_fut  = alt.Chart(df_fut).mark_line(point=True, strokeDash=[4,3]).encode(x="mois:T", y="Incidents:Q", tooltip=["mois","Incidents"])
            st.altair_chart((ch_past + ch_fut).properties(height=240), use_container_width=True)
        else:
            st.caption("Historique trop court pour une régression linéaire (≥6 points recommandé).")
    else:
        st.caption("Pas de dates exploitables.")

    # 🗂️ Historique SQLite
    st.markdown("---")
    st.subheader("🗂️ Historique des prédictions (SQLite)")
    if DB_ENABLED:
        try:
            col_a, col_b = st.columns([2,1])
            search_q = col_a.text_input("Rechercher (nom de fichier ou texte)", value="", key="hist_search_q")
            limit = col_b.number_input("Nb max", min_value=10, max_value=1000, value=200, step=10, key="hist_limit")
            rows = db.search_predictions(search_q.strip(), limit=int(limit), db_path=DB_PATH) if search_q.strip() else db.fetch_recent_predictions(limit=int(limit), db_path=DB_PATH)
            if not rows:
                st.caption("Aucune prédiction enregistrée pour le moment.")
            else:
                dfh = pd.DataFrame([{
                    "Date (UTC)": r["ts"],
                    "Source": r["source"],
                    "Fichier": r.get("file_name") or "",
                    "Modèle": r["model_type"],
                    "Label": r["label"],
                    "Proba": round(float(r["proba"]), 3),
                    "Type détecté": r.get("detected_type") or "",
                    "Langue": r.get("src_lang") or "",
                    "Traduit": "Oui" if r.get("translated") else "Non",
                    "Top mots-clés": ", ".join([f"{w} ({pct}%)" for w, pct in (r.get("top_keywords") or [])][:5]),
                } for r in rows])
                st.dataframe(dfh, use_container_width=True)
                st.download_button(
                    "⬇️ Exporter (CSV)",
                    dfh.to_csv(index=False).encode("utf-8"),
                    file_name="historique_predictions.csv",
                    mime="text/csv",
                    key="dl_hist_csv",
                )
        except Exception as e:
            st.warning(f"Lecture historique indisponible ({e}).")
    else:
        st.caption("Historique désactivé (DB non initialisée).")

# ---- TAB Évaluation ----
with tab_eval:
    st.subheader("📑 Résultats d'évaluation du modèle")
    eval_dir = ROOT_DIR / "assets" / "eval"
    met_path = eval_dir / "metrics.json"
    if met_path.exists():
        try:
            with open(met_path, "r", encoding="utf-8") as f:
                met = json.load(f)
        except Exception as e:
            met = {}
            st.warning(f"Impossible de lire metrics.json ({e})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{met.get('accuracy', float('nan')):.3f}")
        c2.metric("F1-score", f"{met.get('f1', float('nan')):.3f}")
        c3.metric("ROC-AUC", f"{met.get('roc_auc', float('nan')):.3f}")
        c4.metric("N échantillons", f"{met.get('n_samples', 0)}")

        imgs = [
            ("Courbe ROC", eval_dir / "roc.png"),
            ("Precision-Recall", eval_dir / "pr.png"),
            ("Matrice de confusion", eval_dir / "cm.png"),
            ("event_type (barres)", eval_dir / "event_type_bar.png"),
            ("TypeDetecte (camembert)", eval_dir / "typedetecte_pie.png"),
        ]
        for title, p in imgs:
            if p.exists():
                st.markdown(f"**{title}**")
                st.image(str(p))
    else:
        st.info("Aucun résultat d'évaluation trouvé. Lance d’abord : `python notebooks/eval_healthpredict.py`.")
        st.caption("Les figures/metrics seront créées dans assets/eval/.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("© 2025 **HealthPredict AI** — Contact : gemimakerenondelepourou@gmail.com — Licence MIT")
st.markdown("**GitHub :** HealthPredict AI &nbsp;|&nbsp; **LinkedIn :** Gemima Ondele")
st.markdown("**Version :** 1.0.0 — 2025-01-01 &nbsp;|&nbsp; **Contributeurs :** Gemima Ondele")
st.markdown("**Technologies :** Streamlit, Pandas, Altair, Matplotlib, Scikit-learn, SHAP, Transformers (optionnel)")
st.markdown("**Remerciements :** OpenAI, Hugging Face (CamemBERT), et la communauté.")
st.markdown("**Note :** Démo éducative — ne pas utiliser pour décisions médicales sans validation experte.")
st.markdown("**Avertissement :** Basé sur données historiques, pas de garantie de précision future.")
st.markdown("**Confidentialité :** Pas de données personnelles stockées.")
st.markdown("**Licence :** MIT — voir LICENSE.")
st.markdown("**Contribuer :** Issues & PR bienvenues.")
st.markdown("**Support :** Contact email/LinkedIn.")
