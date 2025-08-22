# -*- coding: utf-8 -*-
# HealthPredict AI — Dashboard + Prédiction + 📎 Documents + 📚 Historique & Prévisions
# OCR images & PDF (Tesseract + Poppler), Similarité historique, Traduction FR, Exports CSV/Excel, Historique SQLite

# app/healthpredict_app.py
import os, io, sys, json, joblib, yaml, warnings, unicodedata, re
from pathlib import Path
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt  # noqa: F401 (parfois utile)
import numpy as np
import datetime as dt  # noqa: F401
from math import sqrt
from typing import Optional

# ========= Fix chemins: ajouter la racine du projet au PYTHONPATH =========
APP_DIR = Path(__file__).resolve().parent           # .../app
ROOT_DIR = APP_DIR.parent                           # projet/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ✅ DOIT ÊTRE AVANT TOUT AUTRE st.* :
st.set_page_config(page_title="HealthPredict AI", layout="wide")

# =========================
# Options lourdes (désactivables)
# =========================
USE_CAMEMBERT = os.getenv("HP_USE_CAMEMBERT", "0") == "1"  # active aussi Transformers/torch si 1

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
from dotenv import load_dotenv  # type: ignore
load_dotenv()

def _safe_ensure_assets():
    """Appelle scripts/download_assets.ensure_assets() si dispo."""
    try:
        from scripts.download_assets import ensure_assets  # nécessite ROOT_DIR dans sys.path
    except Exception as e:
        st.warning(f"Téléchargement des assets ignoré: {e}")
        return
    try:
        if os.environ.get("HP_AUTO_DOWNLOAD", "1") == "1":
            ensure_assets()
    except Exception as e:
        st.warning(f"Téléchargement des assets échoué: {e}")

_safe_ensure_assets()

# ----- Base SQLite -----
DB_ENABLED = True
try:
    import db  # fichier db.py à la racine
    DB_PATH = os.environ.get("HP_DB", str(ROOT_DIR / "data" / "app.db"))
    db.init_db(DB_PATH)
except Exception as e:
    DB_ENABLED = False
    st.warning(f"Historique SQLite désactivé ({e}).")

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

# --- spaCy optionnel (pour extras UI, pas dans le pipeline TF-IDF) ---
USE_SPACY = os.getenv("HP_USE_SPACY", "0") == "1"
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
    """Retourne une chaîne sûre ('' pour NaN/None)."""
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

def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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

    # 1) processed_csv (config puis fallback assets/)
    cand_processed = []
    if paths.get("processed_csv"):
        cand_processed.append(paths["processed_csv"])
    cand_processed.append(str(ROOT_DIR / "assets" / "data" / "processed" / "medical_imaging_text_labeled.csv"))

    for p in cand_processed:
        df = load_csv_safe(p)
        if not df.empty:
            return df

    # 2) raw_csv (config puis fallback assets/)
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
    if vec is None:
        return []
    X = vec.transform([cleaned_text])
    feats = vec.get_feature_names_out()
    idx = X.nonzero()[1]
    scores = X.data
    pairs = [(feats[i], float(scores[j])) for j, i in enumerate(idx)]
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:topk]
    s = sum(p for _, p in pairs) or 1.0
    return [(w, round(100.0 * p / s, 1)) for w, p in pairs]

# ======================= OCR helpers =======================
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
        return ""  # OCR coté UI

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

def detect_device_type_row(generic, brand, text) -> str:
    for s in (generic, brand, text):
        lab = detect_device_type_text(s)
        if lab:
            return lab
    return "Autre"

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

    # construire des Series vides si la colonne n'existe pas
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
    if vec is None:
        return None, None, None
    corpus = df_hist[text_col].fillna("").astype(str).tolist()
    X = vec.transform([clean_text(x) for x in corpus])
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
    vq = vec.transform([clean_text(text_query)])
    # norme requête robuste
    try:
        qn = float(np.sqrt(vq.multiply(vq).sum()))
    except Exception:
        qn = float(np.linalg.norm(vq.toarray()))
    if qn == 0.0:
        return []
    # produit puis division protégée
    sims = (X_hist @ vq.T).toarray().ravel()
    denom = (row_norms * qn)
    denom[denom == 0] = 1.0
    sims = sims / denom
    if not np.any(np.isfinite(sims)):
        return []
    top_idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), df_hist.iloc[int(i)]) for i in top_idx]

# ======================= Chargement & UI =======================
st.title("🔧 HealthPredict AI - Maintenance Prédictive des Équipements Médicaux")
st.markdown("**Optimisez la disponibilité, la sécurité et les coûts grâce à l’IA.**")

# Charger la config puis le dataset
cfg = load_cfg()
df_main = load_main_dataset(cfg)
if df_main.empty:
    st.error("Aucune donnée chargée (processed/raw). Vérifiez vos chemins de données dans config.yaml.")
    st.stop()

# ✅ Limitation douce pour préserver la RAM (modifiable via variable d'env)
MAX_ROWS_UI = int(os.getenv("HP_MAX_ROWS_UI", "200000"))
if len(df_main) > MAX_ROWS_UI:
    st.info(f"Aperçu limité aux {MAX_ROWS_UI} lignes sur {len(df_main)} (pour préserver la mémoire).")
    df_main = df_main.head(MAX_ROWS_UI).copy()

text_col_main = pick_text_column(df_main)
if "Recommandation" not in df_main.columns and "Alerte" in df_main.columns:
    df_main["Recommandation"] = df_main.apply(reco, axis=1)

df_main = add_detected_type(df_main)
df_main = add_parsed_date(df_main)

tab_dash, tab_pred, tab_docs, tab_hist = st.tabs(
    ["📊 Tableau de bord", "🤖 Prédiction", "📎 Documents", "📚 Historique & Prévisions"]
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

# ---- TAB Prédiction (texte saisi) ----
@st.cache_resource
def _pipe_tfidf(path):
    # chemin config puis fallback /assets
    p = path
    if not os.path.exists(p):
        fb = str(ROOT_DIR / "assets" / "models" / "healthpredict_model.joblib")
        if os.path.exists(fb):
            p = fb
    if not os.path.exists(p):
        st.error("Modèle TF-IDF introuvable. Activez HP_AUTO_DOWNLOAD=1 (scripts/download_assets.py) "
                 "ou placez assets/models/healthpredict_model.joblib.")
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
        st.error("Modèle CamemBERT introuvable. Activez HP_AUTO_DOWNLOAD=1 (scripts/download_assets.py) "
                 "ou placez assets/models/healthpredict_camembert_model.joblib.")
    return load_camembert(p)

def _predict_text(txt_for_model, which):
    cleaned = clean_text(txt_for_model)
    if which == "CamemBERT + IA":
        if not USE_CAMEMBERT:
            st.info("CamemBERT désactivé — utilisation du modèle TF-IDF.")
        else:
            try:
                tok, cam, clf = _camembert(cfg["paths"]["camembert_model"])
                proba = float(camembert_proba_fn([cleaned], tok, cam, clf)[0])
                label = "Critique" if proba >= 0.5 else "Pas critique"
                return label, proba, None
            except Exception as e:
                st.info(f"CamemBERT indisponible ({e}). Bascule TF-IDF.")
    pipe = _pipe_tfidf(cfg["paths"]["tfidf_model"])
    proba = float(pipe.predict_proba([cleaned])[0][1])
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

        label, proba, kw = _predict_text(txt_for_model, model_type)
        st.markdown("🔴 **ALERTE CRITIQUE** (p=%.2f)" % proba if label == "Critique" else "🟢 **Pas critique** (p=%.2f)" % proba)

        # 💾 Enregistrer dans SQLite (si activée)
        if DB_ENABLED:
            try:
                model_name = "CamemBERT" if model_type.startswith("CamemBERT") and USE_CAMEMBERT else "TFIDF"
                db.insert_prediction(
                    source="input",
                    file_name=None,
                    input_text=txt_for_model,
                    cleaned_text=clean_text(txt_for_model),
                    model_type=model_name,
                    label=label,
                    proba=proba,
                    detected_type=detect_device_type_text(txt_for_model) or "Autre",
                    src_lang=("en" if translated else "fr") if lang_choice == "Auto" else ("fr" if lang_choice=="Français" else "en"),
                    translated=bool(translated),
                    top_keywords=(kw or []),
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
                st.info("Vectorizer indisponible (modèle TF-IDF).")
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

    st.markdown("**Paramètres OCR**")
    use_ocr = st.toggle("Activer l’OCR pour images", value=True, key="toggle_use_ocr")
    use_pdf_ocr = st.toggle("Activer l’OCR pour PDF scannés", value=True, key="toggle_use_pdf_ocr")
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

                label, proba, _ = _predict_text(txt_for_model, model_docs)
                type_doc = detect_device_type_text(text) or "Autre"
                st.markdown(
                    f"**Résultat :** {'🔴 Critique' if label=='Critique' else '🟢 Pas critique'} — "
                    f"**Confiance : {proba:.2f}** — **Type détecté : {type_doc}**"
                )

                cleaned = clean_text(txt_for_model)
                kw = top_keywords_tfidf_from_pipe(pipe_cached, cleaned, topk=15)
                if kw:
                    df_kw = pd.DataFrame(kw, columns=["Mot-clé", "Contribution %"])
                    st.dataframe(df_kw, use_container_width=True)
                    for mot, pct in kw:
                        all_kw_rows.append({"Fichier": up.name, "Mot-clé": mot, "Contribution %": pct})

                if use_sim_docs and pipe_hist is not None:
                    st.caption("🔎 Cas similaires (top-5)")
                    sims = top_similar(txt_for_model, pipe_hist, X_hist, norms_hist, df_main, text_col_main, k=5)
                    if sims:
                        rows = []
                        for s, r in sims:
                            extrait = ensure_str(r.get(text_col_main, ""))
                            rows.append({
                                "Similarité": round(100*s,1),
                                "Date": ensure_str(r.get("date_received", r.get("date_dt", ""))),
                                "Type détecté": ensure_str(r.get("TypeDetecte", "")),
                                "Extrait": (extrait[:200] + "…") if len(extrait) > 200 else extrait,
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    else:
                        st.caption("Aucun similaire trouvé.")

                # 💾 Enregistrer dans SQLite (si activée)
                if DB_ENABLED:
                    try:
                        model_name = "CamemBERT" if model_docs.startswith("CamemBERT") and USE_CAMEMBERT else "TFIDF"
                        db.insert_prediction(
                            source="doc",
                            file_name=up.name,
                            input_text=txt_for_model,
                            cleaned_text=clean_text(txt_for_model),
                            model_type=model_name,
                            label=label,
                            proba=proba,
                            detected_type=type_doc,
                            src_lang=("en" if translated else "fr") if lang_docs == "Auto" else ("fr" if lang_docs=="Français" else "en"),
                            translated=bool(translated),
                            top_keywords=(kw or []),
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
        gr = m.groupby("mois").size().reset_index(name="Incidents")
        chart = alt.Chart(gr).mark_area(opacity=0.4).encode(x="mois:T", y="Incidents:Q").properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("Pas de dates exploitables.")

    st.markdown("**⏱️ Estimation du temps avant prochain incident (approx. Poisson)**")
    if "date_dt" in dft.columns and dft["date_dt"].notna().any():
        series = dft["date_dt"].dropna().sort_values()
        if len(series) >= 5:
            days = (series.max() - series.min()).days or 1
            lam = len(series) / days
            expected_gap = 1.0 / lam if lam > 0 else float("inf")
            last_date = series.max().date()
            next_expected = (pd.Timestamp(last_date) + pd.Timedelta(days=expected_gap)).date()
            lam_se = sqrt(len(series)) / days
            lam_low, lam_high = max(lam - 1.96 * lam_se, 1e-9), lam + 1.96 * lam_se
            gap_low, gap_high = 1.0 / lam_high, 1.0 / lam_low
            st.info(
                f"Taux observé λ ≈ **{lam:.3f}** inc./jour — écart moyen ≈ **{expected_gap:.1f} jours**.\n"
                f"Dernier incident: **{last_date}** → Prochain attendu autour du **{next_expected}** "
                f"(intervalle env. **{gap_low:.1f} – {gap_high:.1f} jours**)."
            )
        else:
            st.caption("Historique trop court pour estimer.")
    else:
        st.caption("Pas de dates exploitables.")

    st.markdown("**🌐 Traduction du dataset (EN → FR)**")
    col_tr1, col_tr2 = st.columns([2, 1])
    n_rows = col_tr1.slider("Nombre de lignes à traduire (pour test)", 100, 5000, 1000, step=100, key="tr_nrows")
    do_all = col_tr2.toggle("Traduire tout le dataset (attention aux perfs)", value=False, key="tr_all")
    if st.button("Lancer la traduction et exporter (Excel)", key="btn_trad_excel"):
        txtcol = pick_text_column(df_main)
        df_to_tr = df_main[[txtcol]].copy()
        if not do_all:
            df_to_tr = df_to_tr.head(n_rows).copy()
        texts = df_to_tr[txtcol].fillna("").astype(str).tolist()
        try:
            trad = translate_text(texts, src="en", tgt="fr")
        except Exception as e:
            st.error(f"Erreur de traduction: {e}")
            trad = texts
        out = df_to_tr.copy()
        out["texte_fr"] = trad
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="DatasetFR")
        st.download_button(
            "⬇️ Télécharger dataset traduit (Excel)",
            data=buf.getvalue(),
            file_name="dataset_traduit_fr.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_dataset_trad_xlsx",
        )

    # 🗂️ Historique SQLite
    st.markdown("---")
    st.subheader("🗂️ Historique des prédictions (SQLite)")
    if DB_ENABLED:
        try:
            search_q = st.text_input("Rechercher (nom de fichier ou texte)", value="", key="hist_search_q")
            limit = st.number_input("Nb max", min_value=10, max_value=1000, value=200, step=10, key="hist_limit")
            rows = db.search_predictions(search_q.strip(), limit=int(limit)) if search_q.strip() else db.fetch_recent_predictions(limit=int(limit))
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
