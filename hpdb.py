# hpdb.py
# Mini couche SQLite: init, insertion, requêtes d'historique

from __future__ import annotations
import sqlite3, json, os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Dossier data/ par défaut à la racine du projet
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Chemin DB configurable via HP_DB, sinon data/app.db
DB_PATH_ENV = os.environ.get("HP_DB", str(DATA_DIR / "app.db"))

def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH_ENV
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[str] = None) -> None:
    """Crée le fichier DB + la table si besoin."""
    path = db_path or DB_PATH_ENV
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            source TEXT,
            file_name TEXT,
            input_text TEXT,
            cleaned_text TEXT,
            model_type TEXT,
            label TEXT,
            proba REAL,
            detected_type TEXT,
            src_lang TEXT,
            translated INTEGER,
            top_keywords TEXT
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_source ON predictions(source);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_file ON predictions(file_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions(label);")
        con.commit()

def insert_prediction(
    *,
    source: str,
    file_name: Optional[str],
    input_text: str,
    cleaned_text: str,
    model_type: str,
    label: str,
    proba: float,
    detected_type: Optional[str] = None,
    src_lang: Optional[str] = None,
    translated: bool = False,
    top_keywords: Optional[List] = None,
    db_path: Optional[str] = None,
) -> int:
    """Insère une prédiction et renvoie l'id."""
    payload = json.dumps(top_keywords or [], ensure_ascii=False)
    with _connect(db_path) as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO predictions
            (source, file_name, input_text, cleaned_text, model_type, label, proba,
             detected_type, src_lang, translated, top_keywords)
            VALUES
            (:source, :file_name, :input_text, :cleaned_text, :model_type, :label, :proba,
             :detected_type, :src_lang, :translated, :top_keywords)
        """, {
            "source": source,
            "file_name": file_name,
            "input_text": input_text,
            "cleaned_text": cleaned_text,
            "model_type": model_type,
            "label": label,
            "proba": float(proba) if proba is not None else None,
            "detected_type": detected_type,
            "src_lang": src_lang,
            "translated": 1 if translated else 0,
            "top_keywords": payload,
        })
        con.commit()
        return int(cur.lastrowid)

def fetch_recent_predictions(limit: int = 200, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with _connect(db_path) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT * FROM predictions
            ORDER BY datetime(ts) DESC
            LIMIT ?
        """, (int(limit),))
        rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["top_keywords"] = json.loads(d.get("top_keywords") or "[]")
        except Exception:
            d["top_keywords"] = []
        out.append(d)
    return out

def search_predictions(query: str, limit: int = 200, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    q = f"%{query.strip()}%"
    with _connect(db_path) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT * FROM predictions
            WHERE COALESCE(file_name,'') LIKE ? OR COALESCE(input_text,'') LIKE ?
            ORDER BY datetime(ts) DESC
            LIMIT ?
        """, (q, q, int(limit)))
        rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["top_keywords"] = json.loads(d.get("top_keywords") or "[]")
        except Exception:
            d["top_keywords"] = []
        out.append(d)
    return out
