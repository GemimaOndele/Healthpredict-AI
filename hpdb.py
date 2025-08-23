# hpdb.py
# Mini couche SQLite: init, insertion, requêtes d'historique (avec ts auto)
from __future__ import annotations
import sqlite3, json, os, pathlib, datetime as dt
from typing import Iterable, Dict, Any, List, Optional

_DB_PATH: Optional[str] = None

def _ensure_parent_dir(path: str) -> None:
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

def _connect(path_override: Optional[str] = None) -> sqlite3.Connection:
    dbp = path_override or _DB_PATH
    if not dbp:
        raise RuntimeError("DB non initialisée. Appelez init_db() d’abord.")
    conn = sqlite3.connect(dbp, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = "data/app.db") -> None:
    """Crée le fichier + tables si besoin puis retient le chemin par défaut."""
    global _DB_PATH
    _DB_PATH = db_path
    _ensure_parent_dir(db_path)
    with _connect() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT NOT NULL,
            source        TEXT NOT NULL,     -- 'input' | 'doc'
            file_name     TEXT,
            input_text    TEXT,
            cleaned_text  TEXT,
            model_type    TEXT,              -- 'TFIDF' | 'CamemBERT'
            label         TEXT,              -- 'Critique' | 'Pas critique'
            proba         REAL,
            detected_type TEXT,
            src_lang      TEXT,
            translated    INTEGER,           -- 0/1
            top_keywords  TEXT               -- JSON: [["mot", pct], ...]
        )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts DESC)")
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
    detected_type: str,
    src_lang: str,
    translated: bool,
    top_keywords: Optional[Iterable] = None,
    db_path: Optional[str] = None,
) -> int:
    """Insère une ligne et renvoie l'id (ts auto en UTC)."""
    if top_keywords is None:
        top_keywords = []
    payload = json.dumps(list(top_keywords), ensure_ascii=False)

    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with _connect(db_path) as con:
        cur = con.execute("""
            INSERT INTO predictions
            (ts, source, file_name, input_text, cleaned_text, model_type, label, proba,
             detected_type, src_lang, translated, top_keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts,
            source,
            file_name,
            input_text,
            cleaned_text,
            model_type,
            label,
            float(proba),
            detected_type,
            src_lang,
            1 if translated else 0,
            payload
        ))
        con.commit()
        return int(cur.lastrowid)

def fetch_recent_predictions(limit: int = 200, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    with _connect(db_path) as con:
        rows = con.execute("""
            SELECT id, ts, source, file_name, model_type, label, proba,
                   detected_type, src_lang, translated, top_keywords
            FROM predictions
            ORDER BY ts DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["top_keywords"] = json.loads(d.get("top_keywords") or "[]")
        except Exception:
            d["top_keywords"] = []
        out.append(d)
    return out

def search_predictions(q: str, limit: int = 100, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    like = f"%{q}%"
    with _connect(db_path) as con:
        rows = con.execute("""
            SELECT id, ts, source, file_name, model_type, label, proba,
                   detected_type, src_lang, translated, top_keywords
            FROM predictions
            WHERE COALESCE(file_name,'') LIKE ? OR COALESCE(input_text,'') LIKE ?
            ORDER BY ts DESC
            LIMIT ?
        """, (like, like, int(limit))).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["top_keywords"] = json.loads(d.get("top_keywords") or "[]")
        except Exception:
            d["top_keywords"] = []
        out.append(d)
    return out
