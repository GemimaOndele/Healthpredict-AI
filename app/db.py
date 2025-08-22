# db.py
from __future__ import annotations
import sqlite3, json, os, pathlib, datetime as dt
from typing import Iterable, Dict, Any, List, Optional

_DB_PATH: Optional[str] = None

def _ensure_parent_dir(path: str) -> None:
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

def _connect() -> sqlite3.Connection:
    if not _DB_PATH:
        raise RuntimeError("DB non initialisée. Appelez init_db() d’abord.")
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = "data/app.db") -> None:
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
            model_type    TEXT,
            label         TEXT,
            proba         REAL,
            detected_type TEXT,
            src_lang      TEXT,
            translated    INTEGER,
            top_keywords  TEXT
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
) -> int:
    if top_keywords is None:
        top_keywords = []
    payload = json.dumps(list(top_keywords), ensure_ascii=False)
    with _connect() as con:
        cur = con.execute("""
            INSERT INTO predictions
            (ts, source, file_name, input_text, cleaned_text, model_type, label, proba,
             detected_type, src_lang, translated, top_keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            source, file_name, input_text, cleaned_text, model_type, label, float(proba),
            detected_type, src_lang, 1 if translated else 0, payload
        ))
        con.commit()
        return int(cur.lastrowid)

def fetch_recent_predictions(limit: int = 200) -> List[Dict[str, Any]]:
    with _connect() as con:
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

def search_predictions(q: str, limit: int = 100) -> List[Dict[str, Any]]:
    like = f"%{q}%"
    with _connect() as con:
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
