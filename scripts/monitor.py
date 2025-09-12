# 🧩 Fichier `scripts/monitor.py`
#Il est autonome.

from __future__ import annotations
import argparse
import os
import time
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests

DEFAULT_API = "http://localhost:8000/health"
DEFAULT_DB = os.environ.get("HP_DB", str(Path("data") / "app.db"))

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("healthcheck")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

def check_api(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return True, f"API OK ({r.status_code})"
        return False, f"API BAD STATUS ({r.status_code})"
    except Exception as e:
        return False, f"API ERROR: {e}"

def check_db(db_path: str) -> tuple[bool, str]:
    try:
        con = sqlite3.connect(db_path)
        try:
            tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            names = {t[0] for t in tables}
            if "predictions" in names:
                return True, "DB OK (table predictions trouvée)"
            else:
                return False, f"DB OK mais table 'predictions' absente (tables: {sorted(names)})"
        finally:
            con.close()
    except Exception as e:
        return False, f"DB ERROR: {e}"

def once(api_url: str, db_path: str, logger: logging.Logger) -> int:
    ok_api, msg_api = check_api(api_url)
    ok_db, msg_db = check_db(db_path)
    if ok_api and ok_db:
        logger.info(f"{msg_api} | {msg_db}")
        return 0
    if not ok_api and ok_db:
        logger.warning(f"{msg_api} | {msg_db}")
        return 1
    if ok_api and not ok_db:
        logger.warning(f"{msg_api} | {msg_db}")
        return 2
    logger.error(f"{msg_api} | {msg_db}")
    return 3

def main():
    p = argparse.ArgumentParser(description="HealthPredict monitoring (API & DB).")
    p.add_argument("--api-url", default=DEFAULT_API, help="URL /health de l'API FastAPI")
    p.add_argument("--db", default=DEFAULT_DB, help="Chemin SQLite (HP_DB)")
    p.add_argument("--interval", type=int, default=300, help="Intervalle en secondes (0 = exécuter une fois)")
    p.add_argument("--log", default=str(Path("logs") / "healthcheck.log"), help="Fichier de log (rotation)")
    args = p.parse_args()

    logger = setup_logger(Path(args.log))
    logger.info(f"Start monitor | api={args.api_url} | db={args.db} | interval={args.interval}s")

    if args.interval <= 0:
        code = once(args.api_url, args.db, logger)
        raise SystemExit(code)

    while True:
        code = once(args.api_url, args.db, logger)
        time.sleep(max(1, args.interval))

if __name__ == "__main__":
    main()
