# -*- coding: utf-8 -*-
import requests, time, json, csv, os, math
from datetime import datetime
import yaml

CONFIG = "config/config.yaml"

def load_cfg():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def build_query(keyword, start, end):
    # IMPORTANT: pour device/event, le champ de date est `date_received` (YYYYMMDD).
    # Les textes narratifs sont dans l'array `mdr_text[].text`.
    date_range = f"(date_received:[{start.replace('-','')}+TO+{end.replace('-','')}])"
    # On cible brand_name / generic_name et le narratif mdr_text.text
    text_filter = f"(device.generic_name:\"{keyword}\" OR device.brand_name:\"{keyword}\" OR mdr_text.text:\"{keyword}\")"
    return f"{date_range}+AND+{text_filter}"

def fetch_page(base_url, query, limit, skip):
    url = f"{base_url}?search={query}&limit={limit}&skip={skip}"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        # OpenFDA renvoie 404 quand skip dépasse le nombre total de résultats
        return None, 0
    r.raise_for_status()
    data = r.json()
    total = data.get("meta", {}).get("results", {}).get("total", 0)
    results = data.get("results", [])
    return results, total

def extract_rows(results):
    rows = []
    for it in results:
        # champs utiles (robuste aux absences)
        date_received = it.get("date_received")
        event_type = it.get("event_type")
        brand_names = []
        generic_names = []
        for dev in it.get("device", []) or []:
            if "brand_name" in dev and dev["brand_name"]:
                brand_names.append(dev["brand_name"])
            if "generic_name" in dev and dev["generic_name"]:
                generic_names.append(dev["generic_name"])

        # concat narratifs
        texts = []
        for t in it.get("mdr_text", []) or []:
            tx = t.get("text")
            if tx:
                texts.append(tx)
        narrative = " ".join(texts).strip()

        rows.append({
            "date_received": date_received,
            "event_type": event_type,
            "brand_name": "; ".join(sorted(set(brand_names))) if brand_names else "",
            "generic_name": "; ".join(sorted(set(generic_names))) if generic_names else "",
            "event_text": narrative
        })
    return rows

def main():
    cfg = load_cfg()
    base_url = cfg["openfda"]["base_url"]
    start = cfg["openfda"]["start_date"]
    end = cfg["openfda"]["end_date"]
    limit = int(cfg["openfda"]["limit"])
    max_records = int(cfg["openfda"]["max_records"])
    sleep_sec = float(cfg["openfda"]["sleep_sec"])
    keywords = cfg["openfda"]["keywords"]

    out_csv = cfg["paths"]["raw_csv"]
    out_jsonl = cfg["paths"]["raw_jsonl"]
    ensure_dirs(out_csv)
    ensure_dirs(out_jsonl)

    all_rows = []
    with open(out_jsonl, "w", encoding="utf-8") as fj:
        pass  # clear file

    for kw in keywords:
        print(f"[INFO] Téléchargement pour mot-clé: {kw}")
        q = build_query(kw, start, end)
        skip = 0
        total_seen = 0
        try:
            while True:
                results, total = fetch_page(base_url, q, limit, skip)
                if results is None or not results:
                    break
                rows = extract_rows(results)
                all_rows.extend(rows)
                # append jsonl pour archivage
                with open(out_jsonl, "a", encoding="utf-8") as fj:
                    for r in results:
                        fj.write(json.dumps(r, ensure_ascii=False) + "\n")
                total_seen += len(rows)
                skip += limit
                if total_seen >= max_records:
                    break
                time.sleep(sleep_sec)
        except requests.HTTPError as e:
            print(f"[WARN] HTTP pour {kw}: {e}")
        except Exception as e:
            print(f"[WARN] Erreur {kw}: {e}")

    if not all_rows:
        print("[INFO] Aucun enregistrement collecté.")
        return

    # dédoublonnage simple par (date_received, brand_name, generic_name, hash_text)
    seen = set()
    dedup = []
    for r in all_rows:
        key = (r["date_received"], r["brand_name"], r["generic_name"], hash(r["event_text"]))
        if key not in seen and r["event_text"]:
            seen.add(key)
            dedup.append(r)

    # export CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date_received","event_type","brand_name","generic_name","event_text"])
        w.writeheader()
        for r in dedup:
            w.writerow(r)

    print(f"[OK] Sauvegardé {len(dedup)} lignes -> {out_csv}")

if __name__ == "__main__":
    main()
