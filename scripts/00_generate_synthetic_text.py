# Génère un mini CSV d'exemple (FR/EN) pour tester rapidement le pipeline/app.
import os, random, pandas as pd
os.makedirs("data/samples", exist_ok=True)

samples = [
    ("Critique - IRM arrêtée, surchauffe détectée, fumée au niveau de l'alimentation", 1),
    ("Modérée - Scanner CT produit un bruit anormal, redémarrage requis", 0),
    ("Faible - Message d'alarme intermittent, aucun incident patient", 0),
    ("Critique - X-ray panel short circuit causing shutdown", 1),
    ("Modérée - Fluoroscopy latency observed, device rebooted", 0),
]
df = pd.DataFrame({"text": [s[0] for s in samples], "label_critique": [s[1] for s in samples]})
df["text_clean"] = df["text"].str.lower()
df.to_csv("data/samples/synthetic_small.csv", index=False, encoding="utf-8")
print("[OK] data/samples/synthetic_small.csv")
