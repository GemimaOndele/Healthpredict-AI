# Collecte de jeux de données (Kaggle & co)

Pour constituer un corpus textuel médical (incidents, maintenance, rapports) :

- Kaggle
  - "Medical Device Adverse Events", "MAUDE device events", "FDA Medical Device Reports"
  - "Radiology reports", "MIMIC-CXR reports" (text), "RSNA" (plutôt images)
- PhysioNet
  - "MIMIC-CXR-JPG" + "MIMIC-CXR Reports" (nécessite accès, licence)
- OpenFDA (script 01_download_openfda.py)
  - Alimente un CSV textuel depuis la base MAUDE via l’API.

⚠️ Respecter les licences (usage académique/recueil anonymisé). Éviter toute donnée patient non anonymisée.
