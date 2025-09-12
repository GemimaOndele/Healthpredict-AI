#docs/data_dictionary.md (schéma minimal)

Fichier : assets/data/processed/medical_imaging_text_labeled.csv

- event_text : str — description de l’incident (UTF-8, nettoyé).
- event_type : str — catégorie si disponible (Death/Injury/Malfunction/Other...).
- label : int — 1 = critique (death/injury), 0 = non critique (sinon).
- date_received : str (YYYY-MM-DD) — si présente dans la source OpenFDA.

Qualité (post-validation) :
- Lignes : ~33k
- Duplicats event_text retirés : yes (compteur dans le validateur)
- Longueur médiane (car.) : ~2 264

# Dictionnaire de données — `medical_imaging_text_labeled.csv`

| Colonne         | Type     | Obligatoire | Description                                                                 |
|-----------------|----------|-------------|------------------------------------------------------------------------------|
| event_text      | string   | Oui         | Texte descriptif de l’incident / alerte.                                    |
| event_type      | string   | Non         | Catégorie FDA : `Death`, `Injury`, `Malfunction`, `Other`, etc.              |
| label           | int (0/1)| Oui         | Cible binaire : 1=critique (mort/blessure), 0=non critique.                 |
| date_received   | date/str | Non         | Date de réception (format `YYYYMMDD` ou ISO si convertie).                  |
| brand_name      | string   | Non         | Marque de l’équipement (si disponible).                                     |
| generic_name    | string   | Non         | Type générique (si disponible).                                             |
| TypeDetecte     | string   | Non         | Type déduit (IRM, Scanner, Radiologie, …) — ajouté par l’application.       |

**Remarque :** Le minimum requis pour l’entraînement TF-IDF est `event_text` + `label`.
