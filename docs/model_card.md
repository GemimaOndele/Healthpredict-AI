# Model Card – HealthPredict AI

## Aperçu
- Deux familles: TF-IDF + Logistic Regression (baseline), CamemBERT + LR (avancé).
- Tâche: classification binaire "Critique / Non critique" d’un rapport d’incident.

## Données
- Source: OpenFDA (device events) + textes de test anonymisés.
- Taille traitée: ~33k lignes (voir `assets/data/processed/...`).

## Métriques (exemple récent)
- TF-IDF: Accuracy ~0.90, F1 ~0.92, ROC-AUC ~0.96 (voir `assets/eval/metrics.json`).
- CamemBERT (run local CPU): résultats variables selon ressources.

## Limites
- Label “critique” défini par proxy (Death/Injury) → simplification.
- Sensibles au bruit OCR et aux textes très courts.

## Usage responsable
- Indicateur d’aide à la priorisation, pas une décision médicale.
- Exiger revue humaine avant action.

# En résumé : 

But : Prédire la criticité (critique / non critique) d’un rapport texte.

Jeux de données : OpenFDA (Device Event) + rapports de test anonymisés (Joigny).
Prétraitements : normalisation, nettoyage, tokenisation simple.

Modèles :
- TF-IDF + LogisticRegression (baseline, interprétable)
- CamemBERT + LogisticRegression (FR, plus coûteux)

Métriques (val jeu de test) :
- Accuracy ~0.85 | F1 ~0.82 | AUC ~0.87 (baseline)
Limites : bruit, déséquilibres, OCR imparfait, dataset US centré.
Usage : démonstration pédagogique, pas d’usage clinique.
