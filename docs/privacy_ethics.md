# Confidentialité, Éthique & Biais – HealthPredict AI

## Données
- Données publiques OpenFDA + documents de test anonymisés (ex. Joigny).
- Pas de données personnelles dans le repo. Traces locales uniquement (`data/app.db`).

## Finalité
- Démonstrateur académique de maintenance prédictive sur textes d’incidents.
- Non destiné au diagnostic médical ni à une décision clinique.

## Biais & Limites
- Données OpenFDA (majoritairement EN) => traduction automatique FR possible.
- Classe "critique" = heuristique (Death/Injury) → peut sur-représenter la gravité.
- OCR bruyant ⇒ risques d’erreurs de lecture.

## Mesures d’atténuation
- Échantillonnage équilibré pour l’entraînement.
- Seuil ajustable (0.5 par défaut).
- Explicabilité TF-IDF (mots-clés contributifs).
- Journalisation des versions de modèles.

## Base légale (cadre académique)
- Usage pédagogique, sans traitement de données personnelles.
- Si déploiement réel : réaliser une AIPD (DPIA) et contractualiser.
