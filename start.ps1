#requires -Version 5.1
<# start.ps1 — HealthPredict AI (Windows PowerShell)
   1) (option) active le venv .henv/venv/.venv
   2) télécharge les modèles spaCy (si absents)
   3) télécharge les assets (si absents)
   4) (option) évalue si HP_RUN_EVAL=1
   5) lance Streamlit
#>

$ErrorActionPreference = 'Stop'

# --- Racine projet ---
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

# --- (option) Activer venv ---
$venvCandidates = @(
    (Join-Path $ROOT ".henv\Scripts\Activate.ps1")
    (Join-Path $ROOT "venv\Scripts\Activate.ps1")
    (Join-Path $ROOT ".venv\Scripts\Activate.ps1")
)
$activated = $false
foreach ($cand in $venvCandidates) {
    if (Test-Path $cand) {
        Write-Host "🔹 Activation venv: $cand"
        . $cand
        $activated = $true
        break
    }
}
if (-not $activated) {
    Write-Host "ℹ️ Aucun venv détecté (.henv/venv/.venv). Python global utilisé."
}

# --- Vars d'env utiles ---
$env:HP_USE_CAMEMBERT = "1"   # active CamemBERT si dispo

# --- 1) spaCy models (idempotent) ---
Write-Host "⬇️ spaCy: en_core_web_sm"
& python -m spacy download en_core_web_sm
if ($LASTEXITCODE -ne 0) { Write-Warning "en_core_web_sm non installé (ok si déjà présent)." }

Write-Host "⬇️ spaCy: fr_core_news_sm"
& python -m spacy download fr_core_news_sm
if ($LASTEXITCODE -ne 0) { Write-Warning "fr_core_news_sm non installé (ok si déjà présent)." }

# --- 2) Assets (CSV/modèles) ---
if (Test-Path (Join-Path $ROOT "scripts\download_assets.py")) {
    Write-Host "⬇️ Téléchargement des assets…"
    & python -m scripts.download_assets
    if ($LASTEXITCODE -ne 0) { Write-Warning "download_assets a renvoyé un code ≠ 0 (peut être normal si déjà présents)." }
} else {
    Write-Warning "scripts\download_assets.py introuvable — étape ignorée."
}

# --- 3) (option) Évaluation ---
if ($env:HP_RUN_EVAL -eq "1") {
    $evalScript = Join-Path $ROOT "notebooks\eval_healthpredict.py"
    if (Test-Path $evalScript) {
        Write-Host "🧪 Évaluation du modèle…"
        & python $evalScript
        if ($LASTEXITCODE -ne 0) { Write-Warning "Évaluation terminée avec un code ≠ 0 (vérifie config/config.yaml & assets)." }
    } else {
        Write-Warning "notebooks\eval_healthpredict.py introuvable — étape ignorée."
    }
} else {
    Write-Host "⏭️ Évaluation sautée (définis HP_RUN_EVAL=1 pour l’activer)."
}

# --- 4) Streamlit ---
$addr = "0.0.0.0"
$port = if ($env:PORT) { $env:PORT } else { "8501" }
$app  = Join-Path $ROOT "app\healthpredict_app.py"
if (-not (Test-Path $app)) { throw "Fichier app introuvable: $app" }

Write-Host "🚀 Lancement Streamlit sur ${addr}:${port}"
& python -m streamlit run $app --server.address=$addr --server.port=$port
