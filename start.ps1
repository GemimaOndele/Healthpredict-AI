#requires -Version 5.1
<# start.ps1 — HealthPredict AI (Windows PowerShell)
   1) (option) active le venv .henv/venv/.venv
   2) télécharge les modèles spaCy (si absents)
   3) télécharge les assets (si absents)
   4) (option) évalue si HP_RUN_EVAL=1
   5) lance Streamlit
#>
Param(
  [int]$Port = 8501
)

Write-Host "🔹 Activation venv (.henv)" -ForegroundColor Cyan
$venv = Join-Path $PSScriptRoot ".henv\Scripts\Activate.ps1"
if (!(Test-Path $venv)) {
  Write-Host "⚠️ .henv introuvable, création..." -ForegroundColor Yellow
  python -m venv .henv
}
. $venv

Write-Host "📦 pip install -r requirements.txt" -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

# (optionnel) Torch CPU déjà installé chez vous, sinon:
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# spaCy modèles: uniquement si demandé
if ($env:HP_USE_SPACY -eq "1") {
  Write-Host "⬇️ spaCy models (fr/en)..." -ForegroundColor Cyan
  python -c "import spacy" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "spaCy non installé ? (pip l’a pourtant installé). On continue." -ForegroundColor Yellow
  }
  python -m spacy download fr_core_news_sm
  python -m spacy download en_core_web_sm
} else {
  Write-Host "⏭️ HP_USE_SPACY≠1 → on n'installe pas les modèles spaCy." -ForegroundColor DarkGray
}

# Assets (modèles déjà publiés sur Hugging Face)
Write-Host "⬇️ Téléchargement des assets…" -ForegroundColor Cyan
python -c "import huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) { pip install huggingface_hub }
python scripts/download_assets.py

# DB par défaut si non définie
if (-not $env:HP_DB) { $env:HP_DB = (Join-Path $PSScriptRoot "data\app.db") }

# Lancer Streamlit
Write-Host ""
Write-Host "🚀 Lancement Streamlit sur 0.0.0.0:$Port" -ForegroundColor Green
streamlit run app/healthpredict_app.py --server.address=0.0.0.0 --server.port=$Port
