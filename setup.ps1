<#
.SYNOPSIS
    MagazineIQ Setup Script
.DESCRIPTION
    Installs required pip packages, downloads the SpaCy en_core_web_sm model, 
    and caches the Hugging Face summarization model so that it is available offline.
#>

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MagazineIQ Environment Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Install pip requirements
Write-Host "[1/3] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install pip dependencies. Please check your internet connection."
    exit $LASTEXITCODE
}
Write-Host "Dependencies installed successfully.`n" -ForegroundColor Green

# 2. Download SpaCy Model
Write-Host "[2/3] Downloading SpaCy language model (en_core_web_sm)..." -ForegroundColor Yellow
python -m spacy download en_core_web_sm
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download SpaCy model."
    exit $LASTEXITCODE
}
Write-Host "SpaCy model downloaded successfully.`n" -ForegroundColor Green

# 3. Cache Hugging Face Model
Write-Host "[3/3] Downloading and caching Hugging Face summarization model (facebook/bart-large-cnn)..." -ForegroundColor Yellow
Write-Host "      This might take a few minutes as the model is large (~1.6GB)."
$pythonScript = @"
import os
import sys

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    
    print('Downloading summarizer (facebook/bart-base)...')
    model_name = 'facebook/bart-base'
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print('Downloading topic classifier (typeform/distilbert-base-uncased-mnli)...')
    pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli', device=-1)
    
    print('Downloading sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...')
    pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=-1)
    
    print('All models downloaded and cached successfully!')
except Exception as e:
    print(f'Error caching model: {e}', file=sys.stderr)
    sys.exit(1)
"@

$pythonScript | python -
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Failed to cache the Hugging Face model. The application will use the offline TextRank fallback."
} else {
    Write-Host "Hugging Face model cached successfully.`n" -ForegroundColor Green
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! You can now run:" -ForegroundColor Cyan
Write-Host "  python -m streamlit run app.py" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
