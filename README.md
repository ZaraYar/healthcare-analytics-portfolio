
# Zahra Farzanyar — Healthcare Data Science Portfolio

This repository contains two recruiter-ready projects with documentation and runnable code:

1. **readmission-prediction/** — Binary classification for 30-day hospital readmission using XGBoost with SHAP interpretability.
2. **llm-note-summarization/** — Clinical note summarization using Azure OpenAI (synthetic, de-identified samples) with prompt templates and an evaluation script.

> **Privacy Notice:** All data in this repo is synthetic and de-identified. No PHI (Protected Health Information) is included.

## Quick Start
```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install per-project requirements and run
cd readmission-prediction
pip install -r requirements.txt
python src/train_model.py

cd ../llm-note-summarization
pip install -r requirements.txt
# Set Azure OpenAI environment variables before running (see README)
python src/summarize_notes.py --input data/sample_notes.json --output outputs/summaries.json
python src/evaluate_summaries.py --input outputs/summaries.json
```

## How to publish to GitHub
```bash
git init
git add .
git commit -m "Add readmission prediction and LLM summarization projects"
# Create a new GitHub repo, then:
git remote add origin https://github.com/ZaraYar/healthcare-analytics-portfolio.git
git branch -M main
git push -u origin main
```

---
Generated on 2025-12-09 19:22 UTC.
