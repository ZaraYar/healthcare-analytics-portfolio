
# Readmission Prediction (XGBoost; SHAP)

Predict 30-day hospital readmission from synthetic encounter features.

## Highlights
- **Model:** XGBoost classifier (fallback to scikit-learn GradientBoosting if XGBoost not installed)
- **Interpretability:** SHAP values with summary and feature-importance plots
- **Metrics:** AUC, PR-AUC, F1, confusion matrix
- **Reproducible pipeline:** deterministic seeds, saved model/artifacts in `outputs/`

## Data
Synthetic dataset generated for demonstration: `data/synthetic_readmission.csv` with columns such as:
- `age`, `length_of_stay`, `prior_admissions`, `num_diagnoses`, `chronic_index`, `medication_count`, `lab_abnormality_score`, `utilization_last_6m`, `wellness_engagement_score`, `enrollment_tenure_months`, `readmitted_30d` (target)

> Replace with your de-identified real data if available.

## Run
```bash
pip install -r requirements.txt
python src/train_model.py
```
Artifacts will be saved in `outputs/`:
- `model.pkl`, `metrics.json`, `shap_summary.png`, `roc_pr_curves.png`

## Requirements
See `requirements.txt`.

## License
MIT (see `LICENSE`).
