
import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import matplotlib.pyplot as plt

# Try to import XGBoost; fallback if unavailable
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# SHAP optional
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

DATA_PATH = Path('data/synthetic_readmission.csv')
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop('readmitted_30d', axis=1)
y = df['readmitted_30d']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Model
if HAS_XGB:
    model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
else:
    model = GradientBoostingClassifier(random_state=42)

pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('clf', model)])
pipe.fit(X_train, y_train)

# Metrics
proba = pipe.predict_proba(X_test)[:,1]
pred = (proba>=0.5).astype(int)
metrics = {
    'auc_roc': float(roc_auc_score(y_test, proba)),
    'auc_pr': float(average_precision_score(y_test, proba)),
    'f1': float(f1_score(y_test, pred)),
    'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
    'model': 'XGBoost' if HAS_XGB else 'GradientBoostingClassifier'
}
with open(OUT/'metrics.json','w') as f: json.dump(metrics, f, indent=2)
print('Metrics:', metrics)

# Plots
from sklearn.metrics import roc_curve, precision_recall_curve
fpr, tpr, _ = roc_curve(y_test, proba)
prec, rec, _ = precision_recall_curve(y_test, proba)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(fpr,tpr,label='ROC')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.subplot(1,2,2)
plt.plot(rec,prec,label='PR')
plt.title('Precision-Recall')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.tight_layout(); plt.savefig(OUT/'roc_pr_curves.png', dpi=180)

# SHAP summary
if HAS_SHAP and HAS_XGB:
    explainer = shap.TreeExplainer(pipe.named_steps['clf'])
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout(); plt.savefig(OUT/'shap_summary.png', dpi=180)
else:
    with open(OUT/'shap_note.txt','w') as f:
        f.write('Install xgboost and shap to generate SHAP plots: pip install xgboost shap')

# Save model
joblib.dump(pipe, OUT/'model.pkl')
