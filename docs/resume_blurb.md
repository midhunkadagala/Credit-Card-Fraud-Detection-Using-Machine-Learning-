# Resume-Ready Project Description

## Short version (2–3 lines)

**Explainable Credit Card Fraud Detection Using Machine Learning** — Built a local Streamlit application that validates a user-uploaded transaction CSV, trains Logistic Regression, Random Forest, and Isolation Forest with leakage-safe SMOTE preprocessing, evaluates models with imbalance-aware metrics (F1, ROC-AUC, PR-AUC), and explains predictions using SHAP.

## Bullet version (resume project section)

- Designed an end-to-end fraud detection pipeline in Python (Pandas, Scikit-learn, Imbalanced-learn) with mandatory runtime CSV upload and schema validation (`Time`, `Amount`, `V1`–`V28`, `Class`).
- Compared supervised and unsupervised models (Logistic Regression, Random Forest, Isolation Forest); applied SMOTE only on training data to prevent leakage.
- Implemented evaluation dashboards (confusion matrices, ROC/PR curves) and SHAP-based global/local explanations via an interactive Streamlit UI.

## LinkedIn / GitHub About blurb

Developed an explainable credit card fraud detection system for academic demonstration: manual dataset upload, multi-model comparison under severe class imbalance, and SHAP interpretability — fully runnable offline on Windows.
