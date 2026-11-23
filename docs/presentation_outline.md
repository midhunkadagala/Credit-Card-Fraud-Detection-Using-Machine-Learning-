# Project Presentation Outline (10–15 slides)

## Slide 1 — Title
- **Explainable Credit Card Fraud Detection Using Machine Learning**
- Student name, M.Tech, guide name, year

## Slide 2 — Agenda
- Problem → Objectives → Dataset → Method → Models → Results → XAI → Demo → Conclusion

## Slide 3 — Problem Statement
- Rare fraud events, high financial cost
- Need detection **and** explanation
- Local academic system; no cloud

## Slide 4 — Objectives
- Manual CSV upload + validation
- Compare LR, RF, Isolation Forest
- SMOTE without leakage
- SHAP explanations
- Streamlit dashboard

## Slide 5 — Dataset (Manual Upload)
- Kaggle Credit Card Fraud Detection schema
- Features: `Time`, `Amount`, `V1`–`V28`
- Target: `Class` (0 legit, 1 fraud)
- **Not bundled** — uploaded at runtime
- Extreme imbalance (~0.17% fraud typical)

## Slide 6 — System Architecture
- Show diagram from `docs/diagrams.md`
- UI → validation → preprocess → train → evaluate → SHAP / predict

## Slide 7 — Methodology
- Stratified split
- Scale on train only
- SMOTE on train only
- Isolation Forest without SMOTE
- Honest metrics from actual runs

## Slide 8 — Algorithms
- Logistic Regression (baseline, scaled)
- Random Forest (non-linear, importances)
- Isolation Forest (unsupervised anomalies)

## Slide 9 — Evaluation Metrics
- Why accuracy fails under imbalance
- Precision, Recall, F1, ROC-AUC, PR-AUC
- Confusion matrix interpretation

## Slide 10 — Experimental Results
- Paste **your** comparison table from the app
- Highlight best model by F1 / PR-AUC
- Show ROC + PR curve screenshots

## Slide 11 — Explainable AI (SHAP)
- Global feature importance
- Individual transaction explanation
- Note: PCA features are anonymized

## Slide 12 — Live Demo Flow
1. Upload CSV
2. EDA page
3. Train models
4. Compare
5. Predict + download report
6. SHAP explanation

## Slide 13 — Limitations
- Anonymized features limit business interpretability
- Offline batch demo, not production streaming
- Predictions require human review

## Slide 14 — Future Scope
- Cost-sensitive thresholds, boosting models, drift monitoring

## Slide 15 — Conclusion & Q&A
- Complete local explainable fraud-detection pipeline
- Thank you / questions
