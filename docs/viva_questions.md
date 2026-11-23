# Viva Questions and Answers (30+)

Use these for M.Tech viva preparation. Prefer answering in your own words.

---

### 1. What is the objective of your project?
To detect potentially fraudulent credit card transactions using ML, compare multiple algorithms under severe class imbalance, and explain predictions with SHAP through a local Streamlit dashboard.

### 2. Why is the dataset not included in the project folder?
To respect dataset licensing/distribution practices, keep the repository lightweight, and force a realistic workflow where data is supplied at runtime. The app never auto-downloads or fabricates data.

### 3. What happens if the user does not upload a dataset?
The UI shows: “Please upload your credit card fraud detection dataset to continue.” Training, comparison, prediction, and SHAP remain blocked.

### 4. What columns are required?
`Time`, `Amount`, `V1` through `V28`, and target `Class`.

### 5. What do Class 0 and Class 1 mean?
`0` = legitimate transaction, `1` = fraudulent transaction.

### 6. Why are features named V1–V28?
They are anonymized PCA components from the original dataset to protect confidentiality. They are not raw merchant/location fields.

### 7. Why is the dataset imbalanced?
Fraud is rare compared with normal spending. In the classic Kaggle set, fraud is roughly 0.17% of transactions.

### 8. Why is accuracy a poor standalone metric here?
A model predicting all transactions as legitimate can get >99% accuracy while detecting zero fraud. Precision, Recall, F1, ROC-AUC, and PR-AUC are more informative.

### 9. What is Precision in fraud detection?
Of all transactions predicted as fraud, the fraction that are actually fraud. High precision means fewer false alarms.

### 10. What is Recall in fraud detection?
Of all actual frauds, the fraction correctly detected. High recall means fewer missed frauds.

### 11. What is the F1-score?
Harmonic mean of Precision and Recall; balances false alarms and missed frauds.

### 12. Why prefer PR-AUC for imbalance?
Precision-Recall curves focus on performance on the positive (fraud) class and are more sensitive under skew than ROC-AUC alone.

### 13. What is SMOTE?
Synthetic Minority Over-sampling Technique — creates synthetic minority-class samples between nearest neighbors to balance training data.

### 14. Why apply SMOTE only on training data?
Applying SMOTE before the split (or on test data) leaks information and produces overly optimistic metrics. Test data must stay realistic.

### 15. Do you apply SMOTE to Isolation Forest?
No. Isolation Forest is unsupervised anomaly detection; SMOTE would distort the natural anomaly structure.

### 16. How are Isolation Forest outputs mapped?
Model output `-1` (anomaly) → Class `1` (fraud); `1` (normal) → Class `0` (legitimate).

### 17. What is contamination in Isolation Forest?
Expected proportion of anomalies. In this project it is estimated from the training fraud rate when labels exist.

### 18. Why use StandardScaler?
Logistic Regression (and many linear/distance-based methods) benefit when features are on comparable scales. The scaler is fit on train data only.

### 19. What is stratified sampling?
Train/test split that preserves class proportions so the rare fraud class appears in both sets.

### 20. Why choose Logistic Regression?
Strong, fast, interpretable baseline; probability outputs; works well with scaled tabular features.

### 21. Why choose Random Forest?
Handles non-linear interactions, resistant to overfitting relative to single trees, provides feature importances.

### 22. What is SHAP?
SHapley Additive exPlanations — attributes a prediction to feature contributions relative to a baseline, based on cooperative game theory.

### 23. Difference between global and local SHAP?
Global: average importance across many rows (e.g., mean |SHAP|). Local: explanation for one transaction.

### 24. Can SHAP recover merchant name importance?
No. Those fields are not in the dataset; only PCA features, Time, and Amount are available.

### 25. What is data leakage? Give an example in this project context.
Using future information or test-set statistics during training — e.g., fitting a scaler on the full dataset including test rows, or SMOTE before splitting.

### 26. How do you prevent leakage?
Fit preprocessing on train only; transform test with the fitted objects; SMOTE after split on train only; evaluate only on untouched test labels.

### 27. What libraries did you use?
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, SHAP, Streamlit, Plotly, Matplotlib/Seaborn, Joblib.

### 28. Why Streamlit?
Rapid interactive dashboards for academic demos without building a separate web stack; runs locally.

### 29. Are model predictions confirmed fraud?
No. They are risk flags for human review. The UI states this disclaimer explicitly.

### 30. How are models saved?
With Joblib into the local `models/` folder after successful training.

### 31. What if uploaded prediction CSV lacks V columns?
Validation fails and Streamlit shows which feature columns are missing; prediction is blocked.

### 32. How do you select the “best” model?
Primarily by F1-score, with Recall and PR-AUC as tie-breakers — metrics aligned with fraud detection goals.

### 33. Limitations of your project?
Offline batch demo; anonymized features reduce business interpretability; not a production real-time payment system; no regulatory compliance stack.

### 34. Future improvements?
Threshold optimization, cost-sensitive learning, gradient boosting, cross-validation, drift detection, analyst workflow UI.

### 35. What is the confusion matrix for binary fraud detection?
TN / FP in row Actual Legit; FN / TP in row Actual Fraud. FP = false alarms; FN = missed frauds.

### 36. Which error is usually more costly: FP or FN?
Often FN (missed fraud) is more costly financially, but excessive FP harms customer experience. The trade-off is business-dependent.

### 37. Why both supervised and unsupervised models?
Supervised models use labels for high detection quality; unsupervised Isolation Forest shows how anomalies can be flagged when labels are scarce.

### 38. What validation do you perform on Class values?
Class must be numeric and only {0, 1}, with both classes present for training.

### 39. How would you explain your pipeline in 60 seconds?
Upload & validate CSV → clean/EDA → stratified split → scale → SMOTE on train for LR/RF → train three models → evaluate with imbalance-aware metrics → explain with SHAP → score new rows as predictions for review.

### 40. Is this deployed on AWS/Azure?
No. It runs fully locally on a Windows laptop with no paid cloud services.
