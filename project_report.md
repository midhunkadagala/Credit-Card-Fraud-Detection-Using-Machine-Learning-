# Explainable Credit Card Fraud Detection Using Machine Learning

**A Project Report**

Submitted in the partial fulfilment of the requirements for the award of the degree of

**Master of Technology in Department of Computer Science Engineering**

Submitted By

**KADAGALA MIDHUN SAI LAKSHMI GANESH**  
**2601050074**

Course: Computational Thinking and Programming  
Academic Year: 2026-2027

under the supervision of

**Dr. P PRABAKARAN**  
[Assistant Professor]  
Dept of Computer Science and Engineering

Koneru Lakshmaiah Education Foundation  
(Deemed to be University estd., u/s 3 of UGC Act 1956)  
Green Fields, Vaddeswaram, Guntur (Dist.), Andhra Pradesh - 522302  
Academic Year 2026-2027

---

## Declaration

The Individual Project Report entitled **"Explainable Credit Card Fraud Detection Using Machine Learning"** is a record of Bonafide work of **KADAGALA MIDHUN SAI LAKSHMI GANESH - 2601050074** submitted in partial fulfillment for the award of M. Tech in Computer Science Engineering at K L University.

KADAGALA MIDHUN SAI LAKSHMI GANESH  
2601050074

---

## Certificate

This is to certify that the Individual Project entitled **"Explainable Credit Card Fraud Detection Using Machine Learning"** is being submitted by **KADAGALA MIDHUN SAI LAKSHMI GANESH - 2601050074** in partial fulfillment for the award of M. Tech in Department of Computer Science Engineering at the K L University and is a record of Bonafide work carried out under our guidance and supervision.

The results embodied in this report have not been copied from any other departments/University/Institute.

Signature of the Supervisor  
**Dr. P PRABAKARAN**  
[Assistant Professor]

Project Co Ordinator  
**Dr. P PRABAKARAN**  
[Assistant Professor]

Signature of the HOD _________________  
Signature of the External Examiner _________________

---

## Acknowledgement

It is great pleasure for me to express my gratitude to our honorable president Sri. Koneru Satyanarayana, for giving me the opportunity and platform with facilities in accomplishing the project-based laboratory report.

I express sincere gratitude to our HOD **DR. T PAVAN KUMAR** for his administration towards our academic growth. I record it as my privilege to deeply thank you for providing us with the efficient faculty and facilities to make our ideas into reality.

I express my sincere thanks to our project supervisor **Dr. P PRABAKARAN** for his novel association of ideas, encouragement, appreciation, and intellectual zeal which motivated us to venture this report successfully.

Finally, I am pleased to acknowledge the indebtedness to all those who devoted themselves directly or indirectly to make this project report success.

---

## Abstract

Credit card fraud is rare but costly. In real transaction data, fraudulent cases usually form less than one percent of all records. A model that always predicts “legitimate” can still report very high accuracy while catching no fraud at all. Therefore accuracy alone is not a reliable measure for this problem. Banks also need more than a yes/no decision: they need an explanation of why a transaction was flagged, so that analysts can review it with confidence.

This project implements an **Explainable Credit Card Fraud Detection** system in Python. The user manually uploads a CSV dataset at runtime; the dataset is never bundled with the source code and is never downloaded automatically. After validation of the expected schema (`Time`, `V1`–`V28`, `Amount`, `Class`), the system analyses class imbalance, splits the data with stratified sampling, applies scaling, and uses **SMOTE only on the training set** for supervised models. Three algorithms are trained and compared: **Logistic Regression**, **Random Forest** and **Isolation Forest**. Isolation Forest is treated as an unsupervised anomaly detector and does not use SMOTE. Model quality is measured with Accuracy, Precision, Recall, F1-score, ROC-AUC, Precision–Recall AUC and confusion matrices. **SHAP** is used to explain which features push a prediction toward fraud or toward legitimate. Results are presented in an interactive **Streamlit** dashboard.

On the Kaggle Credit Card Fraud Detection dataset (283,726 rows after cleaning; 473 fraud, 283,253 legitimate), Random Forest achieved the best overall trade-off (**F1 = 0.7957**, **ROC-AUC = 0.9701**, **PR-AUC = 0.7962**). Logistic Regression achieved high recall but low precision. Isolation Forest was weaker on Precision–Recall metrics. The project follows the complete computational-thinking workflow: problem decomposition and abstraction, algorithm and data-structure selection, a modular Python implementation, an interactive application, validation and error handling, performance evaluation and an honest statement of limitations (anonymised PCA features, heuristic imbalance handling and model predictions that are not confirmed fraud).

---

## Table of Contents

1. Introduction  
2. Computational Thinking  
3. Methodology  
4. Implementation  
5. Execution and Results  
6. Testing and Quality Engineering  
7. Discussion of Results  
8. Conclusion  
9. Future Improvements  
10. References  

---

# 1. INTRODUCTION

Credit card payments generate a continuous stream of transactions. Most of them are legitimate, but a small fraction may be fraudulent. Missing a fraud case causes financial loss; raising too many false alarms wastes analyst time and annoys genuine customers. The practical task is therefore to build a detection system that is sensitive to rare fraud patterns, fair under severe class imbalance, and transparent enough for human review.

This project presents an explainable machine-learning application that loads a user-provided transaction dataset, trains and compares multiple models, reports imbalance-aware metrics, and explains individual predictions with SHAP through a Streamlit interface.

## 1.1 Project Overview

The system expects a CSV in the popular Kaggle Credit Card Fraud Detection format:

- Features: `Time`, anonymised PCA components `V1`–`V28`, and `Amount`
- Target: `Class` where `0` = legitimate and `1` = fraudulent

At startup the application asks the user to upload the dataset manually. It validates columns and types, shows a preview and class counts, prepares train/test splits, trains selected models, saves them locally with Joblib, compares metrics, supports batch prediction, and produces SHAP explanations for Logistic Regression and Random Forest.

What makes the system “explainable” is the last stage: instead of only printing a predicted label, it shows which features increased or decreased the fraud score for a chosen transaction.

The project is developed as a computational-thinking exercise. It moves from a real-world fraud problem to decomposition and abstraction, algorithm selection, a Python implementation, modular design, interactive demonstration, testing and performance evaluation.

## 1.2 Problem Statement

A bank or payment processor receives a large number of transactions every day. Fraudulent transactions are rare, feature values may be anonymised for privacy, and decisions must be audited. If a naive classifier is used without care for imbalance and explanation, the following problems occur:

- A high-accuracy model may still miss almost all frauds.
- Precision and recall are not examined, so false alarms or missed frauds remain hidden.
- SMOTE or resampling may be applied incorrectly to test data, causing data leakage and inflated scores.
- Analysts receive a black-box decision with no feature-level explanation.
- Students and practitioners have no single local tool that combines training, comparison, prediction and explainability with a mandatory manual dataset upload.

The goal is a simple, local, explainable and academically transparent fraud-detection application that handles imbalance correctly, compares multiple algorithms, and justifies predictions with SHAP.

## 1.3 Project Goals and Objectives

- Accept a manually uploaded CSV and validate the expected fraud-detection schema.
- Analyse missing values, duplicates and class imbalance visually.
- Split data with stratified sampling and prevent leakage (SMOTE on train only).
- Train Logistic Regression, Random Forest and Isolation Forest.
- Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC and confusion matrices.
- Save trained models locally and support batch prediction with a downloadable report.
- Explain model decisions using SHAP (global and individual).
- Deliver an interactive Streamlit dashboard suitable for M.Tech demonstration and portfolio use.

---

# 2. COMPUTATIONAL THINKING

Computational thinking is used to turn the fraud-detection problem into a form that a program can solve. The problem is decomposed into small steps, abstracted into simple models, and recognised as a known pattern in imbalanced classification and explainable AI.

## 2.1 Problem Decomposition

| Step | Task |
|------|------|
| 1 | Ask the user to upload a CSV dataset manually |
| 2 | Validate required columns (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| 3 | Clean missing values and duplicate rows; report counts |
| 4 | Separate features `X` and target `y` |
| 5 | Stratified train/test split |
| 6 | Scale features using statistics learned on the training set only |
| 7 | Apply SMOTE on the training set for supervised models (never on test) |
| 8 | Train Logistic Regression / Random Forest / Isolation Forest |
| 9 | Predict on the untouched test set and compute metrics |
| 10 | Save models; generate comparison charts and SHAP explanations |
| 11 | Serve results through the Streamlit pages |

**Table 2.1:** Decomposition of the fraud-detection problem

## 2.2 Abstraction

The real banking environment is reduced to a few simple models:

- **Transaction row:** numeric feature vector (`Time`, `V1`–`V28`, `Amount`) plus optional `Class`.
- **DatasetInfo:** filename, row/column counts, fraud/legitimate counts and fraud percentage.
- **PreparedData:** train/test matrices, optional SMOTE-resampled training data, fitted scaler and feature names.
- **TrainedModelBundle:** model object, scaler, parameters, test predictions/scores and notes.
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix and classification report.
- **SHAP explanation:** feature contributions that increase or decrease the fraud score.

## 2.3 Pattern Recognition

The problem is a classic **imbalanced binary classification** problem with rare positive events [1][2]. Supervised models learn from labelled transactions; Isolation Forest follows the **anomaly-detection** pattern and scores unusual points without using SMOTE [3]. Explainability follows the **additive feature-attribution** pattern of SHAP [4]. These observations lead directly to the design: one preprocessing pipeline, three model trainers behind a common evaluation interface, and a separate explainability module.

---

# 3. METHODOLOGY

This section explains the algorithms chosen for the project, the imbalance-handling strategy, the evaluation metrics, the data structures used, and a brief complexity view.

## 3.1 Overall Pipeline

1. Manual CSV upload (no auto-download, no bundled dataset).
2. Schema validation and basic cleaning.
3. Exploratory analysis of class distribution.
4. Stratified split into train and test sets.
5. Standard scaling fitted on train data only.
6. SMOTE on train data for Logistic Regression and Random Forest.
7. Model training and test-set evaluation.
8. Model comparison and selection by F1-score (with Recall / PR-AUC as tie-breakers).
9. Prediction and SHAP explanation in the dashboard.

**Why several models?** A single algorithm is not always best for fraud detection. Linear models are interpretable and fast; tree ensembles often capture non-linear patterns better; Isolation Forest provides an unsupervised baseline. Comparing them side by side gives evidence instead of a guess.

## 3.2 Algorithms Used

| Algorithm | Type | SMOTE | Role in this project |
|-----------|------|-------|----------------------|
| Logistic Regression | Supervised classifier | Yes (train only) | Linear baseline with probability scores |
| Random Forest | Supervised ensemble | Yes (train only) | Non-linear model with feature importance |
| Isolation Forest | Unsupervised anomaly detector | No | Flags rare/unusual transactions |

**Table 3.1:** Algorithms used by the system

**Isolation Forest label mapping used in this project:**

- Model output `-1` → anomaly → predicted fraud (`Class = 1`)
- Model output `1` → normal → predicted legitimate (`Class = 0`)

## 3.3 Handling Class Imbalance

Fraud datasets are highly skewed. In the experimental dataset used here, fraud was only about **0.17%** of transactions. The project therefore:

- Uses **stratified splitting** so both train and test preserve class ratios.
- Applies **SMOTE** only to the training fold for supervised models.
- Never applies SMOTE to the test set.
- Reports Precision, Recall, F1 and PR-AUC in addition to Accuracy.

Accuracy alone is insufficient: a trivial “always legitimate” predictor would score above 99% accuracy while detecting zero fraud.

## 3.4 Performance Metrics

| Metric | Definition / role |
|--------|-------------------|
| Accuracy | Overall correct predictions (misleading alone under imbalance) |
| Precision | Of predicted frauds, how many were truly fraud |
| Recall | Of actual frauds, how many were caught |
| F1-score | Harmonic mean of Precision and Recall |
| ROC-AUC | Ranking quality across thresholds |
| PR-AUC | Precision–Recall area; especially useful for rare positives |
| Confusion matrix | Counts of TP, FP, TN, FN |

**Table 3.2:** Performance metrics

## 3.5 Data Structures and Modules

| Module / structure | Purpose |
|--------------------|---------|
| `data_loader.py` | CSV reading, schema validation, dataset summary |
| `preprocessing.py` | Split, scale, SMOTE, leakage-safe preparation |
| `train_model.py` | Train/save/load model bundles with Joblib |
| `evaluation.py` | Metrics, comparison table, Plotly charts |
| `prediction.py` | Batch/single prediction and downloadable report |
| `explainability.py` | SHAP global and individual explanations |
| `app.py` | Streamlit multi-page dashboard |

**Table 3.3:** Project modules and their purpose

## 3.6 Complexity Notes

Let `n` be the number of transactions and `d` the number of features (`d = 30` here).

| Operation | Typical cost |
|-----------|--------------|
| Reading CSV | O(n · d) |
| Stratified split | O(n) |
| Scaling | O(n · d) |
| SMOTE | Depends on minority size and neighbours |
| Logistic Regression training | Roughly iterative O(n · d) per iteration |
| Random Forest training | Grows with trees × samples × features |
| Isolation Forest training | Grows with trees × samples |
| SHAP (tree / linear) | Extra cost on a sampled subset for interactive use |

For academic demonstration, Random Forest was limited to a practical number of estimators and SHAP was computed on a sampled subset to keep the interface responsive.

---

# 4. IMPLEMENTATION

The solution is implemented in Python 3 and runs locally on Windows. The application layer is Streamlit; scientific computing uses Pandas, NumPy, Scikit-learn, Imbalanced-learn, SHAP, Plotly, Matplotlib/Seaborn and Joblib. No cloud service is required.

## 4.1 Folder Structure

```text
credit_card_fraud_detection/
├── app.py
├── requirements.txt
├── README.md
├── project_report.md
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── prediction.py
│   ├── evaluation.py
│   └── explainability.py
├── models/          # trained models saved here
└── outputs/
    ├── charts/
    ├── reports/
    └── shap_plots/
```

The dataset is **not** placed inside this folder. The user supplies it at runtime.

## 4.2 Manual Dataset Upload Mechanism

When the application starts, the Home page displays:

> Please upload your credit card fraud detection dataset to continue.

The user may:

1. Browse and upload a CSV file, or  
2. Paste a full local path (useful for large files such as ~150 MB).

Validation checks:

- File is non-empty CSV
- Required columns exist
- `Class` contains only 0/1 and both classes are present for training
- Feature columns are numeric

If validation fails, training and prediction are blocked and a clear message lists missing or incorrect columns.

## 4.3 Preprocessing Design

- `train_test_split(..., stratify=y)` preserves class balance.
- `StandardScaler` is fitted on training features only, then applied to test/prediction data.
- `SMOTE` is fitted on training data only for Logistic Regression and Random Forest.
- Isolation Forest uses the same split and scaling pattern but **without** SMOTE.

## 4.4 Model Training and Persistence

Each successful training run creates a `TrainedModelBundle` containing the estimator, scaler, feature names, parameters, test predictions and scores. The bundle is saved with Joblib under `models/`, for example:

- `logistic_regression.joblib`
- `random_forest.joblib`
- `isolation_forest.joblib`

## 4.5 Streamlit Dashboard Pages

| Page | Purpose |
|------|---------|
| Home Dashboard | Introduction, upload, summary metrics |
| Data Analysis | Preview, missing/duplicate analysis, distributions |
| Model Training | Train one selected model and save it |
| Model Comparison | Side-by-side metrics, ROC/PR curves, best model |
| Fraud Prediction | Row/batch prediction and CSV download |
| Explainable AI | SHAP global importance and individual explanation |

## 4.6 OOP / Modular Design

| Component | Responsibility |
|-----------|----------------|
| Dataset validation helpers | Protect the pipeline from bad files |
| PreparedData container | Carry leakage-safe train/test artefacts |
| TrainedModelBundle | Single object for evaluate / predict / explain |
| Evaluation helpers | Compute genuine metrics from actual predictions |
| Explainability helpers | Translate SHAP values into simple language |

Encapsulation keeps UI code in `app.py` and scientific logic in `src/`, which makes the project easier to test and demonstrate.

## 4.7 AI-Assisted Development

AI tools were used as an engineering aid. Every generated part was reviewed against the project specification: mandatory manual upload, no bundled dataset, no test-set SMOTE, genuine metrics and local execution only.

| Stage | Practice in this project |
|-------|--------------------------|
| Specification first | Upload rules, algorithms, metrics and pages were fixed before coding |
| Modular design | Separate modules for load, preprocess, train, evaluate, predict, explain |
| Critical review | Checked leakage, Isolation Forest label mapping and Streamlit error messages |
| Validation | Ran the pipeline on the real Kaggle CSV and verified saved models |
| Improvement | Added path loader for large CSVs and multi-model prediction dropdown |

---

# 5. EXECUTION AND RESULTS

## 5.1 Experimental Dataset

The experiments used the publicly known Credit Card Fraud Detection CSV provided by the user at runtime (not packaged in the repository).

| Item | Value |
|------|-------|
| Source file (user-provided) | `creditcard.csv` |
| Raw rows | 284,807 |
| Rows after duplicate removal | 283,726 |
| Columns | 31 |
| Legitimate (`Class = 0`) | 283,253 |
| Fraudulent (`Class = 1`) | 473 |
| Fraud percentage | 0.1667% |
| Missing values | 0 |
| Train/test split | 80% / 20%, stratified |
| Random state | 42 |

**Table 5.1:** Dataset summary after cleaning

## 5.2 Training Configuration

| Model | Key settings |
|-------|--------------|
| Logistic Regression | `max_iter=1000`, `C=1.0`, StandardScaler + SMOTE (train) |
| Random Forest | `n_estimators=100`, `max_depth=20`, StandardScaler + SMOTE (train) |
| Isolation Forest | `n_estimators=100`, contamination estimated from train fraud rate, **no SMOTE** |

**Table 5.2:** Training configuration

## 5.3 Quantitative Results

All numbers below come from actual model execution on the held-out test set. No metric was hardcoded.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Uses SMOTE |
|-------|----------|-----------|--------|----------|---------|--------|------------|
| Logistic Regression | 0.9737 | 0.0530 | 0.8737 | 0.1000 | 0.9626 | 0.6750 | Yes |
| **Random Forest** | **0.9993** | **0.8132** | **0.7789** | **0.7957** | **0.9701** | **0.7962** | Yes |
| Isolation Forest | 0.9974 | 0.2245 | 0.2316 | 0.2280 | 0.9413 | 0.1180 | No |

**Table 5.3:** Model comparison on the test set

**Best model by F1-score (primary), then Recall / PR-AUC:** Random Forest.

## 5.4 Result Interpretation

- **Random Forest** gave the strongest practical balance: high precision and solid recall, with the best F1 and PR-AUC.
- **Logistic Regression** caught many frauds (high recall ≈ 0.87) but produced many false alarms (precision ≈ 0.05), which lowers F1.
- **Isolation Forest** remained weaker on precision/recall for this labelled evaluation, which is expected because it is unsupervised and tuned through contamination rather than direct class learning.
- **Accuracy** stayed high for all models because legitimate transactions dominate; this confirms why Accuracy is not enough by itself.

## 5.5 Application Output

Running the Streamlit app and loading the dataset shows:

- Sidebar status: `Loaded: creditcard.csv` with row and fraud counts
- Home summary cards for total / fraud / legitimate / fraud percentage
- Comparison tables and ROC / Precision–Recall curves after models are loaded
- Prediction page supporting Logistic Regression, Random Forest and Isolation Forest
- Explainable AI page with SHAP global importance and individual contribution charts

## 5.6 System Architecture (Logical)

```text
User CSV (manual upload)
        │
        ▼
   Validation + EDA
        │
        ▼
 Stratified Split + Scaling
        │
        ├── SMOTE (train only) ──► Logistic Regression / Random Forest
        │
        └── No SMOTE ───────────► Isolation Forest
        │
        ▼
 Evaluation (Precision/Recall/F1/ROC/PR) + Joblib save
        │
        ▼
 Prediction + SHAP Explainability
        │
        ▼
 Streamlit Dashboard
```

---

# 6. TESTING AND QUALITY ENGINEERING

## 6.1 Test Plan

| Test | Input / action | Expected result | Result |
|------|----------------|-----------------|--------|
| TC1 | No file uploaded | Informative message; training blocked | Pass |
| TC2 | Non-CSV file | Rejected with clear error | Pass |
| TC3 | CSV missing `Class` or `V1`–`V28` | Lists missing columns | Pass |
| TC4 | Valid Kaggle CSV | Loads with correct counts/preview | Pass |
| TC5 | Duplicate rows present | Duplicates removed and reported | Pass |
| TC6 | Train Logistic Regression | Metrics computed; model saved | Pass |
| TC7 | Train Random Forest | Metrics computed; model saved | Pass |
| TC8 | Train Isolation Forest | No SMOTE; `-1/1` mapped to `1/0` | Pass |
| TC9 | Compare all three models | Comparison table + best model | Pass |
| TC10 | Predict before training | Error asking user to train first | Pass |
| TC11 | Prediction CSV with wrong columns | Incompatible-column error | Pass |
| TC12 | SHAP on RF/LR | Global and individual explanations | Pass |

**Table 6.1:** Test cases

## 6.2 Levels of Testing

- **Unit level:** schema validation, missing-column messages, metric computation helpers.
- **Integration level:** preprocessing → training → evaluation → save/load Joblib bundles.
- **End-to-end level:** Streamlit upload → analysis → training/comparison → prediction → SHAP.
- **Regression check:** reloading saved models reproduces the stored test predictions/metrics.

## 6.3 Quality Practices

| Purpose | Practice / tool |
|---------|-----------------|
| Dependency management | `requirements.txt` + virtual environment |
| Local reproducibility | Fixed `random_state=42` |
| Leakage control | Fit scaler/SMOTE on train only |
| User-facing errors | Streamlit messages instead of raw tracebacks |
| Model artefacts | Joblib files under `models/` |
| Reports | Classification reports under `outputs/reports/` |

**Table 6.2:** Quality practices

---

# 7. DISCUSSION OF RESULTS

## 7.1 Observations

- Severe imbalance (≈ 0.17% fraud) makes Accuracy look strong for every model; Precision, Recall, F1 and PR-AUC are the metrics that matter.
- Random Forest was the best overall model for this experiment.
- Logistic Regression remains useful as an interpretable high-recall baseline, but its low precision would create many false alerts in production.
- Isolation Forest is valuable pedagogically as an unsupervised approach, but supervised learning with careful imbalance handling performed better here.
- SHAP explanations are especially important because `V1`–`V28` are anonymised PCA components; the system must not invent readable fields such as merchant name or city.

## 7.2 Limitations and Technical Defence

- The feature set is anonymised; explanations refer to PCA components, not business attributes.
- SMOTE is a heuristic oversampling method; it improves training balance but does not create real fraud cases.
- Isolation Forest contamination is an approximation of the fraud rate and affects sensitivity.
- Model outputs are **predictions**, not confirmed fraud decisions; human review remains necessary.
- Results depend on the uploaded dataset and split seed; another sample or bank’s private data may rank models differently.
- SHAP computation on the full test set would be slow, so interactive explanations use sampled rows.

## 7.3 Live Demonstration Flow

1. Introduce the real-world fraud-detection problem and class imbalance.
2. Open the project in VS Code and start Streamlit.
3. Upload / load `creditcard.csv` manually and show validation + summary.
4. Open Data Analysis and discuss the imbalance chart.
5. Train or load the three models.
6. Show the comparison table and explain why Accuracy is not enough.
7. Run Fraud Prediction on a selected row and download a report.
8. Open Explainable AI and interpret SHAP contributions in simple language.
9. State limitations and future improvements.
10. Conclude with the best model and learning outcomes.

## 7.4 Mapping to the Defence Requirements

| Requirement | Where it is covered |
|-------------|---------------------|
| Real-world problem | Section 1.2 |
| Computational thinking, decomposition, abstraction | Section 2 |
| Algorithm and data-structure selection | Sections 3.1 to 3.5 |
| Python implementation | Section 4 |
| Interactive application / demonstration | Sections 4.5, 5.5, 7.3 |
| Testing and quality engineering | Section 6 |
| AI-assisted development | Section 4.7 |
| Performance evaluation | Sections 5.3 and 5.4 |
| Technical defence / limitations | Section 7.2 |

**Table 7.1:** Mapping of the project to the defence requirements

---

# 8. CONCLUSION

The Explainable Credit Card Fraud Detection system applies decomposition, careful preprocessing and multiple machine-learning algorithms to a realistic imbalanced classification problem. For the experimental dataset, Random Forest achieved the best F1-score and Precision–Recall performance, while Logistic Regression offered high recall at the cost of many false positives, and Isolation Forest provided a weaker unsupervised baseline.

Because accuracy alone is misleading under imbalance, the application emphasises Precision, Recall, F1, ROC-AUC and PR-AUC, and adds SHAP-based explanations so that predictions can be reviewed rather than blindly trusted. The dataset remains outside the repository and must be supplied manually, which keeps the project honest for academic evaluation and portfolio presentation.

The project covers the complete workflow expected for defence: problem formulation, computational thinking, algorithm selection, modular Python implementation, interactive demonstration, testing, performance evaluation and an explicit discussion of limitations.

---

# 9. FUTURE IMPROVEMENTS

- Hyperparameter tuning with cross-validation focused on PR-AUC / F1.
- Additional models (XGBoost, LightGBM, calibrated classifiers).
- Cost-sensitive learning and threshold optimisation for analyst workload.
- Better unsupervised baselines and hybrid supervised–unsupervised ensembles.
- Model-card style documentation for each saved model.
- Broader automated pytest suite with coverage reporting.
- Optional Docker packaging for reproducible demos.
- Extension to streaming/near-real-time scoring for new transactions.
- Richer explainability views (dependence plots, cohort explanations).

---

# 10. REFERENCES

[1] Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE Symposium Series on Computational Intelligence*.

[2] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

[3] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE International Conference on Data Mining*.

[4] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

[6] Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox. *JMLR*, 18(17), 1–5.

[7] Streamlit Inc. Streamlit Documentation. https://docs.streamlit.io/

[8] Python Software Foundation. Python 3 Documentation. https://docs.python.org/3/

[9] Credit Card Fraud Detection Dataset (ULB / Kaggle). https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

[10] Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.

---

## Appendix A — How to Run (Windows / VS Code)

```powershell
cd "D:\AIML WITH APPLIED MATHEMATICS\credit_card_fraud_detection"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501**, then manually upload/load `creditcard.csv`.

Alternatively, double-click `Start_Fraud_Detection_App.bat`.

## Appendix B — Resume-Ready Project Description

Built an explainable credit-card fraud detection system in Python with Streamlit, comparing Logistic Regression, Random Forest and Isolation Forest under severe class imbalance. Implemented leakage-safe preprocessing with stratified splits and train-only SMOTE, evaluated models using Precision/Recall/F1/ROC-AUC/PR-AUC, and added SHAP explanations for transparent decision support. Delivered a local multi-page dashboard with manual dataset upload, model persistence and batch prediction reporting.
