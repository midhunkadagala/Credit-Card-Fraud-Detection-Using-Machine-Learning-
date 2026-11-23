"""Generate a KL-University-style PDF project report."""

from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Explainable_Credit_Card_Fraud_Detection_Project_Report.pdf",
)


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "TitleCenter",
            parent=base["Title"],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyJust",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "center": ParagraphStyle(
            "Center",
            parent=base["BodyText"],
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "boldcenter": ParagraphStyle(
            "BoldCenter",
            parent=base["BodyText"],
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "right": ParagraphStyle(
            "Right",
            parent=base["BodyText"],
            fontSize=10,
            leading=13,
            alignment=TA_RIGHT,
        ),
    }
    return s


def p(text: str, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.15, 0.25, 0.45)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.97)]),
            ]
        )
    )
    return t


def build():
    s = styles()
    story = []

    # -------------------- Title page --------------------
    story.append(Spacer(1, 0.6 * inch))
    story.append(p("Explainable Credit Card Fraud Detection<br/>Using Machine Learning", s["title"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(p("A Project Report", s["boldcenter"]))
    story.append(
        p(
            "Submitted in the partial fulfilment of the requirements for the award of the degree of",
            s["center"],
        )
    )
    story.append(
        p(
            "<b>Master of Technology in Department of<br/>Computer Science Engineering</b>",
            s["center"],
        )
    )
    story.append(Spacer(1, 0.25 * inch))
    story.append(p("Submitted By", s["center"]))
    story.append(p("<b>KADAGALA MIDHUN SAI LAKSHMI GANESH</b>", s["center"]))
    story.append(p("<b>2601050074</b>", s["center"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(p("Course: Computational Thinking and Programming", s["center"]))
    story.append(p("Academic Year: 2026-2027", s["center"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(p("under the supervision of", s["center"]))
    story.append(p("<b>Dr. P PRABAKARAN</b>", s["center"]))
    story.append(p("[Assistant Professor]<br/>Dept of Computer Science and Engineering", s["center"]))
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        p(
            "<b>Koneru Lakshmaiah Education Foundation</b><br/>"
            "(Deemed to be University estd., u/s 3 of UGC Act 1956)<br/>"
            "Green Fields, Vaddeswaram, Guntur (Dist.), Andhra Pradesh - 522302<br/>"
            "Academic Year 2026-2027",
            s["center"],
        )
    )
    story.append(PageBreak())

    # -------------------- Declaration --------------------
    story.append(p("Declaration", s["title"]))
    story.append(
        p(
            'The Individual Project Report entitled "Explainable Credit Card Fraud Detection Using Machine Learning" '
            "is a record of Bonafide work of KADAGALA MIDHUN SAI LAKSHMI GANESH - 2601050074 submitted in partial "
            "fulfillment for the award of M. Tech in Computer Science Engineering at K L University.",
            s["body"],
        )
    )
    story.append(Spacer(1, 0.4 * inch))
    story.append(p("KADAGALA MIDHUN SAI LAKSHMI GANESH<br/>2601050074", s["right"]))
    story.append(PageBreak())

    # -------------------- Certificate --------------------
    story.append(p("Certificate", s["title"]))
    story.append(
        p(
            'This is to certify that the Individual Project entitled "Explainable Credit Card Fraud Detection Using Machine Learning" '
            "is being submitted by KADAGALA MIDHUN SAI LAKSHMI GANESH - 2601050074 in partial fulfillment for the award of "
            "M. Tech in Department of Computer Science Engineering at the K L University and is a record of Bonafide work "
            "carried out under our guidance and supervision.",
            s["body"],
        )
    )
    story.append(
        p(
            "The results embodied in this report have not been copied from any other departments/University/Institute.",
            s["body"],
        )
    )
    story.append(Spacer(1, 0.5 * inch))
    story.append(p("Signature of the Supervisor<br/><b>Dr. P PRABAKARAN</b><br/>[Assistant Professor]", s["small"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(p("Project Co Ordinator<br/><b>Dr. P PRABAKARAN</b><br/>[Assistant Professor]", s["small"]))
    story.append(Spacer(1, 0.35 * inch))
    story.append(p("Signature of the HOD _______________________", s["small"]))
    story.append(p("Signature of the External Examiner _______________________", s["small"]))
    story.append(PageBreak())

    # -------------------- Acknowledgement --------------------
    story.append(p("Acknowledgement", s["title"]))
    story.append(
        p(
            "It is great pleasure for me to express my gratitude to our honorable president Sri. Koneru Satyanarayana, "
            "for giving me the opportunity and platform with facilities in accomplishing the project-based laboratory report.",
            s["body"],
        )
    )
    story.append(
        p(
            "I express sincere gratitude to our HOD DR. T PAVAN KUMAR for his administration towards our academic growth. "
            "I record it as my privilege to deeply thank you for providing us with the efficient faculty and facilities to make our ideas into reality.",
            s["body"],
        )
    )
    story.append(
        p(
            "I express my sincere thanks to our project supervisor Dr. P PRABAKARAN for his novel association of ideas, "
            "encouragement, appreciation, and intellectual zeal which motivated us to venture this report successfully.",
            s["body"],
        )
    )
    story.append(
        p(
            "Finally, I am pleased to acknowledge the indebtedness to all those who devoted themselves directly or "
            "indirectly to make this project report success.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # -------------------- Abstract --------------------
    story.append(p("Abstract", s["title"]))
    story.append(
        p(
            "Credit card fraud is rare but costly. In real transaction data, fraudulent cases usually form less than one percent "
            "of all records. A model that always predicts “legitimate” can still report very high accuracy while catching no fraud "
            "at all. Therefore accuracy alone is not a reliable measure for this problem. Banks also need explanations for flagged "
            "transactions so that analysts can review decisions with confidence.",
            s["body"],
        )
    )
    story.append(
        p(
            "This project implements an Explainable Credit Card Fraud Detection system in Python. The user manually uploads a CSV "
            "dataset at runtime; the dataset is never bundled with the source code and is never downloaded automatically. After "
            "schema validation (Time, V1–V28, Amount, Class), the system analyses class imbalance, uses stratified sampling, applies "
            "scaling, and uses SMOTE only on the training set for supervised models. Three algorithms are compared: Logistic Regression, "
            "Random Forest and Isolation Forest. Isolation Forest does not use SMOTE. Evaluation uses Accuracy, Precision, Recall, "
            "F1-score, ROC-AUC, Precision–Recall AUC and confusion matrices. SHAP explains which features push predictions toward fraud "
            "or legitimate. Results are shown in a Streamlit dashboard.",
            s["body"],
        )
    )
    story.append(
        p(
            "On the experimental dataset (283,726 rows after cleaning; 473 fraud, 283,253 legitimate), Random Forest achieved the best "
            "overall trade-off (F1 = 0.7957, ROC-AUC = 0.9701, PR-AUC = 0.7962). Logistic Regression achieved high recall but low precision. "
            "Isolation Forest was weaker on Precision–Recall metrics. The project follows the complete computational-thinking workflow and "
            "honestly states its limitations.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # -------------------- 1 Introduction --------------------
    story.append(p("1. INTRODUCTION", s["h1"]))
    story.append(
        p(
            "Credit card payments generate a continuous stream of transactions. Most are legitimate, but a small fraction may be fraudulent. "
            "Missing a fraud causes financial loss; too many false alarms waste analyst time. This project presents an explainable "
            "machine-learning application that loads a user-provided dataset, trains and compares multiple models, reports imbalance-aware "
            "metrics, and explains predictions with SHAP through Streamlit.",
            s["body"],
        )
    )
    story.append(p("1.1 Project Overview", s["h2"]))
    story.append(
        p(
            "The system expects a CSV in the Kaggle Credit Card Fraud Detection format: features Time, V1–V28 and Amount, and target Class "
            "(0 = legitimate, 1 = fraudulent). At startup the application asks for a manual upload, validates the schema, shows preview and "
            "class counts, prepares train/test splits, trains models, saves them with Joblib, compares metrics, supports batch prediction, "
            "and produces SHAP explanations.",
            s["body"],
        )
    )
    story.append(p("1.2 Problem Statement", s["h2"]))
    story.append(
        p(
            "A payment system receives many transactions daily. Fraud is rare, features may be anonymised, and decisions must be auditable. "
            "Without careful imbalance handling and explanation, high accuracy can hide missed frauds, SMOTE may leak into test data, and "
            "analysts receive black-box decisions. The goal is a local, explainable application with mandatory manual dataset upload.",
            s["body"],
        )
    )
    story.append(p("1.3 Project Goals and Objectives", s["h2"]))
    for item in [
        "Accept a manually uploaded CSV and validate the expected schema.",
        "Analyse missing values, duplicates and class imbalance.",
        "Use stratified split and train-only SMOTE (no leakage).",
        "Train Logistic Regression, Random Forest and Isolation Forest.",
        "Evaluate with Precision, Recall, F1, ROC-AUC, PR-AUC and confusion matrices.",
        "Support prediction reports and SHAP explanations in Streamlit.",
    ]:
        story.append(p(f"• {item}", s["small"]))
    story.append(PageBreak())

    # -------------------- 2 CT --------------------
    story.append(p("2. COMPUTATIONAL THINKING", s["h1"]))
    story.append(
        p(
            "Computational thinking turns the fraud-detection problem into programmable steps through decomposition, abstraction and pattern recognition.",
            s["body"],
        )
    )
    story.append(p("2.1 Problem Decomposition", s["h2"]))
    story.append(
        table(
            [
                ["Step", "Task"],
                ["1", "Manual CSV upload"],
                ["2", "Validate Time, V1–V28, Amount, Class"],
                ["3", "Clean missing values and duplicates"],
                ["4", "Separate X and y"],
                ["5", "Stratified train/test split"],
                ["6", "Scale using train statistics only"],
                ["7", "SMOTE on train for supervised models"],
                ["8", "Train LR / RF / Isolation Forest"],
                ["9", "Evaluate on untouched test set"],
                ["10", "Save models, compare, explain with SHAP"],
                ["11", "Serve results in Streamlit pages"],
            ],
            [0.7 * inch, 5.5 * inch],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(p("Table 2.1: Decomposition of the fraud-detection problem", s["small"]))
    story.append(p("2.2 Abstraction", s["h2"]))
    story.append(
        p(
            "Key abstractions are Transaction row, DatasetInfo, PreparedData, TrainedModelBundle, Metrics and SHAP explanation. "
            "These hide banking complexity while preserving the information needed for training, evaluation and review.",
            s["body"],
        )
    )
    story.append(p("2.3 Pattern Recognition", s["h2"]))
    story.append(
        p(
            "The problem is imbalanced binary classification. Supervised models learn from labels; Isolation Forest follows anomaly detection; "
            "SHAP follows additive feature attribution. The design therefore uses one preprocessing pipeline, three trainers and a shared evaluation interface.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # -------------------- 3 Methodology --------------------
    story.append(p("3. METHODOLOGY", s["h1"]))
    story.append(p("3.1 Overall Pipeline", s["h2"]))
    story.append(
        p(
            "Manual upload → validation/EDA → stratified split → scaling → SMOTE (train only for LR/RF) → training → "
            "test evaluation → comparison → prediction/SHAP. Isolation Forest skips SMOTE. Multiple models are compared because "
            "no single algorithm is best for every fraud scenario.",
            s["body"],
        )
    )
    story.append(p("3.2 Algorithms Used", s["h2"]))
    story.append(
        table(
            [
                ["Algorithm", "Type", "SMOTE", "Role"],
                ["Logistic Regression", "Supervised", "Train only", "Linear baseline with probabilities"],
                ["Random Forest", "Supervised ensemble", "Train only", "Non-linear model + importance"],
                ["Isolation Forest", "Unsupervised", "No", "Anomaly baseline (-1→fraud, 1→legit)"],
            ],
            [1.5 * inch, 1.4 * inch, 1.0 * inch, 2.3 * inch],
        )
    )
    story.append(p("Table 3.1: Algorithms used by the system", s["small"]))
    story.append(p("3.3 Handling Class Imbalance", s["h2"]))
    story.append(
        p(
            "Fraud was only about 0.17% of transactions in the experiment. The project uses stratified splitting, train-only SMOTE "
            "for supervised models, and reports Precision/Recall/F1/PR-AUC because Accuracy alone is misleading.",
            s["body"],
        )
    )
    story.append(p("3.4 Performance Metrics", s["h2"]))
    story.append(
        p(
            "Accuracy, Precision, Recall, F1-score, ROC-AUC, Precision–Recall AUC and confusion matrices are computed from actual "
            "test predictions. No metric values are hardcoded.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # -------------------- 4 Implementation --------------------
    story.append(p("4. IMPLEMENTATION", s["h1"]))
    story.append(
        p(
            "The solution is implemented in Python 3 for local Windows execution using Streamlit, Pandas, NumPy, Scikit-learn, "
            "Imbalanced-learn, SHAP, Plotly, Matplotlib/Seaborn and Joblib. No cloud service is required.",
            s["body"],
        )
    )
    story.append(p("4.1 Folder Structure", s["h2"]))
    story.append(
        p(
            "credit_card_fraud_detection/ contains app.py, requirements.txt, README.md, project_report.md, src/ "
            "(data_loader, preprocessing, train_model, prediction, evaluation, explainability), models/ and outputs/. "
            "The dataset is not placed inside this folder.",
            s["body"],
        )
    )
    story.append(p("4.2 Manual Dataset Upload", s["h2"]))
    story.append(
        p(
            "The Home page shows: “Please upload your credit card fraud detection dataset to continue.” The user may browse a CSV "
            "or paste a local path. Validation checks required columns, Class labels 0/1, and numeric features. Invalid files block training/prediction.",
            s["body"],
        )
    )
    story.append(p("4.3 Streamlit Pages", s["h2"]))
    story.append(
        table(
            [
                ["Page", "Purpose"],
                ["Home Dashboard", "Introduction, upload, summary metrics"],
                ["Data Analysis", "Preview, missing/duplicates, distributions"],
                ["Model Training", "Train and save a selected model"],
                ["Model Comparison", "Metrics, ROC/PR curves, best model"],
                ["Fraud Prediction", "Row/batch prediction + CSV download"],
                ["Explainable AI", "SHAP global and individual explanations"],
            ],
            [1.7 * inch, 4.5 * inch],
        )
    )
    story.append(p("Table 4.1: Dashboard pages", s["small"]))
    story.append(PageBreak())

    # -------------------- 5 Results --------------------
    story.append(p("5. EXECUTION AND RESULTS", s["h1"]))
    story.append(p("5.1 Experimental Dataset", s["h2"]))
    story.append(
        table(
            [
                ["Item", "Value"],
                ["User-provided file", "creditcard.csv"],
                ["Raw rows", "284,807"],
                ["Rows after cleaning", "283,726"],
                ["Columns", "31"],
                ["Legitimate (Class=0)", "283,253"],
                ["Fraudulent (Class=1)", "473"],
                ["Fraud percentage", "0.1667%"],
                ["Train/test split", "80%/20% stratified, seed=42"],
            ],
            [2.3 * inch, 3.9 * inch],
        )
    )
    story.append(p("Table 5.1: Dataset summary after cleaning", s["small"]))
    story.append(p("5.2 Quantitative Results", s["h2"]))
    story.append(
        p(
            "All numbers below come from actual model execution on the held-out test set.",
            s["body"],
        )
    )
    story.append(
        table(
            [
                ["Model", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "PR-AUC"],
                ["Logistic Regression", "0.9737", "0.0530", "0.8737", "0.1000", "0.9626", "0.6750"],
                ["Random Forest", "0.9993", "0.8132", "0.7789", "0.7957", "0.9701", "0.7962"],
                ["Isolation Forest", "0.9974", "0.2245", "0.2316", "0.2280", "0.9413", "0.1180"],
            ],
            [1.6 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.85 * inch, 0.85 * inch],
        )
    )
    story.append(p("Table 5.2: Model comparison on the test set", s["small"]))
    story.append(
        p(
            "<b>Best model by F1-score:</b> Random Forest.",
            s["body"],
        )
    )
    story.append(p("5.3 Result Interpretation", s["h2"]))
    story.append(
        p(
            "Random Forest gave the strongest practical balance of precision and recall. Logistic Regression caught many frauds "
            "(high recall) but created many false alarms (low precision). Isolation Forest was weaker on Precision–Recall metrics "
            "as an unsupervised baseline. Accuracy stayed high for all models because legitimate transactions dominate, confirming "
            "why Accuracy alone is insufficient.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # -------------------- 6 Testing --------------------
    story.append(p("6. TESTING AND QUALITY ENGINEERING", s["h1"]))
    story.append(
        table(
            [
                ["Test", "Expected result"],
                ["TC1 No upload", "Training blocked with clear message"],
                ["TC2 Invalid/missing columns", "Error lists missing fields"],
                ["TC3 Valid CSV", "Preview and counts displayed"],
                ["TC4 Train LR/RF/IF", "Genuine metrics + Joblib save"],
                ["TC5 Compare models", "Comparison table and best model"],
                ["TC6 Predict before train", "Error asking to train first"],
                ["TC7 Bad prediction CSV", "Incompatible-column error"],
                ["TC8 SHAP on LR/RF", "Global and individual explanations"],
            ],
            [2.0 * inch, 4.2 * inch],
        )
    )
    story.append(p("Table 6.1: Representative test cases", s["small"]))
    story.append(
        p(
            "Quality practices include a virtual environment, fixed random_state, train-only scaler/SMOTE fitting, Streamlit error "
            "messages, and persisted model/report artefacts under models/ and outputs/.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # -------------------- 7 Discussion --------------------
    story.append(p("7. DISCUSSION OF RESULTS", s["h1"]))
    story.append(p("7.1 Observations", s["h2"]))
    story.append(
        p(
            "Severe imbalance makes Accuracy look strong for every model; Precision, Recall, F1 and PR-AUC matter more. "
            "Random Forest was best overall. Logistic Regression is a useful high-recall baseline. Isolation Forest is pedagogically "
            "important but weaker here. SHAP is essential because V1–V28 are anonymised PCA features.",
            s["body"],
        )
    )
    story.append(p("7.2 Limitations and Technical Defence", s["h2"]))
    story.append(
        p(
            "Features are anonymised; SMOTE is heuristic; Isolation Forest contamination is approximate; predictions are not confirmed "
            "fraud; results depend on the uploaded dataset and split seed; SHAP uses sampled rows for interactive speed.",
            s["body"],
        )
    )
    story.append(p("7.3 Live Demonstration Flow", s["h2"]))
    for step in [
        "Introduce fraud detection and class imbalance.",
        "Start Streamlit in VS Code and load creditcard.csv manually.",
        "Show Home summary and Data Analysis charts.",
        "Train/load three models and open Model Comparison.",
        "Explain why Accuracy is not enough.",
        "Run Fraud Prediction and download a report.",
        "Show SHAP explanations in simple language.",
        "State limitations and conclude with best model.",
    ]:
        story.append(p(f"• {step}", s["small"]))
    story.append(PageBreak())

    # -------------------- 8-10 --------------------
    story.append(p("8. CONCLUSION", s["h1"]))
    story.append(
        p(
            "The Explainable Credit Card Fraud Detection system applies decomposition, careful preprocessing and multiple algorithms "
            "to a realistic imbalanced classification problem. Random Forest achieved the best F1 and PR-AUC in the experiment. "
            "The application emphasises imbalance-aware metrics and SHAP explanations, keeps the dataset outside the repository, "
            "and remains suitable for M.Tech demonstration and portfolio presentation.",
            s["body"],
        )
    )
    story.append(p("9. FUTURE IMPROVEMENTS", s["h1"]))
    for item in [
        "Hyperparameter tuning focused on PR-AUC / F1",
        "Additional models such as XGBoost or LightGBM",
        "Cost-sensitive thresholds for analyst workload",
        "Broader automated pytest suite and coverage",
        "Streaming/near-real-time scoring",
        "Richer SHAP visualisations and model cards",
    ]:
        story.append(p(f"• {item}", s["small"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(p("10. REFERENCES", s["h1"]))
    refs = [
        "[1] Dal Pozzolo et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. IEEE SSCI.",
        "[2] Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321–357.",
        "[3] Liu, Ting & Zhou (2008). Isolation Forest. IEEE ICDM.",
        "[4] Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
        "[5] Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.",
        "[6] Lemaître et al. (2017). Imbalanced-learn: A Python Toolbox. JMLR.",
        "[7] Streamlit Documentation. https://docs.streamlit.io/",
        "[8] Python 3 Documentation. https://docs.python.org/3/",
        "[9] Credit Card Fraud Detection Dataset (Kaggle/ULB).",
        "[10] Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.",
    ]
    for r in refs:
        story.append(p(r, s["small"]))

    story.append(PageBreak())
    story.append(p("Appendix A — How to Run (Windows / VS Code)", s["h1"]))
    story.append(
        p(
            "cd \"D:\\AIML WITH APPLIED MATHEMATICS\\credit_card_fraud_detection\"<br/>"
            "python -m venv venv<br/>"
            ".\\venv\\Scripts\\Activate.ps1<br/>"
            "pip install -r requirements.txt<br/>"
            "streamlit run app.py --server.address 127.0.0.1 --server.port 8501",
            s["small"],
        )
    )
    story.append(
        p(
            "Open http://127.0.0.1:8501 and manually upload/load creditcard.csv. "
            "Or double-click Start_Fraud_Detection_App.bat.",
            s["body"],
        )
    )
    story.append(p("Appendix B — Resume-Ready Project Description", s["h1"]))
    story.append(
        p(
            "Built an explainable credit-card fraud detection system in Python with Streamlit, comparing Logistic Regression, "
            "Random Forest and Isolation Forest under severe class imbalance. Implemented leakage-safe preprocessing with stratified "
            "splits and train-only SMOTE, evaluated models using Precision/Recall/F1/ROC-AUC/PR-AUC, and added SHAP explanations "
            "for transparent decision support. Delivered a local multi-page dashboard with manual dataset upload, model persistence "
            "and batch prediction reporting.",
            s["body"],
        )
    )

    doc = SimpleDocTemplate(
        OUT,
        pagesize=A4,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Explainable Credit Card Fraud Detection Project Report",
        author="KADAGALA MIDHUN SAI LAKSHMI GANESH",
    )
    doc.build(story)
    print(f"PDF_CREATED={OUT}")


if __name__ == "__main__":
    build()
