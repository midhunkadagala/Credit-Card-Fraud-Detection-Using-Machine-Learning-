# Explainable Credit Card Fraud Detection Using Machine Learning

Local Streamlit app for M.Tech demonstration. Detects fraudulent transactions, compares Logistic Regression / Random Forest / Isolation Forest, balances training data with SMOTE, and explains predictions with SHAP.

## Important: Manual Dataset Upload

**The dataset is NOT included in this repository.**

1. Download the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) CSV yourself.
2. Keep it **outside** this project folder.
3. When the app starts, upload the CSV through the Streamlit file uploader.

Expected columns: `Time`, `V1`–`V28`, `Amount`, `Class` (`0` = legitimate, `1` = fraud).

No automatic Kaggle download. No bundled or fake sample data.

## Setup (Windows)

```powershell
cd "D:\AIML WITH APPLIED MATHEMATICS\credit_card_fraud_detection"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501) and upload your CSV on the Home Dashboard.

## Pages

1. Home Dashboard — upload + summary  
2. Data Analysis — preview, missing/duplicates, distributions  
3. Model Training — train and save models  
4. Model Comparison — metrics, ROC, PR curves  
5. Fraud Prediction — batch / row prediction + download report  
6. Explainable AI — SHAP global and individual explanations  
