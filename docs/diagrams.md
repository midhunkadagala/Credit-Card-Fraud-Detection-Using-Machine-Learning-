# System Diagrams

Copy these Mermaid blocks into the report, presentation, or GitHub README.
They also render on GitHub.

---

## 1. System Architecture Diagram

```mermaid
flowchart TB
    User[User on Windows Laptop]
    UI[Streamlit Dashboard app.py]
    DL[data_loader.py\nValidate CSV Schema]
    PP[preprocessing.py\nStratified Split + Scale + SMOTE train-only]
    TR[train_model.py]
    EV[evaluation.py]
    PR[prediction.py]
    XP[explainability.py SHAP]
    MD[(models/*.joblib)]
    OUT[(outputs/ charts reports shap)]

    User -->|Manual CSV Upload| UI
    UI --> DL
    DL --> PP
    PP --> TR
    TR --> MD
    TR --> EV
    EV --> OUT
    MD --> PR
    MD --> XP
    PR --> UI
    XP --> UI
    EV --> UI
```

---

## 2. Data Flow Diagram

```mermaid
flowchart LR
    A[Uploaded creditcard.csv] --> B{Schema valid?}
    B -->|No| C[Show missing/incorrect columns\nBlock training]
    B -->|Yes| D[Clean missing + duplicates]
    D --> E[EDA: counts + charts]
    E --> F[Train / Test stratified split]
    F --> G[Fit StandardScaler on Train]
    G --> H{Model type?}
    H -->|LR / RF| I[SMOTE on Train only]
    H -->|Isolation Forest| J[No SMOTE]
    I --> K[Train supervised model]
    J --> L[Train Isolation Forest]
    K --> M[Predict on Test]
    L --> M
    M --> N[Metrics + Plots]
    N --> O[Save joblib model]
    O --> P[Batch prediction / SHAP]
```

---

## 3. Application Flowchart

```mermaid
flowchart TD
    Start([Start: streamlit run app.py]) --> Home[Home Dashboard]
    Home --> Upload{CSV uploaded?}
    Upload -->|No| Msg[Show: Please upload dataset]
    Msg --> Home
    Upload -->|Yes| Validate{Valid schema?}
    Validate -->|No| Err[Explain missing columns]
    Err --> Home
    Validate -->|Yes| Ready[Enable all pages]
    Ready --> EDA[Data Analysis]
    Ready --> Train[Model Training]
    Train --> Save[Save model to models/]
    Ready --> Comp[Model Comparison]
    Ready --> Pred[Fraud Prediction]
    Ready --> Shap[Explainable AI]
    Pred --> End([Download report / review flags])
    Shap --> End
    Comp --> End
```

---

## 4. Train vs Test Leakage Prevention

```mermaid
flowchart TB
    subgraph Train Path
      T1[X_train] --> T2[Fit Scaler]
      T2 --> T3[Transform X_train]
      T3 --> T4[SMOTE resample]
      T4 --> T5[model.fit]
    end
    subgraph Test Path
      S1[X_test] --> S2[Transform with fitted Scaler only]
      S2 --> S3[model.predict]
      S3 --> S4[Compare to y_test]
    end
    T5 -.->|same model| S3
```

**Rule:** Never fit the scaler on test data. Never apply SMOTE to test data.
