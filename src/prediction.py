"""
prediction.py
-------------
Batch and single-row prediction using previously trained models.

Predictions are model outputs — NOT confirmed fraud. Always communicate
this distinction clearly in the UI.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import REQUIRED_FEATURES, validate_prediction_csv
from src.preprocessing import transform_features
from src.train_model import (
    MODEL_ISOLATION_FOREST,
    TrainedModelBundle,
)


DISCLAIMER = (
    "These outputs are MODEL PREDICTIONS, not confirmed fraud decisions. "
    "A predicted fraud label means the model flagged the transaction as "
    "suspicious and it should be reviewed by a human analyst."
)


def predict_dataframe(
    bundle: TrainedModelBundle,
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    """
    Run batch prediction on a DataFrame with required feature columns.

    Returns
    -------
    (result_df, message)
    """
    if bundle is None or bundle.model is None:
        raise ValueError("Model is not trained. Please train a model before prediction.")

    validation = validate_prediction_csv(df, require_class=False)
    if not validation.is_valid:
        missing = ", ".join(validation.missing_columns) if validation.missing_columns else ""
        raise ValueError(
            f"{validation.message} Missing: {missing}. {validation.extra_info}"
        )

    X = df[REQUIRED_FEATURES].copy()
    X_transformed = transform_features(X, bundle.scaler)

    if bundle.model_name == MODEL_ISOLATION_FOREST:
        raw = bundle.model.predict(X_transformed)
        preds = np.where(raw == -1, 1, 0)
        # anomaly score: higher => more anomalous
        scores = -bundle.model.decision_function(X_transformed)
        proba = None
    else:
        preds = bundle.model.predict(X_transformed)
        if hasattr(bundle.model, "predict_proba"):
            proba = bundle.model.predict_proba(X_transformed)[:, 1]
            scores = proba
        else:
            proba = None
            scores = None

    result = df.copy()
    result["Predicted_Class"] = preds.astype(int)
    result["Predicted_Label"] = np.where(
        result["Predicted_Class"] == 1, "Fraud (Predicted)", "Legitimate (Predicted)"
    )

    if proba is not None:
        result["Fraud_Probability"] = np.round(proba, 6)
    elif scores is not None:
        result["Anomaly_Score"] = np.round(scores, 6)

    result["Note"] = DISCLAIMER
    return result, DISCLAIMER


def predict_single_row(
    bundle: TrainedModelBundle,
    row: pd.Series,
) -> dict:
    """Predict a single transaction row selected from the uploaded dataset."""
    df = pd.DataFrame([row])
    result, disclaimer = predict_dataframe(bundle, df)
    out = result.iloc[0].to_dict()
    out["disclaimer"] = disclaimer
    return out


def build_downloadable_report(result_df: pd.DataFrame) -> bytes:
    """Return CSV bytes for Streamlit download_button."""
    return result_df.to_csv(index=False).encode("utf-8")
