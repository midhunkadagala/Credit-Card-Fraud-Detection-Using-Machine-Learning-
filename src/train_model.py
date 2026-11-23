"""
train_model.py
--------------
Train Logistic Regression, Random Forest, and Isolation Forest.

Models are saved to the local `models/` folder with joblib ONLY after
successful training. Nothing is downloaded from the cloud.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocessing import PreparedData

# Project-relative models directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_LOGISTIC = "Logistic Regression"
MODEL_RANDOM_FOREST = "Random Forest"
MODEL_ISOLATION_FOREST = "Isolation Forest"

SUPPORTED_MODELS = [MODEL_LOGISTIC, MODEL_RANDOM_FOREST, MODEL_ISOLATION_FOREST]


@dataclass
class TrainedModelBundle:
    """Everything needed later for evaluation, prediction, and SHAP."""

    model_name: str
    model: Any
    scaler: Any
    feature_names: list
    trained_at: str
    params: Dict[str, Any] = field(default_factory=dict)
    training_notes: str = ""
    uses_smote: bool = False
    supports_proba: bool = True
    feature_importances: Optional[Dict[str, float]] = None
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None
    y_pred: Optional[np.ndarray] = None
    y_score: Optional[np.ndarray] = None  # probability of fraud or anomaly score


def ensure_models_dir() -> str:
    """Create models/ if it does not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR


def _model_filename(model_name: str) -> str:
    safe = model_name.lower().replace(" ", "_")
    return os.path.join(MODELS_DIR, f"{safe}.joblib")


def train_logistic_regression(
    prepared: PreparedData,
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: Optional[str] = None,
) -> TrainedModelBundle:
    """
    Train Logistic Regression on (optionally SMOTE-resampled) training data.

    Scaling should already be applied in PreparedData when scale_features=True.
    """
    if prepared.X_train_resampled is not None and prepared.y_train_resampled is not None:
        X_fit = prepared.X_train_resampled
        y_fit = prepared.y_train_resampled
        used_smote = True
    else:
        X_fit = prepared.X_train
        y_fit = prepared.y_train
        used_smote = False

    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        random_state=prepared.random_state,
        solver="lbfgs",
    )
    model.fit(X_fit, y_fit)

    y_pred = model.predict(prepared.X_test)
    y_score = model.predict_proba(prepared.X_test)[:, 1]

    # Coefficient magnitudes as a simple importance proxy
    importances = {
        name: float(abs(coef))
        for name, coef in zip(prepared.feature_names, model.coef_[0])
    }

    return TrainedModelBundle(
        model_name=MODEL_LOGISTIC,
        model=model,
        scaler=prepared.scaler,
        feature_names=prepared.feature_names,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        params={"max_iter": max_iter, "C": C, "class_weight": class_weight},
        training_notes=(
            "Trained with StandardScaler + SMOTE on training data only."
            if used_smote
            else "Trained with StandardScaler; SMOTE was not applied."
        ),
        uses_smote=used_smote,
        supports_proba=True,
        feature_importances=importances,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        y_pred=y_pred,
        y_score=y_score,
    )


def train_random_forest(
    prepared: PreparedData,
    n_estimators: int = 100,
    max_depth: Optional[int] = 20,
    min_samples_split: int = 2,
    class_weight: Optional[str] = None,
) -> TrainedModelBundle:
    """Train Random Forest classifier (SMOTE on train if available)."""
    if prepared.X_train_resampled is not None and prepared.y_train_resampled is not None:
        X_fit = prepared.X_train_resampled
        y_fit = prepared.y_train_resampled
        used_smote = True
    else:
        X_fit = prepared.X_train
        y_fit = prepared.y_train
        used_smote = False

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=prepared.random_state,
        n_jobs=-1,
    )
    model.fit(X_fit, y_fit)

    y_pred = model.predict(prepared.X_test)
    y_score = model.predict_proba(prepared.X_test)[:, 1]

    importances = {
        name: float(imp)
        for name, imp in zip(prepared.feature_names, model.feature_importances_)
    }

    return TrainedModelBundle(
        model_name=MODEL_RANDOM_FOREST,
        model=model,
        scaler=prepared.scaler,
        feature_names=prepared.feature_names,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        params={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "class_weight": class_weight,
        },
        training_notes=(
            "Random Forest trained on SMOTE-balanced training data."
            if used_smote
            else "Random Forest trained without SMOTE."
        ),
        uses_smote=used_smote,
        supports_proba=True,
        feature_importances=importances,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        y_pred=y_pred,
        y_score=y_score,
    )


def train_isolation_forest(
    prepared: PreparedData,
    n_estimators: int = 100,
    contamination: float = 0.0017,
    max_samples: Union[str, int] = "auto",
) -> TrainedModelBundle:
    """
    Train Isolation Forest (unsupervised).

    Mapping:
      model output -1  -> fraud (anomaly)  -> Class 1
      model output  1  -> normal           -> Class 0

    SMOTE is NOT used. Contamination should roughly match the expected
    fraud rate in the training data when known.
    """
    # Fit primarily on majority (legitimate) + overall train; standard IF
    # is trained on X_train without labels, then evaluated on X_test labels.
    X_fit = prepared.X_train
    y_train = prepared.y_train

    # Auto-estimate contamination from training labels if possible
    fraud_rate = float((y_train == 1).mean()) if len(y_train) else contamination
    # Clamp to a sensible range for sklearn
    cont = float(np.clip(fraud_rate if fraud_rate > 0 else contamination, 0.0001, 0.5))

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=cont,
        max_samples=max_samples,
        random_state=prepared.random_state,
        n_jobs=-1,
    )
    model.fit(X_fit)

    raw_pred = model.predict(prepared.X_test)  # -1 anomaly, 1 normal
    y_pred = np.where(raw_pred == -1, 1, 0)  # map to Class

    # decision_function: higher = more normal; invert so higher = more anomalous/fraud
    decision = model.decision_function(prepared.X_test)
    y_score = -decision  # higher score => more likely fraud

    return TrainedModelBundle(
        model_name=MODEL_ISOLATION_FOREST,
        model=model,
        scaler=prepared.scaler,
        feature_names=prepared.feature_names,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        params={
            "n_estimators": n_estimators,
            "contamination": cont,
            "max_samples": max_samples,
        },
        training_notes=(
            "Isolation Forest trained WITHOUT SMOTE. "
            f"Contamination set to {cont:.6f}. "
            "Predictions mapped: -1→fraud(1), 1→legit(0)."
        ),
        uses_smote=False,
        supports_proba=False,
        feature_importances=None,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        y_pred=y_pred,
        y_score=y_score,
    )


def train_model(
    model_name: str,
    prepared: PreparedData,
    **kwargs,
) -> TrainedModelBundle:
    """Dispatch training by model name."""
    if model_name == MODEL_LOGISTIC:
        return train_logistic_regression(prepared, **kwargs)
    if model_name == MODEL_RANDOM_FOREST:
        return train_random_forest(prepared, **kwargs)
    if model_name == MODEL_ISOLATION_FOREST:
        return train_isolation_forest(prepared, **kwargs)
    raise ValueError(
        f"Unsupported model: {model_name}. Choose from {SUPPORTED_MODELS}."
    )


def save_model_bundle(bundle: TrainedModelBundle) -> str:
    """Persist a trained model bundle to models/ using joblib."""
    ensure_models_dir()
    path = _model_filename(bundle.model_name)
    # Do not pickle huge test matrices into disk if extremely large —
    # but for academic demos keeping them enables reload + metrics.
    payload = {
        "model_name": bundle.model_name,
        "model": bundle.model,
        "scaler": bundle.scaler,
        "feature_names": bundle.feature_names,
        "trained_at": bundle.trained_at,
        "params": bundle.params,
        "training_notes": bundle.training_notes,
        "uses_smote": bundle.uses_smote,
        "supports_proba": bundle.supports_proba,
        "feature_importances": bundle.feature_importances,
        "X_test": bundle.X_test,
        "y_test": bundle.y_test,
        "y_pred": bundle.y_pred,
        "y_score": bundle.y_score,
    }
    joblib.dump(payload, path)
    return path


def load_model_bundle(model_name: str) -> TrainedModelBundle:
    """Load a previously saved model bundle from models/."""
    path = _model_filename(model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found for '{model_name}'. Please train the model first."
        )
    payload = joblib.load(path)
    return TrainedModelBundle(**payload)


def list_saved_models() -> list:
    """Return names of models that have a corresponding .joblib file."""
    ensure_models_dir()
    found = []
    for name in SUPPORTED_MODELS:
        if os.path.exists(_model_filename(name)):
            found.append(name)
    return found
