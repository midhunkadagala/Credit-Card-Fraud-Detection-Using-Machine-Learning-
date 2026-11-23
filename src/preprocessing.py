"""
preprocessing.py
----------------
Train/test splitting and preprocessing helpers.

Critical rules for fraud detection:
1. Stratified train/test split (preserve class ratio).
2. Fit scalers ONLY on training data.
3. Apply SMOTE ONLY on training data (never on test data).
4. Isolation Forest does NOT use SMOTE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import REQUIRED_FEATURES, REQUIRED_TARGET


@dataclass
class PreparedData:
    """Container for train/test arrays after preprocessing."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_resampled: Optional[pd.DataFrame]
    y_train_resampled: Optional[pd.Series]
    feature_names: list
    scaler: Optional[StandardScaler]
    test_size: float
    random_state: int
    smote_applied: bool


def separate_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y."""
    if REQUIRED_TARGET not in df.columns:
        raise ValueError("Target column 'Class' is missing from the dataset.")

    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[REQUIRED_FEATURES].copy()
    y = df[REQUIRED_TARGET].astype(int).copy()
    return X, y


def drop_duplicates_and_report(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop duplicate rows and return cleaned frame + count removed."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> Tuple[pd.DataFrame, int]:
    """
    Handle missing values.

    strategy='drop' removes rows with any NaN (safest for this dataset,
    which typically has no missing values).
    """
    n_missing_rows = int(df.isna().any(axis=1).sum())
    if strategy == "drop":
        cleaned = df.dropna().reset_index(drop=True)
    else:
        cleaned = df.fillna(df.median(numeric_only=True)).reset_index(drop=True)
    return cleaned, n_missing_rows


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    apply_smote: bool = True,
    scale_features: bool = True,
    smote_k_neighbors: int = 5,
) -> PreparedData:
    """
    Full preprocessing pipeline for supervised models (LR / RF).

    Steps:
      1. Separate X, y
      2. Stratified train_test_split
      3. Optional StandardScaler fit on train only
      4. Optional SMOTE on train only
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty. Please upload a valid dataset first.")

    X, y = separate_features_target(df)

    if len(X) < 10:
        raise ValueError("Dataset is too small to split into train and test sets.")

    minority_count = int(y.value_counts().min())
    if minority_count < 2:
        raise ValueError(
            "Need at least 2 samples of each class for stratified splitting."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler: Optional[StandardScaler] = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=REQUIRED_FEATURES,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=REQUIRED_FEATURES,
            index=X_test.index,
        )
        X_train, X_test = X_train_scaled, X_test_scaled

    X_train_resampled = None
    y_train_resampled = None
    smote_applied = False

    if apply_smote:
        # Ensure k_neighbors does not exceed minority class size - 1
        n_minority = int(y_train.value_counts().min())
        k = min(smote_k_neighbors, max(1, n_minority - 1))
        if n_minority < 2:
            raise ValueError(
                "Not enough minority-class samples in the training set to apply SMOTE."
            )
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_train_resampled = pd.DataFrame(X_res, columns=REQUIRED_FEATURES)
        y_train_resampled = pd.Series(y_res, name=REQUIRED_TARGET)
        smote_applied = True

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_resampled=X_train_resampled,
        y_train_resampled=y_train_resampled,
        feature_names=list(REQUIRED_FEATURES),
        scaler=scaler,
        test_size=test_size,
        random_state=random_state,
        smote_applied=smote_applied,
    )


def prepare_data_for_isolation_forest(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> PreparedData:
    """
    Preprocessing for Isolation Forest.

    - Stratified split (so we can evaluate against true labels)
    - Scaling optional but recommended
    - NO SMOTE (unsupervised anomaly detection)
    """
    return prepare_data(
        df=df,
        test_size=test_size,
        random_state=random_state,
        apply_smote=False,
        scale_features=scale_features,
    )


def get_class_distribution(y: pd.Series) -> dict:
    """Return counts and percentages for Class 0 / Class 1."""
    counts = y.value_counts().to_dict()
    total = int(len(y))
    return {
        "legit": int(counts.get(0, 0)),
        "fraud": int(counts.get(1, 0)),
        "total": total,
        "fraud_pct": (counts.get(1, 0) / total * 100.0) if total else 0.0,
        "legit_pct": (counts.get(0, 0) / total * 100.0) if total else 0.0,
    }


def transform_features(
    X: pd.DataFrame,
    scaler: Optional[StandardScaler],
) -> pd.DataFrame:
    """Apply a fitted scaler to new feature rows (for prediction)."""
    if scaler is None:
        return X[REQUIRED_FEATURES].copy()
    scaled = scaler.transform(X[REQUIRED_FEATURES])
    return pd.DataFrame(scaled, columns=REQUIRED_FEATURES, index=X.index)
