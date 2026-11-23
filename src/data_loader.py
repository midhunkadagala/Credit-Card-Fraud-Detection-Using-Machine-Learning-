"""
data_loader.py
--------------
Responsible for reading and validating a user-uploaded CSV dataset.

IMPORTANT:
- The dataset is NEVER bundled with this project.
- The user must upload a CSV at runtime through the Streamlit UI.
- No automatic download from Kaggle or any other source.
- No silent fallback to sample / fake data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

# Expected columns for the Kaggle Credit Card Fraud Detection dataset
REQUIRED_TARGET = "Class"
REQUIRED_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS = REQUIRED_FEATURES + [REQUIRED_TARGET]


@dataclass
class DatasetValidationResult:
    """Result of validating an uploaded CSV."""

    is_valid: bool
    message: str
    missing_columns: List[str] = field(default_factory=list)
    extra_info: str = ""


@dataclass
class DatasetInfo:
    """Summary statistics shown after a successful upload."""

    filename: str
    n_rows: int
    n_columns: int
    n_fraud: int
    n_legit: int
    fraud_pct: float
    n_missing: int
    n_duplicates: int
    columns: List[str]


def validate_dataset_schema(df: pd.DataFrame) -> DatasetValidationResult:
    """
    Validate that the uploaded DataFrame matches the expected Kaggle schema.

    Expected:
      - Target column: Class (0 = legitimate, 1 = fraudulent)
      - Features: Time, Amount, V1..V28
    """
    if df is None or df.empty:
        return DatasetValidationResult(
            is_valid=False,
            message="The uploaded file is empty. Please upload a non-empty CSV dataset.",
        )

    # Normalize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    present = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = sorted(expected - present)

    if missing:
        return DatasetValidationResult(
            is_valid=False,
            message=(
                "The uploaded CSV does not match the expected Credit Card Fraud "
                "Detection format. Missing required columns are listed below."
            ),
            missing_columns=missing,
            extra_info=(
                f"Expected columns: {', '.join(EXPECTED_COLUMNS)}. "
                f"Found columns: {', '.join(df.columns.tolist())}."
            ),
        )

    # Target column checks
    if REQUIRED_TARGET not in df.columns:
        return DatasetValidationResult(
            is_valid=False,
            message="Missing target column 'Class'.",
            missing_columns=[REQUIRED_TARGET],
        )

    unique_classes = set(pd.Series(df[REQUIRED_TARGET]).dropna().unique().tolist())
    # Allow numeric 0/1 or equivalent
    try:
        numeric_classes = {int(x) for x in unique_classes}
    except (TypeError, ValueError):
        return DatasetValidationResult(
            is_valid=False,
            message=(
                "Column 'Class' must contain numeric labels: "
                "0 = legitimate, 1 = fraudulent. "
                f"Found values: {sorted(unique_classes)}."
            ),
        )

    if not numeric_classes.issubset({0, 1}):
        return DatasetValidationResult(
            is_valid=False,
            message=(
                "Column 'Class' must only contain 0 (legitimate) and 1 (fraudulent). "
                f"Found values: {sorted(numeric_classes)}."
            ),
        )

    if 0 not in numeric_classes or 1 not in numeric_classes:
        return DatasetValidationResult(
            is_valid=False,
            message=(
                "Column 'Class' should contain both classes for training: "
                "0 (legitimate) and 1 (fraudulent). "
                f"Found values: {sorted(numeric_classes)}."
            ),
        )

    # Basic dtype check for feature columns
    bad_types = []
    for col in REQUIRED_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            bad_types.append(col)

    if bad_types:
        return DatasetValidationResult(
            is_valid=False,
            message=(
                "Some required feature columns are not numeric. "
                f"Non-numeric columns: {', '.join(bad_types)}."
            ),
        )

    return DatasetValidationResult(
        is_valid=True,
        message="Dataset validated successfully. Schema matches the expected format.",
    )


def load_uploaded_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], DatasetValidationResult]:
    """
    Read a Streamlit UploadedFile (or file-like object) as a CSV and validate it.

    Returns
    -------
    (dataframe, validation_result)
        dataframe is None when validation fails.
    """
    if uploaded_file is None:
        return None, DatasetValidationResult(
            is_valid=False,
            message="Please upload your credit card fraud detection dataset to continue.",
        )

    filename = getattr(uploaded_file, "name", "uploaded.csv")
    if not str(filename).lower().endswith(".csv"):
        return None, DatasetValidationResult(
            is_valid=False,
            message="Invalid file type. Please upload a CSV file (.csv).",
        )

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001 — surface friendly message in UI
        return None, DatasetValidationResult(
            is_valid=False,
            message=f"Could not read the CSV file. Error: {exc}",
        )

    if df.empty:
        return None, DatasetValidationResult(
            is_valid=False,
            message="The uploaded CSV has no rows. Please upload a valid dataset.",
        )

    validation = validate_dataset_schema(df)
    if not validation.is_valid:
        return None, validation

    # Ensure Class is integer 0/1
    df = df.copy()
    df[REQUIRED_TARGET] = df[REQUIRED_TARGET].astype(int)

    # Drop exact duplicate rows after validation (report count separately in get_dataset_info)
    return df, validation


def get_dataset_info(df: pd.DataFrame, filename: str) -> DatasetInfo:
    """Build a DatasetInfo summary for the dashboard."""
    n_fraud = int((df[REQUIRED_TARGET] == 1).sum())
    n_legit = int((df[REQUIRED_TARGET] == 0).sum())
    n_rows = int(len(df))
    fraud_pct = (n_fraud / n_rows * 100.0) if n_rows > 0 else 0.0
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())

    return DatasetInfo(
        filename=filename,
        n_rows=n_rows,
        n_columns=int(df.shape[1]),
        n_fraud=n_fraud,
        n_legit=n_legit,
        fraud_pct=fraud_pct,
        n_missing=n_missing,
        n_duplicates=n_duplicates,
        columns=df.columns.tolist(),
    )


def validate_prediction_csv(df: pd.DataFrame, require_class: bool = False) -> DatasetValidationResult:
    """
    Validate a CSV used for batch prediction.

    Prediction files must contain the feature columns (Time, Amount, V1..V28).
    The Class column is optional for prediction (labels may be unknown).
    """
    if df is None or df.empty:
        return DatasetValidationResult(
            is_valid=False,
            message="The prediction CSV is empty.",
        )

    df.columns = [str(c).strip() for c in df.columns]
    present = set(df.columns)
    missing_features = sorted(set(REQUIRED_FEATURES) - present)

    if missing_features:
        return DatasetValidationResult(
            is_valid=False,
            message="Prediction file is missing required feature columns.",
            missing_columns=missing_features,
            extra_info=f"Found columns: {', '.join(df.columns.tolist())}.",
        )

    if require_class and REQUIRED_TARGET not in present:
        return DatasetValidationResult(
            is_valid=False,
            message="Missing target column 'Class'.",
            missing_columns=[REQUIRED_TARGET],
        )

    bad_types = [c for c in REQUIRED_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_types:
        return DatasetValidationResult(
            is_valid=False,
            message=f"Non-numeric feature columns: {', '.join(bad_types)}.",
        )

    return DatasetValidationResult(
        is_valid=True,
        message="Prediction file validated successfully.",
    )
