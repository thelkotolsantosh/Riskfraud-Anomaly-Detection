"""Synthetic dataset generation and CSV validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "user_id"
LABEL_COLUMN = "is_fraud"
FEATURE_COLUMNS = (
    "transaction_count_7d",
    "avg_transaction_amount",
    "night_transaction_ratio",
    "unique_merchants_7d",
)


def generate_synthetic_data(
    rows: int = 1000, fraud_rate: float = 0.03, seed: int = 42
) -> pd.DataFrame:
    """Create reproducible, illustrative user activity with synthetic labels."""
    if rows < 20:
        raise ValueError("rows must be at least 20")
    if not 0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be greater than 0 and less than 0.5")

    fraud_count = max(2, round(rows * fraud_rate))
    if fraud_count > rows - 2:
        raise ValueError("rows and fraud_rate must leave at least two normal rows")

    rng = np.random.default_rng(seed)
    labels = np.zeros(rows, dtype=int)
    labels[:fraud_count] = 1
    rng.shuffle(labels)
    fraud = labels == 1
    normal = ~fraud

    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"user_{index:05d}" for index in range(rows)],
            "transaction_count_7d": np.where(
                fraud, rng.integers(45, 91, rows), rng.integers(1, 21, rows)
            ),
            "avg_transaction_amount": np.where(
                fraud, rng.uniform(600, 2000, rows), rng.uniform(10, 250, rows)
            ),
            "night_transaction_ratio": np.where(
                fraud, rng.uniform(0.65, 1.0, rows), rng.uniform(0.0, 0.30, rows)
            ),
            "unique_merchants_7d": np.where(
                fraud, rng.integers(25, 61, rows), rng.integers(1, 11, rows)
            ),
            LABEL_COLUMN: labels,
        }
    )
    return frame


def validate_dataset(frame: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    """Validate required columns and return a defensive copy."""
    required = {ID_COLUMN, *FEATURE_COLUMNS}
    if require_label:
        required.add(LABEL_COLUMN)

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("dataset must contain at least one row")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("user_id values must be present and unique")

    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite numeric values")

    if require_label:
        labels = frame[LABEL_COLUMN]
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError("is_fraud must contain only 0 or 1")
        if labels.nunique() != 2:
            raise ValueError("is_fraud must contain both classes")

    return frame.copy()


def read_dataset(path: str | Path, require_label: bool = True) -> pd.DataFrame:
    """Read and validate a CSV dataset."""
    frame = pd.read_csv(path)
    return validate_dataset(frame, require_label=require_label)
