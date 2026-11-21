"""Isolation Forest training, ranking, evaluation, and persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .data import FEATURE_COLUMNS, ID_COLUMN, LABEL_COLUMN, validate_dataset


class RiskModel:
    """Fit an Isolation Forest and return higher scores for stronger anomalies."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be greater than 0 and at most 0.5")
        self.contamination = contamination
        self.random_state = random_state
        self.estimator = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, frame: pd.DataFrame) -> "RiskModel":
        checked = validate_dataset(frame, require_label=True)
        self.estimator.fit(checked.loc[:, FEATURE_COLUMNS])
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        checked = validate_dataset(frame, require_label=False)
        # Isolation Forest's decision_function is lower for anomalies.
        return -self.estimator.decision_function(checked.loc[:, FEATURE_COLUMNS])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "contamination": self.contamination,
                "random_state": self.random_state,
                "estimator": self.estimator,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RiskModel":
        stored = joblib.load(path)
        instance = cls(
            contamination=stored["contamination"],
            random_state=stored["random_state"],
        )
        instance.estimator = stored["estimator"]
        return instance


def evaluate_scores(
    labels: pd.Series | np.ndarray,
    scores: np.ndarray,
    alert_rate: float = 0.05,
) -> dict[str, float | int]:
    """Evaluate ranking quality at an alert budget; scores are not probabilities."""
    if not 0 < alert_rate <= 1:
        raise ValueError("alert_rate must be greater than 0 and at most 1")
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    if len(y) != len(values) or len(y) == 0:
        raise ValueError("labels and scores must have the same nonzero length")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("labels must contain both 0 and 1")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")

    k = max(1, int(np.ceil(len(values) * alert_rate)))
    top_indices = np.argsort(values)[-k:]
    true_positives = int(y[top_indices].sum())
    positive_count = int(y.sum())
    return {
        "roc_auc": float(roc_auc_score(y, values)),
        "alert_count": k,
        "precision_at_budget": true_positives / k,
        "recall_at_budget": true_positives / positive_count,
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    contamination: float = 0.05,
    test_size: float = 0.25,
    random_state: int = 42,
    alert_rate: float = 0.05,
) -> tuple[RiskModel, dict[str, float | int]]:
    """Split labeled data, fit without labels, and evaluate on held-out rows."""
    checked = validate_dataset(frame, require_label=True)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    train, test = train_test_split(
        checked,
        test_size=test_size,
        random_state=random_state,
        stratify=checked[LABEL_COLUMN],
    )
    model = RiskModel(contamination=contamination, random_state=random_state)
    model.fit(train)
    metrics = evaluate_scores(
        test[LABEL_COLUMN], model.score(test), alert_rate=alert_rate
    )
    metrics["train_rows"] = len(train)
    metrics["test_rows"] = len(test)
    return model, metrics
