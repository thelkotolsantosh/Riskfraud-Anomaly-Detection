import numpy as np
import pytest

from riskfraud.data import generate_synthetic_data
from riskfraud.model import RiskModel, evaluate_scores, train_and_evaluate


def test_training_ranks_synthetic_anomalies_higher_than_normal_rows():
    frame = generate_synthetic_data(rows=400, fraud_rate=0.1, seed=3)
    model, metrics = train_and_evaluate(
        frame, test_size=0.25, random_state=3, alert_rate=0.1
    )

    assert metrics["roc_auc"] > 0.9
    assert metrics["precision_at_budget"] > 0.8
    assert len(model.score(frame)) == len(frame)


def test_model_save_and_load_preserves_scores(tmp_path):
    frame = generate_synthetic_data(rows=200, fraud_rate=0.1)
    model = RiskModel().fit(frame)
    path = tmp_path / "models" / "model.joblib"
    model.save(path)

    loaded = RiskModel.load(path)
    np.testing.assert_allclose(model.score(frame), loaded.score(frame))


def test_alert_budget_flags_top_fraction():
    result = evaluate_scores([0, 0, 0, 1], np.array([0.1, 0.2, 0.3, 0.9]), 0.25)
    assert result["alert_count"] == 1
    assert result["precision_at_budget"] == 1.0
    assert result["recall_at_budget"] == 1.0


def test_invalid_contamination_and_alert_rate_are_rejected():
    with pytest.raises(ValueError, match="contamination"):
        RiskModel(contamination=0)
    with pytest.raises(ValueError, match="alert_rate"):
        evaluate_scores([0, 1], np.array([0.1, 0.9]), alert_rate=0)
