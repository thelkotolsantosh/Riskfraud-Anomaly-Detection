import pandas as pd
import pytest

from riskfraud.data import generate_synthetic_data, validate_dataset


def test_generator_is_reproducible_and_contains_two_classes():
    first = generate_synthetic_data(rows=100, fraud_rate=0.1, seed=7)
    second = generate_synthetic_data(rows=100, fraud_rate=0.1, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["is_fraud"].nunique() == 2


def test_generator_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least 20"):
        generate_synthetic_data(rows=10)
    with pytest.raises(ValueError, match="fraud_rate"):
        generate_synthetic_data(rows=100, fraud_rate=0.8)


def test_validation_rejects_missing_columns_and_duplicate_ids():
    frame = generate_synthetic_data(rows=100, fraud_rate=0.1)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(frame.drop(columns=["avg_transaction_amount"]))

    frame.loc[1, "user_id"] = frame.loc[0, "user_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(frame)


def test_scoring_input_does_not_require_label():
    frame = generate_synthetic_data(rows=100, fraud_rate=0.1).drop(columns="is_fraud")
    assert len(validate_dataset(frame, require_label=False)) == 100
