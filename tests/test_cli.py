import pandas as pd

from riskfraud.cli import main


def test_generate_train_and_score_commands_run_end_to_end(tmp_path):
    training = tmp_path / "generated.csv"
    model = tmp_path / "models" / "model.joblib"
    metrics = tmp_path / "metrics.json"
    scored_input = tmp_path / "scoring.csv"
    output = tmp_path / "results" / "scores.csv"

    main(["generate", "--rows", "200", "--fraud-rate", "0.1", "--output", str(training)])
    main(
        [
            "train",
            "--input", str(training),
            "--model", str(model),
            "--metrics", str(metrics),
            "--test-size", "0.25",
            "--alert-rate", "0.1",
        ]
    )
    pd.read_csv(training).drop(columns="is_fraud").to_csv(scored_input, index=False)
    main(
        [
            "score",
            "--input", str(scored_input),
            "--model", str(model),
            "--alert-rate", "0.1",
            "--output", str(output),
        ]
    )

    result = pd.read_csv(output)
    assert len(result) == 200
    assert result["flagged"].sum() == 20
    assert metrics.exists()
