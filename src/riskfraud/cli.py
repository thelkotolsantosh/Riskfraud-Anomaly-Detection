"""Command-line interface for generating, training, and scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import generate_synthetic_data, read_dataset, validate_dataset
from .model import RiskModel, train_and_evaluate


def _write_json(path: str, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rank synthetic user activity by anomaly score."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="create a labeled synthetic CSV")
    generate.add_argument("--rows", type=int, default=1000)
    generate.add_argument("--fraud-rate", type=float, default=0.03)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", required=True)

    train = commands.add_parser("train", help="train and evaluate an Isolation Forest")
    train.add_argument("--input", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--metrics")
    train.add_argument("--contamination", type=float, default=0.05)
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--alert-rate", type=float, default=0.05)
    train.add_argument("--seed", type=int, default=42)

    score = commands.add_parser("score", help="score a CSV and flag top-ranked rows")
    score.add_argument("--input", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--alert-rate", type=float, default=0.05)
    score.add_argument("--output", required=True)

    args = parser.parse_args(argv)

    if args.command == "generate":
        frame = generate_synthetic_data(args.rows, args.fraud_rate, args.seed)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"Wrote {len(frame)} synthetic rows to {destination}")
        return

    if args.command == "train":
        frame = read_dataset(args.input, require_label=True)
        model, metrics = train_and_evaluate(
            frame,
            contamination=args.contamination,
            test_size=args.test_size,
            random_state=args.seed,
            alert_rate=args.alert_rate,
        )
        model.save(args.model)
        if args.metrics:
            _write_json(args.metrics, metrics)
        print(json.dumps(metrics, indent=2))
        return

    if not 0 < args.alert_rate <= 1:
        parser.error("--alert-rate must be greater than 0 and at most 1")
    frame = read_dataset(args.input, require_label=False)
    model = RiskModel.load(args.model)
    scores = model.score(frame)
    count = max(1, int(__import__("math").ceil(len(frame) * args.alert_rate)))
    top_indices = pd.Series(scores).nlargest(count).index
    result = pd.DataFrame(
        {
            "user_id": frame["user_id"],
            "risk_score": scores,
            "flagged": 0,
        }
    )
    result.loc[top_indices, "flagged"] = 1
    result = result.sort_values("risk_score", ascending=False)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Wrote scores for {len(result)} rows to {destination}")


if __name__ == "__main__":
    main()
