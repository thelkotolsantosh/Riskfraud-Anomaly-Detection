# Risk and Fraud Anomaly Detection

A small, reproducible portfolio project that uses Isolation Forest to rank synthetic user activity by anomaly score. It demonstrates a complete workflow: generate data, train a model, measure ranking quality, and score a batch for review.

This is an educational prototype. The scores are anomaly rankings, not calibrated fraud probabilities, and the synthetic data does not represent real customer behavior.

## What it does

- Generates deterministic synthetic user-level activity with a small labeled anomaly group.
- Trains an Isolation Forest using numeric behavior features.
- Reports ROC AUC, precision at the selected alert budget, and recall at that budget on a held-out split.
- Scores a CSV and flags the highest-scoring fraction for review.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/thelkotolsantosh/Riskfraud-Anomaly-Detection.git
cd Riskfraud-Anomaly-Detection
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the project and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Generate a dataset, train the model, and score the sample:

```bash
python -m riskfraud.cli generate --rows 1000 --fraud-rate 0.03 --seed 42 --output data/generated.csv
python -m riskfraud.cli train --input data/generated.csv --model models/isolation_forest.joblib --test-size 0.25
python -m riskfraud.cli score --input data/sample_features.csv --model models/isolation_forest.joblib --alert-rate 0.10 --output results/scores.csv
```

The generated training CSV contains a synthetic `is_fraud` label for evaluation. The scoring input may omit that label. The score output contains `user_id`, `risk_score`, and `flagged`; higher scores mean more anomalous activity.

You can also install a command-line entry point and use `riskfraud --help`.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest
```

## Input format

The training file must contain these columns:

| Column | Meaning |
|---|---|
| `user_id` | Unique synthetic user identifier |
| `transaction_count_7d` | Transactions in the previous seven days |
| `avg_transaction_amount` | Average transaction amount |
| `night_transaction_ratio` | Share of activity during the synthetic night-time window |
| `unique_merchants_7d` | Distinct merchants in the previous seven days |
| `is_fraud` | Synthetic label used only for evaluation |

The four behavior columns must be numeric and finite. The scoring file needs the identifier and behavior columns; `is_fraud` is optional.

## How evaluation works

The training command makes a stratified train/test split. Isolation Forest is fit on the training rows without using labels. Labels are used only to calculate held-out metrics. The alert budget selects the top-scoring fraction of the held-out rows.

- **ROC AUC** measures whether labeled anomalies tend to rank above normal rows.
- **Precision at budget** is the share of flagged rows with a positive synthetic label.
- **Recall at budget** is the share of positive synthetic labels that are flagged.

These metrics describe this generated dataset only. They do not predict performance on real transactions.

## Project layout

```text
src/riskfraud/       Data validation, model, and CLI
tests/               Unit and end-to-end CLI tests
data/sample_features.csv
docs/                Business-impact calculation and limitations
pyproject.toml        Package metadata and dependency ranges
```

## Limitations

The generator creates simple, deliberately separated behavior patterns so the workflow is easy to reproduce. Real fraud is more varied, labels are delayed and biased, and unusual behavior is not necessarily fraud. Isolation Forest produces an anomaly score, not a probability or an explanation. Do not use this project to make real financial or customer decisions.

See [Limitations](Limitations.MD) and [Business Impact](Business%20Impact.MD) for details.

## License

MIT. See [LICENSE](LICENSE).
