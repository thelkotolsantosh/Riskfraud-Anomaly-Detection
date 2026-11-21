# Limitations

- **Synthetic data:** The generator uses a small set of deliberately separated patterns. It does not model real customers, transactions, fraud rings, or changing attacker behavior.
- **Synthetic labels:** Labels are created by the generator and are clean by construction. Real fraud labels are delayed, incomplete, and affected by investigation choices.
- **Anomaly is not fraud:** Unusual behavior can be legitimate. A score should not be used alone to deny service or make a financial decision.
- **Uncalibrated scores:** Isolation Forest scores rank observations. They are not probabilities and do not explain why a row was flagged.
- **Offline evaluation:** A random stratified split does not test future-time performance, new-user generalization, investigator workload, or production drift.
- **Limited feature set:** The example uses four numeric aggregates and has no streaming, graph, device, or external risk signals.

Before adapting the approach to a real problem, define the review process, use appropriately governed data, establish time-aware and user-separated evaluation, compare with a simple baseline, and monitor false positives and missed cases.
