# Business impact: how to read the example

This repository does not contain real transaction data or measured business results. The generated labels and model metrics are for demonstrating a workflow only.

A team evaluating a real alerting system can estimate net value with an explicit set of assumptions:

```text
net value = (true alerts × average loss avoided)
            - (false alerts × investigation cost)
```

The inputs should come from the organization’s own confirmed outcomes, review costs, and operating constraints. Alert budget controls how many records are sent for review; it does not establish that an alert is fraud.

The project reports ROC AUC, precision at the selected alert budget, and recall at that budget on a synthetic held-out split. None of these values should be presented as an expected production result.
