# Fairness-aware student-risk demo

A minimal Python project that trains a baseline classifier on student's success or failure data, then audits subgroup fairness with **Aequitas** and tests simple **re-sampling** and **post-hoc thresholding** strategies.

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn aequitas fastapi
```

## Data layout

Put your splits here (CSV headers must include `label_value` and the features):

```
app/train_with_proxy.csv
app/test_with_proxy.csv
```

Sensitive attributes used in audits (already mapped to readable labels):
`["Age Code", "Educational special needs", "Gender", "Debtor"]`

## Quick start (Python REPL / notebook)

```python
from project import train_and_eval_variant, retrain_and_evaluate, posthoc_group_threshold

# 1) Baseline model (LogisticRegression by default)
res = train_and_eval_variant(exclude_cols=None, threshold=0.5)
print(res["accuracy"], res["f1_score"])

# 2) Re-sample the train set to balance a protected attribute
rows = __import__("pandas").read_csv("app/test_with_proxy.csv").to_dict("records")
fair = retrain_and_evaluate(rows, method="uniform_both", feature="Gender", threshold=0.5)
print(fair["fairness_at_threshold"])

# 3) Post-hoc subgroup list-size thresholds (top-k style)
ph = posthoc_group_threshold(feature="Gender", k=1000, method=None)
print(ph["group_list_sizes"])
```

