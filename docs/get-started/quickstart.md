---
description: Validate a model with trustcv in a few lines, with independent rows and with repeated patients.
---

# Quickstart

## 1. Independent rows

When every row is a different patient and time order does not matter, stratified K-fold is appropriate. `recommend_cv` reaches that conclusion only from what you tell it, so say so explicitly with `declare_independent_samples=True`.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustcv import TrustCV, recommend_cv

X, y = load_breast_cancer(return_X_y=True)
plan = recommend_cv(X, y)
print(plan.category, plan.method)  # iid stratified_kfold

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
result = TrustCV(method=plan.method, declare_independent_samples=True).validate(
    model=model, X=X, y=y, cv=plan.splitter
)
print(result.overall_status)
print(result.mean_scores["roc_auc"], result.confidence_intervals["roc_auc"])
```

Keep scaling, imputation, feature selection and resampling inside the `Pipeline`. Anything applied to `X` before `validate()` is fitted on the test folds too, and no check can see it.

## 2. Several rows per patient

Repeated visits, several images per patient or several samples per hospital make rows dependent. Pass the patient (or site) IDs as `groups=` and use a group-aware method, so a patient is never in both training and test.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from trustcv import TrustCV, recommend_cv

rng = np.random.default_rng(0)
patients = np.repeat(np.arange(60), 4)            # 60 patients, 4 visits each
X = rng.normal(size=(60, 6))[patients] + rng.normal(0, 0.1, (240, 6))
y = np.repeat(rng.integers(0, 2, 60), 4)

plan = recommend_cv(X, y, groups=patients)
print(plan.method)  # stratified_grouped_kfold

result = TrustCV(method="stratified_group_kfold").validate(
    model=RandomForestClassifier(n_estimators=100, random_state=0),
    X=X, y=y, groups=patients,
)
print(result.checks["group_leakage"].status)  # PASSED: no patient on both sides
print(result.overall_status)
```

If you leave out `groups=` here, trustcv notices that rows come in tight clusters and reports `NEEDS_REVIEW` with a hint to pass the patient IDs.

## 3. Read the result

`result.summary()` prints the scores with confidence intervals, every check and the overall status. [Reading results](reading-results.md) explains each line.

## Next steps

- [Choosing a split](../guides/choosing-cv.md) for time series, sites and spatial data
- [Data leakage](../guides/data-leakage.md) for what the checks catch and miss
- [Command line and examples](cli.md) to run complete recipes
