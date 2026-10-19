---
description: How trustcv computes confidence intervals for cross-validated scores, why bootstrapping fold scores is too narrow, and how to resample whole patients.
---

# Confidence intervals

A cross-validated score is an estimate, and its interval says how far the estimate could plausibly be from the performance of a model trained the same way. The folds share most of their training data, so their scores are correlated. An interval that treats fold scores as independent is too narrow.

`TrustCV` reports an interval for every metric in `result.confidence_intervals`. Choose the method with `ci_method=`:

| `ci_method` | What it resamples | Use when |
| --- | --- | --- |
| `"corrected_t"` (default) | Nothing; a t interval on fold scores with the Nadeau–Bengio variance correction | Any splitter, including repeated K-fold |
| `"oof_bootstrap"` | Pooled out-of-fold predictions; whole groups when `groups=` is given | Each row is tested exactly once, and you want the interval of the pooled metric |
| `"bootstrap"` (legacy) | The fold scores | Comparison with older results only; it warns |

`ci_level=` sets the level (default 0.95) and `n_bootstrap=` the number of resamples (default 1000).

## Corrected resampled t (`corrected_t`)

With `k` folds and `r` repeats, the plain variance of the mean fold score, `s²/(r·k)`, ignores that the training sets overlap. Nadeau and Bengio (2003) inflate it to

`(1/(r·k) + n_test/n_train) · s²`

where `n_test` and `n_train` are the actual mean test and training fold sizes, and use a t quantile with `r·k − 1` degrees of freedom. The interval is centred on the mean fold score.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from trustcv import TrustCV

rng = np.random.default_rng(0)
patients = np.repeat(np.arange(80), 3)  # 80 patients, 3 rows each
X = rng.normal(size=(80, 5))[patients] + rng.normal(0, 0.3, (240, 5))
y = np.repeat((rng.random(80) < 0.4).astype(int), 3)
X[:, 0] += y

result = TrustCV(method="stratified_group_kfold", metrics=["roc_auc", "accuracy"]).validate(
    model=LogisticRegression(), X=X, y=y, groups=patients
)
low, high = result.confidence_intervals["roc_auc"]
print(f"AUC {result.mean_scores['roc_auc']:.3f} (95% CI {low:.3f} to {high:.3f})")
```

With 5 folds the interval is wide. That reflects how little 5 correlated fold scores can tell you, not a fault of the method.

## Out-of-fold bootstrap (`oof_bootstrap`)

`oof_bootstrap` pools the out-of-fold predictions from every fold and bootstraps the metric over them. When `groups=` is passed, it resamples whole groups (cluster bootstrap), so all rows of a patient are in or out of a resample together. Without groups, classification resamples are stratified by class so every resample keeps both classes.

```python
result_oof = TrustCV(
    method="stratified_group_kfold", metrics=["roc_auc", "accuracy"], ci_method="oof_bootstrap"
).validate(model=LogisticRegression(), X=X, y=y, groups=patients)
print(result_oof.confidence_intervals["roc_auc"])
print(result_oof.diagnostics["confidence_intervals"]["roc_auc"])  # skipped resamples
```

- The interval describes the pooled out-of-fold metric. `result.mean_scores` is still the mean of the fold scores, so the two can differ a little.
- The test folds must cover every row exactly once. Splitters that do not (Monte Carlo, bootstrap, repeated K-fold, hold-out) fall back to `corrected_t` with a warning.
- `validate(..., ci_cluster=False)` resamples rows even when groups are given. Use it only to show how much narrower a row-level interval is; with repeated measures it is too narrow.
- Resamples in which a metric is undefined (for example, only one class) are skipped. The count is in `result.diagnostics["confidence_intervals"]`, and a warning is raised when more than 10% are skipped.

## Why not bootstrap the fold scores

Bootstrapping 5 or 10 fold scores treats them as independent draws. They are not: any two folds share most of their training data. The resulting interval reflects only the spread between folds and is too narrow. Before 1.1.0 this was the default. It is still available as `ci_method="bootstrap"`, and `validate()` warns when it is used.

A percentile over 5 fold scores is not a confidence interval either, for the same reason.

## Bounded metrics

For built-in metrics bounded to [0, 1], such as accuracy, sensitivity, specificity, F1 and ROC AUC, intervals are clipped to that range. The unclipped bounds are kept in `result.diagnostics["confidence_intervals"][metric]["clipped_from"]`. Intervals of custom scorers passed with `scoring=` are not clipped, because trustcv cannot know their range.

## Patient-level AUC with `ClinicalMetrics`

`ClinicalMetrics.calculate_all` computes sensitivity, specificity, PPV, NPV, AUC and related metrics from one set of predictions, for example a final hold-out set or pooled out-of-fold predictions. Its AUC interval is a percentile bootstrap. Pass `groups=` to resample whole patients instead of rows:

```python
from trustcv import ClinicalMetrics

proba = LogisticRegression().fit(X[:150], y[:150]).predict_proba(X[150:])[:, 1]
y_test, groups_test = y[150:], patients[150:]

metrics = ClinicalMetrics().calculate_all(
    y_test, (proba >= 0.5).astype(int), y_proba=proba, groups=groups_test
)
print(metrics["auc_roc"], metrics["auc_roc_ci"])
```

- `groups` must be 1-D with one entry per prediction; otherwise `calculate_all` raises `ValueError`.
- If no resample contains both classes, the interval is `(nan, nan)`.
- The intervals for sensitivity, specificity, PPV, NPV and accuracy are Wilson score intervals on counts of rows. They do not use `groups`, so with several rows per patient they are too narrow.

## Reporting an interval

State the method with the interval, for example "AUC 0.81 (95% CI 0.74 to 0.87, corrected resampled t, 5-fold patient-grouped CV)". `result.summary()` names the method it used. For how to record this alongside the split, see [documenting validation](../regulatory/documentation-guide.md).

## References

- Nadeau, C. and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning* 52, 239–281.
- Bouckaert, R. R. and Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. *PAKDD 2004*, 3–12.
