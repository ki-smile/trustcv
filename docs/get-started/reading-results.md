---
description: What trustcv's overall status and each integrity check mean, and what a PASSED status does not prove.
---

# Reading results

`TrustCV(...).validate(...)` returns a `ValidationResult` with scores, confidence intervals, a set of integrity checks and one overall status.

## Overall status

| Status | Meaning | What to do |
| --- | --- | --- |
| <span class="tcv-status failed">FAILED</span> | A leakage-relevant check failed or errored. | Fix the cause; the scores are not trustworthy. |
| <span class="tcv-status review">NEEDS_REVIEW</span> | A check found signs of possible leakage (`WARNING`) that cannot be proven automatically. | Read the check's message and recommendation, and confirm or rule it out. |
| <span class="tcv-status unverified">NOT_FULLY_VERIFIED</span> | Nothing suspicious, but some condition could not be checked. | Supply what was missing (often `groups=`) or document why it does not apply. |
| <span class="tcv-status passed">PASSED</span> | Every relevant check passed or did not apply. | Report it, together with what PASSED does not cover (below). |

The values are available as `trustcv.checks.OVERALL_STATUSES`.

## The checks

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustcv import TrustCV

X, y = load_breast_cancer(return_X_y=True)
result = TrustCV(declare_independent_samples=True).validate(
    model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X=X, y=y
)
for name, check in result.checks.items():
    print(f"{name:26} {check.status:15} {check.message[:60]}")
```

| Check | Counts towards status | What it establishes |
| --- | --- | --- |
| `duplicate_samples` | yes | Exact duplicate rows (values only), and how many test rows have an identical training row in the scored folds. |
| `group_leakage` | yes | With `groups=`: no group in both training and test, on every scored fold. Without groups: `WARNING` when rows look clustered or an identifier column repeats. |
| `preprocessing_leakage` | yes | `PASSED` only when the model is an sklearn `Pipeline`, so its steps are refit per fold. |
| `external_leakage_detector` | yes | Duplicates, near-duplicates and other patterns, checked on the folds that were scored. |
| `target_leakage_features` | yes | `WARNING` names features that almost determine the target. Very strong single predictors are listed without changing the status. |
| `permutation_sanity` | when enabled | Opt-in (`permutation_check=True`): shuffled labels must score at chance inside the CV loop. |
| `class_balance` | no | Describes imbalance; never a leakage failure. |
| `covariate_shift` | no | Features whose distribution differs between folds; shift is not leakage. |

Each check has a `status`, a one-line `message` and `details` for programmatic use. `result.recommendations` lists fixes, such as the method to use when groups overlap.

## What PASSED does not prove

Some leaks happen before `validate()` receives your data, so no check can see them:

- feature selection, scaling, imputation, encoding or oversampling applied to the whole dataset first;
- hyperparameters tuned on the same folds you report (use nested CV);
- early stopping or checkpoint choice on the test fold;
- missing patient IDs: trustcv can flag clustered rows, but cannot invent the grouping;
- features recorded after the outcome that are only moderately predictive.

A `PASSED` status means every applicable supported check passed. It is not proof that no leakage exists.

## Legacy fields

`result.leakage_check` remains for compatibility. Its `has_leakage` key is inverted (`True` means no leakage was found) and deprecated; use `result.checks` or `leakage_check["external_leakage_detected"]`.
