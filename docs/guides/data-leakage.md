---
description: The forms of data leakage common in medical machine learning, which ones trustcv checks on the scored folds, and the ones no check can see.
---

# Data leakage

Data leakage is information from the test data reaching model training or model selection. The cross-validation score then overstates how the model will do on new patients. In clinical work this is the most common reason a model that validated well fails in use.

trustcv detects the supported forms of leakage on the folds that were actually scored, and says what it could not check. Some leaks happen before trustcv receives the data; no check can see those, and they are listed [at the end of this page](#what-the-checks-cannot-see).

## Forms of leakage

| Form | Example | Prevention | Checked by trustcv |
| --- | --- | --- | --- |
| Patient | Several images or visits of one patient on both sides | Group by patient (`groups=`) | `group_leakage` with groups; independence scan and identifier hints without |
| Temporal | Training on 2024 to predict 2023; no gap before the test period | Forward splits with a gap | Splitter design; `DataLeakageChecker.check(timestamps=...)` |
| Spatial | Adjacent tissue patches or neighbouring regions on both sides | Spatial blocks with a buffer | Splitter design; `DataLeakageChecker.check(coordinates=...)` |
| Site and device | One scanner's artefacts in training and test | Group by site or scanner | `group_leakage` with site IDs as `groups=` |
| Duplicates | The same record exported twice, or augmented copies | De-duplicate; augment inside the training fold | `duplicate_samples`, `external_leakage_detector` |
| Target proxy | A feature recorded after the outcome, such as a discharge code | Remove features not available at prediction time | `target_leakage_features` (univariate, strong proxies only) |
| Preprocessing before the split | Scaling, imputation or feature selection fitted on all rows | Put every step in an sklearn `Pipeline` | Only whether the model is a `Pipeline` |
| Oversampling before the split | SMOTE on the full dataset | Resample inside the training fold | Possibly the independence scan |
| Tuning on the reported folds | Choosing hyperparameters by the same CV score you report | Nested CV | Not visible |
| Early stopping on the test fold | Picking the best epoch by test-fold loss | Inner validation split inside each training fold | Warning from trustcv's own callbacks |

## How trustcv checks

`TrustCV(...).validate(...)` runs its checks on the train/test splits that produced the scores, not on splits of its own. Group overlap and exact duplicates are checked on every scored fold; the distance-based detector audits up to `max_audit_folds` folds (default 10), and its coverage is in `result.diagnostics["fold_audit"]`.

Every check ends in a status, and the checks combine into one overall status:

| Overall status | Meaning |
| --- | --- |
| <span class="tcv-status failed">FAILED</span> | A check found leakage, such as the same patient in training and test. |
| <span class="tcv-status review">NEEDS_REVIEW</span> | A check found signs of possible leakage (`WARNING`) that cannot be proven automatically. A person should confirm or rule them out. |
| <span class="tcv-status unverified">NOT_FULLY_VERIFIED</span> | Nothing suspicious, but something could not be checked, for example no patient IDs were given or the model is not a `Pipeline`. |
| <span class="tcv-status passed">PASSED</span> | Every applicable supported check passed. Not proof that no leakage exists. |

[Reading results](../get-started/reading-results.md) describes each check.

### Patient leakage

Rows from the same patient are more alike than rows from different patients. A row-level split lets the model recognise patients instead of learning the disease.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustcv import TrustCV, KFoldMedical, GroupKFoldMedical

rng = np.random.default_rng(0)
patients = np.repeat(np.arange(50), 4)                 # 50 patients, 4 rows each
X = rng.normal(size=(50, 5))[patients] + rng.normal(0, 0.1, (200, 5))
y = np.repeat(rng.integers(0, 2, 50), 4)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# Row-level folds: the same patients appear on both sides
wrong = TrustCV().validate(model=model, X=X, y=y, groups=patients,
                           cv=KFoldMedical(n_splits=5, shuffle=True, random_state=0))
print(wrong.overall_status)                    # FAILED
print(wrong.checks["group_leakage"].message)

# Patient-level folds
right = TrustCV().validate(model=model, X=X, y=y, groups=patients,
                           cv=GroupKFoldMedical(n_splits=5, random_state=0))
print(right.overall_status)                    # PASSED
```

`TrustCV(method="stratified_group_kfold")` with `groups=` does the same without building the splitter yourself.

### No groups supplied: the independence scan

When `validate()` gets no `groups=`, it scans for rows that come in unusually tight clusters: repeated subjects, augmentation or oversampling. A hit makes `group_leakage` a `WARNING` and the overall status <span class="tcv-status review">NEEDS_REVIEW</span>, also when `declare_independent_samples=True`.

```python
result = TrustCV().validate(model=model, X=X, y=y)    # the same data, groups left out
print(result.overall_status)                          # NEEDS_REVIEW
print(result.checks["group_leakage"].message)
```

It is evidence, not proof: tight clusters of genuinely independent samples look the same. Limits:

- only complete rows and columns with many distinct values are used; binary, coarse-integer or mostly missing data is reported as not applicable;
- rows of one subject that differ a lot (for example ICU time series) are not near-copies and are not flagged;
- clusters larger than 30 rows are not flagged.

Turn the scan off with `TrustCV(check_independence=False)` once you have established independence another way.

### Identifier columns

For a DataFrame, trustcv looks for columns named like identifiers (`patient_id`, `PatientID`, `mrn`, and similar). Repeated values without `groups=` make `group_leakage` a `WARNING`, and the recommendations say to drop the column and pass it as `groups=`. An identifier left in `X` also lets a model memorise patients.

```python
import pandas as pd

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
df.insert(0, "patient_id", patients)

result = TrustCV(check_independence=False).validate(model=model, X=df, y=y)
print(result.checks["group_leakage"].status)    # WARNING
print(result.recommendations[0])

# The fix: groups from the column, the column out of X
fixed = TrustCV(method="stratified_group_kfold").validate(
    model=model, X=df.drop(columns="patient_id"), y=y, groups=df["patient_id"]
)
print(fixed.overall_status)
```

### Duplicates and target proxies

```python
Xd = np.c_[X, y + rng.normal(0, 0.01, 200)]     # a feature that almost equals the label
Xd = np.r_[Xd, Xd[:20]]                         # 20 duplicated rows
yd = np.r_[y, y[:20]]

result = TrustCV(declare_independent_samples=True, check_independence=False).validate(
    model=model, X=Xd, y=yd
)
for name in ("duplicate_samples", "target_leakage_features", "external_leakage_detector"):
    print(name, result.checks[name].status, result.checks[name].message)
print(result.overall_status)                    # FAILED
```

`duplicate_samples` counts identical rows (values only, ignoring the index) and how many test rows have an identical training row. `target_leakage_features` uses univariate ROC-AUC for binary targets and absolute Spearman correlation for regression; it names features above `target_auc_threshold` (default 0.99) and lists strong single predictors (`target_strong_threshold`, default 0.90) without changing the status. It cannot see multivariate encodings of the target or moderately predictive post-outcome features.

## Auditing your own splits

`DataLeakageChecker.check(folds=...)` audits splits you made yourself, for example in a custom training loop. Pass the same `(train_idx, test_idx)` pairs you score.

```python
from trustcv import DataLeakageChecker

folds = list(KFoldMedical(n_splits=5, shuffle=True, random_state=0).split(X))
report = DataLeakageChecker(verbose=False).check(X, y, groups=patients, folds=folds)
print(report.has_leakage, report.leakage_types, report.severity)   # True ['patient'] critical
print(report.recommendations[0])
```

`check()` also accepts `timestamps=` and `coordinates=`. For runner-based training, `LeakageDetectionCallback(data=(X, y), groups=patient_ids)` runs these checks on each fold of `UniversalCVRunner`; see [Deep learning](deep-learning.md).

## Preventing structural leakage

### Temporal gaps

`purge_gap` (`PurgedKFoldCV`, `PurgedGroupTimeSeriesSplit`, `CombinatorialPurgedCV`) and `gap` (`TimeSeriesSplit`, `RollingWindowCV`, `ExpandingWindowCV`) count samples in time order, not days. With one row per day they coincide. Pass time with `timestamps=`; do not pass timestamps as `groups=`.

```python
from trustcv import PurgedKFoldCV

ts = np.arange(120)                               # one row per day
X_t = rng.normal(size=(120, 4))

cv = PurgedKFoldCV(n_splits=4, purge_gap=5)       # drop 5 samples next to each test fold
for train, test in cv.split(X_t, timestamps=ts):
    before = train[train < test.min()]
    if before.size:
        print(test.min() - before.max() - 1)      # 5 samples purged
```

### Spatial buffers

`buffer_size` in `BufferedSpatialCV` is in the units of the coordinates you pass to `split()`. A value below 1.0 is read as a fraction of the largest distance. Coordinates in degrees give a buffer in degrees; project them first if you need metres.

```python
from trustcv import BufferedSpatialCV

coords = rng.uniform(0, 100, size=(120, 2))        # here: kilometres
cv = BufferedSpatialCV(n_splits=4, buffer_size=10, random_state=0)
for train, test in cv.split(X_t, coordinates=coords):
    pass                                           # no training point within 10 km of the test block
```

### Hierarchies, sites and scanners

Pass the hierarchy to `split(hierarchy=...)` and choose the level with `hierarchy_level=`. Keeping whole hospitals together also keeps their patients together.

```python
from trustcv import HierarchicalGroupKFold, LeaveOneGroupOut

hospital = np.repeat(np.arange(6), 20)
patient = np.repeat(np.arange(40), 3)
cv = HierarchicalGroupKFold(n_splits=3, hierarchy_level="hospital", random_state=0)
for train, test in cv.split(X_t, hierarchy={"hospital": hospital, "patient": patient}):
    assert not set(hospital[train]) & set(hospital[test])

# Scanner or device effects: group by scanner and test each one in turn
scanner = np.repeat(["A", "B", "C"], 40)
for train, test in LeaveOneGroupOut().split(X_t, groups=scanner):
    pass
```

Stratifying by scanner keeps every scanner in training, so it does not test whether the model transfers to a new one.

## What the checks cannot see

These leaks happen before `validate()` receives `X`, or in decisions made outside it. Each one inflates scores, even on data with no signal, and the overall status can still be <span class="tcv-status passed">PASSED</span>.

| Done before or outside CV | Fix |
| --- | --- |
| Feature selection on the full dataset (`SelectKBest` on 2,000 noise features gives an AUC far above 0.5) | Put the selector in the `Pipeline` |
| Oversampling (SMOTE) or augmentation | Resample inside the training fold (for example `imblearn.pipeline.Pipeline`); the independence scan may flag it |
| Imputation, scaling, target encoding | Put the transformer in the `Pipeline` |
| Hyperparameter tuning on the same folds you report | Nested CV; see [Best practices](best-practices.md#tune-inside-the-training-folds) |
| Early stopping or checkpoint choice on the test fold | Hold out a validation split inside each training fold (`run(..., inner_validation=0.1)` on any trustcv runner); see [Deep learning](deep-learning.md) |
| Missing patient, site or time metadata | Supply `groups=` or a temporal or spatial splitter; trustcv cannot infer them |

The feature-selection case, run end to end:

```python
from sklearn.feature_selection import SelectKBest, f_classif
from trustcv import StratifiedKFoldMedical

X_noise = rng.normal(size=(100, 2000))
y_noise = rng.integers(0, 2, 100)                  # no relation to X
cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=0)

X_leaky = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)   # sees every row
leaky = TrustCV(declare_independent_samples=True).validate(model=model, X=X_leaky, y=y_noise, cv=cv)

honest_model = make_pipeline(SelectKBest(f_classif, k=20), StandardScaler(), LogisticRegression(max_iter=1000))
honest = TrustCV(declare_independent_samples=True).validate(model=honest_model, X=X_noise, y=y_noise, cv=cv)

print(leaky.overall_status, round(leaky.mean_scores["roc_auc"], 2))    # PASSED, far above 0.5
print(honest.overall_status, round(honest.mean_scores["roc_auc"], 2))  # PASSED, near 0.5 (chance)
```

Both runs pass every check. Only the second number describes the model.
