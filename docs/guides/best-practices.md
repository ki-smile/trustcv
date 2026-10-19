---
description: Practical rules for cross-validating medical machine learning models, with the common mistakes and how to avoid them in trustcv and scikit-learn.
---

# Best practices

Most optimistic validation results come from a handful of mistakes: the wrong unit in the split, preprocessing fitted on all rows, and model choices made on the folds that are then reported. This page lists them with a correct alternative for each.

## Quick reference

| Do | Avoid |
| --- | --- |
| Split by the unit you want to generalise to: patient, site, period or region | Row-level splits when a patient has several rows |
| Put scaling, imputation, feature selection and resampling in a `Pipeline` | Fitting any of them on the full dataset before CV |
| Tune hyperparameters inside each training fold (nested CV) | Reporting the best score from a hyperparameter search |
| Early-stop on a validation split inside the training fold | Early stopping or checkpoint choice on the test fold |
| Stratify classification folds, especially for rare outcomes | Unstratified folds that leave some test folds with no events |
| Respect time order when the model predicts the future | Shuffled splits of longitudinal data |
| Report confidence intervals and the overall status | A mean score on its own |
| Fix and record random seeds | Unrecorded seeds and splitter settings |

## Common mistakes

### Preprocessing before the split

<!-- notest: deliberately wrong pattern -->
```python
# Wrong: the scaler sees the test rows
X_scaled = StandardScaler().fit_transform(X)
result = TrustCV().validate(model=LogisticRegression(), X=X_scaled, y=y)
```

Put every fitted step in the model, so it is refit on each training fold:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustcv import TrustCV

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
result = TrustCV(declare_independent_samples=True).validate(model=model, X=X, y=y)
print(result.checks["preprocessing_leakage"].status)  # PASSED: steps refit per fold
```

The `preprocessing_leakage` check can only confirm that the model is a `Pipeline`. It cannot see anything done to `X` before `validate()`.

### Feature selection outside the pipeline

Selecting features on all rows is the same mistake with a larger effect. On pure noise it produces an apparently useful model:

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from trustcv import StratifiedKFoldMedical

rng = np.random.default_rng(0)
X_noise = rng.normal(size=(100, 2000))
y_noise = rng.integers(0, 2, 100)                  # no relation to X
cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=0)

X_sel = SelectKBest(f_classif, k=20).fit_transform(X_noise, y_noise)
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y_noise, cv=cv, scoring="roc_auc")

pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
honest = cross_val_score(pipe, X_noise, y_noise, cv=cv, scoring="roc_auc")
print(round(leaky.mean(), 2), round(honest.mean(), 2))   # far above 0.5, near 0.5
```

### Patient data on both sides

`KFold(shuffle=True)` splits rows, so a patient with several rows lands in training and test.

<!-- notest: deliberately wrong pattern -->
```python
# Wrong: rows, not patients
cv = KFoldMedical(n_splits=5, shuffle=True)
for train, test in cv.split(X):
    ...  # the same patient can be in both
```

Group by patient. `TrustCV` also checks the scored folds for overlap:

```python
from trustcv import StratifiedGroupKFold

patients = np.repeat(np.arange(60), 4)
X_rep = rng.normal(size=(60, 5))[patients] + rng.normal(0, 0.1, (240, 5))
y_rep = np.repeat(rng.integers(0, 2, 60), 4)

cv = StratifiedGroupKFold(n_splits=5, random_state=0)
result = TrustCV().validate(
    model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    X=X_rep, y=y_rep, groups=patients, cv=cv,
)
print(result.checks["group_leakage"].status)   # PASSED
```

### Unstratified folds with a rare outcome

With 5% positives, unstratified folds can leave a test fold with very few or no events, and AUC or sensitivity is then undefined or unstable. (`cross_val_score(..., cv=5)` already stratifies for classifiers; the problem is an explicit shuffled `KFold`.)

```python
from trustcv import KFoldMedical

y_rare = np.r_[np.ones(10, int), np.zeros(190, int)]
X_rare = rng.normal(size=(200, 3))

unstratified = [int(y_rare[te].sum()) for _, te in KFoldMedical(5, shuffle=True, random_state=1).split(X_rare)]
stratified = [int(y_rare[te].sum()) for _, te in StratifiedKFoldMedical(5, shuffle=True, random_state=1).split(X_rare, y_rare)]
print(unstratified, stratified)   # uneven event counts vs 2 per fold
```

### Ignoring time order

If the model will predict later outcomes, a shuffled split lets it train on the future. Use a forward split and pass time as `timestamps=`. `gap` and `purge_gap` count samples, not days.

```python
from trustcv import TimeSeriesSplit

dates = np.datetime64("2023-01-01") + np.arange(200).astype("timedelta64[D]")
for train, test in TimeSeriesSplit(n_splits=4, gap=7).split(X_rare, timestamps=dates):
    assert dates[train].max() < dates[test].min()
```

### Tune inside the training folds

Choosing hyperparameters by cross-validation and then reporting that same cross-validation score is optimistic: the score was used to pick the model. Put the search inside each outer training fold (nested CV), so the outer test fold is only scored:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    {"logisticregression__C": [0.1, 1, 10]}, cv=3, scoring="roc_auc",
)
result = TrustCV(declare_independent_samples=True).validate(model=search, X=X, y=y)
print(result.mean_scores["roc_auc"], result.confidence_intervals["roc_auc"])
```

The outer folds score a search refit on each training fold. Because the model is a `GridSearchCV`, not a `Pipeline`, `preprocessing_leakage` is `NOT_CHECKED` and the overall status is <span class="tcv-status unverified">NOT_FULLY_VERIFIED</span>; record that the preprocessing sits inside the searched pipeline.

The inner `cv=3` above splits rows. With repeated patients, the inner splits must be grouped too. `NestedGroupedCV` groups both loops:

```python
from trustcv import NestedGroupedCV, GroupKFoldMedical

nested = NestedGroupedCV(
    outer_cv=GroupKFoldMedical(n_splits=5, random_state=0),
    inner_cv=GroupKFoldMedical(n_splits=3, random_state=0),
    n_jobs=1,  # inner grid search in this process; -1 (default) uses all cores
)
outer_scores, chosen = nested.fit_predict(
    LogisticRegression(max_iter=1000), X_rep, y_rep, groups=patients, param_grid={"C": [0.1, 1, 10]}
)
print(np.round(outer_scores, 2), chosen)
```

The parameters chosen in each outer fold are part of the result: if they vary widely, say so.

### Early stopping on the test fold

In deep learning, choosing the best epoch or checkpoint by the held-out fold's loss is model selection on the test data. Keep the held-out fold for scoring only, and early-stop on a split of the training fold. `KerasCVRunner.run(inner_validation=0.1)` does this, group-aware with `groups=`; trustcv's `EarlyStopping` and `ModelCheckpoint(save_best_only=True)` warn when they select on a `val_*` metric computed on the held-out fold. See [Deep learning](deep-learning.md).

## Medical data

### Sample size and events per fold

There is no fixed minimum number of samples per class. What matters is whether each test fold holds enough events to estimate the metrics you report. trustcv warns when a fold cannot support a metric (`TrustCV(warn_metric_feasibility=True)`, the default), and `check_fold_metric_feasibility` runs the same check on any folds:

```python
from trustcv.metrics.diagnostics import check_fold_metric_feasibility

folds = [te for _, te in StratifiedKFoldMedical(5, shuffle=True, random_state=1).split(X_rare, y_rare)]
report = check_fold_metric_feasibility(y_rare, folds, metric_names=["roc_auc", "sensitivity"])
print(report["has_warnings"], [f["class_counts"] for f in report["folds"]])  # False, 2 events per fold
```

Feasible is a low bar: two events per test fold make AUC computable, not stable. With few events, use fewer folds, repeat stratified k-fold with different seeds, and report the interval width alongside the estimate. Leave-one-out does not reduce variance.

### Multi-site data

Splitting by site tests transfer to a new hospital. Pass the hierarchy to `split()` and choose the level:

```python
from trustcv import HierarchicalGroupKFold

site = np.repeat(np.arange(6), 40)               # 6 sites of 10 patients (40 rows) each
cv = HierarchicalGroupKFold(n_splits=3, hierarchy_level="site", random_state=0)
for train, test in cv.split(X_rep, y_rep, hierarchy={"site": site, "patient": patients}):
    assert not set(site[train]) & set(site[test])
```

With only a few sites, `LeaveOneGroupOut` on site IDs reports each site separately, which is often more informative than a pooled mean.

### Longitudinal patient data

Repeated visits need both patient grouping and time order. `PurgedGroupTimeSeriesSplit(group_exclusive=True)` removes a test patient's earlier visits from training. `validate()` calls `split(X, y, groups)` without timestamps, so sort the rows by time first and pass the splitter via `cv=`:

```python
from trustcv import PurgedGroupTimeSeriesSplit

n_pat = 60
start = rng.integers(0, 720, n_pat)
pid = np.repeat(np.arange(n_pat), 4)
day = np.repeat(start, 4) + np.tile([0, 30, 60, 90], n_pat)
order = np.argsort(day, kind="stable")              # time order
pid, day = pid[order], day[order]
X_long = rng.normal(size=(len(day), 5))
y_long = rng.integers(0, 2, len(day))

cv = PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=8, group_exclusive=True)
result = TrustCV().validate(
    model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    X=X_long, y=y_long, groups=pid, cv=cv,
)
print(result.checks["group_leakage"].status)
```

### Rare outcomes and oversampling

Oversample only the training fold. With imbalanced-learn, put the sampler in an `imblearn.pipeline.Pipeline` so it runs inside each fold and never touches the test rows:

<!-- notest: needs imbalanced-learn, which is not a trustcv dependency -->
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

model = make_imb_pipeline(StandardScaler(), SMOTE(random_state=0), LogisticRegression(max_iter=1000))
result = TrustCV(method="stratified_kfold", declare_independent_samples=True).validate(model=model, X=X, y=y)
```

Oversampling before `validate()` puts synthetic neighbours of test rows into training. The independence scan may flag it, but it is not designed to catch every case.

## A complete example

Grouped data, preprocessing in the pipeline, stratified group folds, and the default `corrected_t` intervals:

```python
from sklearn.ensemble import RandomForestClassifier

pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=50, random_state=0))
validator = TrustCV(method="stratified_group_kfold", n_splits=5, random_state=42)
result = validator.validate(model=pipeline, X=X_rep, y=y_rep, groups=patients)

print(result.overall_status)
print(result.summary())
```

Record the method, `n_splits`, `random_state`, the group column, the overall status and every check that was not `PASSED`. [Reading results](../get-started/reading-results.md) explains each check, and [Confidence intervals](confidence-intervals.md) explains the interval methods.

## Documentation

trustcv helps document validation choices: the split, the checks, their statuses and the intervals. It can support a regulatory submission, but it does not make a model or device compliant with any regulation. [Documenting validation](../regulatory/documentation-guide.md) covers what to record.

## Checklist

- [ ] The split matches the deployment question (new patients, sites, periods or regions).
- [ ] Every patient's rows are on one side of each split, and the IDs were passed as `groups=`.
- [ ] Time order is respected where the model predicts later outcomes; gaps are stated in samples.
- [ ] All fitted preprocessing, feature selection and resampling sit inside the pipeline.
- [ ] Hyperparameters were tuned inside the training folds; no choice used the reported test folds.
- [ ] Early stopping and checkpoints used a validation split inside the training fold.
- [ ] Each test fold has enough events for the reported metrics.
- [ ] Confidence intervals are reported with their method (`corrected_t` or `oof_bootstrap`).
- [ ] The overall status and every non-`PASSED` check are reported and explained.
- [ ] Seeds, splitter settings and the trustcv version are recorded.
- [ ] Subgroup performance (for example by sex, age band or site) was examined.
