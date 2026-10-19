---
description: How to choose a cross-validation split that matches how your data was collected, with repeated patients, time order, sites and geography.
---

# Choosing a split

A cross-validation estimate is only as good as its resemblance to deployment. Ask how the model will meet new data: new patients, a later time period, a new hospital, a new region. Then choose the split that keeps exactly that unit out of training.

The structure of the data decides the family. Sample size and class balance only refine the choice within it.

## Start from the data structure

| Your data | Unit that must not cross train and test | Family | Typical splitter |
| --- | --- | --- | --- |
| One row per patient, order irrelevant | Row | Independent | `StratifiedKFoldMedical` |
| Several rows per patient (visits, images, slides) | Patient | Grouped | `StratifiedGroupKFold` |
| Several sites, hospitals or scanners | Site | Grouped | `LeaveOneGroupOut` or `GroupKFoldMedical` on site IDs |
| Patients nested in hospitals | The level you want to generalise to | Grouped | `HierarchicalGroupKFold` |
| Time order matters (forecasting, drift) | Future period | Temporal | Forward-only: `TimeSeriesSplit`, `ExpandingWindowCV`, `RollingWindowCV` |
| Repeated patients over time | Patient and future period | Temporal | `PurgedGroupTimeSeriesSplit(group_exclusive=True)` |
| Locations or spatially correlated samples | Region | Spatial | `SpatialBlockCV`, `BufferedSpatialCV` |
| Several labels per row | Row (or patient) | Multilabel | `MultilabelStratifiedKFold`, `MultilabelStratifiedGroupKFold` |

If more than one row applies, the stricter constraint wins. Repeated patients in a time series need both grouping and time order.

## Let `recommend_cv` make the first choice

`recommend_cv` chooses a family from what you pass it. It cannot see structure you do not supply: without `groups=` it cannot know that rows share a patient.

```python
import numpy as np
from trustcv import recommend_cv

rng = np.random.default_rng(0)
patients = np.repeat(np.arange(50), 4)        # 50 patients, 4 rows each
X = rng.normal(size=(200, 5))
y = rng.integers(0, 2, 200)

plan = recommend_cv(X, y, groups=patients)
print(plan.category, plan.method)  # grouped stratified_grouped_kfold
print(plan.rationale)
print(plan.code)
```

The result has `category`, `method`, a configured `splitter`, a `rationale`, `warnings` and a `code` string you can paste into a methods section or script. With both `groups=` and `timestamps=` it returns `PurgedGroupTimeSeriesSplit(group_exclusive=True)`, and its warnings remind you to sort rows by time, because `validate()` has no `timestamps` argument.

## Independent rows

Use these only when every row is a different patient and nothing links rows (no shared site effect you care about, no time order). Declare it with `TrustCV(declare_independent_samples=True)`; trustcv still scans for rows that look clustered.

| Method | Bias of the estimate | Variance | Cost | Use when |
| --- | --- | --- | --- | --- |
| `HoldOut` | Pessimistic: the model is trained on only part of the data | High: one split | 1 fit | Very large datasets, or a final locked test set |
| `KFoldMedical`, `StratifiedKFoldMedical` (k = 5 or 10) | Slightly pessimistic; less so with more folds | Moderate | k fits | The default for independent rows |
| `RepeatedKFold` | As k-fold | Lower than one k-fold run: averages over partitions | k × repeats fits | Small datasets, or when one partition looks unstable |
| `LOOCV` | Close to unbiased | High | n fits | Very small datasets with cheap models; not a variance remedy |
| `LPOCV` | Close to unbiased | High | Combinatorial | Tiny datasets only |
| `MonteCarloCV` | Pessimistic, depending on `test_size` | Moderate | `n_iterations` fits | Flexible train/test ratio |
| `BootstrapValidation` | Through `validate()`: the plain out-of-bag mean (pessimistic). The `.632`/`.632+` corrections are not applied automatically | Low | `n_iterations` fits | Small datasets |
| `NestedCV` | Less biased when hyperparameters are tuned | Moderate | outer × inner × grid fits | Any tuning: see [Best practices](best-practices.md#tune-inside-the-training-folds) |

Stratify classification folds (`StratifiedKFoldMedical`, or `RepeatedKFold(stratify=True)`) so every fold has both classes in similar proportion. This matters most for rare outcomes.

**How many folds.** Five or ten folds is the usual choice. More folds train each model on more data, so the estimate is less pessimistic, at the cost of more fits; the variance of the estimate does not reliably fall as k grows. To reduce the variance that comes from one particular partition, repeat k-fold with different seeds rather than increasing k. With very few events, use fewer folds so each test fold still holds enough positives to compute the metrics you report.

**Bootstrap .632 is a point estimate, computed separately.** `validate()` reports the out-of-bag mean whatever `estimator` you choose. To get the `.632` or `.632+` value, collect training and out-of-bag errors yourself and call `BootstrapValidation.compute_error(train_errors, test_errors)`. The result is a corrected point estimate, not a confidence interval. For intervals, use trustcv's `ci_method` (default `corrected_t`); intervals built naively from overlapping folds or resamples are too narrow. See [Confidence intervals](confidence-intervals.md).

## Grouped data

When a patient contributes several rows, those rows are more alike than rows from different patients. A row-level split puts a patient on both sides and the model is scored partly on people it has seen.

| Method | Splits on | Use when |
| --- | --- | --- |
| `GroupKFoldMedical` | Groups | Many groups, classes roughly balanced |
| `StratifiedGroupKFold` | Groups, balancing class proportions per fold | Many groups and a classification target; the usual default |
| `LeaveOneGroupOut` | One group per test fold | Few groups (sites, scanners), each tested in turn |
| `LeavePGroupsOut` | p groups per test fold | Few groups, larger test folds |
| `RepeatedGroupKFold` | Groups, repeated with new seeds | More stable estimates with grouped data |
| `HierarchicalGroupKFold` | A chosen level of a hierarchy | Patients within hospitals within countries |
| `NestedGroupedCV` | Groups, in both outer and inner loops | Hyperparameter tuning with grouped data |
| `MultilabelStratifiedGroupKFold` | Groups, balancing several labels | Multilabel targets with repeated patients |

Choose the group by the question. To claim generalisation to new patients, group by patient. To claim generalisation to new hospitals or scanners, group by hospital or scanner; stratifying by site keeps every site in training and does not test transfer.

## Temporal data

When the model will predict the future, test on periods after the training data.

| Method | What it does | Units |
| --- | --- | --- |
| `TimeSeriesSplit` | Expanding training window, test on the next block | `gap` is a number of samples |
| `RollingWindowCV` | Fixed-size training window that slides forward | `window_size`, `gap` in samples |
| `ExpandingWindowCV` | Training window grows from `initial_train_size` | `gap` in samples |
| `BlockedTimeSeries` | Contiguous blocks by `block_size` | An integer is a number of samples; `"day"` or `"week"` uses `timestamps=` |
| `PurgedKFoldCV` | K-fold over time with purged samples around each test fold and an embargo after it. **Not forward-only**: it also trains on data after the test fold, so do not use it to estimate forecasting performance | `purge_gap` in samples; `embargo_size` a fraction |
| `CombinatorialPurgedCV` | Several test blocks per split, purged | `purge_gap` in samples |
| `PurgedGroupTimeSeriesSplit` | Forward splits with patient exclusion (`group_exclusive=True`) | `purge_gap` in samples |
| `NestedTemporalCV` | Nested CV with temporal outer and inner splitters | As its splitters |

`purge_gap` and `gap` count rows in time order, not days. With one row per day they coincide; with irregular visits, convert the gap you need in time into a number of rows. Pass time with `timestamps=` to `split()`, not as `groups=`.

## Spatial data

Nearby locations share environment, tissue or population. Random splits put neighbours on both sides.

| Method | What it does | Units |
| --- | --- | --- |
| `SpatialBlockCV` | Grid or k-means blocks of coordinates (`block_shape`) | `n_splits` blocks |
| `BufferedSpatialCV` | Spatial blocks, plus a buffer around each test block removed from training | `buffer_size` in coordinate units; a value below 1.0 is read as a fraction of the largest distance |
| `SpatiotemporalBlockCV` | Blocks in space and time | `buffer_space` in coordinate units, `buffer_time` in temporal blocks |
| `EnvironmentalHealthCV` | Spatial blocks with seasonal or other temporal strategies | As configured |

Coordinates in degrees give a buffer in degrees. Project to metres first if you want a buffer in metres.

## Common scenarios

The examples use small synthetic data so they run quickly. Replace the arrays with your own.

### Multi-site study

Goal: performance at a hospital the model has not seen. With few sites, test each in turn.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustcv import TrustCV, LeaveOneGroupOut

rng = np.random.default_rng(1)
site = np.repeat(["A", "B", "C", "D"], 50)
X = rng.normal(size=(200, 5))
y = rng.integers(0, 2, 200)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

result = TrustCV().validate(model=model, X=X, y=y, groups=site, cv=LeaveOneGroupOut())
print(result.checks["group_leakage"].status)  # PASSED: no site on both sides
```

If patients are also repeated within sites, grouping by site already keeps each patient on one side, provided no patient attends two sites.

### ICU monitoring

Goal: predict deterioration for new patients. Each patient contributes many hourly rows, so a plain `TimeSeriesSplit` on the pooled rows would put the same patient's earlier and later hours on both sides. If the claim is about new patients at a later time, exclude test patients from training as well:

```python
from trustcv import PurgedGroupTimeSeriesSplit

n_pat = 40
start = rng.integers(0, 500, n_pat)                     # admission hour
patient = np.repeat(np.arange(n_pat), 6)
hour = np.repeat(start, 6) + np.tile(np.arange(6), n_pat)
order = np.argsort(hour, kind="stable")                 # sort rows by time
patient, hour = patient[order], hour[order]
X_icu = rng.normal(size=(len(hour), 8))
y_icu = rng.integers(0, 2, len(hour))

cv = PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=6, group_exclusive=True)
for train, test in cv.split(X_icu, y_icu, groups=patient, timestamps=hour):
    assert not set(patient[train]) & set(patient[test])
```

If time drift is not part of the claim, `StratifiedGroupKFold` on patient IDs is simpler. Either way the patient IDs must be passed: rows of one ICU patient differ too much to be flagged by the independence scan.

### Diagnosis from images, one image per patient

Rows are independent, so stratified k-fold is appropriate. If some patients have several images, this becomes the grouped case.

```python
from sklearn.datasets import load_breast_cancer
from trustcv import StratifiedKFoldMedical

X_bc, y_bc = load_breast_cancer(return_X_y=True)
result = TrustCV(declare_independent_samples=True).validate(
    model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    X=X_bc, y=y_bc, cv=StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=0),
)
print(result.overall_status)
```

### Longitudinal study

Goal: validate on new patients at later visits. Sort rows by visit date, pass `timestamps=`, and set `group_exclusive=True` so a test patient's earlier visits are removed from training. `purge_gap=8` drops the 8 rows immediately before each test block.

```python
from trustcv import PurgedGroupTimeSeriesSplit

n_pat = 60
start = rng.integers(0, 720, n_pat)                     # enrolment day
pid = np.repeat(np.arange(n_pat), 4)
day = np.repeat(start, 4) + np.tile([0, 30, 60, 90], n_pat)
order = np.argsort(day, kind="stable")
pid, day = pid[order], day[order]
visit_dates = np.datetime64("2022-01-01") + day.astype("timedelta64[D]")
X_long = rng.normal(size=(len(day), 5))
y_long = rng.integers(0, 2, len(day))

cv = PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=8, group_exclusive=True)
for train, test in cv.split(X_long, y_long, groups=pid, timestamps=visit_dates):
    assert visit_dates[train].max() < visit_dates[test].min()

# validate() calls split(X, y, groups) without timestamps, so rows must already be in time order
result = TrustCV().validate(model=model, X=X_long, y=y_long, groups=pid, cv=cv)
print(result.checks["group_leakage"].message)
```

### Geographic spread

Goal: predict in regions with no training data. Pass coordinates to `split()`. Here the coordinates are in kilometres, so `buffer_size=10` removes training points within 10 km of the test block.

```python
from scipy.spatial.distance import cdist
from trustcv import BufferedSpatialCV

coords_km = rng.uniform(0, 100, size=(150, 2))
X_geo = rng.normal(size=(150, 4))

cv = BufferedSpatialCV(n_splits=4, buffer_size=10, random_state=0)
for train, test in cv.split(X_geo, coordinates=coords_km):
    print(round(cdist(coords_km[train], coords_km[test]).min(), 1))  # always above 10
```

`BufferedSpatialCV` takes coordinates only in `split()`, so iterate it directly as above. `SpatialBlockCV(coordinates=...)` stores them in the constructor and can be passed to `validate(cv=...)`.

## Record the choice

Reviewers will ask why this split. Write down:

- the deployment question (new patients, new sites, a later period, new regions);
- the unit kept out of training, and the column that identifies it;
- the splitter, its parameters with units (`purge_gap` in samples, `buffer_size` in coordinate units) and the random seed;
- methods considered and rejected, and why;
- `result.overall_status` and any check that was not `PASSED`.

`plan.rationale` and `plan.code` from `recommend_cv` are a starting point. [Documenting validation](../regulatory/documentation-guide.md) shows how to turn this into a report section. The [API reference](../reference/api.md) lists every splitter's parameters.
