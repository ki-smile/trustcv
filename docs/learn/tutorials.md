---
description: Short tutorials with code that runs - data leakage from repeated patients, choosing a split, patient grouping, temporal, nested and spatial cross-validation.
---

<link rel="stylesheet" href="../assets/css/learn.css">

# Tutorials

Six short tutorials, from the most common mistake to the specialised cases. Every code block on this page runs as written with the current trustcv release and scikit-learn (each is executed on every documentation build), and finishes in a few seconds.

| Tutorial | Level | You will learn |
| --- | --- | --- |
| [Understanding data leakage](#understanding-data-leakage) | Beginner | How row-level folds flatter a model when patients repeat |
| [Choosing a split](#choosing-a-split) | Beginner | Which splitter matches your data |
| [Patient-level grouping](#patient-level-grouping) | Intermediate | What to pass as `groups=` |
| [Temporal cross-validation](#temporal-cross-validation) | Intermediate | Forward-only splits, gaps and purging |
| [Nested cross-validation](#nested-cross-validation) | Advanced | Tuning without touching the test folds |
| [Spatial cross-validation](#spatial-cross-validation) | Advanced | Blocks and buffers for geographic data |

## Understanding data leakage

Leakage is when the test folds tell the model something it would not know in practice. The most common case in medical data is repeated patients: if visits of one patient are on both sides of the split, the model can recognise the patient instead of learning the condition.

The example builds 100 patients with 5 visits each. One feature carries a weak real signal; eight others are a patient "fingerprint" that says nothing about the outcome. Row-level K-fold and patient-grouped K-fold then score the same model.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from trustcv import GroupKFoldMedical, KFoldMedical

rng = np.random.default_rng(0)
n_patients, n_visits = 100, 5
patient_ids = np.repeat(np.arange(n_patients), n_visits)
signal = rng.normal(size=n_patients)                       # weak real signal
y = np.repeat((signal + rng.normal(0, 1.5, n_patients) > 0).astype(int), n_visits)
fingerprint = rng.normal(size=(n_patients, 8))             # identifies the patient, not the outcome
X = np.column_stack([
    signal[patient_ids] + rng.normal(0, 0.5, len(y)),
    fingerprint[patient_ids] + rng.normal(0, 0.1, (len(y), 8)),
])

model = RandomForestClassifier(n_estimators=100, random_state=0)
row_level = cross_val_score(
    model, X, y, cv=KFoldMedical(n_splits=5, shuffle=True, random_state=0), scoring="roc_auc"
)
grouped = cross_val_score(
    model, X, y, cv=GroupKFoldMedical(n_splits=5, random_state=0), groups=patient_ids, scoring="roc_auc"
)
print(f"Row-level K-fold AUC: {row_level.mean():.2f}")   # close to 1: the model recognises patients
print(f"Patient-grouped AUC:  {grouped.mean():.2f}")     # about 0.7: what new patients would see
```

`scoring="roc_auc"` matters: `cross_val_score` reports accuracy by default, not AUC.

To put a gap like this in context: if row-level K-fold reports an AUC of 0.92 and the patient-grouped estimate is 0.73, the row-level figure is too high by 0.19 AUC, about 26% in relative terms. The model you would deploy is the 0.73 one. How large the gap is depends on the data; here the fingerprint features make it extreme.

trustcv catches this case when you give it the patient IDs: `TrustCV(...).validate(..., groups=patient_ids)` fails the group-leakage check if a patient is on both sides. Without IDs it looks for tightly clustered rows and reports `NEEDS_REVIEW`. See [Reading results](../get-started/reading-results.md).

<div data-tcv-quiz="tutorial" class="tcv-quiz" id="tutorial-quiz-1"></div>

## Choosing a split

Start from how the data were collected, not from the model.

| Your data | Use | Why |
| --- | --- | --- |
| Independent rows, one per patient | `StratifiedKFoldMedical` | Keeps the class balance in every fold |
| Several rows per patient or site | `StratifiedGroupKFold`, with `groups=` | No patient on both sides |
| Transfer to new sites | `LeaveOneGroupOut` with `groups=site` | Each site is the test set once |
| Time-ordered, predicting the future | `TimeSeriesSplit` or `PurgedGroupTimeSeriesSplit` | Never trains on later rows |
| Geographic or spatial | `SpatialBlockCV` or `BufferedSpatialCV` | Test areas are not surrounded by training neighbours |

`recommend_cv` applies these rules from what you tell it about the data:

```python
import numpy as np
from trustcv import recommend_cv

rng = np.random.default_rng(1)
patients = np.repeat(np.arange(40), 3)
X = rng.normal(size=(120, 4))
y = np.repeat(rng.integers(0, 2, 40), 3)

plan = recommend_cv(X, y, groups=patients)
print(plan.method)        # a grouped method, because groups were given
print(plan.splitter)
```

When in doubt, prefer the stricter split. An estimate that is a little pessimistic is safer than one that is optimistic. The [guide to choosing a split](../guides/choosing-cv.md) covers the harder cases.

## Patient-level grouping

Here is a small table with three patients and nine visits:

```python
import pandas as pd

data = pd.DataFrame({
    "patient_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
    "visit_date": ["2023-01", "2023-02", "2023-03", "2023-01", "2023-02",
                   "2023-01", "2023-02", "2023-03", "2023-04"],
    "blood_pressure": [120, 125, 130, 110, 115, 140, 145, 150, 155],
    "diagnosis": [0, 0, 1, 0, 0, 1, 1, 1, 1],
})
print(data.head())
```

Which column should you pass as `groups=`?

??? question "Show the answer"
    `patient_id`. All rows of a patient must be on the same side of every split. `visit_date` is a time order, not a group, and `diagnosis` is the label.

```python
from trustcv import GroupKFoldMedical

cv = GroupKFoldMedical(n_splits=3, random_state=0)
X = data[["blood_pressure"]].to_numpy()
for train, test in cv.split(X, data["diagnosis"], groups=data["patient_id"]):
    print("test patients:", sorted(set(data["patient_id"].iloc[test])))
```

The group is whatever makes rows dependent. It is usually the patient, but it can be the hospital, the scanner, the family or the sequencing batch. With more than one level, `HierarchicalGroupKFold(hierarchy_level="hospital")` holds out whole hospitals; pass the hierarchy to `split(hierarchy=...)`.

## Temporal cross-validation

Random K-fold on time-ordered data lets the model train on tomorrow to predict today. Forward-only splits avoid that: every training row comes before every test row.

```python
import numpy as np
from trustcv import PurgedKFoldCV, TimeSeriesSplit

X = np.zeros((24, 1))
for train, test in TimeSeriesSplit(n_splits=3, gap=2).split(X):
    print(f"train 0-{train.max()}, test {test.min()}-{test.max()}")   # always forward

for train, test in PurgedKFoldCV(n_splits=3, purge_gap=2).split(X):
    print(f"test {test.min()}-{test.max()}, rows after test in training: {(train > test.max()).sum()}")
```

`gap` and `purge_gap` count rows, not days. If you have one row per hour, `gap=24` leaves one day.

Purged K-fold does not make a split forward-only. It removes `purge_gap` rows on each side of the test block, which helps when labels overlap in time, but its training set still includes later blocks (the second loop above prints how many rows). For an estimate of how a model will do on future patients, use a forward-only splitter: `TimeSeriesSplit`, `ExpandingWindowCV`, `RollingWindowCV`, or `PurgedGroupTimeSeriesSplit` when patients repeat. The [split diagrams](visualisations.md) show the difference.

## Nested cross-validation

If you tune hyperparameters on the same folds you report, the reported score is optimistic: you picked the setting that happened to do best on those test rows. Nested cross-validation tunes inside each outer training set and scores on an outer test fold that tuning never saw.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from trustcv import GroupKFoldMedical, NestedGroupedCV

rng = np.random.default_rng(2)
groups = np.repeat(np.arange(40), 3)
X = rng.normal(size=(120, 5))
y = (X[:, 0] + rng.normal(0, 1, 120) > 0).astype(int)
param_grid = {"C": [0.1, 1.0, 10.0]}

# Option 1: by hand, keeping patients whole at both levels
outer_cv, inner_cv = GroupKFoldMedical(n_splits=4, random_state=0), GroupKFoldMedical(n_splits=3, random_state=0)
scores = []
for train, test in outer_cv.split(X, y, groups):
    inner = list(inner_cv.split(X[train], y[train], groups[train]))
    search = GridSearchCV(LogisticRegression(), param_grid, cv=inner)
    search.fit(X[train], y[train])
    scores.append(search.score(X[test], y[test]))
print(f"outer accuracy: {np.mean(scores):.2f}")

# Option 2: NestedGroupedCV does the same loop
nested = NestedGroupedCV(outer_cv=GroupKFoldMedical(n_splits=4, random_state=0),
                         inner_cv=GroupKFoldMedical(n_splits=3, random_state=0))
outer_scores, best_params = nested.fit_predict(LogisticRegression(), X, y, groups, param_grid)
print(f"outer accuracy: {np.mean(outer_scores):.2f}", best_params[0])
```

`NestedGroupedCV.fit_predict` scores accuracy. Feature selection and resampling are tuning too: keep them inside the inner loop, in a `Pipeline`.

## Spatial cross-validation

Nearby locations share exposures and outcomes, so a random split puts near-duplicates of each test point into training. Spatial blocks hold out whole areas; a buffer also drops training points close to the test block.

```python
import numpy as np
from trustcv import BufferedSpatialCV, SpatialBlockCV

rng = np.random.default_rng(3)
coords = rng.uniform(0, 100, size=(200, 2))        # x, y in kilometres
X = rng.normal(size=(200, 3))

for train, test in SpatialBlockCV(n_splits=4, random_state=0).split(X, coordinates=coords):
    print("block:", len(test), "test,", len(train), "train")

for train, test in BufferedSpatialCV(n_splits=4, buffer_size=10, random_state=0).split(X, coordinates=coords):
    print("buffered:", len(test), "test,", len(train), "train")   # fewer training points
```

Coordinates go to `split(coordinates=...)`. `buffer_size` is in the units of the coordinates (kilometres here); a value below 1 is read as a fraction of the largest distance.

## Test your knowledge

<div data-tcv-quiz="tutorial-final" class="tcv-quiz" id="tutorial-quiz-2"></div>

For twenty more questions, take the [quiz](quiz.md).

<script src="../assets/js/tutorials.js"></script>
