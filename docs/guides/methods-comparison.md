---
description: How trustcv's splitters and checks relate to scikit-learn and to the cross-validation support of other machine learning toolkits.
---

# Compared with other toolkits

trustcv builds on scikit-learn rather than replacing it. Its splitters follow the scikit-learn `split(X, y, groups)` interface, `TrustCV.validate()` accepts any scikit-learn estimator or `Pipeline`, and any scikit-learn splitter can be passed as `cv=`. What trustcv adds is splitters for data structures scikit-learn does not cover, checks on the folds that were scored, and corrected confidence intervals.

## Compared with scikit-learn

| | scikit-learn 1.7 | trustcv |
| --- | --- | --- |
| Splitters | 15 (K-fold, stratified, grouped, shuffle, leave-out, repeated, `TimeSeriesSplit`, `PredefinedSplit`) | 32 splitters: independent, grouped, temporal, spatial and multilabel |
| Patient grouping | `GroupKFold`, `StratifiedGroupKFold`, `LeaveOneGroupOut`, `LeavePGroupsOut`, `GroupShuffleSplit` | The same ideas, plus `RepeatedGroupKFold`, `HierarchicalGroupKFold` and `NestedGroupedCV` |
| Time order | `TimeSeriesSplit` with `gap` and `max_train_size` | Adds rolling and expanding windows, blocked splits, purging and embargo (`PurgedKFoldCV`, `CombinatorialPurgedCV`, `PurgedGroupTimeSeriesSplit`) |
| Spatial data | None | `SpatialBlockCV`, `BufferedSpatialCV`, `SpatiotemporalBlockCV`, `EnvironmentalHealthCV` |
| Leakage checks | None | Group overlap, duplicates, target proxies, identifier columns, clustered rows, preprocessing outside the `Pipeline` |
| Confidence intervals | None (fold scores only) | `corrected_t` (default) and `oof_bootstrap`; see [confidence intervals](confidence-intervals.md) |
| Choosing a split | Up to you | `recommend_cv` suggests one from the data structure you describe |

scikit-learn's grouped splitters are correct and widely used; if your data only needs patient grouping, `sklearn.model_selection.StratifiedGroupKFold` is a sound choice. trustcv's `StratifiedGroupKFold` does the same job: it keeps each group in one fold and balances the class distribution across folds as far as the groups allow.

The two work together. A trustcv splitter can drive `cross_val_score`, and `TrustCV` can check a scikit-learn splitter:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from trustcv import StratifiedGroupKFold, TrustCV

rng = np.random.default_rng(0)
patients = np.repeat(np.arange(60), 4)  # 60 patients, 4 rows each
X = rng.normal(size=(60, 5))[patients] + rng.normal(0, 0.3, (240, 5))
y = np.repeat(rng.integers(0, 2, 60), 4)

scores = cross_val_score(
    LogisticRegression(), X, y, groups=patients,
    cv=StratifiedGroupKFold(n_splits=5, random_state=0), scoring="roc_auc",
)
print(scores.round(3))

# A row-level split puts the same patient in training and test; trustcv reports it.
result = TrustCV(method="kfold").validate(
    model=LogisticRegression(), X=X, y=y, groups=patients,
    cv=KFold(n_splits=5, shuffle=True, random_state=0),
)
print(result.checks["group_leakage"].status)  # FAILED
```

## Compared with other toolkits

Most training libraries leave cross-validation to scikit-learn splitters or a hand-written loop. The table summarises built-in support as described in each project's documentation; check the version you use.

| Toolkit | Built-in cross-validation | Patient grouping |
| --- | --- | --- |
| XGBoost | `xgboost.cv`: K-fold, `stratified=`, custom `folds=` | Pass folds from a grouped splitter |
| LightGBM | `lightgbm.cv`: K-fold, stratified, custom `folds=` | Pass folds from a grouped splitter |
| CatBoost | `catboost.cv`: `fold_count=`, `type=` (Classical, Inverted, TimeSeries), `stratified=` | Pass folds from a grouped splitter (`folds=`) |
| H2O | `nfolds=` with `fold_assignment` (Auto, Random, Modulo, Stratified) | `fold_column=` |
| PyCaret | `fold_strategy=` (K-fold, stratified, group, time series, or a scikit-learn splitter) | `fold_groups=` |
| PyTorch, TensorFlow/Keras, JAX | No cross-validation module | Manual loop over splitter indices |
| MONAI | `CrossValidation` dataset helper and `partition_dataset` | Manual, by partitioning patients |

Any of these can take trustcv folds, because a splitter returns plain index arrays. For PyTorch, Keras, MONAI and JAX, trustcv's runners do the loop: see [deep learning](deep-learning.md).

## What trustcv does not do

- It does not train or tune models for you beyond the fold loop. Nested helpers (`NestedCV`, `NestedGroupedCV`, `NestedTemporalCV`) give a less biased estimate after tuning than tuning and scoring on the same folds; they are not unbiased.
- It cannot see leakage that happened before `validate()` receives `X`, such as feature selection on the full dataset.
- It can support documentation of validation choices; it does not make a model or device compliant with any regulation. See [documenting validation](../regulatory/documentation-guide.md).

## Next steps

- [Choosing a split](choosing-cv.md) for which splitter fits your data
- [Data leakage](data-leakage.md) for what the checks catch and miss
