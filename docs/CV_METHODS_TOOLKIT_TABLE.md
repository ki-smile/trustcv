# TrustCV Cross-Validation Methods

This file lists every primary cross-validation method currently exposed by the toolkit.

Scope:
- `29` core methods verified in `tests/test_all_29_methods.py`
- `3` multilabel extensions exported from `trustcv.splitters`
- `TrustCV(method=...)` shortcuts taken from `trustcv/validators.py`

Notes:
- Alias names such as `KFold`, `GroupKFold`, `TemporalClinical`, `BlockedTimeSeriesSplit`, and similar convenience exports are not listed separately as distinct methods.
- `TrustCV.validate()` only forwards `X`, `y`, and optional `groups` to the splitter. If a method needs extra arrays such as `timestamps`, `coordinates`, `environmental_data`, or a full `hierarchy` dictionary, use `UniversalCVRunner` or call the splitter directly.
- Nested helpers such as `NestedCV`, `NestedGroupedCV`, and `NestedTemporalCV` are best used through their `fit_predict(...)` helpers.

## Quick Patterns

Basic shortcut:

```python
from trustcv import TrustCV

res = TrustCV(method="stratified_kfold", n_splits=5).validate(
    model=model,
    X=X,
    y=y,
)
```

Direct splitter with `TrustCV.validate(cv=...)`:

```python
from trustcv import TrustCV
from trustcv.splitters import RepeatedGroupKFold

validator = TrustCV(method="group_kfold", n_splits=5)
cv = RepeatedGroupKFold(n_splits=5, n_repeats=2, random_state=42)

res = validator.validate(
    model=model,
    X=X,
    y=y,
    groups=patient_ids,
    cv=cv,
)
```

Advanced splitter with extra inputs:

```python
from trustcv.core.runner import UniversalCVRunner
from trustcv.splitters import SpatiotemporalBlockCV

runner = UniversalCVRunner(
    cv_splitter=SpatiotemporalBlockCV(n_spatial_blocks=3, n_temporal_blocks=3),
    verbose=1,
)

res = runner.run(
    model=model,
    data=(X, y),
    coordinates=coords,
    timestamps=timestamps,
)
```

Nested helper:

```python
from trustcv.splitters import NestedCV, KFoldMedical

cv = NestedCV(
    outer_cv=KFoldMedical(n_splits=5),
    inner_cv=KFoldMedical(n_splits=3),
)

scores, best_params = cv.fit_predict(estimator, X, y, param_grid)
```

## I.I.D. Methods

| Category | Method class | `TrustCV(method=...)` shortcut | Recommended toolkit usage | Extra data | Typical use |
| --- | --- | --- | --- | --- | --- |
| IID | `HoldOut` | `holdout` | `TrustCV(method="holdout", test_size=0.2)` | none | Single train/test split |
| IID | `KFoldMedical` | `kfold` | `TrustCV(method="kfold", n_splits=5)` | none | Standard k-fold |
| IID | `StratifiedKFoldMedical` | `stratified_kfold` | `TrustCV(method="stratified_kfold", n_splits=5)` | `y` | Preserve class balance |
| IID | `RepeatedKFold` | `repeated_kfold` | `TrustCV(method="repeated_kfold", n_splits=5, n_repeats=2)` | `y` if stratified repeats | Reduce variance with repeated folds |
| IID | `LOOCV` | `loocv` | `TrustCV(method="loocv")` | none | Very small datasets |
| IID | `LPOCV` | `lpocv` | `TrustCV(method="lpocv", p=2)` | none | Exhaustive leave-`p`-out |
| IID | `BootstrapValidation` | `bootstrap` | `TrustCV(method="bootstrap", bootstrap_iterations=200)` | none | Bootstrap/OOB estimation |
| IID | `MonteCarloCV` | `monte_carlo` | `TrustCV(method="monte_carlo", n_iterations=50, test_size=0.2)` | none | Random subsampling |
| IID | `NestedCV` | none | `NestedCV(...).fit_predict(estimator, X, y, param_grid)` | `param_grid` | Nested tuning + unbiased evaluation |

## Temporal Methods

| Category | Method class | `TrustCV(method=...)` shortcut | Recommended toolkit usage | Extra data | Typical use |
| --- | --- | --- | --- | --- | --- |
| Temporal | `TimeSeriesSplit` | `temporal` | `TrustCV(method="temporal", n_splits=5)` or `TrustCV(...).validate(cv=TimeSeriesSplit(...))` | optional `timestamps` when using runner/direct splitter | Forward-chaining time splits |
| Temporal | `RollingWindowCV` | none | `TrustCV(...).validate(cv=RollingWindowCV(window_size=50, step_size=10))` | none | Fixed-length train window |
| Temporal | `ExpandingWindowCV` | none | `TrustCV(...).validate(cv=ExpandingWindowCV(initial_train_size=30, step_size=10))` | none | Growing train window |
| Temporal | `BlockedTimeSeries` | none | `UniversalCVRunner(cv_splitter=BlockedTimeSeries(...)).run(..., timestamps=timestamps)` | `timestamps` recommended | Seasonal or blocked time splits |
| Temporal | `PurgedKFoldCV` | none | `UniversalCVRunner(cv_splitter=PurgedKFoldCV(...)).run(..., timestamps=timestamps)` | `timestamps` recommended | Purge/embargo leakage control |
| Temporal | `CombinatorialPurgedCV` | none | `TrustCV(...).validate(cv=CombinatorialPurgedCV(...))` | none | Multiple purged test combinations |
| Temporal | `PurgedGroupTimeSeriesSplit` | none | `UniversalCVRunner(cv_splitter=PurgedGroupTimeSeriesSplit(...)).run(..., groups=groups, timestamps=timestamps)` | `groups`, `timestamps` | Group-aware temporal splitting |
| Temporal | `NestedTemporalCV` | none | `NestedTemporalCV(...).fit_predict(estimator, X, y, param_grid, timestamps=timestamps)` | `param_grid`, optional `timestamps` | Nested time-series tuning |

## Grouped Methods

| Category | Method class | `TrustCV(method=...)` shortcut | Recommended toolkit usage | Extra data | Typical use |
| --- | --- | --- | --- | --- | --- |
| Grouped | `GroupKFoldMedical` | `group_kfold` or `patient_grouped_kfold` | `TrustCV(method="group_kfold", n_splits=5)` | `groups` | Keep each patient/group in one fold |
| Grouped | `StratifiedGroupKFold` | `stratified_group_kfold` or `stratified_grouped_kfold` | `TrustCV(method="stratified_group_kfold", n_splits=5)` | `groups`, `y` | Grouped + class-balanced folds |
| Grouped | `LeaveOneGroupOut` | none | `TrustCV(...).validate(cv=LeaveOneGroupOut(), groups=groups)` | `groups` | One group held out at a time |
| Grouped | `LeavePGroupsOut` | none | `TrustCV(...).validate(cv=LeavePGroupsOut(n_groups=2), groups=groups)` | `groups` | Exhaustive grouped hold-out |
| Grouped | `RepeatedGroupKFold` | `repeated_group_kfold` | `TrustCV(method="repeated_group_kfold", n_splits=5, n_repeats=2)` | `groups` | Repeated patient/group CV |
| Grouped | `HierarchicalGroupKFold` | `hierarchical_group_kfold` | `TrustCV(method="hierarchical_group_kfold", hierarchy_level="patient")` for flat `groups`; use direct splitter for full `hierarchy` | flat `groups` or `hierarchy` dict | Nested site/department/patient structures |
| Grouped | `MultilevelCV` | none | `cv = MultilevelCV(...); list(cv.split(X, y, groups=hierarchy_dict))` | `groups` as hierarchy dict | Validate at a chosen hierarchy level |
| Grouped | `NestedGroupedCV` | none | `NestedGroupedCV(...).fit_predict(estimator, X, y, groups, param_grid)` | `groups`, `param_grid` | Nested grouped tuning |

## Spatial Methods

| Category | Method class | `TrustCV(method=...)` shortcut | Recommended toolkit usage | Extra data | Typical use |
| --- | --- | --- | --- | --- | --- |
| Spatial | `SpatialBlockCV` | none | `UniversalCVRunner(cv_splitter=SpatialBlockCV(n_splits=5)).run(..., coordinates=coords)` | `coordinates` | Spatial autocorrelation control |
| Spatial | `BufferedSpatialCV` | none | `UniversalCVRunner(cv_splitter=BufferedSpatialCV(...)).run(..., coordinates=coords)` | `coordinates` | Add buffer around test region |
| Spatial | `SpatiotemporalBlockCV` | none | `UniversalCVRunner(cv_splitter=SpatiotemporalBlockCV(...)).run(..., coordinates=coords, timestamps=timestamps)` | `coordinates`, `timestamps` | Joint space-time blocking |
| Spatial | `EnvironmentalHealthCV` | none | `UniversalCVRunner(cv_splitter=EnvironmentalHealthCV(...)).run(..., coordinates=coords, timestamps=timestamps, environmental_data=env_data)` | `coordinates`, `timestamps`, `environmental_data` | Environmental health studies |

## Multilabel Extensions

| Category | Method class | `TrustCV(method=...)` shortcut | Recommended toolkit usage | Extra data | Typical use |
| --- | --- | --- | --- | --- | --- |
| Multilabel | `MultilabelStratifiedKFold` | `multilabel_stratified_kfold` | `TrustCV(method="multilabel_stratified_kfold", n_splits=5)` | multilabel `y` | Preserve multilabel prevalence |
| Multilabel | `MultilabelStratifiedGroupKFold` | `multilabel_stratified_group_kfold` | `TrustCV(method="multilabel_stratified_group_kfold", n_splits=5)` or direct splitter | multilabel `y`, `groups` | Group-safe multilabel balancing |
| Multilabel | `MultiLabelGroupSplitter` | `multilabel_group_kfold` | `TrustCV(method="multilabel_group_kfold", n_splits=5)` or direct splitter | multilabel `y`, `groups` | Iterative group-level multilabel split |

## Alias Summary

Common aliases accepted by the toolkit:

| Primary method | Common aliases |
| --- | --- |
| `KFoldMedical` | `KFold` |
| `StratifiedKFoldMedical` | `StratifiedKFold` |
| `GroupKFoldMedical` | `GroupKFold`, `PatientGroupKFold`, `group_kfold`, `patient_grouped_kfold` |
| `StratifiedGroupKFold` | `stratified_group_kfold`, `stratified_grouped_kfold` |
| `TimeSeriesSplit` | `TemporalClinical`, `temporal` |
| `RepeatedGroupKFold` | `repeated_group_kfold`, `RepeatedGroupKFold` |
| `HierarchicalGroupKFold` | `hierarchical_group_kfold`, `HierarchicalGroupKFold` |
| `MultilabelStratifiedKFold` | `multilabel_stratified_kfold` |
| `MultilabelStratifiedGroupKFold` | `multilabel_stratified_group_kfold` |
| `MultiLabelGroupSplitter` | `multilabel_group_kfold`, `MultiLabelGroupKFold` |

## Recommended Rule of Thumb

Use:
- `TrustCV(method="...")` for the shortcut-supported IID, grouped, and basic temporal methods.
- `TrustCV(...).validate(cv=...)` for splitters that only need `X`, `y`, and optional `groups`.
- `UniversalCVRunner(cv_splitter=...).run(...)` when the splitter also needs `timestamps`, `coordinates`, or other extra arrays.
- `*.fit_predict(...)` for nested CV helpers.
