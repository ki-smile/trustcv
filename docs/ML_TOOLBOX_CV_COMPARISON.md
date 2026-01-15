
# ML/DL Toolbox Cross-Validation Support Comparison

## Top 20 Machine Learning & Deep Learning Training Toolboxes

A comparative overview of how common ML/DL frameworks handle cross-validation (CV), with a focus on medical and clinical use cases (grouped patients, temporal validation, class imbalance).

---

## 📈 CV Method Count Comparison

| Toolbox | Built-in CV Splitters | Notes |
|---------|----------------------|-------|
| **TrustCV** | **29** | IID (9), Grouped (8), Temporal (8), Spatial (4) |
| **Scikit-learn** | **15** | KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut, ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit, RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit, PredefinedSplit |
| **CatBoost** | **3 types** | Classical, Inverted, TimeSeries + stratified option |
| **PyCaret** | **5 strategies** | kfold, stratifiedkfold, groupkfold, timeseries, sliding + custom |
| **H2O.ai** | **4 assignments** | Auto, Random, Modulo, Stratified + fold_column |
| **XGBoost** | **2** | KFold, StratifiedKFold via `xgb.cv` + custom folds |
| **LightGBM** | **2** | KFold, StratifiedKFold via `lgb.cv` + custom folds |
| **TensorFlow/Keras** | **0** | No CV module; manual loops with sklearn |
| **PyTorch** | **0** | No CV module; use sklearn splitters |
| **MONAI** | **1** | CrossValidation dataset + partition_dataset utility |
| **Fast.ai** | **0** | IndexSplitter only; K-fold requires sklearn |

### What TrustCV Provides That Scikit-learn Lacks

| Category | TrustCV Methods | Scikit-learn Equivalent |
|----------|-----------------|------------------------|
| **Spatial CV** | SpatialBlockCV, BufferedSpatialCV, SpatiotemporalBlockCV, EnvironmentalHealthCV | ❌ None |
| **Purged Temporal** | PurgedKFoldCV, CombinatorialPurgedCV, EmbargoCV | ❌ None |
| **Bootstrap/Monte Carlo** | BootstrapValidation, MonteCarloCV | ❌ None (only ShuffleSplit) |
| **Hierarchical** | HierarchicalCV, NestedGroupedCV | ❌ None |
| **Medical-specific** | Rolling/Expanding Window CV, BlockedTimeSeries | ❌ Basic TimeSeriesSplit only |
| **Leakage Detection** | 6 types (patient, temporal, spatial, preprocessing, duplicate, feature-target) | ❌ None |
| **Regulatory Compliance** | FDA/CE MDR reporting | ❌ None |

---

## 📊 Summary Table

**Interpretation notes**

* **Default CV** = behaviour when you use the “standard” helper (e.g. `cv=None` in scikit-learn, or leaving resampling at defaults).
* **Built-in CV support** = what the library actually provides, not what is usually hand-implemented around it.
* “Medical CV support” is qualitative: can it natively express *patient-level grouping*, *time-respecting splits*, etc., without a lot of custom plumbing?

| Toolbox               | CV Methods | Default CV (typical)                                                  | Built-in CV support                                                                                       | Medical CV support             | Rating |
| --------------------- | ---------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------ | ------ |
| **TrustCV (ours)**    | **29**     | Configurable                                                          | IID (9), Grouped (8), Temporal (8), Spatial (4) + leakage detection                                       | ⭐⭐⭐ Excellent                   | 🥇     |
| **Scikit-Learn**      | **15**     | 5-fold in `GridSearchCV`, `cross_val_score`, etc. ([Scikit-learn][1]) | KFold, Stratified, Group, TimeSeries, Leave-One-Out, Repeated, PredefinedSplit                            | ⭐⭐ Good (groups + basic time)  | 🥈     |
| **PyCaret**           | **5**      | 10-fold (`fold=10` in `setup`) ([AutoML][6])                          | KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit, sliding + custom                                     | ⭐⭐⭐ Strong (simple group/time) | 🥉     |
| **H2O.ai**            | **4**      | No CV by default (`nfolds=0`) ([H2O Release][5])                      | Auto, Random, Modulo, Stratified via `fold_assignment` + `fold_column`                                    | ⭐⭐ Good (fold_column)          | ⭐⭐⭐    |
| **CatBoost**          | **3**      | 3-fold in `catboost.cv` (`fold_count=3`) ([CatBoost][3])              | Classical, Inverted, TimeSeries types + stratified + GroupId                                              | ⭐⭐ Moderate (native groups)    | ⭐⭐     |
| **XGBoost**           | **2**      | 3-fold in `xgboost.cv` (`nfold=3`)                                    | K-fold, stratified via `stratified=True`, custom folds via `folds`                                        | ⭐⭐ Moderate (via custom folds) | ⭐⭐     |
| **LightGBM**          | **2**      | 5-fold in `lightgbm.cv` (`nfold=5`) ([LightGBM Documentation][2])     | K-fold, stratified, custom folds via `folds`, early stopping                                              | ⭐⭐ Moderate (via custom folds) | ⭐⭐     |
| **MONAI**             | **1**      | None (manual)                                                         | CrossValidation dataset + partition_dataset utility                                                       | ⭐⭐ Moderate (medical imaging)  | ⭐⭐     |
| **TensorFlow/Keras**  | **0**      | None (single train/val split; CV is manual)                           | No dedicated CV module; manual loops or sklearn wrappers                                                  | ⭐ Basic (manual only)          | ⭐⭐     |
| **PyTorch**           | **0**      | None (manual)                                                         | No CV module; use `DataLoader` + external splitters                                                       | ⭐ Basic (manual only)          | ⭐⭐     |
| **Fast.ai**           | **0**      | Single train/valid split ([Walk with fastai][4])                      | IndexSplitter only; full K-fold CV requires sklearn                                                       | ⭐ Basic (manual K-fold)        | ⭐⭐     |
| **MLflow**            | **0**      | None                                                                  | No CV engine; tracks any CV you run                                                                       | ⭐ Basic (logging only)         | ⭐⭐     |
| **Optuna**            | **0**      | User-defined                                                          | Integrates with any CV loop you define                                                                    | ⭐⭐ Moderate (via callbacks)    | ⭐⭐⭐    |
| **Weights & Biases**  | **0**      | None                                                                  | Logging/monitoring; no CV engine                                                                          | ⭐ Basic (logging only)         | ⭐⭐     |
| **Auto-sklearn**      | **0**      | Holdout 67/33 (`resampling_strategy="holdout"`) ([AutoML][6])         | Optional K-fold CV (uses sklearn internally)                                                              | ⭐⭐ Moderate (custom splitter)  | ⭐⭐     |
| **TPOT**              | **0**      | 5-fold (default `cv=5`)                                               | Uses sklearn CV internally                                                                                | ⭐ Basic (via sklearn)          | ⭐⭐     |
| **scikit-multilearn** | **0**      | 3-fold (typical)                                                      | Wraps sklearn; supports multi-label CV                                                                    | ⭐ Basic                        | ⭐⭐     |
| **Dask-ML**           | **0**      | Often 3-fold in `GridSearchCV` ([ml.dask.org][7])                     | Distributed versions of sklearn CV/GS                                                                     | ⭐ Basic                        | ⭐⭐     |
| **Spark MLlib**       | **1**      | 3-fold in `CrossValidator` (default)                                  | K-fold CV only                                                                                            | ⭐ Basic                        | ⭐⭐     |
| **Vowpal Wabbit**     | **0**      | None                                                                  | Online learning; CV is manual                                                                             | ⭐ Basic                        | ⭐⭐     |
| **JAX/Flax**          | **0**      | None                                                                  | No CV module; rely on sklearn ecosystem                                                                   | ⭐ Basic                        | ⭐⭐     |
| **Hugging Face**      | **0**      | None                                                                  | Datasets + Trainer use single splits; K-fold is manual                                                    | ⭐ Basic                        | ⭐⭐     |

**Rating key:**
🥇 Best for medical CV; 🥈 Very good; 🥉 Good
⭐⭐⭐ Strong medical support; ⭐⭐ Moderate; ⭐ Basic

---

## 🔍 Detailed Notes on Key Toolboxes

### 1. Scikit-Learn 🥇

* **Default CV**

  * `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `cross_validate` use 5-fold CV when `cv=None`. ([Scikit-learn][1])
* **Built-in splitters (15 total)**

  * **Basic**: `KFold`, `StratifiedKFold`, `ShuffleSplit`, `StratifiedShuffleSplit`
  * **Group-aware**: `GroupKFold`, `StratifiedGroupKFold`, `GroupShuffleSplit`
  * **Leave-out**: `LeaveOneOut`, `LeavePOut`, `LeaveOneGroupOut`, `LeavePGroupsOut`
  * **Repeated**: `RepeatedKFold`, `RepeatedStratifiedKFold`
  * **Temporal**: `TimeSeriesSplit`
  * **Other**: `PredefinedSplit`
* **What sklearn lacks**

  * ❌ Spatial CV methods (geographic/imaging data)
  * ❌ Purged/embargo temporal methods (financial/ICU monitoring)
  * ❌ Bootstrap validation or Monte Carlo CV as splitter classes
  * ❌ Built-in data leakage detection
  * ❌ Regulatory compliance features (FDA/CE MDR)
  * ❌ Hierarchical/multilevel CV for nested patient structures
* **Medical orientation**

  * **Patient-level grouping**: `GroupKFold` / `GroupShuffleSplit` with `patient_id`.
  * **Temporal**: basic `TimeSeriesSplit`, but no native purging/embargo.
  * **Metrics**: full sklearn metrics; medical metrics (e.g. sensitivity/specificity, PR-AUC) easily composed.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

### 2. TensorFlow / Keras 🥈

* **Default CV**: none. You typically do one train/validation split.
* **CV strategy**:

  * Manual loops with `KFold`/`StratifiedKFold` from sklearn.
  * Or wrap `tf.keras.wrappers.scikit_learn.KerasClassifier` and use sklearn’s CV.

```python
from sklearn.model_selection import KFold
from tensorflow.keras.models import clone_model

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(X):
    model = clone_model(base_model)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    model.fit(X[train_idx], y[train_idx], epochs=50, verbose=0)
    score = model.evaluate(X[val_idx], y[val_idx], verbose=0)[1]  # AUC
    scores.append(score)

print(f"Keras AUC: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

### 3. PyTorch 🥈

* **Default CV**: none.
* **Typical pattern**:

  * Use sklearn splitters to generate indices.
  * Wrap subsets using `torch.utils.data.Subset` or `SubsetRandomSampler`.

```python
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(range(len(dataset))):
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), batch_size=64)
    val_loader   = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx),   batch_size=64)
    # your training loop here
```

---

### 4. XGBoost 🥉

* **Built-in CV**: `xgboost.cv` with `nfold=3` by default. 
* **Capabilities**:

  * K-fold and stratified CV (for classification) via `nfold` + `stratified=True`.
  * Custom folds via `folds` argument (list of `(train_idx, test_idx)` pairs).
* **Medical usage**:

  * Patient-group CV via sklearn `GroupKFold` and passing `folds`.

```python
import xgboost as xgb
from sklearn.model_selection import GroupKFold

dtrain = xgb.DMatrix(X, label=y)

# simple stratified CV (classification)
results = xgb.cv(
    params=params,
    dtrain=dtrain,
    nfold=5,
    stratified=True,
    seed=42
)

# patient-level CV
gkf = GroupKFold(n_splits=5)
custom_folds = list(gkf.split(X, y, groups=patient_ids))

results_med = xgb.cv(params, dtrain, folds=custom_folds)
```

---

### 5. LightGBM 🥉

* **Built-in CV**: `lightgbm.cv` with `nfold=5` by default. ([LightGBM Documentation][2])
* **Capabilities**:

  * K-fold & stratified CV (`nfold`, `stratified=True`).
  * Custom folds via `folds` (sklearn splitter or list of index tuples).
  * Early stopping, custom metrics.
* **Medical usage**:

  * Use `GroupKFold` (sklearn) and pass to `folds`.

```python
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

train_data = lgb.Dataset(X, label=y)
gkf = GroupKFold(n_splits=5)
custom_folds = list(gkf.split(X, y, groups=patient_ids))

results = lgb.cv(
    params=params,
    train_set=train_data,
    folds=custom_folds,
    seed=42
)
```

---

### 6. CatBoost ⭐⭐

* **Default CV**: `catboost.cv` uses `fold_count=3` unless overridden. ([CatBoost][3])
* **Capabilities**:

  * K-fold and stratified K-fold.
  * Group-aware CV: if your dataset contains a `GroupId` column, all rows from the same group go to the same fold in CV. ([CatBoost][9])
  * No separate “TimeSeriesSplit” object; time-respecting CV is done by disabling shuffling and ordering data yourself.
* **Medical usage**:

  * Use `GroupId=patient_id` to ensure patient-level CV.

```python
from catboost import Pool, cv

pool = Pool(
    X,
    y,
    cat_features=categorical_features,
    group_id=patient_ids   # ensures grouped CV
)

cv_results = cv(
    pool=pool,
    params=params,
    fold_count=5,
    stratified=True,
    shuffle=False,  # important for temporal data
    seed=42
)
```

---

### 7. MLflow

* **Role**: experiment tracking & model registry.
* **CV**: none; it simply logs whatever CV you run.

```python
import mlflow
from sklearn.model_selection import cross_val_score

with mlflow.start_run():
    scores = cross_val_score(model, X, y, cv=group_kfold, scoring="roc_auc")
    mlflow.log_metric("cv_mean_auc", scores.mean())
    mlflow.log_metric("cv_std_auc",  scores.std())
```

---

### 8. Optuna ⭐⭐⭐

* **Role**: hyperparameter optimisation.
* **CV**: entirely user-defined – you plug in any splitter or CV scheme.

```python
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score

gkf = GroupKFold(n_splits=5)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth    = trial.suggest_int("max_depth", 3, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    scores = cross_val_score(model, X, y, cv=gkf, scoring="roc_auc", groups=patient_ids)
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

---

### 9. PyCaret ⭐⭐⭐

* **Default CV**: 10-fold (`fold=10`) in `setup` for classification. 
* **Built-in CV options**:

  * `fold_strategy = "kfold"`, `"stratifiedkfold"`, `"groupkfold"`, `"timeseries"`, or custom sklearn splitter.
* **Medical usage**:

  * Very convenient for grouped CV: `fold_strategy="groupkfold"`, `fold_groups="patient_id"`.

```python
import pycaret.classification as pc

clf = pc.setup(
    data=df,
    target="disease",
    fold_strategy="groupkfold",
    fold_groups="patient_id",
    fold=5,
    session_id=42
)

best_models = pc.compare_models(
    include=["rf", "xgboost", "lightgbm"],
    sort="AUC",
    n_select=3
)
```

---

### 10. H2O.ai ⭐⭐⭐

* **Default CV**: `nfolds=0` by default – i.e., no CV unless you set it. ([H2O Release][5])
* **Capabilities**:

  * K-fold CV via `nfolds >= 2`.
  * Custom grouping (including patient-like grouping) via `fold_column`.
  * No dedicated time-series CV; time-based folds are expressed via `fold_column` or manual splitting. ([Cross Validated][10])

```python
import h2o
from h2o.estimators import H2ORandomForestEstimator

h2o.init()
hf = h2o.H2OFrame(df)

# standard k-fold CV
model = H2ORandomForestEstimator(
    nfolds=5,
    fold_assignment="Stratified",
    keep_cross_validation_predictions=True,
    seed=42
)

# patient-level CV via fold_column
hf["fold_id"] = patient_based_fold_ids
model_grouped = H2ORandomForestEstimator(
    nfolds=5,
    fold_column="fold_id",
    keep_cross_validation_predictions=True,
    seed=42
)
```

---

## 🏥 Medical-Specific Capabilities

### Patient Grouping Support

| Toolbox                   | Native support                      | Workaround needed? | Implementation difficulty |
| ------------------------- | ----------------------------------- | ------------------ | ------------------------- |
| Scikit-Learn              | ✅ `GroupKFold`, `GroupShuffleSplit` | ❌                  | Easy                      |
| XGBoost                   | ✅ Custom `folds` indices            | ❌                  | Easy                      |
| LightGBM                  | ✅ Custom `folds` indices            | ❌                  | Easy                      |
| PyCaret                   | ✅ `fold_strategy="groupkfold"`      | ❌                  | Very easy                 |
| H2O.ai                    | ✅ `fold_column`                     | ❌                  | Easy                      |
| CatBoost                  | ✅ `GroupId`/group columns           | ❌                  | Easy–Moderate             |
| TensorFlow                | ❌ (no CV engine)                    | ✅ via sklearn      | Moderate                  |
| PyTorch                   | ❌ (no CV engine)                    | ✅ via sklearn      | Moderate                  |
| Others (AutoML, tracking) | Indirect via sklearn/etc            | ✅                  | Varies                    |

---

### Temporal Validation Support

Here "temporal" means respecting time order (no leakage from future to past). Some libraries have dedicated objects, others rely on manual folds.

| Toolbox            | Native temporal CV object              | Methods available                                        | Medical suitability |
| ------------------ | -------------------------------------- | -------------------------------------------------------- | ------------------- |
| Scikit-Learn       | ✅ `TimeSeriesSplit` (1 method)         | Expanding window only                                    | Basic               |
| **TrustCV (ours)** | ✅ **8 temporal splitters**             | TimeSeriesSplit, RollingWindowCV, ExpandingWindowCV, BlockedTimeSeries, PurgedKFoldCV, CombinatorialPurgedCV, NestedTemporalCV, EmbargoCV | Excellent           |
| CatBoost           | ✅ TimeSeries type (1 method)           | Forward-chaining only                                    | Moderate            |
| XGBoost            | ⚠️ Via custom `folds` only             | Any temporal scheme you implement                        | Moderate            |
| LightGBM           | ⚠️ Via custom `folds` only             | Any temporal scheme you implement                        | Moderate            |
| H2O.ai             | ⚠️ `fold_column` built from timestamps | Time-based folds via user-generated `fold_column`        | Moderate            |
| Fast.ai            | ⚠️ Time-aware splitters, no full CV    | Single temporal split; full K-fold temporal CV is manual | Poor–Moderate       |
| Others             | ❌ No direct support                    | Manual only                                              | Poor                |

> In TrustCV, temporal splitters are designed explicitly to avoid leakage (e.g. *purged k-fold* CV removes data around the test window to prevent look-ahead bias in financial/ICU monitoring scenarios).

---

### Class Imbalance & Thresholding

| Toolbox      | Stratified CV support                  | Medical metrics (e.g. sensitivity, specificity) | Threshold optimisation    |
| ------------ | -------------------------------------- | ----------------------------------------------- | ------------------------- |
| Scikit-Learn | ✅ Strong (`Stratified*`)               | ⚠️ Basic but flexible (`make_scorer`)           | ✅ Good (manual / custom)  |
| PyCaret      | ✅ Strong (default stratified)          | ✅ Built-in AUC, F1, recall, etc.                | ✅ Good (plot-based tools) |
| H2O.ai       | ✅ Good (`balance_classes`, stratified) | ✅ Excellent (full report, gains, lift)          | ✅ Good                    |
| XGBoost      | ✅ Good (stratified CV)                 | ⚠️ Basic (via external metrics)                 | ⚠️ Limited (manual)       |
| CatBoost     | ✅ Good                                 | ⚠️ Basic out-of-the-box                         | ⚠️ Limited (manual)       |
| TensorFlow   | ⚠️ Manual (class weights, sampling)    | ⚠️ Manual via custom Keras metrics              | ⚠️ Manual                 |

---

## 🔧 Integration Recommendations

### For Medical Research Pipelines

**Suggested stack:**

1. **Data & splitting**: `pandas` + **trustcv** (patient, temporal, spatial splitters).
2. **Model training**: scikit-learn, XGBoost, LightGBM, CatBoost.
3. **Deep learning**: PyTorch (with trustcv/split indices).
4. **Hyperparameter tuning**: Optuna (using the same trustcv splits).
5. **Experiment tracking**: MLflow or Weights & Biases.

### For Clinical / Production-oriented Systems

1. **Validation logic**: encode in **trustcv** and persist folds.
2. **Robust tabular models**: XGBoost / LightGBM / CatBoost on trustcv folds.
3. **AutoML for baseline**: PyCaret or H2O AutoML.
4. **Deployment & monitoring**: MLflow Model Registry + tracking.

### For Academic Work

1. **Experiments**: scikit-learn + trustcv (explicit split reproducibility).
2. **Deep models**: PyTorch or TensorFlow with *the same* split indices.
3. **Novel CV methods**: implement as new `trustcv` splitter classes.

---

## 📝 Implementation Cheatsheet

### Passing the Same Splits Across Libraries

```python
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups=patient_ids))

# XGBoost
dtrain = xgb.DMatrix(X, label=y)
xgb_cv = xgb.cv(params, dtrain, folds=splits)

# LightGBM
train_data = lgb.Dataset(X, label=y)
lgb_cv = lgb.cv(params, train_set=train_data, folds=splits)

# PyTorch / Keras / any DL
for fold_id, (train_idx, test_idx) in enumerate(splits):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    # train your DL model here
```

---

### Medical CV Template with `trustcv`

```python
import numpy as np

def medical_cv_template(X, y, patient_ids=None, timestamps=None, coordinates=None):
    """
    Generic medical CV template using trustcv.
    Chooses an appropriate splitter based on the structure of the data.
    """

    # 1) Patient-level grouping
    if patient_ids is not None:
        from trustcv.splitters.grouped import GroupKFoldMedical
        cv = GroupKFoldMedical(n_splits=5)
        splits = cv.split(X, y, groups=patient_ids)

    # 2) Temporal data (e.g. longitudinal, follow-up)
    elif timestamps is not None:
        from trustcv.splitters.temporal import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=5)
        # TimeSeriesSplit usually operates on ordered indices; ensure X is sorted by time
        splits = cv.split(X)

    # 3) Spatial data (e.g. imaging sites / geographic blocks)
    elif coordinates is not None:
        from trustcv.splitters.spatial import SpatialBlockCV
        cv = SpatialBlockCV(n_splits=5, block_size=0.1)
        splits = cv.split(X, coordinates=coordinates)

    # 4) IID fallback (stratified k-fold)
    else:
        from trustcv.splitters.iid import StratifiedKFoldMedical
        cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X, y)

    return list(splits)


# Example usage with any model
splits = medical_cv_template(X, y, patient_ids=patient_ids)
scores = []

for train_idx, test_idx in splits:
    model = make_model()  # sklearn, XGBoost, LightGBM, CatBoost, DL wrapper, etc.
    model.fit(X[train_idx], y[train_idx])
    score = evaluate_model(model, X[test_idx], y[test_idx])  # e.g. AUC
    scores.append(score)

print(f"Medical CV score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

## 🎯 Conclusions

### CV Method Count Summary

| Category | TrustCV | Scikit-learn | Others |
|----------|---------|--------------|--------|
| **Total splitters** | **29** | 15 | 0-5 |
| **IID methods** | 9 | 9 | varies |
| **Grouped methods** | 8 | 5 | 0-2 |
| **Temporal methods** | 8 | 1 | 0-1 |
| **Spatial methods** | 4 | 0 | 0 |
| **Leakage detection** | ✅ 6 types | ❌ | ❌ |

### Recommendations

* **Most comprehensive CV for medical ML**:

  * **TrustCV** provides 29 methods covering all data types (IID, grouped, temporal, spatial) with built-in leakage detection and regulatory compliance features.

* **Best general-purpose foundation**:

  * **Scikit-Learn** (15 methods) offers excellent coverage for standard use cases; combine with TrustCV for medical-specific needs.

* **Most convenient "batteries included" option**:

  * **PyCaret** when you want a high-level API with good grouped/temporal support out of the box.

* **Best for large-scale / distributed data**:

  * **H2O.ai**, **Dask-ML**, **Spark MLlib** with carefully designed folds (often generated by TrustCV / sklearn).

* **High-performance gradient boosting**:

  * **XGBoost**, **LightGBM**, **CatBoost** all work very well once you standardise your folds.

**Key practical rule for medical ML**

> Decide your cross-validation design (grouping, temporal windows, spatial blocking) **first**, encode it in a single source of truth (e.g. TrustCV splitter), and then reuse the exact same folds across all toolboxes you compare.

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?utm_source=chatgpt.com "GridSearchCV — scikit-learn 1.7.2 documentation"
[2]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html?utm_source=chatgpt.com "lightgbm.cv — LightGBM 4.6.0.99 documentation"
[3]: https://catboost.ai/docs/en/concepts/python-reference_cv?utm_source=chatgpt.com "cv"
[4]: https://walkwithfastai.com/Cross_Validation?utm_source=chatgpt.com "Lesson 3 - Cross-Validation"
[5]: https://h2o-release.s3.amazonaws.com/h2o/rel-vajda/2/docs-website/h2o-py/docs/modeling.html?utm_source=chatgpt.com "Modeling In H2O — H2O documentation"

[7]: https://ml.dask.org/modules/generated/dask_ml.model_selection.GridSearchCV.html?utm_source=chatgpt.com "dask_ml.model_selection.GridSearchCV"
[8]: https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/estimators/stackedensemble.html?utm_source=chatgpt.com "h2o.estimators.stackedensemble"
[9]: https://catboost.ai/docs/en/features/cross-validation?utm_source=chatgpt.com "Cross-validation"
[10]: https://stats.stackexchange.com/questions/278810/how-does-h2o-handle-time-series-cross-validation?utm_source=chatgpt.com "How does h2o handle time-series cross validation?"
