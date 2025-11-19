## IID splitters – imports and usage

TrustCV v0.1 focuses on **IID cross-validation**, with splitters that mirror scikit-learn’s
API but add clearer semantics and medical/industrial defaults.

### 1. Basic imports

All IID splitters are available directly from the top-level `trustcv` package:

```python
from trustcv import (
    HoldOut,
    KFoldMedical,          # alias: KFold
    StratifiedKFoldMedical, # alias: StratifiedKFold
    RepeatedKFold,
    LOOCV,                 # alias: LeaveOneOut
    LPOCV,                 # alias: LeavePOut
    BootstrapValidation,
    MonteCarloCV,
    NestedCV,
)
```

If you prefer, you can import them from the IID module explicitly:

```python
from trustcv.splitters.iid import StratifiedKFoldMedical, HoldOut, RepeatedKFold
```

All splitters follow the scikit-learn pattern:

- `get_n_splits(X=None, y=None, groups=None)`
- `split(X, y=None, groups=None)` → yields `(train_idx, test_idx)` index arrays

You can either:

- Use them manually in your own loop, **or**
- Pass them into `UniversalCVRunner` or `TrustCVValidator`.

------

### 2. `HoldOut`

**What it does:**
 Single train/test split with optional stratification. Think `train_test_split`, but as an object.

**Constructor:**

```python
cv = HoldOut(
    test_size=0.2,
    stratify=True,          # or array-like of labels, or False/None
    shuffle=True,
    random_state=42,
)
```

**Manual usage:**

```python
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

**With `UniversalCVRunner`:**

```python
from trustcv import UniversalCVRunner, HoldOut

cv = HoldOut(test_size=0.2, stratify=True, random_state=42)
runner = UniversalCVRunner(cv_splitter=cv, framework="auto")
results = runner.run(model=clf, data=(X, y), metrics=["accuracy", "roc_auc"])
```

------

### 3. `KFoldMedical` (alias: `KFold`)

**What it does:**
 Standard K-fold CV for IID data (no stratification). Equivalent to scikit-learn’s `KFold`.

**Constructor:**

```python
cv = KFoldMedical(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
# or via alias:
from trustcv import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

**Manual usage:**

```python
for train_idx, test_idx in cv.split(X, y):
    ...
```

**With runner:**

```python
runner = UniversalCVRunner(cv_splitter=cv, framework="auto")
results = runner.run(model=clf, data=(X, y), metrics=["accuracy", "roc_auc"])
```

------

### 4. `StratifiedKFoldMedical` (alias: `StratifiedKFold`)

**What it does:**
 K-fold CV that preserves label distribution in each fold. Recommended for **imbalanced classification**.

**Constructor:**

```python
from trustcv import StratifiedKFoldMedical
cv = StratifiedKFoldMedical(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
# alias:
from trustcv import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**With runner (typical path in docs):**

```python
from trustcv import UniversalCVRunner, StratifiedKFoldMedical

cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
runner = UniversalCVRunner(cv_splitter=cv, framework="auto")
results = runner.run(model=clf, data=(X, y), metrics=["accuracy", "roc_auc", "f1"])
print(results.summary())
```

------

### 5. `RepeatedKFold`

**What it does:**
 Runs K-fold CV multiple times with different random splits to reduce variance of the estimate.

**Constructor:**

```python
from trustcv import RepeatedKFold

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42,
)
```

**Usage with runner:**

```python
runner = UniversalCVRunner(cv_splitter=cv, framework="auto")
results = runner.run(model=clf, data=(X, y), metrics=["accuracy", "roc_auc"])
```

------

### 6. `LOOCV` (alias: `LeaveOneOut`)

**What it does:**
 Leave-one-out CV: each sample is used once as a test case. Useful for very small datasets or when every sample is critical.

**Constructor:**

```python
from trustcv import LOOCV
cv = LOOCV()
# alias:
from trustcv import LeaveOneOut
cv = LeaveOneOut()
```

**Usage:**

```python
for train_idx, test_idx in cv.split(X, y):
    ...
```

------

### 7. `LPOCV` (alias: `LeavePOut`)

**What it does:**
 Leave-**P**-out CV: all combinations of P samples as the test set (combinatorial; only for very small datasets).

**Constructor:**

```python
from trustcv import LPOCV
cv = LPOCV(p=2)
# alias:
from trustcv import LeavePOut
cv = LeavePOut(p=2)
```

------

### 8. `BootstrapValidation`

**What it does:**
 Bootstrap-based validation: repeatedly resamples the dataset to estimate performance and uncertainty (e.g., .632/.632+ style).

**Typical usage:**

```python
from trustcv import BootstrapValidation, UniversalCVRunner

cv = BootstrapValidation(
    n_splits=200,
    test_size=0.2,
    random_state=42,
)
runner = UniversalCVRunner(cv_splitter=cv, framework="auto")
results = runner.run(model=clf, data=(X, y), metrics=["accuracy", "roc_auc"])
```

------

### 9. `MonteCarloCV`

**What it does:**
 Monte Carlo CV = repeated random train/test splits (also known as repeated random sub-sampling validation).

**Constructor:**

```python
from trustcv import MonteCarloCV
cv = MonteCarloCV(
    n_splits=100,
    test_size=0.2,
    random_state=42,
)
```

**Usage with runner:** same as above.

------

### 10. `NestedCV`

**What it does:**
 Composite splitter to support nested cross-validation (outer splits for evaluation, inner splits for model selection).

**Typical pattern (high-level):**

```python
from trustcv import NestedCV, StratifiedKFoldMedical

inner_cv = StratifiedKFoldMedical(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=2)

nested_cv = NestedCV(
    outer_cv=outer_cv,
    inner_cv=inner_cv,
)

for (outer_train, outer_test), inner_splits in nested_cv.split(X, y):
    # inner_splits is an iterable of (inner_train, inner_val) for tuning
    ...
```

In most real workflows, you’ll combine `NestedCV` with higher-level tooling (e.g. `GridSearchCV` or your own tuning loop).


------

Here are sections **4–8** in a form you can drop into your README or `docs/*.md`.
They assume you already have sections 1–3 (intro, Quickstart, IID splitters).

---

## 4. High-level validator – `TrustCVValidator`

While `UniversalCVRunner` gives you a flexible CV loop, many users prefer a **single entry-point** that feels like `sklearn.model_selection.cross_validate`. That’s what `TrustCVValidator` provides.

### 4.1 Import

```python
from trustcv import TrustCVValidator, ValidationResult
```

> `ValidationResult` is the return type; it’s a small dataclass with `scores`, `mean_scores`, `std_scores`, optional confidence intervals, and a `summary()` method.

### 4.2 What it does

`TrustCVValidator` wraps:

* cross-validation (`sklearn.model_selection.cross_validate`)
* metric aggregation (means, stds, optional confidence intervals)
* **optional** leakage and balance checks
* a human-readable `ValidationResult.summary()` for reporting

In v0.1, we treat the **IID modes** as stable:

* `method="kfold"` → plain K-fold
* `method="stratified_kfold"` → stratified K-fold (recommended for classification)

Grouped and temporal methods (`"patient_grouped_kfold"`, `"temporal"`) exist but are **out of scope** for the v0.1 public docs.

### 4.3 Constructor

```python
validator = TrustCVValidator(
    method: str = "stratified_kfold",
    n_splits: int = 5,
    random_state: int = 42,
    check_leakage: bool = True,
    check_balance: bool = True,
    compliance: str | None = None,
    *,
    metrics: list[str] | None = None,
    return_confidence_intervals: bool = False,
    ci_method: str = "bootstrap",
    n_bootstrap: int = 1000,
)
```

Typical v0.1 configuration:

```python
from trustcv import TrustCVValidator

validator = TrustCVValidator(
    method="stratified_kfold",     # or "kfold"
    n_splits=5,
    random_state=42,
    check_leakage=True,
    check_balance=True,
    metrics=["accuracy", "roc_auc", "f1"],
    return_confidence_intervals=True,
)
```

### 4.4 Running validation

`validate` is keyword-only, so you call it like this:

```python
result: ValidationResult = validator.validate(
    model=model,     # sklearn estimator with fit/predict
    X=X,
    y=y,
    groups=None,     # only used for grouped methods (v0.2+)
    cv=None,         # optional custom BaseCrossValidator
    leakage_checker=None,  # optional custom checker; defaults are fine
    sample_weight=None,    # optional
)

print(result.summary())
```

`ValidationResult` contains:

* `scores: dict[str, np.ndarray]`

  * e.g. `scores["accuracy"]` is an array of per-fold values
* `mean_scores: dict[str, float]`
* `std_scores: dict[str, float]`
* `confidence_intervals: dict[str, tuple[float, float]]` (if enabled)
* convenience: `result.scores["roc_auc"].mean()`, `result.mean_scores["roc_auc"]`, etc.
* `summary()` – a formatted, human-readable text overview.

After `validate(...)`, `TrustCVValidator` also sets `validator.last_result`, which is what the reporting utilities use.

---

## 5. Data integrity – leakage & balance checks

One of TrustCV’s main goals is to **make failure modes visible**. Two core tools in v0.1 are:

* `DataLeakageChecker` – screens for leakage patterns
* `BalanceChecker` – inspects class balance and warns on severe imbalance

### 5.1 Imports

```python
from trustcv import DataLeakageChecker, BalanceChecker
```

### 5.2 `DataLeakageChecker`

**Purpose:** Detect common leakage sources via CV-style splits:

* patient/group overlap
* temporal leakage
* feature/preprocessing leakage
* duplicate samples

**Signature:**

```python
report = DataLeakageChecker().check(
    X,
    y=None,
    groups=None,
    timestamps=None,
    coordinates=None,
    n_splits: int = 5,
    random_state: int | None = 42,
)
```

* If `groups` is provided, it uses `GroupKFold` internally (v0.2+ story).
* Else, if `y` is provided, it uses `StratifiedKFold`.
* Else, it falls back to `KFold`.

The return value is a `LeakageReport` dataclass.

**Example (IID classification):**

```python
from trustcv import DataLeakageChecker

checker = DataLeakageChecker()
leak_report = checker.check(X=X, y=y, n_splits=5, random_state=42)

print("Has leakage:", leak_report.has_leakage)
print("Types:", leak_report.leakage_types)
print("Severity:", leak_report.severity)
print("Summary:", leak_report.summary)
```

`LeakageReport` fields (simplified):

* `has_leakage: bool`
* `leakage_types: list[str]`
* `severity: str` – `'none'`, `'low'`, `'medium'`, `'high'`, or `'critical'`
* `details: dict`
* `recommendations: list[str]`
* `summary` property – short text description

### 5.3 `BalanceChecker`

**Purpose:** Quantify class balance and issue warnings for severe imbalance or rare positive classes.

**Signature:**

```python
report = BalanceChecker(threshold: float = 0.1).check_class_balance(
    y,
    groups=None,   # optional group IDs
)
```

The returned dict contains:

* `n_classes: int`
* `class_distribution: dict[str, {"count": int, "percentage": float}]`
* `imbalance_ratio: float` (largest class / smallest class)
* `minority_percentage: float` (percentage of the rarest class)
* `warnings: list[str]`
* `group_analysis: dict[...]` (only if `groups` was provided; per-group stats)

**Example:**

```python
from trustcv import BalanceChecker

bc = BalanceChecker(threshold=0.1)
bal_report = bc.check_class_balance(y)

print("Classes:", bal_report["class_distribution"])
print("Imbalance ratio:", bal_report["imbalance_ratio"])
print("Warnings:", bal_report["warnings"])
```

In your IID v0.1 story, it’s enough to show:

* run `BalanceChecker` once on the full dataset
* show that severe imbalance triggers a clear text warning
* recommend `StratifiedKFoldMedical` when imbalance is present

---

## 6. Clinical metrics – `ClinicalMetrics` and helpers

TrustCV includes medical-centric metrics so you don’t have to reimplement sensitivity/specificity/PPV/NPV and confidence intervals for every project.

### 6.1 Imports

Top-level convenience:

```python
from trustcv import ClinicalMetrics
```

Or, from the metrics module:

```python
from trustcv.metrics import (
    ClinicalMetrics,
    sensitivity_score,
    specificity_score,
    ppv,
    npv,
    lr_positive,
    lr_negative,
    youden_index,
    net_benefit,
    clinical_utility_score,
)
```

(Those names come from `trustcv.metrics.__init__`.)

### 6.2 `ClinicalMetrics`

**Constructor:**

```python
cm = ClinicalMetrics(
    confidence_level: float = 0.95,
    prevalence: float | None = None,   # override dataset prevalence if desired
)
```

**Main method:**

```python
metrics = cm.calculate_all(
    y_true,
    y_pred,
    y_proba=None,
    sample_weight=None,
)
```

`calculate_all` returns a `dict` that typically includes:

* confusion-matrix-derived metrics:

  * `sensitivity`, `specificity`
  * `ppv` (precision), `npv`
  * `lr_pos`, `lr_neg`
  * `youden_index`
* each of the above often has a corresponding `*_ci` key with a confidence interval
* discrimination metrics (if `y_proba` is provided):

  * `roc_auc`, `roc_auc_ci`
  * `average_precision`, `pr_curve`, etc.
* NNT/NNS if prevalence is known or calculable.

**Example:**

```python
import numpy as np
from trustcv import ClinicalMetrics

# Suppose you collected out-of-fold predictions from TrustCV
# y_true: shape (n_samples,)
# y_proba: shape (n_samples,)
y_pred = (y_proba >= 0.5).astype(int)

cm = ClinicalMetrics(confidence_level=0.95)
clin = cm.calculate_all(y_true=y_true, y_pred=y_pred, y_proba=y_proba)

print("Sensitivity:", clin["sensitivity"], "CI:", clin["sensitivity_ci"])
print("Specificity:", clin["specificity"], "CI:", clin["specificity_ci"])
print("PPV / NPV:", clin["ppv"], clin["npv"])
print("ROC AUC (CI):", clin["roc_auc"], clin["roc_auc_ci"])
```

### 6.3 Functional helpers (optional)

If you prefer a more “metric by metric” style, the following are exported:

* `sensitivity_score(y_true, y_pred)`
* `specificity_score(y_true, y_pred)`
* `ppv(y_true, y_pred)`
* `npv(y_true, y_pred)`
* `lr_positive(y_true, y_pred)`
* `lr_negative(y_true, y_pred)`
* `youden_index(y_true, y_pred)`
* `net_benefit(y_true, y_pred, ...)`
* `clinical_utility_score(...)`

For v0.1 docs, it’s enough to show one or two examples:

```python
from trustcv.metrics import sensitivity_score, specificity_score

sens = sensitivity_score(y_true, y_pred)
spec = specificity_score(y_true, y_pred)
print("Sensitivity:", sens, "Specificity:", spec)
```

---

## 7. Regulatory-style reporting – `RegulatoryReport`

For clinical/regulated contexts, you often need more than a metric table: you need a **structured, reproducible validation report**. That’s what `RegulatoryReport` provides.

### 7.1 Import

```python
from trustcv.reporting import RegulatoryReport
```

### 7.2 What it does

`RegulatoryReport` assembles:

* **Device/model metadata** (name, version, manufacturer, intended use).
* **Dataset description** (number of samples, features, optional demographics).
* **Cross-validation configuration** (method, number of folds, scores).
* **Performance metrics** (often from `ClinicalMetrics`).

and can export them as:

* HTML
* JSON
  (PDF uses the HTML as a base; conversion is left to external tooling.)

### 7.3 Constructor

```python
report = RegulatoryReport(
    model_name: str,
    model_version: str,
    manufacturer: str,
    intended_use: str,
    compliance_standard: str = "FDA",
    project_name: str | None = None,
)
```

Example:

```python
from trustcv.reporting import RegulatoryReport

rep = RegulatoryReport(
    model_name="RandomForestClassifier",
    model_version="0.1.0",
    manufacturer="Your Organization",
    intended_use="Diagnostic support for condition X",
    compliance_standard="FDA",
    project_name="Breast Cancer Example",
)
```

### 7.4 Adding dataset and CV information

You typically call:

```python
rep.add_dataset_info(
    n_patients=len(X),
    n_samples=len(X),
    n_features=X.shape[1],
    demographics=None,    # optional dict (age ranges, sex distribution, etc.)
    data_sources=None,    # optional list of strings
)
```

and

```python
rep.add_cv_results(
    method="StratifiedKFold(5)",
    n_splits=5,
    scores=list(result.scores["roc_auc"]),  # or any primary metric array
    confusion_matrices=None,               # optional, if you have them per fold
)
```

Here `result` could be:

* a `ValidationResult` from `TrustCVValidator`, or
* a `CVResults` from `UniversalCVRunner`.

You usually pick your **primary metric** (e.g. ROC AUC) for the `scores` argument.

### 7.5 Adding clinical metrics

You can either:

* call `reg_report.calculate_clinical_metrics(...)` directly, **or**
* compute metrics via `ClinicalMetrics` and assign into the report’s `performance_metrics`.

Direct usage:

```python
rep.calculate_clinical_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_proba=y_proba,   # optional but recommended
)
```

Internally, this uses standard sklearn metrics (accuracy, recall, precision, ROC AUC, confusion matrix) to populate a performance block.

### 7.6 Generating the report

Finally, you write the report:

```python
output_path = rep.generate_regulatory_report(
    output_path="trustcv_regulatory_report.html",
    format="html",     # "html" or "json"
)
print("Report written to:", output_path)
```

or, if you used `TrustCVValidator`:

```python
from trustcv.reporting import RegulatoryReport

# After validator.validate(...)
rep = RegulatoryReport(
    model_name="RandomForestClassifier",
    model_version="0.1.0",
    manufacturer="Your Organization",
    intended_use="Diagnostic support",
)

rep.add_dataset_info(n_patients=len(X), n_samples=len(X), n_features=X.shape[1])
rep.calculate_clinical_metrics(y_true, y_pred, y_proba)
rep.generate_from_validator(
    validator,                     # TrustCVValidator instance
    run_id="EXP-2025-01-01",       # optional identifier
    output_path="regulatory_report.html",
    format="html",
)
```

> For v0.1, keep the reporting examples simple and tied to **IID** use cases. Grouped/temporal reports can be introduced together with v0.2+.

---

## 8. Roadmap – grouped, temporal, spatial CV (post v0.1)

TrustCV v0.1 intentionally focuses on **IID** evaluation to ship a small, clean, stable core. The codebase already contains grouped, temporal, and spatial splitters and validators, but they are **experimental** and will be formalized in later versions.

### v0.2 – Grouped / participant-aware CV

**Goal:** prevent leakage when multiple samples come from the same entity (patient, subject, asset).

Planned highlights:

* Group-based splitters:

  * `GroupKFoldMedical`, `StratifiedGroupKFold`
  * leave-one-/leave-P-group-out and hierarchical variants
* `TrustCVValidator` methods:

  * `method="patient_grouped_kfold"` (and related options)
* Group-aware leakage checks:

  * explicit detection when the same group appears in train and test
* Group-level metrics:

  * per-patient or per-asset performance summaries

Example domains:

* Multi-visit EHR datasets, repeated imaging per patient.
* Predictive maintenance where multiple measurements come from the same machine.

### v0.3 – Temporal / time-series CV

**Goal:** safe evaluation for time-dependent data (no look-ahead).

Planned highlights:

* Temporal splitters:

  * `TimeSeriesSplit`, `BlockedTimeSeries`, rolling and expanding windows
  * purged K-fold variants (`PurgedKFoldCV`, `PurgedGroupTimeSeriesSplit`)
* `NestedTemporalCV`:

  * nested CV tailored for time-series with embargo windows
* Temporal leakage detection:

  * warn when features or preprocessing leak future information
* Metrics and plots:

  * performance vs. prediction horizon, stability over time

Example domains:

* ICU and ward time-series, readmission prediction.
* Industrial sensor time-series for predictive maintenance and forecasting.

### v0.4 – Spatial CV

**Goal:** honest evaluation when samples are spatially correlated.

Planned highlights:

* Spatial splitters:

  * `SpatialBlockCV`, `BufferedSpatialCV`
  * `SpatiotemporalBlockCV` for combined space–time problems
* Spatial leakage checks:

  * flag overly close train/test samples
* Spatial evaluation:

  * region-wise performance summaries and maps

Example domains:

* Environmental and public health models (air pollution, exposure mapping).
* Infrastructure and geospatial models (railway inspection, satellite imagery).

---



