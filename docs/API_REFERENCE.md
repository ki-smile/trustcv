# trustcv API Reference

## Core Classes

### TrustCVValidator

Note: MedicalValidator is deprecated; use TrustCVValidator.

Main validator class for medical machine learning cross-validation.

```python
from trustcv import TrustCVValidator

validator = TrustCVValidator(
    method='stratified_kfold',
    n_splits=5,
    random_state=42,
    check_leakage=True,
    check_balance=True,
    compliance=None
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | 'stratified_kfold' | Cross-validation method |
| `n_splits` | int | 5 | Number of folds |
| `random_state` | int | 42 | Random seed for reproducibility |
| `check_leakage` | bool | True | Enable data leakage detection |
| `check_balance` | bool | True | Check class balance |
| `compliance` | str/None | None | Report format for regulatory documentation ('FDA', 'CE', None) |
| `metrics` | list[str]/None | ['accuracy','roc_auc','sensitivity','specificity','precision','recall','f1'] | Metrics reported by both validation paths (case-insensitive) |
| `return_confidence_intervals` | bool | False | Enable 95% confidence interval reporting |
| `ci_method` | str | 'bootstrap' | Interval estimator ('bootstrap' or 't-interval') |
| `ci_level` | float | 0.95 | Coverage level for confidence intervals (0-1 range) |
| `n_bootstrap` | int | 1000 | Resamples used when `ci_method='bootstrap'` |

> **Tip:** `metrics` is case-insensitive. Provide a subset like `["accuracy", "roc_auc"]` to limit the reported scores, or leave it as `None` to include the full medical set.

#### Methods

##### fit_validate()

```python
results = validator.fit_validate(
    model, X, y, 
    patient_ids=None,
    timestamps=None,
    scoring=None
)
```

Perform medical cross-validation with comprehensive checks.

**Returns:** `ValidationResult` object containing scores, metrics, and recommendations.

##### validate()

```python
val_result = validator.validate(
    model=model,
    X=X,
    y=y,
    patient_ids=patient_ids,  # optional alias for groups
    cv=None,
    sample_weight=None
)
```

Manually run cross-validation using the configured splitter. Accepts `patient_ids` for grouped splits and honors the validator's `metrics`, `ci_method`, and `ci_level` settings (e.g., set `ci_level=0.90` for 90% intervals, or switch to `ci_method='t-interval'`).

##### suggest_best_method()

```python
method = validator.suggest_best_method(X, y, patient_ids, timestamps)
```

Automatically suggest the best CV method based on data characteristics.

---

## Splitters

### GroupKFoldMedical

Patient-aware K-Fold cross-validator ensuring patient data stays together.

```python
from trustcv.splitters import GroupKFoldMedical

cv = GroupKFoldMedical(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
    # Patient data never splits across folds
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | int | 5 | Number of folds |
| `shuffle` | bool | True | Shuffle patients before splitting |
| `random_state` | int/None | None | Random seed |

### StratifiedGroupKFold

Combines stratification with patient grouping for imbalanced medical data.

```python
from trustcv.splitters import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
    # Maintains class balance AND patient grouping
```

### TimeSeriesSplit

Time-aware splitter for temporal medical data.

```python
from trustcv.splitters import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5, gap=7, test_size=None)

for train_idx, test_idx in cv.split(X, timestamps=dates):
    # Training data always precedes test data
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | int | 5 | Number of temporal splits |
| `gap` | int | 0 | Days between train and test |
| `test_size` | int/float/None | None | Size of test set |
| `max_train_size` | int/None | None | Maximum training set size |

### BlockedTimeSeries

Blocked time series CV for seasonal medical data.

```python
from trustcv.splitters import BlockedTimeSeries

cv = BlockedTimeSeries(n_splits=5, block_size='week')

for train_idx, test_idx in cv.split(X, timestamps=dates):
    # Blocks of time stay together
```

---

## Metrics

### ClinicalMetrics

Calculate medical-relevant metrics with confidence intervals.

```python
from trustcv.metrics import ClinicalMetrics

metrics = ClinicalMetrics(confidence_level=0.95, prevalence=None)
results = metrics.calculate_all(y_true, y_pred, y_proba)
```

#### Methods

##### calculate_all()

```python
results = metrics.calculate_all(
    y_true, 
    y_pred, 
    y_proba=None,
    sample_weight=None
)
```

Calculate comprehensive clinical metrics.

**Returns:** Dictionary containing:
- `sensitivity` with CI
- `specificity` with CI  
- `ppv` (Positive Predictive Value) with CI
- `npv` (Negative Predictive Value) with CI
- `accuracy` with CI
- `auc_roc` with CI (if probabilities provided)
- `nnt` (Number Needed to Treat)
- `nns` (Number Needed to Screen)
- `likelihood_ratios`
- `clinical_significance` assessment

##### format_report()

```python
report_str = metrics.format_report(results)
```

Generate formatted clinical report.

### Utility Functions

#### calculate_nnt()

```python
from trustcv.metrics import calculate_nnt

nnt = calculate_nnt(sensitivity=0.8, specificity=0.9, prevalence=0.1)
```

Calculate Number Needed to Treat.

#### calculate_clinical_significance()

```python
from trustcv.metrics import calculate_clinical_significance

significance = calculate_clinical_significance(
    metric_value=0.85,
    metric_type='sensitivity',
    application='screening'
)
```

Assess clinical significance of metric values.

---

## Checkers

### DataLeakageChecker

Detect various types of data leakage in ML.

```python
from trustcv.checkers import DataLeakageChecker

checker = DataLeakageChecker(verbose=True)

report = checker.check_cv_splits(
    X_train, X_test,
    y_train, y_test,
    patient_ids_train, patient_ids_test,
    timestamps_train, timestamps_test
)
```

#### Methods

##### check_cv_splits()

Check for data leakage between train and test sets.

**Returns:** `LeakageReport` object with:
- `has_leakage`: Boolean indicating if leakage detected
- `leakage_types`: List of detected leakage types
- `severity`: 'none', 'low', 'medium', 'high', 'critical'
- `details`: Detailed findings
- `recommendations`: How to fix issues

##### check_preprocessing_leakage()

```python
has_leakage = checker.check_preprocessing_leakage(
    X_original, 
    X_processed,
    split_indices=(train_idx, test_idx)
)
```

Check if preprocessing was done before splitting.

### BalanceChecker

Check class balance and distribution issues.

```python
from trustcv.checkers import BalanceChecker

checker = BalanceChecker()
report = checker.check_class_balance(y)
```

---

## Datasets

### Data Loaders

#### load_heart_disease()

```python
from trustcv.datasets import load_heart_disease

X, y, patient_ids = load_heart_disease()
```

Load synthetic heart disease dataset with patient grouping.

#### load_diabetic_readmission()

```python
from trustcv.datasets import load_diabetic_readmission

X, y, patient_ids, admission_dates = load_diabetic_readmission()
```

Load diabetic readmission dataset with temporal information.

#### load_cancer_imaging()

```python
from trustcv.datasets import load_cancer_imaging

X, y, patient_ids = load_cancer_imaging()
```

Load simulated cancer imaging features.

### Data Generators

#### generate_synthetic_ehr()

```python
from trustcv.datasets import generate_synthetic_ehr

data = generate_synthetic_ehr(
    n_samples=1000,
    n_features=20,
    n_patients=None,
    temporal=False,
    prevalence=0.3,
    random_state=42
)

X, y, patient_ids = data['X'], data['y'], data['patient_ids']
```

Generate synthetic Electronic Health Record data.

#### generate_temporal_patient_data()

```python
from trustcv.datasets import generate_temporal_patient_data

data = generate_temporal_patient_data(
    n_patients=100,
    n_timepoints=12,
    n_features=10,
    outcome_type='binary',
    missing_rate=0.1,
    random_state=42
)
```

Generate temporal patient trajectories.

---

## ValidationResult

Result object from medical validation.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `scores` | dict | Raw CV scores |
| `mean_scores` | dict | Mean scores across folds |
| `std_scores` | dict | Standard deviation of scores |
| `confidence_intervals` | dict | Confidence intervals per metric (per `ci_level`) |
| `ci_method` | str | Interval estimator used (e.g., `bootstrap`, `t-interval`) |
| `ci_level` | float | Confidence level used when computing intervals |
| `fold_details` | list | Per-fold information |
| `leakage_check` | dict | Data integrity results |
| `recommendations` | list | Actionable suggestions |

### Methods

#### summary()

```python
summary_str = results.summary()
```

Generate human-readable summary.

#### to_dict()

```python
results_dict = results.to_dict()
```

Convert to dictionary for JSON export.

---

## Examples

### Complete Pipeline

```python
from trustcv import TrustCVValidator
from trustcv.datasets import load_heart_disease
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data
X, y, patient_ids = load_heart_disease()

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Initialize validator
validator = TrustCVValidator(
    method='patient_grouped_kfold',
    n_splits=5,
    check_leakage=True,
    compliance='FDA'
)

# Perform validation
results = validator.fit_validate(
    pipeline, X, y,
    patient_ids=patient_ids
)

# View results
print(results.summary())
```

### Custom Cross-Validation

```python
from trustcv.splitters import StratifiedGroupKFold
from sklearn.model_selection import cross_validate

# Custom CV with medical awareness
cv = StratifiedGroupKFold(n_splits=5, random_state=42)

scores = cross_validate(
    estimator=model,
    X=X, y=y,
    groups=patient_ids,
    cv=cv,
    scoring=['accuracy', 'roc_auc', 'sensitivity', 'specificity']
)
```

### Data Leakage Detection

```python
from trustcv.checkers import DataLeakageChecker

checker = DataLeakageChecker()

# Check for various leakage types
report = checker.check_cv_splits(
    X_train, X_test,
    y_train, y_test,
    patient_ids_train, patient_ids_test
)

if report.has_leakage:
    print(f"Warning: {report.severity} severity leakage detected!")
    for recommendation in report.recommendations:
        print(f"  - {recommendation}")
```

---

## Type Hints

trustcv uses comprehensive type hints for better IDE support:

```python
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

def fit_validate(
    self,
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    patient_ids: Optional[Union[np.ndarray, pd.Series]] = None,
    timestamps: Optional[Union[np.ndarray, pd.Series]] = None
) -> ValidationResult:
    ...

def validate(
    self,
    *,
    model,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    patient_ids: Optional[Union[np.ndarray, pd.Series]] = None,
    groups: Optional[Union[np.ndarray, pd.Series]] = None,
    cv: Optional[BaseCrossValidator] = None,
    leakage_checker: Optional[Any] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> ValidationResult:
    ...
```

---

## Error Handling

trustcv provides informative error messages:

```python
try:
    results = validator.fit_validate(model, X, y)
except ValueError as e:
    print(f"Validation error: {e}")
    # e.g., "Patient IDs required for patient_grouped_kfold method"
```

---

## Performance Considerations

- Use `n_jobs=-1` for parallel processing in compatible methods
- Consider `max_train_size` for very large datasets
- Use sampling for initial exploration, full data for final validation

---

## Version Compatibility

| trustcv | Python | scikit-learn | pandas | numpy |
|-----------|--------|--------------|--------|-------|
| 0.1.x | ≥3.8 | ≥1.0 | ≥1.3 | ≥1.20 |

---

*For more examples and tutorials, visit the [documentation](https://trustcv.readthedocs.io).* 
