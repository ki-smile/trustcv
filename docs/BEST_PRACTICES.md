# Cross-Validation Best Practices for Medical Machine Learning

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Method Selection Guide](#method-selection-guide)
3. [Common Pitfalls](#common-pitfalls)
4. [Medical-Specific Considerations](#medical-specific-considerations)
5. [Regulatory Compliance](#regulatory-compliance)
6. [Code Examples](#code-examples)

---

## Quick Reference

### ✅ Always Do
- **Use stratified splits** for imbalanced medical datasets
- **Keep patient data together** - never split records from same patient
- **Check for data leakage** before training
- **Report confidence intervals** not just mean performance
- **Save random seeds** for reproducibility
- **Document preprocessing steps** for regulatory compliance

### ❌ Never Do
- **Don't preprocess before splitting** - causes leakage
- **Don't ignore temporal order** in longitudinal data
- **Don't use regular k-fold** with grouped data
- **Don't trust single train-test split** - high variance
- **Don't mix validation and test sets** - separate holdout needed

---

## Method Selection Guide

### Decision Tree for Choosing CV Method

```
Is your data temporal (time-series)?
├── Yes → Use TemporalClinical or TimeSeriesSplit
│         └── Multiple patients? → Use GroupedTimeSeriesSplit
└── No → Continue
    │
    └── Multiple records per patient?
        ├── Yes → Use PatientGroupKFold
        │         └── Imbalanced? → Use StratifiedGroupKFold
        └── No → Continue
            │
            └── Is dataset imbalanced (>70/30)?
                ├── Yes → Use StratifiedKFold
                └── No → Use standard KFold
```

### Method Comparison Table

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **KFold** | Balanced, independent samples | Simple, unbiased | Ignores class distribution |
| **StratifiedKFold** | Imbalanced datasets | Preserves class ratios | Can't handle groups |
| **PatientGroupKFold** | Multiple records/patient | Prevents patient leakage | May have uneven folds |
| **TemporalClinical** | Time-series medical data | Respects temporal order | Less data for early folds |
| **NestedCV** | Hyperparameter tuning | Unbiased performance | Computationally expensive |

---

## Common Pitfalls

### 1. Data Leakage

#### ❌ Wrong Way
```python
# Preprocessing before splitting - LEAKAGE!
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

#### ✅ Correct Way
```python
# Split first, then preprocess
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform!
```

### 2. Patient Data Mixing

#### ❌ Wrong Way
```python
# Regular k-fold ignores patient grouping
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    # Same patient can be in both sets!
```

#### ✅ Correct Way
```python
from trustcv.splitters import PatientGroupKFold

pgkf = PatientGroupKFold(n_splits=5)
for train, test in pgkf.split(X, groups=patient_ids):
    # Patient data stays together
```

### 3. Ignoring Class Imbalance

#### ❌ Wrong Way
```python
# With 95% negative, 5% positive cases
cv_scores = cross_val_score(model, X, y, cv=5)
# Some folds might have NO positive cases!
```

#### ✅ Correct Way
```python
skf = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=skf)
# Each fold maintains 95/5 ratio
```

### 4. Temporal Leakage

#### ❌ Wrong Way
```python
# Random splitting of time-series data
X_train, X_test = train_test_split(temporal_data, random_state=42)
# Future data leaks into training!
```

#### ✅ Correct Way
```python
from trustcv.splitters import TemporalClinical

tscv = TemporalClinical(n_splits=5)
for train, test in tscv.split(X, timestamps=dates):
    # Always train on past, test on future
```

---

## Medical-Specific Considerations

### 1. Sample Size Requirements

**Minimum samples per class per fold:**
- Binary classification: ≥30 per class
- Multi-class: ≥10 per class
- Rare diseases: Consider LOOCV or bootstrap

```python
def check_sample_size(y, n_splits=5):
    """Check if sample size adequate for CV"""
    unique, counts = np.unique(y, return_counts=True)
    min_class_size = counts.min()
    samples_per_fold = min_class_size // n_splits
    
    if samples_per_fold < 30:
        warnings.warn(
            f"Only {samples_per_fold} samples per fold for minority class. "
            "Consider fewer splits or different method."
        )
    return samples_per_fold
```

### 2. Multi-Site Clinical Trials

For data from multiple hospitals/sites:

```python
from trustcv.splitters import HierarchicalGroupKFold

# Ensure site effects don't bias results
hgkf = HierarchicalGroupKFold(
    n_splits=5,
    hierarchy_level='site'  # Split by site
)

# This prevents overfitting to site-specific patterns
```

### 3. Longitudinal Patient Data

For repeated measurements over time:

```python
# Each patient has multiple visits
df = pd.DataFrame({
    'patient_id': [1, 1, 1, 2, 2, 2],
    'visit_date': ['2021-01', '2021-06', '2022-01'] * 2,
    'measurement': [120, 125, 130, 110, 115, 118]
})

# Must respect both patient grouping AND temporal order
from trustcv import TrustCVValidator

validator = TrustCVValidator(
    method='grouped_temporal',
    patient_grouping=True,
    temporal_ordering=True
)
```

### 4. Rare Disease Classification

For extremely imbalanced datasets (<1% positive):

```python
# Use specialized techniques
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

# 1. Stratified CV to preserve rare class
skf = StratifiedKFold(n_splits=3)  # Fewer splits

# 2. SMOTE only on training data
for train_idx, test_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Apply SMOTE to training only
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train and evaluate
    model.fit(X_train_balanced, y_train_balanced)
    # Test on original imbalanced data
    score = model.score(X_test, y_test)
```

---

## Regulatory Compliance

### FDA Requirements

```python
from trustcv import TrustCVValidator

validator = TrustCVValidator(compliance='FDA')

# Automatically ensures:
# 1. Separate holdout test set (FDA requirement)
# 2. Documentation of all preprocessing
# 3. Confidence intervals for all metrics
# 4. Subgroup analysis (age, sex, ethnicity)
# 5. Failure mode analysis

results = validator.fit_validate(model, X, y)
results.generate_fda_report('validation_report.pdf')
```

### Key FDA Metrics
- **Sensitivity (Recall)**: True Positive Rate
- **Specificity**: True Negative Rate  
- **PPV**: Positive Predictive Value
- **NPV**: Negative Predictive Value
- **AUC-ROC**: With 95% confidence intervals

### CE Mark Requirements
- Transparency and explainability
- GDPR compliance for data handling
- Clinical evaluation metrics
- Risk classification

---

## Code Examples

### Complete Best Practice Pipeline

```python
from trustcv import TrustCVValidator
from trustcv.checkers import DataLeakageChecker
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Check for data leakage
checker = DataLeakageChecker()
leakage_report = checker.check_cv_splits(
    X_train, X_test, 
    patient_ids_train, patient_ids_test
)

if leakage_report.has_leakage:
    raise ValueError(f"Data leakage detected: {leakage_report}")

# 2. Create preprocessing pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# 3. Set up medical-aware validation
validator = TrustCVValidator(
    method='stratified_group_kfold',  # Auto-selected based on data
    n_splits=5,
    check_leakage=True,
    check_balance=True,
    compliance='FDA'  # Generate FDA-ready reports
)

# 4. Perform validation
results = validator.fit_validate(
    pipeline, X, y,
    patient_ids=patient_ids,
    timestamps=admission_dates
)

# 5. Get comprehensive results
print(results.summary())

# 6. Generate regulatory report
results.generate_compliance_report('fda_submission.pdf')

# 7. Save for reproducibility
results.save('validation_results.pkl')
```

### Handling Edge Cases

```python
# Small dataset (<100 samples)
if len(X) < 100:
    # Use Leave-One-Out or Bootstrap
    from sklearn.model_selection import LeaveOneOut
    cv = LeaveOneOut()
    
# Extremely imbalanced (< 10 positive cases)
if y.sum() < 10:
    # Use special techniques
    from sklearn.model_selection import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2)
    
# Multi-label classification
if y.shape[1] > 1:
    from skmultilearn.model_selection import IterativeStratification
    cv = IterativeStratification(n_splits=5)
```

---

## Checklist for Medical ML Projects

Before training any model, verify:

- [ ] Data is split BEFORE any preprocessing
- [ ] Patient records are kept together (no patient in both train/test)
- [ ] Temporal order is preserved (if applicable)
- [ ] Class distribution is maintained across folds
- [ ] Random seed is set for reproducibility
- [ ] Separate holdout test set exists (for final evaluation)
- [ ] Confidence intervals are calculated
- [ ] Subgroup performance is evaluated
- [ ] Clinical significance is considered (not just statistical)
- [ ] Documentation meets regulatory requirements

---

## Additional Resources

- [FDA Guidance on ML/AI](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)
- [TRIPOD Statement](https://www.tripod-statement.org/) - Reporting guidelines
- [trustcv Documentation](https://github.com/ki-smile/trustcv)
- [Clinical ML Best Practices](https://www.nature.com/articles/s41591-018-0316-z)

---

*Last updated: 2025*
*Part of the TrustCV toolkit v1.0.0*