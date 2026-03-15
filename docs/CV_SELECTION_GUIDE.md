# Cross-Validation Selection Guide for Medical ML


## 🎯 Quick Decision Tree

```
START HERE
    │
    ▼
Q1: What type of data structure do you have?
    │
    ├─► Independent samples (each row = different patient)?
    │       │
    │       ├─► Small dataset (<100 samples)?
    │       │     └─► Use: LOOCV or Bootstrap .632
    │       │
    │       ├─► Imbalanced classes?
    │       │     └─► Use: Stratified k-Fold
    │       │
    │       ├─► Need hyperparameter tuning?
    │       │     └─► Use: Nested CV
    │       │
    │       └─► Standard case?
    │             └─► Use: 5-Fold or 10-Fold CV
    │
    ├─► Time series data (temporal order matters)?
    │       │
    │       ├─► Need fixed training size?
    │       │     └─► Use: Rolling Window CV
    │       │
    │       ├─► Want to use all historical data?
    │       │     └─► Use: Expanding Window CV
    │       │
    │       ├─► Have seasonal/periodic patterns?
    │       │     └─► Use: Blocked Time Series CV
    │       │
    │       └─► Standard forecasting?
    │             └─► Use: Time Series Split
    │
    ├─► Grouped data (multiple records per patient)?
    │       │
    │       ├─► Testing generalization to new patients?
    │       │     └─► Use: Leave-One-Group-Out
    │       │
    │       ├─► Need class balance?
    │       │     └─► Use: Stratified Group k-Fold
    │       │
    │       └─► Standard grouped case?
    │             └─► Use: Group k-Fold
    │
    └─► Spatial/Geographic data?
            │
            ├─► Need to prevent spatial leakage?
            │     └─► Use: Buffered Spatial CV
            │
            └─► Standard spatial case?
                  └─► Use: Spatial Block CV
```

## 📊 Detailed Method Selection Matrix

| Data Characteristic | Sample Size | Primary Goal | Recommended Method | Why? |
|---------------------|-------------|--------------|-------------------|------|
| **I.I.D. Data** | | | | |
| Independent samples | >1000 | Quick evaluation | Hold-Out (70/30) | Fast, simple |
| Independent samples | 100-1000 | Robust evaluation | 5-Fold CV | Good bias-variance tradeoff |
| Independent samples | <100 | Max data usage | LOOCV | Uses all data for training |
| Imbalanced classes | Any | Maintain class ratio | Stratified k-Fold | Preserves class distribution |
| Independent samples | Any | Confidence intervals | Bootstrap .632 | Provides CI estimates |
| Independent samples | Any | Model selection | Nested CV | Unbiased hyperparameter tuning |
| **Temporal Data** | | | | |
| Time series | Any | Forecasting | Time Series Split | Respects temporal order |
| Time series | Long series | Fixed window | Rolling Window | Constant training size |
| Time series | Growing data | Use all history | Expanding Window | Increasing training data |
| Time series | With patterns | Preserve patterns | Blocked Time Series | Maintains temporal blocks |
| Financial/Trading | Any | Prevent leakage | Purged K-Fold | Adds temporal gaps |
| **Grouped Data** | | | | |
| Patient records | Many groups | Standard validation | Group k-Fold | No patient in multiple folds |
| Patient records | Few groups | Test generalization | Leave-One-Group-Out | Each group as test |
| Hierarchical | Multi-level | Respect hierarchy | Hierarchical Group CV | Maintains structure |
| Imbalanced groups | Any | Balance + grouping | Stratified Group k-Fold | Preserves both constraints |
| **Spatial Data** | | | | |
| Geographic | Grid-based | Prevent leakage | Buffered Spatial CV | Adds buffer zones |
| Geographic | Continuous | Standard spatial | Spatial Block CV | Creates spatial blocks |
| Spatiotemporal | Both dimensions | Complex patterns | Spatiotemporal Block | Handles both aspects |
| Environmental health | Geographic + health | Environmental factors | Environmental Health CV | Epidemiology studies |
| **Additional Methods** | | | | |
| Independent samples | <50 | Exhaustive validation | LPOCV (Leave-p-Out) | Tests all p-combinations |
| Independent samples | Any | Multiple random splits | Monte Carlo CV | Flexible, confidence intervals |
| Time series | Complex patterns | Multiple test periods | Combinatorial Purged CV | Financial/trading data |
| Time + Groups | Both constraints | Combined validation | Purged Group Time Series | Complex medical studies |
| Time series | Nested optimization | Temporal + hyperparam | Nested Temporal CV | Advanced forecasting |
| Grouped data | Test on p groups | Multiple group testing | Leave-p-Groups-Out | Multi-site validation |
| Grouped data | Multiple runs | Robust group validation | Repeated Group k-Fold | Stable estimates |
| Hierarchical | Multiple levels | Respect all levels | Multi-level CV | Hospital>Dept>Patient |
| Grouped data | Hyperparam tuning | Nested + grouped | Nested Grouped CV | Unbiased selection |

## 🔍 Common Medical ML Scenarios

### Scenario 1: Clinical Trial with Multiple Sites
- **Data**: Patients nested within hospitals
- **Goal**: Test generalization to new sites
- **Method**: Group k-Fold or Leave-One-Site-Out
- **Code Example**:
```python
from trustcv.splitters.grouped import GroupKFoldMedical
cv = GroupKFoldMedical(n_splits=5)
for train, test in cv.split(X, y, groups=site_ids):
    # No site appears in both train and test
```

### Scenario 2: ICU Patient Monitoring
- **Data**: Hourly vital signs over time
- **Goal**: Predict future deterioration
- **Method**: Time Series Split or Rolling Window
- **Code Example**:
```python
from trustcv.splitters.temporal import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)
for train, test in cv.split(X):
    # Always train on past, test on future
```

### Scenario 3: Disease Diagnosis from Images
- **Data**: One image per patient
- **Goal**: Robust performance estimate
- **Method**: Stratified k-Fold (preserves disease prevalence)
- **Code Example**:
```python
from trustcv.splitters.iid import StratifiedKFoldMedical
cv = StratifiedKFoldMedical(n_splits=5)
for train, test in cv.split(X, y):
    # Maintains class balance across folds
```

### Scenario 4: Longitudinal Patient Study
- **Data**: Multiple visits per patient over years
- **Goal**: Validate on new patients
- **Method**: Grouped Time Series CV
- **Code Example**:
```python
from trustcv.splitters.temporal import PurgedGroupTimeSeriesSplit
cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=30)  # 30-day gap
for train, test in cv.split(X, y, groups=patient_ids, times=visit_dates):
    # Respects both patient grouping and temporal order
```

### Scenario 5: Geographic Disease Spread
- **Data**: Cases with GPS coordinates
- **Goal**: Predict spread to new regions
- **Method**: Buffered Spatial CV
- **Code Example**:
```python
from trustcv.splitters.spatial import BufferedSpatialCV
cv = BufferedSpatialCV(n_splits=5, buffer_size=10)  # 10km buffer
for train, test in cv.split(X, coordinates=gps_coords):
    # Buffer prevents spatial leakage
```

## ⚠️ Critical Warnings

### ❌ NEVER DO THIS:
1. **Random splits on time series data** → Future leakage
2. **Random splits on grouped patient data** → Patient in both train/test
3. **Using test set for ANY decisions** → Overfitting to test set
4. **Single hold-out for small datasets** → High variance
5. **Ignoring class imbalance** → Biased to majority class

### ✅ ALWAYS DO THIS:
1. **Check for data leakage** after splitting
2. **Preserve data structure** (temporal, grouped, spatial)
3. **Use stratification** for imbalanced datasets
4. **Apply nested CV** for hyperparameter tuning
5. **Report confidence intervals** not just mean scores

## 📈 Performance vs Computational Cost

| Method | Computational Cost | Variance | Bias | Use When |
|--------|-------------------|----------|------|----------|
| Hold-Out | ⭐ (Fastest) | High | Low | Large datasets, quick tests |
| 5-Fold CV | ⭐⭐⭐ | Medium | Low | Standard choice |
| 10-Fold CV | ⭐⭐⭐⭐ | Low | Low | Need lower variance |
| LOOCV | ⭐⭐⭐⭐⭐ (Slowest) | Lowest | Low | Small datasets |
| Bootstrap | ⭐⭐⭐ | Low | Medium | Need confidence intervals |
| Nested CV | ⭐⭐⭐⭐⭐ | Low | Lowest | Hyperparameter tuning |
| Group k-Fold | ⭐⭐⭐ | Medium | Low | Grouped data |
| Time Series Split | ⭐⭐ | Medium | Low | Temporal data |

## 🎓 Rules of Thumb

1. **Sample Size Rules**:
   - n < 100: Use LOOCV or Bootstrap
   - 100 < n < 1000: Use 10-Fold CV
   - n > 1000: Use 5-Fold CV or Hold-Out

2. **Fold Number Selection**:
   - More folds = Less bias, More variance, More computation
   - Fewer folds = More bias, Less variance, Less computation
   - Sweet spot: 5-10 folds for most cases

3. **Special Cases**:
   - Rare disease (prevalence < 10%): Always use stratification
   - Multi-site data: Always use grouped CV
   - Time series: Never use random splits
   - Small test set: Consider repeated CV for stability

## 💡 Pro Tips

1. **Combine Methods**: Use Repeated Stratified Group k-Fold for robust estimates with grouped imbalanced data

2. **Validation Strategy Hierarchy**:
   ```
   Good:    Hold-Out
   Better:  k-Fold CV
   Best:    Nested k-Fold CV with proper constraints
   ```

3. **Report Multiple Metrics**: Don't just report accuracy
   - Sensitivity/Specificity for medical diagnosis
   - AUROC for ranking performance
   - Calibration for probability estimates

4. **Document Your Choice**: Always justify why you chose a specific CV method in your methods section

## 📚 References & Further Reading

- [Systematic Review of CV Methods in Medical ML](CrossvalidationMethods.pdf)
- [sklearn Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Medical ML Best Practices](https://www.nature.com/articles/s41591-018-0316-z)

---

*Remember: The goal of cross-validation is to estimate how well your model will perform on new, unseen medical data. Choose the method that best mimics your real-world deployment scenario.*