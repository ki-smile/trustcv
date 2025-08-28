# ML/DL Toolbox Cross-Validation Support Comparison

## Top 20 Machine Learning & Deep Learning Training Toolboxes

Comprehensive comparison of CV methods supported by popular ML/DL frameworks.

---

## 📊 Summary Table

| Toolbox | Default CV | Built-in Methods | Medical CV Support | Extensibility | Rating |
|---------|------------|------------------|-------------------|---------------|--------|
| **Scikit-Learn** | 5-Fold | 15+ methods | ⭐⭐ | ⭐⭐⭐ | 🥇 |
| **TensorFlow/Keras** | None | Manual only | ⭐ | ⭐⭐ | 🥈 |
| **PyTorch** | None | Manual only | ⭐ | ⭐⭐ | 🥈 |
| **XGBoost** | 3-Fold | 5 methods | ⭐⭐ | ⭐⭐ | 🥉 |
| **LightGBM** | 5-Fold | 4 methods | ⭐⭐ | ⭐⭐ | 🥉 |
| **CatBoost** | 3-Fold | 3 methods | ⭐ | ⭐⭐ | ⭐⭐ |
| **MLflow** | None | Logging only | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Optuna** | User-defined | Any via callback | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Weights & Biases** | None | Tracking only | ⭐ | ⭐⭐ | ⭐⭐ |
| **Fast.ai** | 5-Fold | 6 methods | ⭐ | ⭐⭐ | ⭐⭐ |
| **H2O.ai** | 5-Fold | 8 methods | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Auto-sklearn** | 10-Fold | 12 methods | ⭐⭐ | ⭐ | ⭐⭐ |
| **TPOT** | 5-Fold | 8 methods | ⭐ | ⭐ | ⭐⭐ |
| **PyCaret** | 10-Fold | 15+ methods | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **scikit-multilearn** | 3-Fold | 5 methods | ⭐ | ⭐⭐ | ⭐⭐ |
| **Dask-ML** | 5-Fold | 10 methods | ⭐ | ⭐⭐ | ⭐⭐ |
| **Spark MLlib** | 3-Fold | 4 methods | ⭐ | ⭐ | ⭐⭐ |
| **Vowpal Wabbit** | None | Manual only | ⭐ | ⭐ | ⭐⭐ |
| **JAX/Flax** | None | Manual only | ⭐ | ⭐⭐ | ⭐⭐ |
| **Hugging Face** | None | Manual only | ⭐ | ⭐⭐ | ⭐⭐ |

**Rating Key:** 🥇 Excellent | 🥈 Very Good | 🥉 Good | ⭐⭐⭐ Great | ⭐⭐ Moderate | ⭐ Basic

---

## 🔍 Detailed Analysis

### 1. **Scikit-Learn** 🥇
- **Default CV**: `KFold(n_splits=5)`
- **Built-in Methods**: 15+ comprehensive methods
  - ✅ KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
  - ✅ LeaveOneOut, LeavePOut, ShuffleSplit
  - ✅ RepeatedKFold, RepeatedStratifiedKFold
  - ✅ StratifiedGroupKFold, GroupShuffleSplit
- **Medical Features**: 
  - ✅ Patient grouping (GroupKFold)
  - ✅ Class stratification
  - ⚠️ Limited temporal methods
- **Integration**: Perfect with sklearn ecosystem
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
```

### 2. **TensorFlow/Keras** 🥈
- **Default CV**: None (manual implementation required)
- **Built-in Methods**: None directly
- **Medical Features**: Manual implementation only
- **Integration**: Requires custom loops
```python
from tensorflow.keras.models import clone_model
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
scores = []
for train_idx, val_idx in kf.split(X):
    model = clone_model(base_model)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    model.fit(X[train_idx], y[train_idx], epochs=50, verbose=0)
    score = model.evaluate(X[val_idx], y[val_idx], verbose=0)[1]
    scores.append(score)
```

### 3. **PyTorch** 🥈
- **Default CV**: None
- **Built-in Methods**: None directly
- **Medical Features**: Manual implementation required
- **Integration**: Custom training loops needed
```python
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(dataset):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, sampler=train_sampler)
    val_loader = DataLoader(dataset, sampler=val_sampler)
    # Custom training loop here
```

### 4. **XGBoost** 🥉
- **Default CV**: `cv.cv(nfolds=3)`
- **Built-in Methods**:
  - ✅ K-Fold
  - ✅ Stratified K-Fold  
  - ✅ Custom folds
  - ⚠️ Limited advanced methods
- **Medical Features**: Group folding via custom folds
```python
import xgboost as xgb
# Built-in CV
results = xgb.cv(
    params=params,
    dtrain=dtrain,
    nfold=5,
    stratified=True,  # For classification
    seed=42
)

# Custom CV for medical data
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
custom_folds = list(gkf.split(X, y, groups=patient_ids))
results = xgb.cv(params, dtrain, folds=custom_folds)
```

### 5. **LightGBM** 🥉
- **Default CV**: `cv(nfolds=5)`
- **Built-in Methods**:
  - ✅ K-Fold
  - ✅ Stratified K-Fold
  - ✅ Custom folds
  - ✅ Early stopping
- **Medical Features**: Custom folds support
```python
import lightgbm as lgb
# Built-in CV
results = lgb.cv(
    params=params,
    train_set=train_data,
    nfold=5,
    stratified=True,
    seed=42
)

# Custom medical CV
custom_folds = [(train_idx, val_idx) for train_idx, val_idx in group_kfold.split(X, y, groups)]
results = lgb.cv(params, train_data, folds=custom_folds)
```

### 6. **CatBoost**
- **Default CV**: `cv(fold_count=3)`
- **Built-in Methods**:
  - ✅ K-Fold
  - ✅ Stratified K-Fold
  - ✅ Time series split
- **Medical Features**: Limited
```python
from catboost import CatBoostClassifier, cv, Pool

pool = Pool(X, y, cat_features=categorical_features)
cv_results = cv(
    pool=pool,
    params=params,
    fold_count=5,
    stratified=True,
    seed=42
)
```

### 7. **MLflow**
- **Default CV**: None (tracking/logging only)
- **Built-in Methods**: Experiment tracking
- **Integration**: Works with any CV method
```python
import mlflow
from sklearn.model_selection import cross_val_score

with mlflow.start_run():
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    mlflow.log_metric("cv_mean_auc", scores.mean())
    mlflow.log_metric("cv_std_auc", scores.std())
```

### 8. **Optuna** ⭐⭐⭐
- **Default CV**: User-defined
- **Built-in Methods**: Flexible integration with any method
- **Medical Features**: Via custom callbacks
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Hyperparameter suggestions
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Use any CV method
    scores = cross_val_score(model, X, y, cv=group_kfold, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 9. **PyCaret** ⭐⭐⭐
- **Default CV**: `fold=10`
- **Built-in Methods**: 15+ methods with simple interface
  - ✅ KFold, StratifiedKFold, GroupKFold
  - ✅ TimeSeriesSplit, Custom folds
- **Medical Features**: Excellent medical data support
```python
import pycaret.classification as pc

# Setup with custom CV
clf = pc.setup(
    data=df,
    target='disease',
    fold_strategy='groupkfold',  # Medical grouping
    fold_groups='patient_id',    # Patient ID column
    fold=5,
    session_id=42
)

# Compare models with automatic CV
best_models = pc.compare_models(
    include=['rf', 'xgboost', 'lightgbm'],
    sort='AUC',
    n_select=3
)
```

### 10. **H2O.ai** ⭐⭐⭐
- **Default CV**: `nfolds=5`
- **Built-in Methods**: 8 comprehensive methods
- **Medical Features**: Good support for grouped data
```python
import h2o
from h2o.estimators import H2ORandomForestEstimator

# Initialize H2O
h2o.init()

# Load data
df = h2o.H2OFrame(your_data)

# Built-in CV
model = H2ORandomForestEstimator(
    nfolds=5,
    fold_assignment="stratified",  # or "modulo", "random"
    keep_cross_validation_predictions=True,
    seed=42
)

# Custom fold assignment for medical data
df['fold_id'] = patient_based_folds  # Your custom fold assignment
model = H2ORandomForestEstimator(
    nfolds=5,
    fold_column='fold_id'  # Use custom folds
)
```

---

## 🏥 Medical-Specific Capabilities Comparison

### Patient Grouping Support
| Toolbox | Native Support | Workaround Required | Implementation Difficulty |
|---------|----------------|-------------------|--------------------------|
| Scikit-Learn | ✅ GroupKFold | ❌ | Easy |
| XGBoost | ✅ Custom folds | ❌ | Easy |
| LightGBM | ✅ Custom folds | ❌ | Easy |
| PyCaret | ✅ GroupKFold | ❌ | Very Easy |
| H2O.ai | ✅ Fold column | ❌ | Easy |
| TensorFlow | ❌ | ✅ | Moderate |
| PyTorch | ❌ | ✅ | Moderate |
| CatBoost | ⚠️ Limited | ✅ | Moderate |

### Temporal Validation Support
| Toolbox | Native Support | Methods Available | Medical Suitability |
|---------|----------------|------------------|-------------------|
| Scikit-Learn | ✅ | TimeSeriesSplit | Good |
| trustcv (Ours) | ✅ | 8 temporal methods | Excellent |
| XGBoost | ⚠️ | Custom implementation | Moderate |
| Fast.ai | ⚠️ | Limited | Poor |
| H2O.ai | ✅ | Time-based splitting | Good |
| Others | ❌ | Manual required | Poor |

### Class Imbalance Handling
| Toolbox | Stratification | Medical Metrics | Threshold Optimization |
|---------|---------------|-----------------|----------------------|
| Scikit-Learn | ✅ Excellent | ⚠️ Basic | ✅ Good |
| PyCaret | ✅ Excellent | ✅ Good | ✅ Good |
| H2O.ai | ✅ Good | ✅ Excellent | ✅ Good |
| XGBoost | ✅ Good | ⚠️ Basic | ⚠️ Limited |
| CatBoost | ✅ Good | ⚠️ Basic | ⚠️ Limited |
| TensorFlow | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual |

---

## 🔧 Integration Recommendations

### For Medical Research
**Recommended Stack:**
1. **Data Preparation**: Pandas + trustcv splitters
2. **Model Training**: Scikit-Learn / PyCaret / H2O.ai
3. **Deep Learning**: PyTorch + custom CV loops
4. **Hyperparameter Tuning**: Optuna
5. **Experiment Tracking**: MLflow or Weights & Biases

### For Clinical Applications
**Recommended Stack:**
1. **Validation**: trustcv (our package)
2. **Traditional ML**: Scikit-Learn + XGBoost/LightGBM
3. **AutoML**: PyCaret or H2O.ai
4. **Production**: MLflow for model management

### For Academic Research
**Recommended Stack:**
1. **Experimentation**: Scikit-Learn + trustcv
2. **Deep Learning**: PyTorch with custom CV
3. **Analysis**: Custom implementations for novel methods
4. **Reproducibility**: Save CV splits + use fixed seeds

---

## 📝 Quick Implementation Cheatsheet

### Convert Between CV Methods
```python
# From sklearn to XGBoost
from sklearn.model_selection import GroupKFold
import xgboost as xgb

gkf = GroupKFold(n_splits=5)
custom_folds = list(gkf.split(X, y, groups=patient_ids))

# Use in XGBoost
dtrain = xgb.DMatrix(X, label=y)
results = xgb.cv(params, dtrain, folds=custom_folds)

# From sklearn to LightGBM
train_data = lgb.Dataset(X, label=y)
results = lgb.cv(params, train_data, folds=custom_folds)

# From sklearn to PyTorch (manual loop)
for train_idx, val_idx in gkf.split(X, y, groups=patient_ids):
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx))
    # Training code here
```

### Medical CV Template
```python
def medical_cv_template(X, y, patient_ids=None, timestamps=None, coordinates=None):
    """Template for medical cross-validation"""
    
    if patient_ids is not None:
        # Grouped data - use GroupKFold
        from trustcv.splitters.grouped import GroupKFoldMedical
        cv = GroupKFoldMedical(n_splits=5)
        splits = cv.split(X, y, groups=patient_ids)
        
    elif timestamps is not None:
        # Temporal data - use TimeSeriesSplit
        from trustcv.splitters.temporal import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=5)
        splits = cv.split(X)
        
    elif coordinates is not None:
        # Spatial data - use SpatialBlockCV
        from trustcv.splitters.spatial import SpatialBlockCV
        cv = SpatialBlockCV(n_splits=5, block_size=0.1)
        splits = cv.split(X, coordinates=coordinates)
        
    else:
        # IID data - use StratifiedKFold
        from trustcv.splitters.iid import StratifiedKFoldMedical
        cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X, y)
    
    return splits

# Usage with any ML framework
splits = medical_cv_template(X, y, patient_ids=patient_ids)
scores = []

for train_idx, test_idx in splits:
    # Use with any framework
    model = your_model()  # sklearn, xgb, pytorch, etc.
    model.fit(X[train_idx], y[train_idx])
    score = evaluate_model(model, X[test_idx], y[test_idx])
    scores.append(score)

print(f"Medical CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

## 🎯 Conclusion

**Best Overall for Medical ML:**
1. **Scikit-Learn + trustcv**: Most comprehensive and medical-specific
2. **PyCaret**: Easiest to use with good medical support  
3. **H2O.ai**: Great for larger datasets with built-in medical features
4. **XGBoost/LightGBM**: Excellent performance with custom CV support

**Key Takeaways:**
- Most frameworks require custom implementation for medical-specific CV
- Scikit-Learn has the best native CV support
- Deep learning frameworks (TensorFlow, PyTorch) need manual CV loops
- AutoML tools (PyCaret, H2O.ai) provide good medical data handling
- Always verify that your chosen method supports your data structure (grouped, temporal, spatial)

Choose your toolbox based on:
- **Data complexity**: Simple → PyCaret, Complex → Scikit-Learn + trustcv
- **Model type**: Traditional ML → Scikit-Learn/XGBoost, Deep Learning → PyTorch
- **Team expertise**: Beginners → PyCaret, Advanced → Custom implementations
- **Deployment needs**: Production → MLflow + robust frameworks