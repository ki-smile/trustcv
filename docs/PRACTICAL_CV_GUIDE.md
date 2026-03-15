# Practical Cross-Validation Implementation Guide

## How to Use Cross-Validation Methods with Your Own Code

This guide shows you how to integrate different CV methods into your existing ML workflows.

## 🚀 Quick Integration Examples

### 1. With Scikit-Learn Models

```python
from trustcv.splitters.iid import StratifiedKFoldMedical
from trustcv.splitters.grouped import GroupKFoldMedical
from trustcv.splitters.temporal import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Your data
X = your_features  # Shape: (n_samples, n_features)
y = your_labels   # Shape: (n_samples,)

# Method 1: Standard stratified CV
cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, test_idx in cv.split(X, y):
    # Train your model
    model = RandomForestClassifier(random_state=42)
    model.fit(X[train_idx], y[train_idx])
    
    # Predict and evaluate
    y_pred = model.predict_proba(X[test_idx])[:, 1]
    score = roc_auc_score(y[test_idx], y_pred)
    scores.append(score)

print(f"CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# Method 2: Patient-grouped CV (when you have patient IDs)
patient_ids = your_patient_ids  # Shape: (n_samples,)
cv = GroupKFoldMedical(n_splits=5)

for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
    # Same training code as above
    # No patient will appear in both train and test
    pass

# Method 3: Temporal CV (when you have time-ordered data)
cv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in cv.split(X):
    # Always trains on past, tests on future
    # train_idx will always be < test_idx
    pass
```

### 2. With PyTorch Models

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from trustcv.splitters.grouped import GroupKFoldMedical

# Your PyTorch model
class MedicalNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Cross-validation with PyTorch
cv = GroupKFoldMedical(n_splits=5)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=patient_ids)):
    print(f"Fold {fold + 1}/5")
    
    # Create data loaders
    train_loader = DataLoader(
        your_dataset, 
        batch_size=32,
        sampler=SubsetRandomSampler(train_idx)
    )
    test_loader = DataLoader(
        your_dataset,
        batch_size=32, 
        sampler=SubsetRandomSampler(test_idx)
    )
    
    # Initialize model
    model = MedicalNet(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    model.train()
    for epoch in range(50):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predictions.extend(outputs.squeeze().numpy())
            true_labels.extend(batch_y.numpy())
    
    # Calculate score
    score = roc_auc_score(true_labels, predictions)
    cv_scores.append(score)

print(f"CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

### 3. With TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from trustcv.splitters.temporal import ExpandingWindowCV

# Your Keras model
def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'auc']
    )
    
    return model

# Cross-validation with Keras
cv = ExpandingWindowCV(min_train_size=100, forecast_horizon=50, step_size=25)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
    print(f"Fold {fold + 1}")
    
    # Prepare data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create and train model
    model = create_model(X.shape[1])
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    y_pred = model.predict(X_test).flatten()
    score = roc_auc_score(y_test, y_pred)
    cv_scores.append(score)

print(f"CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

### 4. With XGBoost

```python
import xgboost as xgb
from trustcv.splitters.iid import BootstrapValidation

# Cross-validation with XGBoost
cv = BootstrapValidation(n_iterations=100, estimator='.632', random_state=42)
cv_scores = []

for train_idx, test_idx in cv.split(X, y):
    # Prepare DMatrix
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
    dtest = xgb.DMatrix(X[test_idx], label=y[test_idx])
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Train model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        verbose_eval=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(dtest)
    score = roc_auc_score(y[test_idx], y_pred)
    cv_scores.append(score)

print(f"Bootstrap .632 CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

### 5. With LightGBM

```python
import lightgbm as lgb
from trustcv.splitters.spatial import SpatialBlockCV

# For spatial data with coordinates
coordinates = your_coordinates  # Shape: (n_samples, 2) - [longitude, latitude]

cv = SpatialBlockCV(n_splits=5, block_size=0.1)  # 0.1 degree blocks
cv_scores = []

for train_idx, test_idx in cv.split(X, coordinates=coordinates):
    # Prepare LightGBM datasets
    train_data = lgb.Dataset(X[train_idx], label=y[train_idx])
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Train model
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Predict and evaluate
    y_pred = model.predict(X[test_idx])
    score = roc_auc_score(y[test_idx], y_pred)
    cv_scores.append(score)

print(f"Spatial CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
```

### 6. Gradient Boosting with TrustCVValidator (Simplified)

XGBoost, LightGBM, and CatBoost all provide **sklearn-compatible APIs**, making them easy to use with `TrustCVValidator`:

```python
from trustcv import TrustCVValidator
from sklearn.datasets import make_classification
import numpy as np

# Create sample data with patient groups
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
patient_ids = np.repeat(np.arange(200), 5)  # 200 patients, 5 samples each

# Initialize validator with patient grouping
validator = TrustCVValidator(
    method='stratified_group_kfold',
    n_splits=5,
    check_leakage=True,
    check_balance=True,
    random_state=42
)

# --- XGBoost (sklearn API) ---
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

results_xgb = validator.validate(model=xgb_model, X=X, y=y, groups=patient_ids)
print("XGBoost Results:")
print(results_xgb.summary())

# --- LightGBM (sklearn API) ---
from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

results_lgbm = validator.validate(model=lgbm_model, X=X, y=y, groups=patient_ids)
print("\nLightGBM Results:")
print(results_lgbm.summary())

# --- CatBoost (sklearn API) ---
from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(
    iterations=100,
    depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)

results_catboost = validator.validate(model=catboost_model, X=X, y=y, groups=patient_ids)
print("\nCatBoost Results:")
print(results_catboost.summary())
```

**Why use sklearn-compatible API?**
- Simpler code - no manual training loops
- Automatic leakage and balance checks
- Consistent results format with confidence intervals
- Works with all TrustCV CV methods

### 7. With JAX/Flax (Neural Networks)

JAX is a high-performance ML framework with automatic differentiation and JIT compilation. Flax provides a high-level neural network API on top of JAX.

**Installation:**
```bash
# CPU only
pip install jax jaxlib flax optax

# GPU support (CUDA 12)
pip install jax[cuda12] flax optax

# See https://github.com/google/jax#installation for more options
```

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np

from trustcv.frameworks.jax import JAXAdapter, JAXCVRunner
from trustcv.splitters import StratifiedKFold, StratifiedGroupKFold

# Define a Flax MLP model
class MLP(nn.Module):
    hidden_dim: int = 64
    n_classes: int = 2

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.1, deterministic=not training)(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_classes)(x)
        return x

# Your data
X = your_features  # Shape: (n_samples, n_features)
y = your_labels    # Shape: (n_samples,)

# Method 1: Using JAXCVRunner (high-level API)
runner = JAXCVRunner(
    model_fn=lambda: MLP(hidden_dim=64, n_classes=2),
    cv_splitter=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    adapter=JAXAdapter(batch_size=32, seed=42)
)

results = runner.run(
    X, y,
    epochs=20,
    optimizer=optax.adam(1e-3)
)

print(results.summary())

# Method 2: With patient grouping (prevents leakage)
patient_ids = your_patient_ids  # Shape: (n_samples,)

runner = JAXCVRunner(
    model_fn=lambda: MLP(hidden_dim=64, n_classes=2),
    cv_splitter=StratifiedGroupKFold(n_splits=5),
    adapter=JAXAdapter(batch_size=32, seed=42)
)

results = runner.run(
    X, y,
    epochs=20,
    groups=patient_ids,  # No patient in both train/test
    optimizer=optax.adam(1e-3)
)

# Method 3: Low-level control with JAXAdapter
adapter = JAXAdapter(batch_size=32, seed=42, use_jit=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    # Create data splits
    train_data, val_data = adapter.create_data_splits((X, y), train_idx, val_idx)

    # Initialize model
    model = MLP(hidden_dim=64, n_classes=2)
    optimizer = optax.adam(1e-3)

    # Training loop
    state = None
    for epoch in range(20):
        result = adapter.train_epoch(model, train_data, optimizer=optimizer, state=state)
        state = result['state']

    # Evaluate
    metrics = adapter.evaluate(model, val_data, state=state)
    scores.append(metrics['val_acc'])
    print(f"Fold {fold + 1}: accuracy = {metrics['val_acc']:.4f}")

print(f"Mean accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
```

**JAX/Flax Key Concepts:**
- **Functional paradigm**: Models are stateless, parameters passed explicitly
- **TrainState**: Flax object that bundles params, apply_fn, and optimizer state
- **JIT compilation**: Use `use_jit=True` for significant speedups
- **PRNG keys**: JAX uses explicit random keys for reproducibility

**When to use JAX:**
- Large-scale neural networks requiring GPU/TPU acceleration
- Research requiring custom gradients or transformations
- When you need maximum performance with JIT compilation

## 🔧 Advanced Integration Patterns

### Hyperparameter Tuning with CV

```python
from trustcv.splitters.grouped import GroupKFoldMedical
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create a custom CV splitter for sklearn compatibility
class GroupKFoldWrapper:
    def __init__(self, groups, n_splits=5):
        self.groups = groups
        self.n_splits = n_splits
        self.cv = GroupKFoldMedical(n_splits=n_splits)
    
    def split(self, X, y=None):
        return self.cv.split(X, y, groups=self.groups)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Use with GridSearchCV
cv_wrapper = GroupKFoldWrapper(groups=patient_ids, n_splits=5)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv_wrapper,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Custom Evaluation Metrics

```python
from sklearn.metrics import precision_recall_curve, auc
from trustcv.splitters.temporal import PurgedKFoldCV

def custom_medical_evaluation(y_true, y_pred_proba):
    """Custom evaluation for ML"""
    # Standard metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # Precision-Recall AUC (better for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Medical-specific metrics
    y_pred = (y_pred_proba > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)  # Precision
    npv = tn / (tn + fn)
    
    return {
        'roc_auc': auc_score,
        'pr_auc': pr_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv
    }

# Use with any CV method
cv = PurgedKFoldCV(n_splits=5, purge_gap=10)
results = []

for train_idx, test_idx in cv.split(X):
    model = your_model()
    model.fit(X[train_idx], y[train_idx])
    y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
    
    fold_metrics = custom_medical_evaluation(y[test_idx], y_pred_proba)
    results.append(fold_metrics)

# Aggregate results
for metric in results[0].keys():
    scores = [r[metric] for r in results]
    print(f"{metric}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### Parallel Processing

```python
from joblib import Parallel, delayed
from trustcv.splitters.iid import RepeatedKFold

def train_and_evaluate_fold(train_idx, test_idx, X, y):
    """Function to train and evaluate one fold"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[train_idx], y[train_idx])
    y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
    return roc_auc_score(y[test_idx], y_pred_proba)

# Parallel cross-validation
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

scores = Parallel(n_jobs=-1)(
    delayed(train_and_evaluate_fold)(train_idx, test_idx, X, y)
    for train_idx, test_idx in cv.split(X, y)
)

print(f"Parallel CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

## 🏥 Medical-Specific Workflows

### ICU Patient Monitoring Pipeline

```python
from trustcv.splitters.temporal import RollingWindowCV
from trustcv import DataLeakageChecker

def icu_monitoring_pipeline(patient_data, vital_signs, outcomes, patient_ids, timestamps):
    """Complete pipeline for ICU patient deterioration prediction"""
    
    # 1. Data preparation
    X = np.column_stack([patient_data, vital_signs])
    y = outcomes
    
    # 2. Temporal CV (important for ICU data!)
    cv = RollingWindowCV(
        window_size=48,  # 48 hours of history
        forecast_horizon=6,  # Predict 6 hours ahead
        step_size=6  # Move 6 hours forward each time
    )
    
    # 3. Leakage checking
    checker = DataLeakageChecker()
    
    results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Check for temporal leakage
        report = checker.check(X, y, timestamps=timestamps)

        if report.has_leakage:
            print(f"⚠️ Warning: Temporal leakage detected in fold {fold}")
            continue
        
        # Train model
        model = your_icu_model()
        model.fit(X[train_idx], y[train_idx])
        
        # Evaluate
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        results.append(score)
        
        print(f"Fold {fold}: AUC = {score:.3f}")
    
    return results

# Usage
icu_scores = icu_monitoring_pipeline(
    patient_data=demographic_features,
    vital_signs=vital_sign_features, 
    outcomes=deterioration_labels,
    patient_ids=patient_identifiers,
    timestamps=measurement_times
)
```

### Multi-site Clinical Trial

```python
from trustcv.splitters.grouped import LeaveOneGroupOut, GroupKFoldMedical

def multisite_trial_analysis(X, y, site_ids):
    """Analyze generalization across clinical sites"""
    
    print("Multi-site Clinical Trial Analysis")
    print("=" * 50)
    
    # 1. Leave-One-Site-Out (test generalization to new sites)
    print("\n1. Leave-One-Site-Out CV (Generalization Test)")
    logo = LeaveOneGroupOut()
    site_scores = {}
    
    for train_idx, test_idx in logo.split(X, y, groups=site_ids):
        test_site = site_ids[test_idx[0]]
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X[train_idx], y[train_idx])
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        
        site_scores[test_site] = score
        print(f"  Test Site {test_site}: AUC = {score:.3f}")
    
    # 2. Group K-Fold (standard validation)
    print("\n2. Group K-Fold CV (Standard Validation)")
    cv = GroupKFoldMedical(n_splits=5)
    group_scores = []
    
    for train_idx, test_idx in cv.split(X, y, groups=site_ids):
        model = RandomForestClassifier(random_state=42)
        model.fit(X[train_idx], y[train_idx])
        y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
        score = roc_auc_score(y[test_idx], y_pred_proba)
        group_scores.append(score)
    
    print(f"  Mean AUC: {np.mean(group_scores):.3f} ± {np.std(group_scores):.3f}")
    
    # 3. Analysis
    print("\n3. Site Generalization Analysis")
    logo_mean = np.mean(list(site_scores.values()))
    logo_std = np.std(list(site_scores.values()))
    group_mean = np.mean(group_scores)
    
    print(f"  Leave-One-Site-Out: {logo_mean:.3f} ± {logo_std:.3f}")
    print(f"  Group K-Fold: {group_mean:.3f} ± {np.std(group_scores):.3f}")
    print(f"  Generalization gap: {group_mean - logo_mean:+.3f}")
    
    if group_mean - logo_mean > 0.05:
        print("  ⚠️ Significant generalization gap - model may not transfer well to new sites")
    else:
        print("  ✅ Good generalization across sites")
    
    return site_scores, group_scores

# Usage
site_results = multisite_trial_analysis(X, y, site_identifiers)
```

## 💡 Pro Tips

1. **Always validate your CV setup**:
   ```python
   # Check train/test contamination
   train_patients = set(patient_ids[train_idx])
   test_patients = set(patient_ids[test_idx])
   assert len(train_patients & test_patients) == 0, "Patient leakage detected!"
   ```

2. **Save CV splits for reproducibility**:
   ```python
   import pickle
   
   # Save splits
   cv_splits = list(cv.split(X, y, groups=patient_ids))
   with open('cv_splits.pkl', 'wb') as f:
       pickle.dump(cv_splits, f)
   
   # Load splits later
   with open('cv_splits.pkl', 'rb') as f:
       cv_splits = pickle.load(f)
   ```

3. **Monitor class balance in each fold**:
   ```python
   for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
       train_balance = y[train_idx].mean()
       test_balance = y[test_idx].mean()
       print(f"Fold {fold}: Train={train_balance:.2%}, Test={test_balance:.2%}")
   ```

4. **Use appropriate metrics for medical data**:
   ```python
   # For imbalanced medical data, use multiple metrics
   metrics = {
       'roc_auc': roc_auc_score(y_true, y_pred_proba),
       'pr_auc': average_precision_score(y_true, y_pred_proba),
       'sensitivity': recall_score(y_true, y_pred),
       'specificity': specificity_score(y_true, y_pred)  # Custom function
   }
   ```

Remember: The key to successful cross-validation is choosing the method that best matches your data structure and deployment scenario!