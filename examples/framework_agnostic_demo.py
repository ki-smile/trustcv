#!/usr/bin/env python3
"""
Framework-Agnostic Cross-Validation Demo

Demonstrates how trustcv v2.0 works seamlessly across different ML frameworks
while maintaining best practices and regulatory compliance.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Import trustcv components
from trustcv import (
    UniversalCVRunner,
    GroupKFoldMedical,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger
)

# Set random seed
np.random.seed(42)

def create_medical_data(n_samples=1000, n_features=20):
    """Create synthetic medical dataset with patient IDs"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=3,
        weights=[0.7, 0.3],  # Imbalanced classes
        random_state=42
    )
    
    # Create patient IDs (some patients have multiple samples)
    patient_ids = np.repeat(np.arange(n_samples // 3), 3)[:n_samples]
    
    return X, y, patient_ids

print("=" * 70)
print("trustcv v2.0 - Framework-Agnostic Cross-Validation Demo")
print("=" * 70)

# Create dataset
X, y, patient_ids = create_medical_data()
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Unique patients: {len(np.unique(patient_ids))}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================================
# Example 1: Scikit-learn (Backward Compatible)
# ============================================================================
print("\n" + "=" * 70)
print("Example 1: Scikit-learn Cross-Validation (Backward Compatible)")
print("=" * 70)

# Traditional scikit-learn workflow still works exactly as before
cv = GroupKFoldMedical(n_splits=5)

# Manual cross-validation (traditional approach)
print("\nManual cross-validation with scikit-learn:")
scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=patient_ids)):
    # Ensure no patient overlap
    train_patients = patient_ids[train_idx]
    val_patients = patient_ids[val_idx]
    assert len(np.intersect1d(train_patients, val_patients)) == 0, "Patient leakage!"
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
    scores.append(score)
    print(f"  Fold {fold + 1}: Accuracy = {score:.3f}")

print(f"Mean accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# ============================================================================
# Example 2: Universal Runner with Auto-Detection
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Universal Runner with Automatic Framework Detection")
print("=" * 70)

# Create universal runner
runner = UniversalCVRunner(
    cv_splitter=GroupKFoldMedical(n_splits=5),
    framework='auto',  # Auto-detect framework
    verbose=1
)

# Run with scikit-learn model
print("\nRunning with RandomForest (auto-detects sklearn):")
results = runner.run(
    model=RandomForestClassifier(n_estimators=100, random_state=42),
    data=(X, y),
    groups=patient_ids
)

# ============================================================================
# Example 3: PyTorch Integration (if available)
# ============================================================================
try:
    import torch
    import torch.nn as nn
    
    print("\n" + "=" * 70)
    print("Example 3: PyTorch Integration")
    print("=" * 70)
    
    # Define PyTorch model
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x
    
    # Model factory function
    def create_pytorch_model():
        return SimpleNN(input_dim=X.shape[1])
    
    # Standardize data for neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use specific PyTorch runner
    from trustcv import TorchCVRunner
    from trustcv.frameworks.pytorch import PyTorchAdapter
    
    torch_runner = TorchCVRunner(
        model_fn=create_pytorch_model,
        cv_splitter=GroupKFoldMedical(n_splits=3),  # Fewer folds for speed
        adapter=PyTorchAdapter(batch_size=32)
    )
    
    print("\nRunning PyTorch model with patient-grouped CV:")
    torch_results = torch_runner.run(
        dataset=(X_scaled, y),
        epochs=20,
        optimizer_fn=lambda m: torch.optim.Adam(m.parameters(), lr=0.001),
        loss_fn=nn.CrossEntropyLoss(),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5),
            ProgressLogger(verbose=1)
        ],
        groups=patient_ids
    )
    
except ImportError:
    print("\n[PyTorch not installed - skipping PyTorch example]")

# ============================================================================
# Example 4: TensorFlow/Keras Integration (if available)
# ============================================================================
try:
    import tensorflow as tf
    from tensorflow import keras
    
    print("\n" + "=" * 70)
    print("Example 4: TensorFlow/Keras Integration")
    print("=" * 70)
    
    # Define Keras model
    def create_keras_model():
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
        return model
    
    # Use specific Keras runner
    from trustcv import KerasCVRunner
    
    keras_runner = KerasCVRunner(
        model_fn=create_keras_model,
        cv_splitter=GroupKFoldMedical(n_splits=3),
        compile_kwargs={
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy',
            'metrics': ['accuracy']
        }
    )
    
    print("\nRunning Keras model with patient-grouped CV:")
    keras_results = keras_runner.run(
        X=X_scaled,
        y=y,
        epochs=20,
        batch_size=32,
        groups=patient_ids,
        verbose=0  # Suppress Keras output
    )
    
    print(f"\nKeras CV Results:")
    for key, value in keras_results.mean_score.items():
        print(f"  {key}: {value:.3f} (+/- {keras_results.std_score[key]:.3f})")
    
except ImportError:
    print("\n[TensorFlow not installed - skipping TensorFlow example]")

# ============================================================================
# Example 5: Framework Comparison
# ============================================================================
print("\n" + "=" * 70)
print("Example 5: Comparing Different Frameworks")
print("=" * 70)

# Use the same CV splitter for fair comparison
cv_splitter = GroupKFoldMedical(n_splits=5)
runner = UniversalCVRunner(cv_splitter=cv_splitter, verbose=0)

# Compare different models
models_to_compare = [
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting", None),  # Will be imported if available
    ("XGBoost", None),  # Will be imported if available
]

# Try to add gradient boosting
try:
    from sklearn.ensemble import GradientBoostingClassifier
    models_to_compare[1] = ("Gradient Boosting", 
                            GradientBoostingClassifier(n_estimators=100, random_state=42))
except ImportError:
    pass

# Try to add XGBoost
try:
    import xgboost as xgb
    models_to_compare[2] = ("XGBoost", 
                           xgb.XGBClassifier(n_estimators=100, random_state=42, 
                                           use_label_encoder=False, eval_metric='logloss'))
except ImportError:
    pass

print("\nComparing models with patient-grouped cross-validation:")
print("-" * 50)

for name, model in models_to_compare:
    if model is not None:
        results = runner.run(
            model=model,
            data=(X, y),
            groups=patient_ids
        )
        
        # Get the main score (usually accuracy for classification)
        scores = results.scores
        accuracies = [s.get('score', s.get('val_accuracy', 0)) for s in scores]
        
        print(f"{name:20} | Accuracy: {np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f})")

# ============================================================================
# Example 6: Best Practices with Callbacks
# ============================================================================
print("\n" + "=" * 70)
print("Example 6: Best Practices with Callbacks and Monitoring")
print("=" * 70)

from trustcv.core.callbacks import RegulatoryComplianceLogger

# Create callbacks for best practices
callbacks = [
    ProgressLogger(log_file='cv_progress.json', verbose=1),
    RegulatoryComplianceLogger(
        output_dir='./regulatory_logs',
        study_name='framework_agnostic_demo'
    )
]

print("\nRunning with regulatory compliance logging:")
runner = UniversalCVRunner(
    cv_splitter=GroupKFoldMedical(n_splits=3),
    verbose=1
)

results = runner.run(
    model=RandomForestClassifier(n_estimators=50, random_state=42),
    data=(X, y),
    groups=patient_ids,
    callbacks=callbacks
)

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\nKey Takeaways:")
print("1. trustcv v2.0 works with any ML framework")
print("2. Backward compatibility maintained - existing code still works")
print("3. Automatic framework detection simplifies usage")
print("4. Patient grouping prevents data leakage across all frameworks")
print("5. Callbacks enable monitoring and regulatory compliance")
print("6. Best practices are enforced by default")