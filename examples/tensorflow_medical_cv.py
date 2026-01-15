#!/usr/bin/env python3
"""
TensorFlow/Keras Medical Cross-Validation Example

Demonstrates how to use trustcv with TensorFlow/Keras for medical classification
with proper patient-level cross-validation.

Designed to run on Google Colab with minimal resources.

Developed at SMAILE (Stockholm Medical AI and Learning Environments)
Karolinska Institutet - https://smile.ki.se

To run on Google Colab:
    !pip install trustcv
    # Then run this script
"""

import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Import trustcv components
from trustcv.splitters import (
    GroupKFold,
    StratifiedGroupKFold,
    RepeatedKFold,
    BootstrapValidation
)

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("TensorFlow/Keras Medical Cross-Validation with trustcv")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")


# =============================================================================
# 1. Create Synthetic Medical Dataset
# =============================================================================
def create_medical_dataset(n_samples=300, n_features=20):
    """
    Create synthetic medical dataset with patient grouping.

    This simulates a longitudinal study where each patient has multiple visits.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[0.65, 0.35],  # Slightly imbalanced
        random_state=42
    )

    # Create patient IDs (3 samples per patient on average)
    n_patients = n_samples // 3
    patient_ids = np.repeat(np.arange(n_patients), 3)[:n_samples]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X.astype(np.float32), y, patient_ids


# =============================================================================
# 2. Define Keras Model
# =============================================================================
def create_keras_model(input_dim, hidden_units=[64, 32], dropout_rate=0.3):
    """
    Create a simple neural network for medical classification.

    Designed to be small enough for quick training on Colab.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units[0], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_units[1], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# =============================================================================
# 3. Cross-Validation Functions
# =============================================================================
def run_cv_keras(X, y, patient_ids, cv_splitter, epochs=20, batch_size=32, verbose=0):
    """
    Run cross-validation with Keras model using trustcv splitter.

    Args:
        X: Features array
        y: Labels array
        patient_ids: Patient group identifiers
        cv_splitter: trustcv CV splitter
        epochs: Training epochs per fold
        batch_size: Batch size for training
        verbose: Keras verbosity level

    Returns:
        List of fold results with metrics
    """
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y, groups=patient_ids)):
        print(f"\n--- Fold {fold + 1} ---")

        # Verify no patient leakage
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        overlap = train_patients & val_patients

        if len(overlap) > 0:
            raise ValueError(f"Patient leakage detected! Overlapping: {overlap}")

        print(f"Train: {len(train_idx)} samples, {len(train_patients)} patients")
        print(f"Val: {len(val_idx)} samples, {len(val_patients)} patients")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Check for single class in validation
        if len(np.unique(y_val)) < 2:
            print("  Warning: Single class in validation set")
            continue

        # Create fresh model for each fold
        model = create_keras_model(input_dim=X.shape[1])

        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )

        # Evaluate
        y_pred_proba = model.predict(X_val, verbose=0).ravel()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except ValueError:
            auc = 0.5

        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'epochs_trained': len(history.history['loss'])
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Epochs: {metrics['epochs_trained']}")

        fold_results.append(metrics)

        # Clear session to free memory
        keras.backend.clear_session()

    return fold_results


def print_summary(results, method_name):
    """Print summary statistics for CV results."""
    if not results:
        print(f"\n{method_name}: No valid results")
        return

    accuracies = [r['accuracy'] for r in results]
    aucs = [r['auc'] for r in results]

    print(f"\n--- {method_name} Summary ---")
    print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
    print(f"AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"Folds completed: {len(results)}")


# =============================================================================
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":
    # Create dataset (small for Colab)
    print("\nCreating medical dataset...")
    X, y, patient_ids = create_medical_dataset(n_samples=300, n_features=20)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Patients: {len(np.unique(patient_ids))}")
    print(f"Class distribution: {np.bincount(y)}")

    # ---------------------------------------------------------------------
    # Example 1: Patient-Grouped K-Fold
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 1: Patient-Grouped K-Fold Cross-Validation")
    print("=" * 70)
    print("Ensures no patient appears in both train and validation sets")

    cv_grouped = GroupKFold(n_splits=5)
    results_grouped = run_cv_keras(
        X, y, patient_ids, cv_grouped,
        epochs=30, batch_size=32
    )
    print_summary(results_grouped, "Patient-Grouped K-Fold")

    # ---------------------------------------------------------------------
    # Example 2: Stratified Group K-Fold
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 2: Stratified Patient-Grouped K-Fold")
    print("=" * 70)
    print("Maintains class balance while respecting patient grouping")

    cv_stratified = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    results_stratified = run_cv_keras(
        X, y, patient_ids, cv_stratified,
        epochs=30, batch_size=32
    )
    print_summary(results_stratified, "Stratified Patient-Grouped K-Fold")

    # ---------------------------------------------------------------------
    # Example 3: Repeated K-Fold (without grouping - for comparison)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 3: Standard Repeated K-Fold (No Grouping)")
    print("=" * 70)
    print("WARNING: May cause patient leakage - shown for comparison only!")

    cv_repeated = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)

    # For repeated KFold, we don't pass groups (no patient protection)
    fold_results_repeated = []
    for fold, (train_idx, val_idx) in enumerate(cv_repeated.split(X, y)):
        if fold >= 3:  # Only show first 3 folds
            break

        # Check for patient overlap (demonstrating the problem)
        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        overlap = train_patients & val_patients

        if len(overlap) > 0:
            print(f"Fold {fold + 1}: {len(overlap)} patients in BOTH train and val!")

    print("\nThis demonstrates why grouped CV is essential for medical data!")

    # ---------------------------------------------------------------------
    # Example 4: Bootstrap Validation
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Example 4: Bootstrap .632 Validation")
    print("=" * 70)
    print("Good for small medical datasets with confidence intervals")

    cv_bootstrap = BootstrapValidation(n_iterations=10, estimator='.632', random_state=42)

    bootstrap_results = []
    for fold, (train_idx, val_idx) in enumerate(cv_bootstrap.split(X, y)):
        if fold >= 5:  # Only 5 iterations for demo
            break

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_keras_model(input_dim=X.shape[1])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        y_pred = (model.predict(X_val, verbose=0).ravel() >= 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        bootstrap_results.append({'accuracy': acc})

        keras.backend.clear_session()

    print_summary([{'accuracy': r['accuracy'], 'auc': 0.5} for r in bootstrap_results],
                 "Bootstrap .632")

    # ---------------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TensorFlow/Keras Cross-Validation Demo Complete!")
    print("=" * 70)

    print("\nKey Takeaways:")
    print("1. Use GroupKFold or StratifiedGroupKFold for patient-level validation")
    print("2. Never let same patient's data appear in both train and validation")
    print("3. Early stopping prevents overfitting")
    print("4. Clear Keras session between folds to manage memory")
    print("5. Bootstrap methods are great for small datasets")

    print("\nRecommended CV Methods for Medical Data:")
    print("- GroupKFold: Standard patient-level validation")
    print("- StratifiedGroupKFold: When class imbalance is a concern")
    print("- LeaveOneGroupOut: For external validation simulation")
    print("- BootstrapValidation: For confidence interval estimation")
