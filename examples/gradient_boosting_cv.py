"""
Gradient Boosting Cross-Validation with TrustCV

This example demonstrates how to use XGBoost, LightGBM, and CatBoost
with TrustCV for medical/healthcare ML applications.

All three libraries provide sklearn-compatible APIs, making them easy
to use with TrustCVValidator without any custom adapters.

Requirements:
    pip install xgboost lightgbm catboost

Usage:
    python gradient_boosting_cv.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

# Check which libraries are available
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"XGBoost not available: {type(e).__name__}")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"LightGBM not available: {type(e).__name__}")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"CatBoost not available: {type(e).__name__}")

from trustcv import TrustCVValidator


def create_medical_dataset(n_patients=200, samples_per_patient=5, n_features=20):
    """
    Create a synthetic medical dataset with patient grouping.

    Returns:
        X: Feature matrix
        y: Labels
        patient_ids: Patient group identifiers
    """
    n_samples = n_patients * samples_per_patient

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=3,
        weights=[0.7, 0.3],  # Imbalanced classes
        random_state=42
    )

    # Create patient IDs (each patient has multiple samples)
    patient_ids = np.repeat(np.arange(n_patients), samples_per_patient)

    return X, y, patient_ids


def compare_gradient_boosting_models():
    """
    Compare XGBoost, LightGBM, and CatBoost using TrustCV.
    """
    print("=" * 60)
    print("Gradient Boosting Cross-Validation with TrustCV")
    print("=" * 60)

    # Create dataset
    print("\n1. Creating synthetic medical dataset...")
    X, y, patient_ids = create_medical_dataset()
    print(f"   - Samples: {len(X)}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Patients: {len(np.unique(patient_ids))}")
    print(f"   - Class distribution: {np.bincount(y)}")

    # Initialize validator with patient grouping
    print("\n2. Initializing TrustCVValidator with patient-grouped CV...")
    validator = TrustCVValidator(
        method='stratified_group_kfold',
        n_splits=5,
        check_leakage=True,
        check_balance=True,
        random_state=42
    )

    results = {}

    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\n3. Evaluating XGBoost...")
        print("-" * 40)

        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )

        results['XGBoost'] = validator.validate(
            model=xgb_model,
            X=X,
            y=y,
            groups=patient_ids
        )
        print(results['XGBoost'].summary())

    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n4. Evaluating LightGBM...")
        print("-" * 40)

        lgbm_model = LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        results['LightGBM'] = validator.validate(
            model=lgbm_model,
            X=X,
            y=y,
            groups=patient_ids
        )
        print(results['LightGBM'].summary())

    # CatBoost
    if CATBOOST_AVAILABLE:
        print("\n5. Evaluating CatBoost...")
        print("-" * 40)

        catboost_model = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=False
        )

        results['CatBoost'] = validator.validate(
            model=catboost_model,
            X=X,
            y=y,
            groups=patient_ids
        )
        print(results['CatBoost'].summary())

    # Summary comparison
    if results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"\n{'Model':<12} {'ROC-AUC':<15} {'Accuracy':<15} {'F1':<15}")
        print("-" * 57)

        for name, result in results.items():
            metrics = result.metrics
            auc = metrics.get('roc_auc', {}).get('mean', 'N/A')
            acc = metrics.get('accuracy', {}).get('mean', 'N/A')
            f1 = metrics.get('f1', {}).get('mean', 'N/A')

            if isinstance(auc, float):
                auc = f"{auc:.4f}"
            if isinstance(acc, float):
                acc = f"{acc:.4f}"
            if isinstance(f1, float):
                f1 = f"{f1:.4f}"

            print(f"{name:<12} {auc:<15} {acc:<15} {f1:<15}")

        print("\nAll models evaluated with:")
        print("  - 5-fold Stratified Group K-Fold CV")
        print("  - Patient-level grouping (no patient in both train/test)")
        print("  - Automatic data leakage detection")
        print("  - Class balance verification")

    return results


def temporal_cv_example():
    """
    Example using temporal CV with gradient boosting for time-series medical data.
    """
    print("\n" + "=" * 60)
    print("Temporal Cross-Validation Example")
    print("=" * 60)

    # Create temporal dataset
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=42)
    timestamps = np.arange(n_samples)  # Sequential timestamps

    # Validator with temporal CV
    validator = TrustCVValidator(
        method='time_series_split',
        n_splits=5,
        check_leakage=True,
        random_state=42
    )

    if LIGHTGBM_AVAILABLE:
        print("\nEvaluating LightGBM with Temporal CV...")

        model = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbose=-1
        )

        results = validator.validate(model=model, X=X, y=y)
        print(results.summary())

        return results
    else:
        print("LightGBM not available for this example.")
        return None


if __name__ == "__main__":
    # Main comparison
    results = compare_gradient_boosting_models()

    # Temporal CV example
    temporal_results = temporal_cv_example()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
