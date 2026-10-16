#!/usr/bin/env python3
"""
Comprehensive demonstration of all cross-validation methods
from the trustcv package
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import all CV methods
from trustcv.splitters import (
    # I.I.D. methods (9)
    HoldOut, KFoldMedical, StratifiedKFoldMedical,
    RepeatedKFold, LOOCV, LPOCV, BootstrapValidation,
    MonteCarloCV, NestedCV,
    # Grouped methods (8)
    GroupKFoldMedical, StratifiedGroupKFold,
    LeaveOneGroupOut, LeavePGroupsOut, RepeatedGroupKFold,
    HierarchicalGroupKFold, MultilevelCV, NestedGroupedCV,
    # Temporal methods (8)
    TimeSeriesSplit, BlockedTimeSeries,
    RollingWindowCV, ExpandingWindowCV,
    PurgedKFoldCV, CombinatorialPurgedCV,
    PurgedGroupTimeSeriesSplit, NestedTemporalCV,
    # Spatial methods (4)
    SpatialBlockCV, BufferedSpatialCV,
    SpatiotemporalBlockCV, EnvironmentalHealthCV
)


def generate_medical_data(n_samples=1000, n_features=20, n_patients=100):
    """Generate synthetic medical data with various structures"""
    
    # Basic features and labels
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=3,
        random_state=42
    )
    
    # Add patient IDs (grouped structure)
    patient_ids = np.repeat(np.arange(n_patients), n_samples // n_patients)
    if len(patient_ids) < n_samples:
        patient_ids = np.concatenate([
            patient_ids,
            np.random.choice(n_patients, n_samples - len(patient_ids))
        ])
    
    # Add timestamps (temporal structure)
    base_date = pd.Timestamp('2020-01-01')
    timestamps = pd.date_range(base_date, periods=n_samples, freq='H')
    
    # Add spatial coordinates
    coordinates = np.random.randn(n_samples, 2) * 10
    
    return X, y, patient_ids, timestamps, coordinates


def demo_iid_methods(X, y):
    """Demonstrate I.I.D. cross-validation methods"""
    print("\n" + "="*60)
    print("I.I.D. CROSS-VALIDATION METHODS")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000)
    
    # 1. Hold-Out Validation
    print("\n1. Hold-Out Validation")
    cv = HoldOut(test_size=0.3, random_state=42)
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        print(f"   Test accuracy: {score:.3f}")
    
    # 2. K-Fold CV
    print("\n2. K-Fold Cross-Validation")
    cv = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Fold {fold}: {score:.3f}")
    print(f"   Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # 3. Stratified K-Fold
    print("\n3. Stratified K-Fold CV")
    cv = StratifiedKFoldMedical(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # 4. Repeated K-Fold
    print("\n4. Repeated K-Fold CV")
    cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
    print(f"   {len(scores)} iterations")
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # 5. Monte Carlo CV
    print("\n5. Monte Carlo CV")
    cv = MonteCarloCV(n_iterations=10, test_size=0.2, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # 6. Bootstrap Validation
    print("\n6. Bootstrap Validation")
    cv = BootstrapValidation(n_iterations=20, estimator='standard', random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
    print(f"   OOB accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # 7. LOOCV (Leave-One-Out)
    print("\n7. Leave-One-Out CV (LOOCV)")
    # Use small subset to avoid long runtime
    X_small, y_small = X[:50], y[:50]
    cv = LOOCV()
    scores = []
    for train_idx, test_idx in cv.split(X_small, y_small):
        model.fit(X_small[train_idx], y_small[train_idx])
        score = accuracy_score(y_small[test_idx], model.predict(X_small[test_idx]))
        scores.append(score)
    print(f"   {len(scores)} iterations (n samples)")
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # 8. LPOCV (Leave-P-Out)
    print("\n8. Leave-P-Out CV (LPOCV, p=2)")
    # Use very small subset due to combinatorial explosion
    X_tiny, y_tiny = X[:20], y[:20]
    cv = LPOCV(p=2)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_tiny, y_tiny)):
        if i >= 50:  # Limit iterations for demo
            break
        model.fit(X_tiny[train_idx], y_tiny[train_idx])
        score = accuracy_score(y_tiny[test_idx], model.predict(X_tiny[test_idx]))
        scores.append(score)
    print(f"   C(20,2) = {cv.get_n_splits(X_tiny)} total iterations")
    print(f"   First 50 iterations accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


def demo_grouped_methods(X, y, patient_ids):
    """Demonstrate grouped cross-validation methods"""
    print("\n" + "="*60)
    print("GROUPED CROSS-VALIDATION METHODS")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000)
    
    # 1. Patient Group K-Fold
    print("\n1. Patient Group K-Fold")
    cv = GroupKFoldMedical(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, patient_ids), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        n_train_patients = len(np.unique(patient_ids[train_idx]))
        n_test_patients = len(np.unique(patient_ids[test_idx]))
        print(f"   Fold {fold}: {score:.3f} (train: {n_train_patients} patients, test: {n_test_patients} patients)")
    print(f"   Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # 2. Stratified Group K-Fold
    print("\n2. Stratified Group K-Fold")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, patient_ids), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # 3. Leave-One-Group-Out
    print("\n3. Leave-One-Group-Out CV")
    cv = LeaveOneGroupOut()
    # Use only first 10 patients for demo (LOGO is expensive)
    mask = np.isin(patient_ids, range(10))
    X_sub, y_sub, groups_sub = X[mask], y[mask], patient_ids[mask]

    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_sub, y_sub, groups_sub)):
        if i >= 5:  # Limit to 5 iterations for demo
            break
        model.fit(X_sub[train_idx], y_sub[train_idx])
        score = accuracy_score(y_sub[test_idx], model.predict(X_sub[test_idx]))
        scores.append(score)
        test_group = groups_sub[test_idx][0]
        print(f"   Patient {test_group}: {score:.3f}")
    print(f"   Mean (first 5): {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # 4. Leave-P-Groups-Out
    print("\n4. Leave-P-Groups-Out CV (p=2)")
    cv = LeavePGroupsOut(n_groups=2)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_sub, y_sub, groups_sub)):
        if i >= 5:  # Limit iterations
            break
        model.fit(X_sub[train_idx], y_sub[train_idx])
        score = accuracy_score(y_sub[test_idx], model.predict(X_sub[test_idx]))
        scores.append(score)
        test_groups = np.unique(groups_sub[test_idx])
        print(f"   Groups {test_groups}: {score:.3f}")
    print(f"   C(10,2) = {cv.get_n_splits(groups=groups_sub)} total iterations")

    # 5. Repeated Group K-Fold
    print("\n5. Repeated Group K-Fold (2 repeats)")
    cv = RepeatedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X, y, patient_ids):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
    print(f"   {len(scores)} iterations (3 folds × 2 repeats)")
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # 6. Hierarchical Group K-Fold
    print("\n6. Hierarchical Group K-Fold")
    # Create hierarchy: hospital -> department -> patient
    n_samples = len(X)
    hospitals = patient_ids // 25  # 4 hospitals
    departments = patient_ids // 10  # 10 departments
    hierarchy = {
        'hospital': hospitals,
        'department': departments,
        'patient': patient_ids
    }
    cv = HierarchicalGroupKFold(n_splits=3, hierarchy_level='hospital', random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, hierarchy=hierarchy), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        test_hospitals = np.unique(hospitals[test_idx])
        print(f"   Fold {fold}: {score:.3f} (test hospitals: {test_hospitals})")
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # 7. Multi-level CV
    print("\n7. Multi-level CV (department level)")
    groups_dict = {
        'hospital': hospitals,
        'department': departments,
        'patient': patient_ids
    }
    cv = MultilevelCV(n_splits=3, validation_level='department')
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups_dict), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
    print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # 8. Nested Grouped CV (split demonstration only)
    print("\n8. Nested Grouped CV")
    outer_cv = GroupKFoldMedical(n_splits=3, random_state=42)
    inner_cv = GroupKFoldMedical(n_splits=2, random_state=42)
    nested_cv = NestedGroupedCV(outer_cv=outer_cv, inner_cv=inner_cv)
    print(f"   Outer splits: {nested_cv.get_n_splits()}")
    print("   (Use fit_predict() for full nested CV with hyperparameter tuning)")


def demo_temporal_methods(X, y, timestamps):
    """Demonstrate temporal cross-validation methods"""
    print("\n" + "="*60)
    print("TEMPORAL CROSS-VALIDATION METHODS")
    print("="*60)
    
    model = LogisticRegression(max_iter=1000)
    
    # 1. Temporal Clinical CV
    print("\n1. Temporal Clinical CV")
    cv = TimeSeriesSplit(n_splits=3, gap=7)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, timestamps=timestamps), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        train_period = f"{timestamps[train_idx].min().date()} to {timestamps[train_idx].max().date()}"
        test_period = f"{timestamps[test_idx].min().date()} to {timestamps[test_idx].max().date()}"
        print(f"   Fold {fold}: {score:.3f}")
        print(f"      Train: {train_period}")
        print(f"      Test:  {test_period}")
    
    # 2. Rolling Window CV
    print("\n2. Rolling Window CV")
    cv = RollingWindowCV(window_size=100, step_size=50, forecast_horizon=20)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if i >= 3:  # Limit iterations for demo
            break
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Window {i+1}: {score:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")
    
    # 3. Expanding Window CV
    print("\n3. Expanding Window CV")
    cv = ExpandingWindowCV(min_train_size=100, step_size=100, forecast_horizon=20)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if i >= 3:  # Limit iterations for demo
            break
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Window {i+1}: {score:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")
    
    # 4. Purged K-Fold CV
    print("\n4. Purged K-Fold CV")
    cv = PurgedKFoldCV(n_splits=3, purge_gap=10, embargo_pct=0.01)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, timestamps=timestamps), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Fold {fold}: {score:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")

    # 5. Combinatorial Purged CV
    print("\n5. Combinatorial Purged CV")
    cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2, purge_gap=5)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        if i >= 3:  # Limit iterations
            break
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Combination {i+1}: {score:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")

    # 6. Purged Group Time Series Split
    print("\n6. Purged Group Time Series Split")
    cv = PurgedGroupTimeSeriesSplit(n_splits=3, purge_gap=10, embargo_size=0.02)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, timestamps=timestamps), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Fold {fold}: {score:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")

    # 7. Nested Temporal CV (split demonstration only)
    print("\n7. Nested Temporal CV")
    outer_cv = ExpandingWindowCV(min_train_size=200, step_size=100, forecast_horizon=50)
    inner_cv = RollingWindowCV(window_size=100, step_size=50, forecast_horizon=20)
    nested_cv = NestedTemporalCV(outer_cv=outer_cv, inner_cv=inner_cv)
    print("   Outer: ExpandingWindowCV, Inner: RollingWindowCV")
    print("   (Use fit_predict() for full nested CV with hyperparameter tuning)")


def demo_spatial_methods(X, y, coordinates, timestamps):
    """Demonstrate spatial cross-validation methods"""
    print("\n" + "="*60)
    print("SPATIAL CROSS-VALIDATION METHODS")
    print("="*60)

    model = LogisticRegression(max_iter=1000)

    # 1. Spatial Block CV
    print("\n1. Spatial Block CV")
    cv = SpatialBlockCV(n_splits=4, block_shape='grid', random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, coordinates=coordinates), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        test_coords = coordinates[test_idx]
        print(f"   Block {fold}: {score:.3f}")
        print(f"      Test region center: ({test_coords[:, 0].mean():.1f}, {test_coords[:, 1].mean():.1f})")

    # 2. Buffered Spatial CV
    print("\n2. Buffered Spatial CV")
    cv = BufferedSpatialCV(n_splits=3, buffer_size=2.0, random_state=42)
    scores = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, coordinates=coordinates), 1):
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Block {fold}: {score:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")

    # 3. Spatiotemporal Block CV
    print("\n3. Spatiotemporal Block CV")
    cv = SpatiotemporalBlockCV(
        n_spatial_blocks=2, n_temporal_blocks=2,
        buffer_space=0, buffer_time=0
    )
    scores = []
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, coordinates=coordinates, timestamps=timestamps), 1
    ):
        if fold > 4:  # Limit iterations
            break
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Spatiotemporal block {fold}: {score:.3f}")

    # 4. Environmental Health CV
    print("\n4. Environmental Health CV (seasonal strategy)")
    # Create environmental data
    environmental_data = {
        'temperature': np.random.randn(len(X)) * 10 + 20,
        'humidity': np.random.randn(len(X)) * 20 + 60
    }
    cv = EnvironmentalHealthCV(
        spatial_blocks=3,
        temporal_strategy='seasonal',
        environmental_vars=['temperature'],
        buffer_config={'temperature': 5.0}
    )
    scores = []
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, coordinates=coordinates, timestamps=timestamps,
                 environmental_data=environmental_data), 1
    ):
        if fold > 4:  # Limit iterations
            break
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[test_idx], model.predict(X[test_idx]))
        scores.append(score)
        print(f"   Block {fold}: {score:.3f} (train: {len(train_idx)}, test: {len(test_idx)})")
    if scores:
        print(f"   Mean accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("COMPREHENSIVE CROSS-VALIDATION METHODS DEMONSTRATION")
    print("trustcv Package - All Methods from PDF")
    print("="*60)
    
    # Generate synthetic medical data
    print("\nGenerating synthetic medical data...")
    X, y, patient_ids, timestamps, coordinates = generate_medical_data(
        n_samples=1000, n_features=20, n_patients=100
    )
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Patients: {len(np.unique(patient_ids))}")
    print(f"  Time range: {timestamps.min().date()} to {timestamps.max().date()}")
    print(f"  Spatial extent: X[{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}], "
          f"Y[{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]")
    
    # Run demonstrations
    demo_iid_methods(X, y)
    demo_grouped_methods(X, y, patient_ids)
    demo_temporal_methods(X, y, timestamps)
    demo_spatial_methods(X, y, coordinates, timestamps)
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nAll 29 cross-validation methods have been implemented!")
    print("Each method handles different data structures and assumptions:")
    print("  - I.I.D. methods: Standard ML scenarios")
    print("  - Grouped methods: Patient-level data")
    print("  - Temporal methods: Time series and longitudinal data")
    print("  - Spatial methods: Geographic and environmental data")
    print("\nFor detailed documentation, see the Jupyter notebooks and API docs.")


if __name__ == "__main__":
    main()