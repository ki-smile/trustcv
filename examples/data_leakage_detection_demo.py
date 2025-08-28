#!/usr/bin/env python3
"""
Data Leakage Detection Demo

This example demonstrates how trustcv automatically detects and prevents
various types of data leakage common in medical machine learning.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from trustcv import KFoldMedical
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Import trustcv components
from trustcv import (
    GroupKFoldMedical,
    PurgedKFoldCV,
    DataLeakageChecker,
    UniversalCVRunner
)

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("trustcv - Data Leakage Detection Demo")
print("=" * 80)
print("\nThis demo shows how data leakage can occur and how trustcv detects it.\n")

# ============================================================================
# SCENARIO 1: Patient Leakage (Most Common in Healthcare)
# ============================================================================
print("=" * 80)
print("SCENARIO 1: Patient Leakage Detection")
print("=" * 80)
print("\nCommon mistake: Multiple samples per patient split randomly\n")

# Create medical dataset with multiple samples per patient
n_patients = 100
samples_per_patient = 5
n_samples = n_patients * samples_per_patient

# Generate features
X, y = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    n_clusters_per_class=3,
    weights=[0.7, 0.3],
    random_state=42
)

# Create patient IDs - multiple samples per patient
patient_ids = np.repeat(np.arange(n_patients), samples_per_patient)
np.random.shuffle(patient_ids)  # Shuffle to mix patient samples

print(f"Dataset: {n_samples} samples from {n_patients} patients")
print(f"Each patient has {samples_per_patient} samples (e.g., multiple visits)\n")

# Initialize leakage checker
checker = DataLeakageChecker(verbose=True)

# Test 1: Standard K-Fold (WRONG - will have leakage)
print("❌ WRONG: Using standard K-Fold:")
print("-" * 40)
standard_cv = KFoldMedical(n_splits=5, shuffle=True, random_state=42)

leakage_found = False
for fold, (train_idx, test_idx) in enumerate(standard_cv.split(X)):
    # Check for patient leakage
    train_patients = patient_ids[train_idx]
    test_patients = patient_ids[test_idx]
    
    # Use the checker's method
    report = checker.check_cv_splits(
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        patient_ids_train=train_patients,
        patient_ids_test=test_patients
    )
    
    if report.has_leakage:
        leakage_found = True
        overlapping = set(train_patients) & set(test_patients)
        print(f"Fold {fold + 1}: ⚠️ LEAKAGE - {len(overlapping)} patients in both sets!")

if leakage_found:
    print("\n💡 Impact: Model learns patient-specific patterns, not disease patterns!")
    print("   Performance will be overestimated and won't generalize to new patients.\n")

# Test 2: Group K-Fold (CORRECT - no leakage)
print("✅ CORRECT: Using GroupKFoldMedical:")
print("-" * 40)
grouped_cv = GroupKFoldMedical(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(grouped_cv.split(X, y, groups=patient_ids)):
    train_patients = patient_ids[train_idx]
    test_patients = patient_ids[test_idx]
    
    report = checker.check_cv_splits(
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        patient_ids_train=train_patients,
        patient_ids_test=test_patients
    )
    
    if not report.has_leakage:
        print(f"Fold {fold + 1}: ✅ Clean - No patient overlap")

print("\n💡 Result: Each patient's data stays together - valid performance estimate!\n")

# ============================================================================
# SCENARIO 2: Temporal Leakage (Using Future to Predict Past)
# ============================================================================
print("=" * 80)
print("SCENARIO 2: Temporal Leakage Detection")
print("=" * 80)
print("\nCommon mistake: Random shuffling of time series data\n")

# Create temporal medical data (e.g., ICU monitoring)
n_days = 365
n_patients_per_day = 10
n_samples = n_days * n_patients_per_day

# Generate temporal features
X_temporal = np.random.randn(n_samples, 15)
y_temporal = np.random.randint(0, 2, n_samples)

# Create timestamps
base_date = datetime(2023, 1, 1)
timestamps = []
for day in range(n_days):
    current_date = base_date + timedelta(days=day)
    timestamps.extend([current_date] * n_patients_per_day)
timestamps = pd.Series(timestamps)

# Add temporal trend to make it realistic
for i in range(n_samples):
    day_num = i // n_patients_per_day
    X_temporal[i] += np.sin(day_num / 30) * 0.5  # Monthly pattern

print(f"Temporal dataset: {n_days} days of data, {n_patients_per_day} patients/day")
print(f"Date range: {timestamps.min().date()} to {timestamps.max().date()}\n")

# Test 1: Standard K-Fold with shuffle (WRONG - temporal leakage)
print("❌ WRONG: Using shuffled K-Fold on time series:")
print("-" * 40)
standard_cv = KFoldMedical(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(standard_cv.split(X_temporal)):
    train_times = timestamps.iloc[train_idx]
    test_times = timestamps.iloc[test_idx]
    
    report = checker.check_cv_splits(
        X_temporal[train_idx], X_temporal[test_idx],
        y_temporal[train_idx], y_temporal[test_idx],
        timestamps_train=train_times,
        timestamps_test=test_times
    )
    
    # Check temporal overlap
    if max(train_times) > min(test_times):
        days_overlap = (max(train_times) - min(test_times)).days
        print(f"Fold {fold + 1}: ⚠️ LEAKAGE - Training on future data ({days_overlap} days overlap)!")

print("\n💡 Impact: Model uses future information to predict past - impossible in practice!\n")

# Test 2: Purged K-Fold (CORRECT - with temporal gap)
print("✅ CORRECT: Using PurgedKFoldCV with gap:")
print("-" * 40)

# Convert timestamps to numeric for splitting
time_groups = np.array([t.toordinal() for t in timestamps])

purged_cv = PurgedKFoldCV(n_splits=5, purge_gap=7)  # 7-day gap

for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X_temporal, groups=time_groups)):
    train_times = timestamps.iloc[train_idx]
    test_times = timestamps.iloc[test_idx]
    
    gap_days = (min(test_times) - max(train_times)).days
    print(f"Fold {fold + 1}: ✅ Clean - {gap_days} day gap between train/test")

print("\n💡 Result: Temporal order preserved with safety gap - realistic evaluation!\n")

# ============================================================================
# SCENARIO 3: Preprocessing Leakage (Normalization Before Split)
# ============================================================================
print("=" * 80)
print("SCENARIO 3: Preprocessing Leakage Detection")
print("=" * 80)
print("\nCommon mistake: Normalizing data before train-test split\n")

# Create dataset
X_preprocessing, y_preprocessing = make_classification(
    n_samples=1000, n_features=10, random_state=42
)

# Test 1: Normalize BEFORE split (WRONG)
print("❌ WRONG: Normalizing entire dataset before splitting:")
print("-" * 40)

scaler_wrong = StandardScaler()
X_normalized_wrong = scaler_wrong.fit_transform(X_preprocessing)  # Fit on ALL data

# Now split
cv = KFoldMedical(n_splits=5)
train_idx, test_idx = next(cv.split(X_normalized_wrong))

# Check for preprocessing leakage
leakage_detected = checker.check_preprocessing_leakage(
    X_preprocessing,
    X_normalized_wrong,
    (train_idx, test_idx)
)

if leakage_detected:
    print("⚠️ PREPROCESSING LEAKAGE DETECTED!")
    print("   Test set statistics influenced training normalization")
    print("   This causes overly optimistic performance estimates\n")

# Test 2: Normalize AFTER split (CORRECT)
print("✅ CORRECT: Normalizing only training data:")
print("-" * 40)

# Split first
train_idx, test_idx = next(cv.split(X_preprocessing))

# Normalize training data only
scaler_correct = StandardScaler()
X_train_normalized = scaler_correct.fit_transform(X_preprocessing[train_idx])
X_test_normalized = scaler_correct.transform(X_preprocessing[test_idx])  # Only transform

print("✅ Clean - Scaler fitted only on training data")
print("   Test data transformed using training statistics")
print("   No information leakage!\n")

# ============================================================================
# SCENARIO 4: Duplicate Sample Detection
# ============================================================================
print("=" * 80)
print("SCENARIO 4: Duplicate Sample Detection")
print("=" * 80)
print("\nCommon issue: Duplicate samples in train and test sets\n")

# Create dataset with some duplicates
X_dup, y_dup = make_classification(n_samples=800, n_features=10, random_state=42)

# Intentionally duplicate some samples
duplicate_indices = np.random.choice(800, 200, replace=False)
X_with_dups = np.vstack([X_dup, X_dup[duplicate_indices]])
y_with_dups = np.hstack([y_dup, y_dup[duplicate_indices]])

print(f"Dataset: {len(X_with_dups)} samples ({len(duplicate_indices)} are duplicates)\n")

# Split with potential duplicates
cv = KFoldMedical(n_splits=5, shuffle=True)
train_idx, test_idx = next(cv.split(X_with_dups))

# Check for duplicates
report = checker.check_cv_splits(
    X_with_dups[train_idx], X_with_dups[test_idx],
    y_with_dups[train_idx], y_with_dups[test_idx]
)

if 'duplicate' in report.leakage_types:
    print("⚠️ DUPLICATE LEAKAGE DETECTED!")
    details = report.details['duplicate_leakage']
    print(f"   {details['duplicate_count']} duplicate samples found")
    print(f"   {details['duplicate_percentage']:.1f}% of test set are duplicates from training!\n")

# ============================================================================
# SCENARIO 5: Feature-Target Leakage (Target Information in Features)
# ============================================================================
print("=" * 80)
print("SCENARIO 5: Feature-Target Leakage Detection")
print("=" * 80)
print("\nDangerous mistake: Features that contain target information\n")

# Create dataset
X_clean, y_target = make_classification(n_samples=1000, n_features=10, random_state=42)

# Add a leaky feature (e.g., "diagnosis_code" that perfectly predicts the target)
X_leaky = np.column_stack([
    X_clean,
    y_target + np.random.normal(0, 0.01, len(y_target))  # Almost perfect correlation
])

feature_names = [f'feature_{i}' for i in range(10)] + ['diagnosis_code_leaked']
X_leaky_df = pd.DataFrame(X_leaky, columns=feature_names)

print("Checking for features with suspicious correlation to target...\n")

# Check for feature-target leakage
leakage_report = checker.check_feature_target_leakage(
    X_leaky_df, y_target, threshold=0.95
)

if leakage_report['has_leakage']:
    print("⚠️ FEATURE-TARGET LEAKAGE DETECTED!")
    for feat in leakage_report['suspicious_features']:
        print(f"   Feature '{feat['name']}' has {feat['correlation']:.3f} correlation with target")
    print("\n💡 These features likely contain information not available at prediction time!\n")

# ============================================================================
# PRACTICAL EXAMPLE: Complete Pipeline with Leakage Prevention
# ============================================================================
print("=" * 80)
print("PRACTICAL EXAMPLE: Safe Cross-Validation Pipeline")
print("=" * 80)

# Create realistic medical dataset
n_patients = 200
n_timepoints = 12  # Monthly data
n_features = 25

# Generate patient data
patient_data = []
for patient_id in range(n_patients):
    # Each patient has multiple timepoints
    for month in range(np.random.randint(3, n_timepoints)):
        features = np.random.randn(n_features)
        # Add patient-specific bias (simulating patient effects)
        features += np.random.randn() * 0.5
        
        patient_data.append({
            'patient_id': f'P{patient_id:04d}',
            'month': month,
            'timestamp': datetime(2023, 1, 1) + timedelta(days=month * 30),
            **{f'feature_{i}': features[i] for i in range(n_features)},
            'outcome': np.random.randint(0, 2)
        })

df = pd.DataFrame(patient_data)
print(f"\nRealistic dataset: {len(df)} samples from {df['patient_id'].nunique()} patients")
print(f"Time range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

# Prepare data
X = df[[f'feature_{i}' for i in range(n_features)]].values
y = df['outcome'].values
patient_ids = df['patient_id'].values
timestamps = df['timestamp'].values

# Safe cross-validation with automatic leakage detection
print("\n🔒 Running safe cross-validation with automatic checks:")
print("-" * 60)

# Use GroupKFold to prevent patient leakage
cv_safe = GroupKFoldMedical(n_splits=5)

# Track performance difference
unsafe_scores = []
safe_scores = []

# Compare unsafe vs safe CV
print("\nUnsafe CV (standard K-Fold):")
unsafe_cv = KFoldMedical(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in unsafe_cv.split(X):
    model = RandomForestClassifier(random_state=42)
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
    unsafe_scores.append(score)

print(f"  Mean accuracy: {np.mean(unsafe_scores):.3f} (+/- {np.std(unsafe_scores):.3f})")

print("\nSafe CV (GroupKFoldMedical):")
for train_idx, test_idx in cv_safe.split(X, y, groups=patient_ids):
    # Verify no leakage
    report = checker.check_cv_splits(
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        patient_ids_train=patient_ids[train_idx],
        patient_ids_test=patient_ids[test_idx],
        timestamps_train=timestamps[train_idx],
        timestamps_test=timestamps[test_idx]
    )
    
    if not report.has_leakage:
        model = RandomForestClassifier(random_state=42)
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[test_idx], y[test_idx])
        safe_scores.append(score)

print(f"  Mean accuracy: {np.mean(safe_scores):.3f} (+/- {np.std(safe_scores):.3f})")

performance_diff = np.mean(unsafe_scores) - np.mean(safe_scores)
print(f"\n⚠️ Performance overestimation due to leakage: {performance_diff:.3f} ({performance_diff/np.mean(safe_scores)*100:.1f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Data Leakage Detection in trustcv")
print("=" * 80)

print("""
Key Takeaways:
1. Patient Leakage: Most common in medical data - use GroupKFoldMedical
2. Temporal Leakage: Critical for time series - use PurgedKFoldCV
3. Preprocessing Leakage: Fit only on training data, never on full dataset
4. Duplicate Detection: Automatically detected by DataLeakageChecker
5. Feature-Target Leakage: Check for suspiciously high correlations

trustcv automatically:
✅ Detects multiple types of leakage
✅ Provides appropriate CV methods for each scenario
✅ Warns about suspicious patterns
✅ Ensures valid performance estimates

Remember: Leakage makes your model look better than it really is!
Always validate with proper CV methods for reliable, deployable models.
""")