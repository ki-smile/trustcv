# Data Leakage Detection in trustcv

## What is Data Leakage?

Data leakage occurs when information from the test set inadvertently influences the training process, leading to overly optimistic performance estimates that don't generalize to real-world data.

## Types of Data Leakage in Medical ML

### 1. Patient Leakage (Most Common in Healthcare)
**Problem**: Same patient's data appears in both training and test sets.
- Multiple samples per patient (repeated measurements, multiple images)
- Longitudinal studies with temporal data
- Multi-modal data (CT + MRI from same patient)

### 2. Temporal Leakage
**Problem**: Using future information to predict past events.
- Training on data from 2024 to predict outcomes in 2023
- Using post-treatment data to predict pre-treatment outcomes
- Time series without proper temporal gaps

### 3. Spatial Leakage
**Problem**: Spatially correlated samples in both sets.
- Adjacent tissue samples
- Neighboring geographic regions
- Overlapping image patches

### 4. Hierarchical Leakage
**Problem**: Related samples through hierarchical structure.
- Same hospital in train/test (hospital-specific practices)
- Same scanner/device (device-specific artifacts)
- Same clinical site or research group

## How trustcv Detects Leakage

### Core Detection Algorithm

> **Note:** The following shows the internal detection logic. The public API uses
> `checker.check()` which runs all checks automatically. See [Practical Examples](#practical-examples) below.

```python
# Internal implementation details (not public API)
from trustcv import DataLeakageChecker

class DataLeakageChecker:
    """
    Comprehensive data leakage detection for ML
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.leakage_report = {}
    
    def check_patient_leakage(self, train_idx, test_idx, patient_ids):
        """
        Detect if same patient appears in both train and test
        
        Algorithm:
        1. Extract patient IDs for training samples
        2. Extract patient IDs for test samples  
        3. Find intersection of patient IDs
        4. Report any overlaps
        """
        train_patients = np.unique(patient_ids[train_idx])
        test_patients = np.unique(patient_ids[test_idx])
        
        # Find overlapping patients
        overlapping = np.intersect1d(train_patients, test_patients)
        
        if len(overlapping) > 0:
            self.leakage_report['patient_leakage'] = {
                'detected': True,
                'overlapping_patients': overlapping.tolist(),
                'num_overlapping': len(overlapping),
                'severity': 'CRITICAL'
            }
            
            if self.verbose:
                print(f"⚠️ PATIENT LEAKAGE DETECTED!")
                print(f"   {len(overlapping)} patients in both train/test")
                print(f"   Patient IDs: {overlapping[:5]}...")
            
            return True
        return False
    
    def check_temporal_leakage(self, train_idx, test_idx, timestamps, 
                               min_gap=0):
        """
        Detect temporal leakage (future → past information flow)
        
        Algorithm:
        1. Get max timestamp in training set
        2. Get min timestamp in test set
        3. Check if test comes after train (with gap)
        4. Detect any temporal overlap
        """
        train_times = timestamps[train_idx]
        test_times = timestamps[test_idx]
        
        # Check for future information in training
        if np.max(train_times) >= np.min(test_times) - min_gap:
            overlap_start = np.min(test_times)
            overlap_end = np.max(train_times)
            
            self.leakage_report['temporal_leakage'] = {
                'detected': True,
                'train_max_time': np.max(train_times),
                'test_min_time': np.min(test_times),
                'overlap_period': [overlap_start, overlap_end],
                'severity': 'HIGH'
            }
            
            if self.verbose:
                print(f"⚠️ TEMPORAL LEAKAGE DETECTED!")
                print(f"   Training uses data up to t={np.max(train_times)}")
                print(f"   Test starts at t={np.min(test_times)}")
                print(f"   Required gap: {min_gap}, Actual: {np.min(test_times) - np.max(train_times)}")
            
            return True
        return False
    
    def check_spatial_leakage(self, train_idx, test_idx, coordinates, 
                             min_distance=0):
        """
        Detect spatial leakage (adjacent/overlapping regions)
        
        Algorithm:
        1. Calculate pairwise distances between train/test samples
        2. Find minimum distance
        3. Check if less than threshold
        4. Identify close sample pairs
        """
        from scipy.spatial.distance import cdist
        
        train_coords = coordinates[train_idx]
        test_coords = coordinates[test_idx]
        
        # Calculate distances
        distances = cdist(train_coords, test_coords)
        min_dist = np.min(distances)
        
        # Find samples that are too close
        close_pairs = np.where(distances < min_distance)
        num_close = len(close_pairs[0])
        
        if num_close > 0:
            self.leakage_report['spatial_leakage'] = {
                'detected': True,
                'min_distance': min_dist,
                'required_distance': min_distance,
                'num_close_pairs': num_close,
                'severity': 'MEDIUM'
            }
            
            if self.verbose:
                print(f"⚠️ SPATIAL LEAKAGE DETECTED!")
                print(f"   {num_close} sample pairs closer than {min_distance}")
                print(f"   Minimum distance: {min_dist:.2f}")
            
            return True
        return False
    
    def check_group_leakage(self, train_idx, test_idx, groups, 
                           group_type="generic"):
        """
        Detect group-based leakage (hospital, scanner, site)
        
        Algorithm:
        1. Identify unique groups in train and test
        2. Find overlapping groups
        3. Calculate contamination percentage
        """
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])
        
        overlapping = np.intersect1d(train_groups, test_groups)
        
        if len(overlapping) > 0:
            # Calculate contamination
            train_contaminated = np.sum(np.isin(groups[train_idx], overlapping))
            test_contaminated = np.sum(np.isin(groups[test_idx], overlapping))
            
            self.leakage_report[f'{group_type}_leakage'] = {
                'detected': True,
                'overlapping_groups': overlapping.tolist(),
                'train_contamination': train_contaminated / len(train_idx),
                'test_contamination': test_contaminated / len(test_idx),
                'severity': 'HIGH' if group_type in ['patient', 'hospital'] else 'MEDIUM'
            }
            
            return True
        return False
```

## Practical Examples

### Example 1: Detecting Patient Leakage in Medical Imaging

```python
import numpy as np
from trustcv import GroupKFoldMedical, DataLeakageChecker

# Simulate medical imaging dataset
n_images = 1000
n_patients = 200

# Multiple images per patient (common in medical imaging!)
patient_ids = np.repeat(np.arange(n_patients), n_images // n_patients)
np.random.shuffle(patient_ids)

X = np.random.randn(n_images, 224, 224, 3)
y = np.random.randint(0, 2, n_images)

# WRONG: Standard K-Fold (will have leakage)
from sklearn.model_selection import KFold
standard_cv = KFold(n_splits=5)

# Detect patient leakage using the public API
checker = DataLeakageChecker()
report = checker.check(X, y, groups=patient_ids, n_splits=5)
print(report.summary)

# CORRECT: Group K-Fold (no leakage)
grouped_cv = GroupKFoldMedical(n_splits=5)
print("\nGroup K-Fold Medical:")
for fold, (train_idx, test_idx) in enumerate(grouped_cv.split(X, y, groups=patient_ids)):
    pass  # No leakage when using GroupKFoldMedical with patient_ids
```

### Example 2: Temporal Leakage in Clinical Time Series

```python
from trustcv import PurgedKFoldCV, TimeSeriesSplit

# ICU monitoring data
n_timepoints = 365  # Daily data for a year
n_patients = 100
n_samples = n_timepoints * n_patients

timestamps = np.repeat(np.arange(n_timepoints), n_patients)
patient_ids = np.tile(np.arange(n_patients), n_timepoints)
X = np.random.randn(n_samples, 20)
y = np.random.randint(0, 2, n_samples)

# WRONG: Standard K-Fold on time series
# Detect temporal leakage using the public API
checker = DataLeakageChecker()
report = checker.check(X, y, timestamps=timestamps)
print(report.summary)

# CORRECT: Purged K-Fold with gap
purged_cv = PurgedKFoldCV(n_splits=5, purge_gap=7)  # 7-day gap
print("\nPurged K-Fold CV:")
for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X, y, groups=timestamps)):
    pass  # No temporal leakage when using PurgedKFoldCV with purge gap
```

### Example 3: Spatial Leakage in Pathology Images

```python
from trustcv import SpatialBlockCV, BufferedSpatialCV

# Whole slide imaging with spatial coordinates
n_patches = 500
coordinates = np.random.randn(n_patches, 2) * 100  # x, y positions
X = np.random.randn(n_patches, 2048)  # Deep features
y = np.random.randint(0, 2, n_patches)

# WRONG: Random split ignores spatial correlation
# Detect spatial leakage using the public API
checker = DataLeakageChecker()
report = checker.check(X, y, coordinates=coordinates)
print(report.summary)

# CORRECT: Spatial blocking with buffer
spatial_cv = BufferedSpatialCV(
    n_splits=5,
    spatial_coordinates=coordinates,
    buffer_size=10  # 10-unit buffer between regions
)

print("\nBuffered Spatial CV:")
for fold, (train_idx, test_idx) in enumerate(spatial_cv.split(X)):
    pass  # No spatial leakage when using BufferedSpatialCV with buffer
```

### Example 4: Hierarchical Leakage in Multi-Center Studies

```python
from trustcv import HierarchicalGroupKFold

# Multi-center clinical trial
n_countries = 5
n_hospitals_per_country = 4  
n_patients_per_hospital = 50
n_samples_per_patient = 3

# Create hierarchical structure
countries = []
hospitals = []
patients = []

for country in range(n_countries):
    for hospital in range(n_hospitals_per_country):
        hospital_id = f"H{country}_{hospital}"
        for patient in range(n_patients_per_hospital):
            patient_id = f"P{country}_{hospital}_{patient}"
            for sample in range(n_samples_per_patient):
                countries.append(country)
                hospitals.append(hospital_id)
                patients.append(patient_id)

countries = np.array(countries)
hospitals = np.array(hospitals)
patients = np.array(patients)

# Check for multiple levels of leakage using the public API
checker = DataLeakageChecker()

# Check at patient level
X_dummy = np.random.randn(len(patients), 10)
y_dummy = np.random.randint(0, 2, len(patients))

report = checker.check(X_dummy, y_dummy, groups=patients, n_splits=5)
print("Patient-level leakage check:")
print(report.summary)

# Correct: Hierarchical grouping
hierarchical_cv = HierarchicalGroupKFold(
    n_splits=5,
    hierarchy_levels=['country', 'hospital', 'patient']
)

print("\nHierarchical Group K-Fold:")
# Would properly separate at chosen hierarchy level
```

## Automated Leakage Detection

### Using the Universal CV Runner

> **Note:** The following shows the conceptual callback pattern, not the exact API.
> In practice, `DataLeakageChecker.check()` runs all relevant checks automatically.

```python
from trustcv import UniversalCVRunner, DataLeakageChecker

# Conceptual callback pattern (illustrative pseudocode)
class LeakageDetectionCallback(CVCallback):
    """
    Automatically detect and report leakage during CV
    """
    def __init__(self, patient_ids=None, timestamps=None,
                 coordinates=None, groups=None):
        self.patient_ids = patient_ids
        self.timestamps = timestamps
        self.coordinates = coordinates
        self.groups = groups
        self.checker = DataLeakageChecker()

    def on_fold_start(self, fold_idx, train_idx, val_idx):
        # The checker.check() method runs all relevant checks automatically
        report = self.checker.check_cv_splits(
            X_train=X[train_idx], X_test=X[val_idx],
            groups=self.patient_ids,
            timestamps=self.timestamps,
            coordinates=self.coordinates
        )
        if report.has_leakage:
            print(f"⚠️ WARNING: Data leakage detected in fold {fold_idx + 1}!")
            print(report.summary)
        else:
            print(f"✅ No data leakage detected in fold {fold_idx + 1}!")

# Use in practice - simplest approach with the public API
checker = DataLeakageChecker()
report = checker.check(
    X, y,
    groups=patient_ids,
    timestamps=timestamps,
    coordinates=spatial_coords,
    n_splits=5
)
print(report.summary)
```

## Prevention Strategies

### 1. Always Use Patient Grouping
```python
# ALWAYS group by patient ID when you have multiple samples per patient
cv = GroupKFoldMedical(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
    # Safe from patient leakage
    pass
```

### 2. Add Temporal Gaps
```python
# Add purge gap for time series
cv = PurgedKFoldCV(n_splits=5, purge_gap=30)  # 30-day gap
```

### 3. Use Spatial Buffers
```python
# Add buffer zones for spatial data
cv = BufferedSpatialCV(buffer_size=100)  # 100m buffer
```

### 4. Respect Hierarchical Structure
```python
# Group by highest level (e.g., hospital)
cv = GroupKFoldMedical(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups=hospital_ids):
    # Entire hospitals are kept together
    pass
```

## Common Pitfalls and Solutions

| Pitfall | Consequence | Solution |
|---------|-------------|----------|
| Using patient's left and right eye images in different sets | Model learns patient-specific features | Group by patient ID |
| Training on 2023 data, testing on 2022 | Impossible in practice | Use TimeSeriesSplit |
| Adjacent tissue samples in train/test | Spatial correlation inflates performance | Use SpatialBlockCV |
| Same MRI scanner in all training data | Model learns scanner artifacts | Stratify by scanner |
| Data augmentation before splitting | Augmented versions in both sets | Augment after splitting |

## Summary

trustcv's leakage detection:
1. **Automatically checks** for patient, temporal, spatial, and group leakage
2. **Reports severity** and specific violations
3. **Integrates with CV** through callbacks
4. **Prevents common mistakes** in ML
5. **Ensures valid** performance estimates

This comprehensive approach helps ensure your model validation reflects real-world performance!