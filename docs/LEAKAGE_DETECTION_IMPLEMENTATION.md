> **Note:** This document describes internal implementation details. The public API uses `DataLeakageChecker.check()` which orchestrates all detection types automatically. See [DATA_LEAKAGE_DETECTION.md](DATA_LEAKAGE_DETECTION.md) for usage examples.

# Data Leakage Detection - Implementation Details

## Core Architecture

The leakage detection in trustcv is implemented through the `DataLeakageChecker` class in `trustcv/checkers/leakage.py`. Here's how each type of leakage is detected:

## 1. Patient Leakage Detection

### Algorithm: Set Intersection
```python
def _check_patient_leakage(self, patient_ids_train, patient_ids_test):
    """
    Core algorithm: Find patients that appear in both sets
    """
    # Step 1: Convert to sets for O(1) lookup
    train_patients = set(patient_ids_train)
    test_patients = set(patient_ids_test)
    
    # Step 2: Find intersection - patients in BOTH sets
    overlap = train_patients.intersection(test_patients)
    
    # Step 3: Calculate metrics
    result = {
        'has_leakage': len(overlap) > 0,
        'overlapping_patients': list(overlap)[:10],  # First 10 for inspection
        'overlap_count': len(overlap),
        'overlap_percentage': len(overlap) / len(train_patients.union(test_patients)) * 100
    }
    
    return result
```

**How it works:**
- **Time Complexity**: O(n) where n is number of samples
- **Space Complexity**: O(p) where p is number of unique patients
- Uses Python sets for efficient intersection operation
- Returns specific patient IDs that are violating the split

### Example Detection:
```python
# Dataset with 3 patients, 5 samples each
patient_ids = ['P001', 'P001', 'P001', 'P001', 'P001',  # Patient 1
               'P002', 'P002', 'P002', 'P002', 'P002',  # Patient 2  
               'P003', 'P003', 'P003', 'P003', 'P003']  # Patient 3

# Bad split - Patient P001 in both
train_idx = [0, 1, 2, 5, 6, 7, 10, 11]  # Has P001, P002, P003
test_idx = [3, 4, 8, 9, 12, 13, 14]     # Has P001, P002, P003

# Detection finds: P001, P002, P003 all appear in both sets!
```

## 2. Temporal Leakage Detection

### Algorithm: Timestamp Comparison
```python
def _check_temporal_leakage(self, timestamps_train, timestamps_test):
    """
    Core algorithm: Check if test data comes before training data
    """
    # Step 1: Convert to datetime if needed
    timestamps_train = pd.to_datetime(timestamps_train)
    timestamps_test = pd.to_datetime(timestamps_test)
    
    # Step 2: Find temporal boundaries
    min_train = timestamps_train.min()
    max_train = timestamps_train.max()
    min_test = timestamps_test.min()
    max_test = timestamps_test.max()
    
    # Step 3: Detect violations
    violations = {
        # Test should come AFTER training
        'has_overlap': min_test < max_train,  # Test starts before training ends
        'test_before_train': min_test < min_train,  # Test entirely before train
        'future_in_train': max_train > max_test,  # Training uses future data
    }
    
    # Step 4: Calculate gap
    temporal_gap = (min_test - max_train).days if min_test > max_train else 0
    
    return {
        'has_leakage': any(violations.values()),
        'violations': violations,
        'temporal_gap_days': temporal_gap,
        'train_period': f"{min_train} to {max_train}",
        'test_period': f"{min_test} to {max_test}"
    }
```

**How it works:**
- Compares temporal boundaries of train/test sets
- Detects if test data occurs before or overlaps with training data
- Calculates temporal gap between sets

### Visual Representation:
```
Good Split (No Leakage):
Train: [Jan|Feb|Mar|Apr|May]     Gap     Test: [Jul|Aug|Sep]
       └─────────────┘                        └──────┘
         max_train < min_test ✓

Bad Split (Leakage):
Train: [Jan|Feb|Mar|Apr|May|Jun|Jul]
              Test: [Apr|May|Jun|Jul|Aug]
                    └── Overlap ──┘
       max_train > min_test ✗
```

## 3. Duplicate Sample Detection

### Algorithm: Hash-Based Comparison
```python
def _check_duplicate_samples(self, X_train, X_test):
    """
    Core algorithm: Hash each row and find duplicates
    """
    # Step 1: Convert to DataFrame for hashing
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    
    # Step 2: Create hash of each sample (row)
    # pandas.util.hash_pandas_object creates unique hash per row
    train_hashes = pd.util.hash_pandas_object(X_train)
    test_hashes = pd.util.hash_pandas_object(X_test)
    
    # Step 3: Find common hashes (duplicate samples)
    common_hashes = set(train_hashes).intersection(set(test_hashes))
    
    # Step 4: Identify actual duplicate rows
    duplicate_train_idx = train_hashes.isin(common_hashes)
    duplicate_test_idx = test_hashes.isin(common_hashes)
    
    return {
        'has_leakage': len(common_hashes) > 0,
        'duplicate_count': len(common_hashes),
        'duplicate_train_indices': np.where(duplicate_train_idx)[0],
        'duplicate_test_indices': np.where(duplicate_test_idx)[0],
        'duplicate_percentage': len(common_hashes) / len(test_hashes) * 100
    }
```

**How it works:**
- Creates unique hash for each sample (row)
- Time Complexity: O(n*m) where n=samples, m=features
- Space Complexity: O(n) for hash storage
- Detects exact duplicates even if row order is different

## 4. Spatial Leakage Detection

### Algorithm: Distance Matrix Calculation
```python
def _check_spatial_leakage(self, train_coords, test_coords, min_distance=0):
    """
    Core algorithm: Calculate pairwise distances
    """
    from scipy.spatial.distance import cdist
    
    # Step 1: Calculate distance matrix (all pairs)
    # cdist uses Euclidean distance by default
    distances = cdist(train_coords, test_coords)  # Shape: (n_train, n_test)
    
    # Step 2: Find minimum distance
    min_dist = np.min(distances)
    
    # Step 3: Find pairs that are too close
    close_pairs = np.where(distances < min_distance)
    
    # Step 4: Get specific violations
    violations = []
    for train_idx, test_idx in zip(close_pairs[0][:10], close_pairs[1][:10]):
        violations.append({
            'train_idx': train_idx,
            'test_idx': test_idx,
            'distance': distances[train_idx, test_idx],
            'train_coord': train_coords[train_idx],
            'test_coord': test_coords[test_idx]
        })
    
    return {
        'has_leakage': min_dist < min_distance,
        'min_distance_found': min_dist,
        'required_distance': min_distance,
        'num_close_pairs': len(close_pairs[0]),
        'violations': violations
    }
```

**How it works:**
- Uses scipy's efficient cdist for distance calculation
- Can use different distance metrics (Euclidean, Manhattan, etc.)
- Identifies specific sample pairs that violate distance threshold

### Visual Example:
```
2D Spatial Data:
     Train samples (•)    Test samples (○)
     
Good (No leakage):          Bad (Leakage):
  •  •  •  │  ○  ○           •  •  ○  •
  •  •  •  │  ○  ○           •  ○  •  ○  
  •  •  •  │  ○  ○           ○  •  •  ○
  ────────┘                  └─ Too close!
  Buffer zone
```

## 5. Preprocessing Leakage Detection

### Algorithm: Statistical Fingerprinting
```python
def check_preprocessing_leakage(self, X_original, X_processed, split_indices):
    """
    Core algorithm: Check if preprocessing used global statistics
    """
    train_idx, test_idx = split_indices
    
    # Step 1: Calculate statistics on processed data
    global_mean = np.mean(X_processed, axis=0)
    global_std = np.std(X_processed, axis=0)
    
    train_mean = np.mean(X_processed[train_idx], axis=0)
    train_std = np.std(X_processed[train_idx], axis=0)
    
    test_mean = np.mean(X_processed[test_idx], axis=0)
    test_std = np.std(X_processed[test_idx], axis=0)
    
    # Step 2: Check for suspicious patterns
    # If data was normalized BEFORE split:
    # - Global mean ≈ 0, Global std ≈ 1
    # - Train mean ≈ 0, Train std ≈ 1
    # - Test mean ≈ 0, Test std ≈ 1
    
    # If data was normalized AFTER split (correct):
    # - Train mean = 0, Train std = 1 (exactly)
    # - Test mean ≠ 0, Test std ≠ 1 (different distribution)
    
    suspicious_patterns = {
        # Check if means are suspiciously similar
        'identical_means': np.allclose(train_mean, test_mean, rtol=1e-10),
        
        # Check if stds are suspiciously similar  
        'identical_stds': np.allclose(train_std, test_std, rtol=1e-10),
        
        # Check if train stats match global stats (indicates global normalization)
        'train_matches_global': np.allclose(train_mean, global_mean, rtol=1e-10),
        
        # Check for standard normalization pattern
        'standard_normalized': np.allclose(global_mean, 0, atol=1e-10) and 
                              np.allclose(global_std, 1, atol=1e-10)
    }
    
    return {
        'has_leakage': suspicious_patterns['train_matches_global'],
        'suspicious_patterns': suspicious_patterns,
        'recommendation': 'Fit preprocessor only on training data'
    }
```

**How it works:**
- Compares statistical properties of train/test/global data
- Detects if normalization used information from test set
- Uses numerical tolerance for floating-point comparison

## 6. Feature-Target Leakage Detection

### Algorithm: Correlation Analysis
```python
def check_feature_target_leakage(self, X, y, threshold=0.95):
    """
    Core algorithm: Find features too correlated with target
    """
    correlations = []
    suspicious_features = []
    
    for feature_idx in range(X.shape[1]):
        feature = X[:, feature_idx]
        
        # Step 1: Calculate correlation based on data type
        if np.issubdtype(y.dtype, np.number):  # Continuous target
            from scipy.stats import pearsonr
            corr, p_value = pearsonr(feature, y)
            correlation = abs(corr)
        else:  # Categorical target
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(feature.reshape(-1, 1), y)[0]
            correlation = min(mi, 1.0)  # Normalize to [0,1]
        
        correlations.append(correlation)
        
        # Step 2: Flag suspicious features
        if correlation > threshold:
            suspicious_features.append({
                'index': feature_idx,
                'correlation': correlation,
                'likely_leakage': correlation > 0.99  # Almost perfect correlation
            })
    
    # Step 3: Analyze patterns
    if suspicious_features:
        # Check if it's likely actual leakage vs strong predictor
        max_corr = max([f['correlation'] for f in suspicious_features])
        
        leakage_indicators = {
            'perfect_correlation': max_corr > 0.999,
            'multiple_suspicious': len(suspicious_features) > 1,
            'threshold_exceeded': max_corr > threshold
        }
    
    return {
        'has_leakage': len(suspicious_features) > 0,
        'suspicious_features': suspicious_features,
        'max_correlation': max(correlations),
        'leakage_confidence': 'HIGH' if max_corr > 0.99 else 'MEDIUM'
    }
```

**How it works:**
- Calculates correlation/mutual information for each feature
- Flags features with suspiciously high correlation to target
- Distinguishes between strong predictors and actual leakage

## Integration with Cross-Validation

### Automatic Checking During CV
```python
class LeakageAwareCVSplitter:
    """
    Wrapper that adds automatic leakage detection to any CV splitter
    """
    def __init__(self, base_splitter, checker=None):
        self.base_splitter = base_splitter
        self.checker = checker or DataLeakageChecker()
        self.leakage_reports = []
    
    def split(self, X, y=None, groups=None):
        for fold_idx, (train_idx, test_idx) in enumerate(
            self.base_splitter.split(X, y, groups)
        ):
            # Run comprehensive leakage checks
            report = self._check_fold(X, y, train_idx, test_idx, groups)
            
            if report.has_leakage:
                # Option 1: Raise exception
                raise ValueError(f"Leakage detected in fold {fold_idx}: {report}")
                
                # Option 2: Skip fold
                # continue
                
                # Option 3: Log warning and continue
                # warnings.warn(f"Leakage in fold {fold_idx}")
            
            self.leakage_reports.append(report)
            yield train_idx, test_idx
    
    def _check_fold(self, X, y, train_idx, test_idx, groups):
        """Run all applicable leakage checks"""
        checks_to_run = []
        
        # Determine which checks are applicable
        if groups is not None:
            if 'patient_id' in str(groups.dtype):
                checks_to_run.append('patient')
            if 'time' in str(groups.dtype):
                checks_to_run.append('temporal')
        
        # Run checks
        return self.checker.check_cv_splits(
            X[train_idx], X[test_idx],
            y[train_idx] if y is not None else None,
            y[test_idx] if y is not None else None,
            patient_ids_train=groups[train_idx] if groups is not None else None,
            patient_ids_test=groups[test_idx] if groups is not None else None
        )
```

## Performance Optimization

### Efficient Implementation Strategies

1. **Hash-based Duplicate Detection**: O(n) average case
2. **Set Operations for Patient Overlap**: O(n) for intersection
3. **Vectorized Distance Calculations**: Uses BLAS-optimized scipy
4. **Early Stopping**: Returns immediately when critical leakage found
5. **Sampling for Large Datasets**: Check subset first for efficiency

```python
def optimized_check(self, X_train, X_test, sample_size=10000):
    """Optimized checking for large datasets"""
    n_train, n_test = len(X_train), len(X_test)
    
    # Sample if dataset is large
    if n_train > sample_size:
        sample_idx = np.random.choice(n_train, sample_size, replace=False)
        X_train_sample = X_train[sample_idx]
    else:
        X_train_sample = X_train
    
    # Quick hash check first (fastest)
    if self._quick_hash_check(X_train_sample, X_test):
        return {'has_leakage': True, 'type': 'duplicate'}
    
    # Then check other types...
```

## Real-time Monitoring

### Callback-based Detection
```python
class LeakageMonitorCallback(CVCallback):
    """Real-time leakage monitoring during training"""
    
    def on_fold_start(self, fold_idx, train_idx, val_idx):
        # Check for leakage before training starts
        self.current_report = self.checker.check_cv_splits(
            self.X[train_idx], self.X[val_idx],
            patient_ids_train=self.patient_ids[train_idx],
            patient_ids_test=self.patient_ids[val_idx]
        )
        
        if self.current_report.has_leakage:
            self.log_leakage(fold_idx, self.current_report)
            if self.strict_mode:
                raise ValueError(f"Leakage detected in fold {fold_idx}")
    
    def on_cv_end(self, results):
        # Generate summary report
        self.generate_leakage_report()
```

## Summary

The leakage detection implementation uses:
1. **Set operations** for patient overlap (O(n))
2. **Hash functions** for duplicate detection (O(n))
3. **Temporal comparisons** for time series (O(n))
4. **Distance matrices** for spatial data (O(n²))
5. **Statistical fingerprinting** for preprocessing leakage
6. **Correlation analysis** for feature-target leakage

All methods are optimized for performance and integrated seamlessly with the CV pipeline!