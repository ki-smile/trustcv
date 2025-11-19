# Recommended README Update for Data Leakage Detection

## Current State
The README mentions data leakage detection briefly:
- Line 14: "🛡️ Safety Checks: Automatic detection of data leakage"
- Lines 138-147: Shows basic usage example
- But lacks comprehensive explanation

## Suggested Addition to README (after line 147)


### 🔍 Comprehensive Data Leakage Detection

trustcv provides automatic detection for 6 types of data leakage common in ML:

#### 1. Patient Leakage Detection
```python
from trustcv import DataLeakageChecker

checker = DataLeakageChecker()

# Automatically detects if same patient in train/test
report = checker.check_cv_splits(
    X_train, X_test, 
    patient_ids_train=patient_ids[train_idx],
    patient_ids_test=patient_ids[test_idx]
)

if report.has_leakage:
    print(report)  # Shows which patients leaked, percentage, severity
```

#### 2. Temporal Leakage Detection
```python
# Detects if using future to predict past
report = checker.check_cv_splits(
    X_train, X_test,
    timestamps_train=timestamps[train_idx],
    timestamps_test=timestamps[test_idx]
)
# Warns if test data comes before training data
```

#### 3. Spatial Leakage Detection
```python
# For geographic or imaging data
checker.check_spatial_leakage(
    train_coords, test_coords, 
    min_distance=100  # Minimum 100m between train/test
)
```

#### 4. Preprocessing Leakage Detection
```python
# Detects if normalization done before split
leakage = checker.check_preprocessing_leakage(
    X_original, X_normalized, 
    split_indices=(train_idx, test_idx)
)
# Warns if global statistics were used
```

#### 5. Duplicate Sample Detection
```python
# Automatically finds duplicate samples
report = checker.check_cv_splits(X_train, X_test)
# Reports exact duplicates between sets
```

#### 6. Feature-Target Leakage Detection
```python
# Finds features suspiciously correlated with target
suspicious = checker.check_feature_target_leakage(
    X, y, threshold=0.95
)
# Warns about potential target leakage in features
```

#### Automatic Detection During Cross-Validation
```python
from trustcv import UniversalCVRunner, GroupKFoldMedical
from trustcv.core.callbacks import LeakageDetectionCallback

# Automatic checking in every fold
runner = UniversalCVRunner(
    cv_splitter=GroupKFoldMedical(n_splits=5),
    verbose=1
)

results = runner.run(
    model=YourModel(),
    data=(X, y),
    callbacks=[
        LeakageDetectionCallback(
            patient_ids=patient_ids,
            timestamps=timestamps,
            strict_mode=True  # Raises exception if leakage found
        )
    ]
)
```

📖 **Full Documentation**: 
- [Data Leakage Detection Guide](docs/DATA_LEAKAGE_DETECTION.md)
- [Implementation Details](docs/LEAKAGE_DETECTION_IMPLEMENTATION.md)
- [Interactive Demo](examples/data_leakage_detection_demo.py)
```

## Additional Documentation Index Update

Add to docs/README.md:

```markdown
# trustcv Documentation

## Core Features Documentation

### 🛡️ Data Leakage Detection
- **[Overview & Types](DATA_LEAKAGE_DETECTION.md)** - Understanding 6 types of leakage
- **[Implementation Details](LEAKAGE_DETECTION_IMPLEMENTATION.md)** - How detection algorithms work
- **[API Reference](API_REFERENCE.md#dataleakagechecker)** - Complete API documentation
- **[Examples](../examples/data_leakage_detection_demo.py)** - Interactive demonstrations

#### Quick Links:
1. [Patient Leakage](DATA_LEAKAGE_DETECTION.md#1-patient-leakage-most-common-in-healthcare)
2. [Temporal Leakage](DATA_LEAKAGE_DETECTION.md#2-temporal-leakage)
3. [Spatial Leakage](DATA_LEAKAGE_DETECTION.md#3-spatial-leakage)
4. [Preprocessing Leakage](DATA_LEAKAGE_DETECTION.md#4-preprocessing-leakage)
5. [Duplicate Detection](DATA_LEAKAGE_DETECTION.md#5-duplicate-sample-detection)
6. [Feature-Target Leakage](DATA_LEAKAGE_DETECTION.md#6-feature-target-leakage)
```

## Website Integration Update

Add to website/index.html navigation:

```html
<nav class="features-nav">
    <a href="#leakage-detection" class="nav-item">
        <i class="fas fa-shield-alt"></i>
        Data Leakage Detection
    </a>
</nav>

<section id="leakage-detection">
    <h2>🛡️ Automatic Data Leakage Detection</h2>
    <div class="feature-grid">
        <div class="detection-type">
            <h3>Patient Leakage</h3>
            <p>Detects if same patient appears in train & test</p>
            <code>checker.check_patient_leakage()</code>
        </div>
        <div class="detection-type">
            <h3>Temporal Leakage</h3>
            <p>Prevents using future to predict past</p>
            <code>checker.check_temporal_leakage()</code>
        </div>
        <div class="detection-type">
            <h3>Spatial Leakage</h3>
            <p>Ensures geographic/image separation</p>
            <code>checker.check_spatial_leakage()</code>
        </div>
        <div class="detection-type">
            <h3>Preprocessing Leakage</h3>
            <p>Detects if normalized before split</p>
            <code>checker.check_preprocessing_leakage()</code>
        </div>
        <div class="detection-type">
            <h3>Duplicate Detection</h3>
            <p>Finds exact duplicates between sets</p>
            <code>Built-in automatic checking</code>
        </div>
        <div class="detection-type">
            <h3>Feature-Target Leakage</h3>
            <p>Identifies suspiciously predictive features</p>
            <code>checker.check_feature_target_leakage()</code>
        </div>
    </div>
</section>
```

## Notebook Update

Create notebooks/11_Data_Leakage_Detection.ipynb:

```python
# Cell 1: Introduction
"""
# Data Leakage Detection in trustcv

This notebook demonstrates:
1. Six types of data leakage
2. How trustcv detects each type
3. Prevention strategies
4. Real-world medical examples
"""

# Cell 2: Setup
import numpy as np
import pandas as pd
from trustcv import DataLeakageChecker
from trustcv import GroupKFoldMedical, PurgedKFoldCV

# Cell 3-8: One cell per leakage type with visualization
# ... (detailed examples for each type)
```

## Summary of Documentation Improvements Needed:

1. ✅ **Main README**: Add comprehensive leakage section
2. ✅ **Dedicated Guides**: Created DATA_LEAKAGE_DETECTION.md and IMPLEMENTATION.md
3. ⚠️ **API Reference**: Need to ensure DataLeakageChecker is fully documented
4. ⚠️ **Website**: Add dedicated leakage detection section
5. ⚠️ **Notebooks**: Create interactive leakage detection tutorial
6. ✅ **Examples**: Created data_leakage_detection_demo.py

The feature IS documented but could be more prominently featured in the main README and website!