# TrustCV v1.0.0 - Framework-Agnostic Cross-Validation Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Custom CV Methods Not in Scikit-learn](#custom-cv-methods)
3. [Framework Integration](#framework-integration)
4. [MONAI Medical Imaging](#monai-medical-imaging)
5. [PyTorch Deep Learning](#pytorch-deep-learning)
6. [TensorFlow/Keras](#tensorflow-keras)
7. [Best Practices](#best-practices)

---

## Introduction

TrustCV v1.0.0 provides framework-agnostic cross-validation, allowing you to use the same CV strategies across scikit-learn, PyTorch, TensorFlow, MONAI, and more. This is particularly important for:

1. **Medical/Clinical Research**: Use specialized CV methods not available in standard libraries
2. **Deep Learning**: Apply proper CV to neural networks with minimal code
3. **Medical Imaging**: MONAI integration for 3D medical image validation
4. **Regulatory Compliance**: Consistent validation across all frameworks

### Key Innovation: Custom CV Methods for Any Framework

**trustcv provides 29 specialized CV methods, many NOT available in scikit-learn:**

```python
# These CV methods are UNIQUE to trustcv and work with ANY framework:

# 1. Temporal Medical Data (NOT in sklearn)
from trustcv import PurgedKFoldCV, CombinatorialPurgedCV

# 2. Hierarchical Patient Grouping (NOT in sklearn)
from trustcv import HierarchicalGroupKFold

# 3. Spatial Medical Data (NOT in sklearn)
from trustcv import SpatialBlockCV, BufferedSpatialCV

# 4. Combined Spatiotemporal (NOT in sklearn)
from trustcv import SpatiotemporalBlockCV

# ALL work with PyTorch, TensorFlow, MONAI, etc!
```

---

## Custom CV Methods Not in Scikit-learn

### 1. Purged K-Fold for Time Series (Financial/Medical Monitoring)

**Problem**: Standard K-fold causes look-ahead bias in time series.  
**Solution**: PurgedKFoldCV adds temporal gaps between train/test sets.

```python
import numpy as np
from trustcv import PurgedKFoldCV, UniversalCVRunner
import torch
import torch.nn as nn

# Example: ICU patient monitoring over time
n_patients = 100
n_timepoints = 500
X = np.random.randn(n_patients * n_timepoints, 20)  # Features
y = np.random.randint(0, 2, n_patients * n_timepoints)  # Binary outcome
timestamps = np.repeat(np.arange(n_timepoints), n_patients)

# This CV method is NOT available in scikit-learn!
cv = PurgedKFoldCV(n_splits=5, purge_gap=10)  # 10 timepoint gap

# Works with PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        # Reshape for LSTM if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Run purged CV with PyTorch
runner = UniversalCVRunner(cv_splitter=cv)
results = runner.run(
    model=lambda: LSTMModel(),
    data=(X, y),
    epochs=20,
    groups=timestamps  # Pass timestamps for purging
)

print(f"Purged CV Results: {results.mean_score}")
```

### 2. Hierarchical Group K-Fold (Multi-level Medical Hierarchies)

**Problem**: Patients nested within hospitals, hospitals within regions.  
**Solution**: Respect hierarchical structure in validation.

```python
from trustcv import HierarchicalGroupKFold

# Example: Multi-center clinical trial
n_regions = 5
n_hospitals_per_region = 4
n_patients_per_hospital = 50

# Create hierarchical structure
regions = np.repeat(np.arange(n_regions), n_hospitals_per_region * n_patients_per_hospital)
hospitals = np.repeat(np.arange(n_regions * n_hospitals_per_region), n_patients_per_hospital)
patients = np.arange(len(regions))

# This preserves hospital AND region structure (NOT in sklearn!)
cv = HierarchicalGroupKFold(
    n_splits=5,
    hierarchy_levels=['region', 'hospital', 'patient']
)

# Works with ANY model - sklearn, PyTorch, TensorFlow
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras

# Same CV for different frameworks!
for model in [
    RandomForestClassifier(),
    keras.Sequential([keras.layers.Dense(10, activation='relu')]),
    LSTMModel()
]:
    runner = UniversalCVRunner(cv_splitter=cv)
    results = runner.run(
        model=model,
        data=(X, y),
        groups={'region': regions, 'hospital': hospitals, 'patient': patients}
    )
```

### 3. Spatial Block CV (Geographic/Imaging Data)

**Problem**: Spatial autocorrelation violates independence assumption.  
**Solution**: Create spatially separated train/test blocks.

```python
from trustcv import SpatialBlockCV, BufferedSpatialCV

# Example: Environmental health study with geographic data
n_samples = 1000
coordinates = np.random.randn(n_samples, 2) * 100  # Lat/long
pollution_features = np.random.randn(n_samples, 10)
health_outcomes = np.random.randint(0, 2, n_samples)

# Spatial blocking (NOT in sklearn!)
cv = SpatialBlockCV(
    n_splits=5,
    spatial_coordinates=coordinates,
    block_size='auto'  # Automatically determine block size
)

# Add buffer zones to prevent spillover
cv_buffered = BufferedSpatialCV(
    n_splits=5,
    spatial_coordinates=coordinates,
    buffer_size=10  # 10 unit buffer between train/test
)

# Use with deep learning for satellite imagery
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(64 * 28 * 28, 2)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

runner = UniversalCVRunner(cv_splitter=cv_buffered)
results = runner.run(model=CNNModel, data=(pollution_features, health_outcomes))
```

### 4. Combinatorial Purged CV (Advanced Time Series)

**Problem**: Need to test on multiple future periods.  
**Solution**: Combinatorial approach with purging.

```python
from trustcv import CombinatorialPurgedCV

# Example: Drug efficacy over multiple time periods
cv = CombinatorialPurgedCV(
    n_splits=5,
    n_test_sets=2,  # Test on 2 future periods simultaneously
    purge_gap=5
)

# This advanced method works with ANY framework
results = runner.run(
    model=YourModel(),  # PyTorch, TF, sklearn - anything!
    data=(X, y),
    groups=timestamps
)
```

---

## MONAI Medical Imaging

### Complete MONAI Example: Brain Tumor Segmentation with Patient-Grouped CV

```python
import numpy as np
from trustcv import MONAICVRunner, GroupKFoldMedical
from trustcv.frameworks.monai import MONAIAdapter
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, 
    ScaleIntensityRanged, CropForegroundd,
    RandFlipd, RandRotate90d, RandShiftIntensityd,
    EnsureTyped, Spacingd, Orientationd
)

# 1. Prepare Data - MONAI format with patient IDs
data_dicts = [
    {"image": f"patient_{i:03d}_t1.nii.gz", 
     "label": f"patient_{i:03d}_seg.nii.gz",
     "patient_id": f"patient_{i:03d}"}
    for i in range(100)
]

# Extract patient IDs for grouping (CRITICAL for medical data!)
patient_ids = np.array([d["patient_id"] for d in data_dicts])

# 2. Define Transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0)),
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-1000, a_max=1000,
        b_min=0.0, b_max=1.0, 
        clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    # Augmentations
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    EnsureTyped(keys=["image", "label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0)),
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-1000, a_max=1000,
        b_min=0.0, b_max=1.0, 
        clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"])
])

# 3. Define Model
def create_unet():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )

# 4. Setup Patient-Grouped Cross-Validation
# This ensures same patient never appears in both train and test!
cv_splitter = GroupKFoldMedical(n_splits=5)

# 5. Create MONAI CV Runner
runner = MONAICVRunner(
    model_fn=create_unet,
    cv_splitter=cv_splitter,
    adapter=MONAIAdapter(
        batch_size=2,  # Small for 3D volumes
        cache_rate=0.5,  # Cache 50% of data
        roi_size=(96, 96, 96),  # For sliding window inference
        device='cuda'
    )
)

# 6. Run Cross-Validation with Medical Best Practices
from trustcv import EarlyStopping, ModelCheckpoint

results = runner.run(
    data_dicts=data_dicts,
    epochs=100,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
    optimizer_fn=lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4),
    loss_fn=DiceLoss(to_onehot_y=True, softmax=True),
    metrics=[
        DiceMetric(include_background=False, reduction="mean"),
    ],
    callbacks=[
        EarlyStopping(monitor='val_dice', mode='max', patience=10),
        ModelCheckpoint(filepath='best_model_fold_{fold}.pth', monitor='val_dice')
    ],
    groups=patient_ids  # CRITICAL: Patient grouping!
)

# 7. Analyze Results
print("\n" + "="*50)
print("MONAI Cross-Validation Results")
print("="*50)
print(f"Mean Dice Score: {results.mean_score['val_dice']:.4f}")
print(f"Std Dice Score: {results.std_score['val_dice']:.4f}")

# Check for patient leakage
for fold_idx, (train_idx, val_idx) in enumerate(results.indices):
    train_patients = patient_ids[train_idx]
    val_patients = patient_ids[val_idx]
    overlap = np.intersect1d(train_patients, val_patients)
    print(f"Fold {fold_idx + 1}: Patient overlap = {len(overlap)} (should be 0)")
```

---

## PyTorch Deep Learning

### Example: Clinical Risk Prediction with Tabular Data

```python
import torch
import torch.nn as nn
from trustcv import TorchCVRunner, StratifiedGroupKFold
from trustcv import EarlyStopping, ProgressLogger

# Create clinical dataset
n_patients = 5000
n_features = 50
X = np.random.randn(n_patients, n_features).astype(np.float32)
y = np.random.randint(0, 2, n_patients)
patient_ids = np.arange(n_patients)
admission_dates = np.random.randint(0, 365, n_patients)

# Deep neural network for tabular data
class ClinicalRiskNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Use stratified group k-fold to maintain class balance
cv = StratifiedGroupKFold(n_splits=5)

# PyTorch-specific runner
runner = TorchCVRunner(
    model_fn=lambda: ClinicalRiskNet(n_features),
    cv_splitter=cv
)

# Advanced training with callbacks
results = runner.run(
    dataset=(X, y),
    epochs=50,
    optimizer_fn=lambda m: torch.optim.AdamW(
        m.parameters(), 
        lr=0.001, 
        weight_decay=1e-5
    ),
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0])),  # Handle imbalance
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best=True),
        ProgressLogger(log_file='training_log.json')
    ],
    groups=patient_ids
)
```

---

## TensorFlow/Keras

### Example: Medical Image Classification with Transfer Learning

```python
import tensorflow as tf
from tensorflow import keras
from trustcv import KerasCVRunner, BlockedTimeSeries

# Prepare medical image data
X = np.random.randn(1000, 224, 224, 3).astype(np.float32)
y = np.random.randint(0, 4, 1000)  # 4 disease classes
timestamps = np.arange(1000)  # Sequential acquisition

# Transfer learning model
def create_transfer_model():
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze base model
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4, activation='softmax')
    ])
    return model

# Use blocked time series CV for temporal data
cv = BlockedTimeSeries(n_splits=5, block_size=200)

runner = KerasCVRunner(
    model_fn=create_transfer_model,
    cv_splitter=cv,
    compile_kwargs={
        'optimizer': keras.optimizers.Adam(1e-4),
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy', keras.metrics.AUC(name='auc')]
    }
)

# Run with TensorFlow callbacks
results = runner.run(
    X=X, y=y,
    epochs=30,
    batch_size=16,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5),
        keras.callbacks.EarlyStopping(patience=5)
    ],
    groups=timestamps
)

print(f"Mean AUC: {results.mean_score['val_auc']:.4f}")
```

---

## Best Practices

### 1. Always Check for Data Leakage

```python
from trustcv import DataLeakageChecker

checker = DataLeakageChecker()

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
    # Check patient leakage
    has_patient_leak = checker.check_patient_leakage(
        train_idx, test_idx, patient_ids
    )
    
    # Check temporal leakage
    has_temporal_leak = checker.check_temporal_leakage(
        train_idx, test_idx, timestamps
    )
    
    if has_patient_leak or has_temporal_leak:
        raise ValueError(f"Data leakage detected in fold {fold_idx}!")
```

### 2. Use Appropriate CV for Your Data Structure

```python
# Decision tree for CV selection
def select_cv_method(data_type, has_groups, has_time, has_space):
    if has_groups and has_time:
        return PurgedGroupTimeSeriesSplit()
    elif has_time and has_space:
        return SpatiotemporalBlockCV()
    elif has_groups:
        return GroupKFoldMedical()
    elif has_time:
        return TimeSeriesSplit()
    elif has_space:
        return SpatialBlockCV()
    else:
        return StratifiedKFoldMedical()
```

### 3. Regulatory Compliance

```python
from trustcv.core.callbacks import RegulatoryComplianceLogger

# For FDA/CE MDR submissions
compliance_logger = RegulatoryComplianceLogger(
    output_dir='./regulatory_docs',
    study_name='ModelValidation_2024',
    include_data_characteristics=True,
    include_model_details=True
)

runner.run(model, data, callbacks=[compliance_logger])
```

---

## Summary

TrustCV v1.0.0 provides:

1. **29 CV methods** - Many NOT available in scikit-learn
2. **Framework agnostic** - Works with PyTorch, TensorFlow, MONAI, etc.
3. **Medical focus** - Patient grouping, temporal purging, spatial blocking
4. **Best practices** - Automatic leakage detection, compliance logging
5. **Production ready** - Callbacks, checkpointing, monitoring

For more examples, see the `examples/` directory in the repository.