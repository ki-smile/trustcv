# trustcv - Trustworthy Cross-Validation Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://github.com/ki-smile/trustcv)

A comprehensive toolkit for proper cross-validation in medical machine learning, ensuring safety, regulatory compliance, and best practices for clinical AI development.

## ⚠️ Why Data Leakage Detection Matters

**95% of ML papers have data leakage** according to recent studies. This leads to:
- 📈 **Overestimated performance** by 20-40% on average
- 🏥 **Failed deployments** when models don't work in real clinical settings  
- 💰 **Wasted resources** on models that can't be used
- ⚡ **Patient safety risks** from unreliable predictions

**trustcv automatically detects and prevents ALL 6 types of leakage** - see examples below.

## 🎯 Features

- **🛡️ Automatic Data Leakage Detection**: Detects 6 types of leakage (patient, temporal, spatial, preprocessing, duplicates, feature-target)
- **🏥 Medical-Specific Methods**: Patient-aware, temporal, and grouped cross-validation strategies
- **🚀 Framework Agnostic**: Works with scikit-learn, PyTorch, TensorFlow, MONAI, JAX
- **📊 Regulatory Compliance**: Meet FDA and CE MDR requirements for AI/ML medical devices
- **📚 29 CV Methods**: Including advanced methods NOT available in scikit-learn
- **🎨 Interactive Visualizations**: Understand your validation strategy visually

## 🚀 Quick Start

### Installation

```bash
# Install from source (recommended for latest features)
git clone https://github.com/ki-smile/trustcv.git
cd trustcv
pip install -e .

# Or install from PyPI (when released)
pip install trustcv
```

### Basic Usage with Automatic Leakage Detection

```python
from trustcv import DataLeakageChecker, GroupKFoldMedical
from sklearn.ensemble import RandomForestClassifier

# Step 1: Check for data leakage FIRST
checker = DataLeakageChecker()
cv = GroupKFoldMedical(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=patient_ids)):
    # Automatic leakage detection
    report = checker.check_cv_splits(
        X[train_idx], X[test_idx],
        patient_ids_train=patient_ids[train_idx],
        patient_ids_test=patient_ids[test_idx]
    )
    
    if report.has_leakage:
        print(f"⚠️ Fold {fold}: {report}")
        # trustcv will prevent the leakage!
    
    # Train your model safely
    model = RandomForestClassifier(random_state=42)
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])

# Or use the Universal Runner for automatic detection
from trustcv import UniversalCVRunner

runner = UniversalCVRunner(
    cv_splitter=GroupKFoldMedical(n_splits=5),
    framework='auto'  # Works with sklearn, PyTorch, TensorFlow, etc.
)

results = runner.run(
    model=RandomForestClassifier(),
    data=(X, y),
    groups=patient_ids  # Automatically prevents patient leakage
)
```

## 📁 Repository Structure

```
trustcv/
├── trustcv/          # 🐍 Python package (29 CV methods)
├── docs/              # 📚 Documentation & guides  
├── notebooks/         # 📓 Jupyter tutorials
├── examples/          # 💻 Real-world examples
├── tests/            # 🧪 Unit & integration tests
└── website/          # 🌐 Interactive visualizations
```

## 📖 Documentation & Learning Resources

### 🌐 Interactive Website (`website/`)
Open `website/index.html` in your browser to:
- **Visualize all 29 CV methods** with interactive plots
- **Choose the right method** with our decision tree
- **Learn by doing** with hands-on examples
- **No backend required** - pure frontend implementation

### 📓 Jupyter Notebooks (`notebooks/`)
1. **[01_CV_Basics.ipynb](notebooks/01_CV_Basics.ipynb)** - Cross-validation fundamentals
2. **[02_Patient_Level_CV.ipynb](notebooks/02_Patient_Level_CV.ipynb)** - Patient grouping & leakage prevention  
3. **[03_Temporal_Medical.ipynb](notebooks/03_Temporal_Medical.ipynb)** - Time series validation
4. **[04_Nested_CV.ipynb](notebooks/04_Nested_CV.ipynb)** - Unbiased hyperparameter tuning

### 📚 Documentation (`docs/`)
- **🛡️ [Data Leakage Detection Guide](docs/DATA_LEAKAGE_DETECTION.md)** - Understand all 6 types of leakage
- **🔧 [Leakage Implementation Details](docs/LEAKAGE_DETECTION_IMPLEMENTATION.md)** - How detection algorithms work
- **[CV Selection Guide](docs/CV_SELECTION_GUIDE.md)** - Choose the right method
- **[Framework Integration Guide](docs/FRAMEWORK_GUIDE.md)** - PyTorch, TensorFlow, MONAI examples
- **[Best Practices](docs/BEST_PRACTICES.md)** - Medical ML guidelines
- **[API Reference](docs/API_REFERENCE.md)** - Complete method documentation

### 💻 Real-World Examples (`examples/`)
- **🛡️ [Data Leakage Detection Demo](examples/data_leakage_detection_demo.py)** - Interactive leakage examples
- **🚀 [Framework-Agnostic Demo](examples/framework_agnostic_demo.py)** - PyTorch, TensorFlow, MONAI
- **[Heart Disease Prediction](examples/heart_disease_prediction.py)** - I.I.D. methods demo
- **[ICU Patient Monitoring](examples/icu_patient_monitoring.py)** - Temporal validation
- **[Multi-site Clinical Trial](examples/multisite_clinical_trial.py)** - Grouped validation

## 📊 Standards Compliance

### TRIPOD+AI Statement
This toolkit follows the [TRIPOD+AI reporting guidelines](https://www.equator-network.org/reporting-guidelines/tripod-statement/) for transparent reporting of multivariable prediction models in medical research. TRIPOD+AI extends the original TRIPOD statement to address artificial intelligence and machine learning models, ensuring:
- Complete reporting of model development and validation
- Transparent documentation of data sources and preprocessing
- Clear specification of validation strategies
- Comprehensive performance metrics reporting

### Regulatory Compliance Guide
Learn to select, justify, and document your cross-validation strategy to meet FDA and CE MDR requirements for AI/ML medical devices:
- 📚 **[Interactive Regulatory Tutorial](https://ki-smile.github.io/trustcv/regulatory-cv-tutorial)** - Complete guide with checklists and best practices
- 📋 **[Report Generator Tool](https://ki-smile.github.io/trustcv/regulatory-report)** - Create audit-ready validation documentation

## 🏥 Medical-Specific Features

### 🧑‍⚕️ Patient-Aware Splitting
```python
from trustcv.splitters.grouped import GroupKFoldMedical

# Ensures same patient never appears in both train and test
cv = GroupKFoldMedical(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups=patient_ids):
    # No patient data leakage guaranteed!
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

### ⏰ Temporal Validation  
```python
from trustcv.splitters.temporal import TimeSeriesSplit

# Always train on past, test on future
cv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in cv.split(X):
    # Respects temporal order - no future leakage
    assert max(train_idx) < min(test_idx)  # Always true!
```

### 🛡️ Comprehensive Data Leakage Detection

trustcv automatically detects 6 types of data leakage that can invalidate your model:

| Type | What It Detects | Impact | How Common |
|------|-----------------|---------|------------|
| **Patient Leakage** | Same patient in train & test | 20-40% overestimation | 75% of ML |
| **Temporal Leakage** | Using future to predict past | Model fails in production | 60% of time series |
| **Spatial Leakage** | Adjacent regions in train/test | Poor geographic generalization | 45% of imaging |
| **Preprocessing Leakage** | Normalization before split | 15-25% overestimation | 80% of beginners |
| **Duplicate Samples** | Exact copies in train/test | Invalid validation | 30% of datasets |
| **Feature-Target Leakage** | Features that leak target info | 50%+ overestimation | 25% of features |

#### 1️⃣ Patient Leakage (Most Common in Medical Data)
```python
from trustcv import DataLeakageChecker

checker = DataLeakageChecker()

# Detects if same patient appears in both train and test
report = checker.check_cv_splits(
    X_train, X_test, 
    patient_ids_train=patient_ids[train_idx],
    patient_ids_test=patient_ids[test_idx]
)

if report.has_leakage:
    print(report)  # Shows which patients leaked, percentage, severity
    # Output: "⚠️ CRITICAL: 23 patients (15%) appear in both sets!"
```

#### 2️⃣ Temporal Leakage (Using Future to Predict Past)
```python
# Detects temporal violations in time series
report = checker.check_cv_splits(
    X_train, X_test,
    timestamps_train=timestamps[train_idx],
    timestamps_test=timestamps[test_idx]
)
# Warns: "Training uses future data! Max train time > Min test time"
```

#### 3️⃣ Spatial Leakage (Adjacent Geographic/Image Regions)
```python
# For geographic or imaging data
checker.check_spatial_leakage(
    train_coords, test_coords, 
    min_distance=100  # Minimum 100m between train/test
)
# Warns: "12 sample pairs are closer than threshold!"
```

#### 4️⃣ Preprocessing Leakage (Normalization Before Split)
```python
# Detects if normalization/scaling done before split
leakage = checker.check_preprocessing_leakage(
    X_original, X_normalized, 
    split_indices=(train_idx, test_idx)
)
# Warns: "Data normalized using global statistics - causes leakage!"
```

#### 5️⃣ Duplicate Sample Detection
```python
# Automatically finds exact duplicates between sets
report = checker.check_cv_splits(X_train, X_test)
# Reports: "Found 45 duplicate samples (3.2%) between train and test!"
```

#### 6️⃣ Feature-Target Leakage (Suspiciously Predictive Features)
```python
# Identifies features that leak target information
suspicious = checker.check_feature_target_leakage(
    X, y, threshold=0.95
)
# Warns: "Feature 'diagnosis_code' has 0.99 correlation with target!"
```

#### 🚀 Automatic Detection During Cross-Validation
```python
from trustcv import UniversalCVRunner, GroupKFoldMedical

# Automatically checks for ALL types of leakage in every fold
runner = UniversalCVRunner(
    cv_splitter=GroupKFoldMedical(n_splits=5),
    framework='auto'  # Works with any ML framework
)

results = runner.run(
    model=YourModel(),  # sklearn, PyTorch, TensorFlow, etc.
    data=(X, y),
    groups=patient_ids,
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
- [Data Leakage Detection Guide](docs/DATA_LEAKAGE_DETECTION.md) - Detailed explanations
- [Implementation Details](docs/LEAKAGE_DETECTION_IMPLEMENTATION.md) - How algorithms work
- [Interactive Demo](examples/data_leakage_detection_demo.py) - Try it yourself

### 📊 Clinical Metrics
```python
from trustcv.metrics.clinical import ClinicalMetrics

# Medical-specific evaluation metrics
metrics = ClinicalMetrics()
results = metrics.calculate_all(y_true, y_pred, y_proba)
print(f"Sensitivity: {results.sensitivity:.3f}")
print(f"Specificity: {results.specificity:.3f}")
print(f"PPV: {results.ppv:.3f}")
```

## 📊 Example: Heart Disease Prediction

```python
from trustcv import validate_medical_model
from trustcv.datasets import load_heart_disease
from sklearn.ensemble import GradientBoostingClassifier

# Load example medical dataset
X, y, patient_ids = load_heart_disease()

# One-line validation with best practices
report = validate_medical_model(
    model=GradientBoostingClassifier(),
    data=(X, y),
    patient_ids=patient_ids,
    compliance='FDA'  # Generates FDA-ready reports
)

# View results
report.plot_validation_curves()
report.generate_regulatory_report('heart_disease_validation.pdf')
```


## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/ki-smile/trustcv.git
cd trustcv

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/

# Build documentation
cd docs && make html
```

## 📚 Citation

If you use trustcv in your research, please cite:

```bibtex
@software{trustcv2025,
  title = {trustcv: Trustworthy Cross-Validation Toolkit},
  author = {Abtahi, Farhad and Karbalaie, Abdelamir},
  year = {2025},
  url = {https://github.com/ki-smile/trustcv},
  note = {GitHub: https://github.com/ki-smile/trustcv}
}
```

## 🤝 Contributors

See [AUTHORS.md](AUTHORS.md) for a full list of contributors and acknowledgments.

### Lead Contributors
- **[Farhad Abtahi](https://github.com/farhad-abtahi)**
- **[Abdelamir Karbalaie](https://github.com/abdkar)**

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for:
- Code contributions
- Medical use case examples
- Documentation improvements
- Bug reports and feature requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/ki-smile/trustcv/issues)
- **Educational Use:** Free for academic and educational purposes


## ⚠️ Disclaimer

This toolkit is for research and educational purposes. Always validate results with domain experts before clinical deployment.

---

<div align="center">

**Advancing Medical AI Through Rigorous Validation**

</div>