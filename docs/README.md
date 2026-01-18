# TrustCV Documentation

**Version 1.0.0** | Framework-Agnostic Cross-Validation for Medical AI

This directory contains comprehensive documentation for TrustCV, a toolkit implementing 29 cross-validation methods with automatic data leakage detection.

---

## Documentation Index

### Getting Started

| Document | Description |
|----------|-------------|
| [Quickstart: IID Cross-Validation](Quickstart:%20IID%20Cross-Validation%20with%20TrustCV.md) | Step-by-step introduction to basic CV workflows |
| [CV Selection Guide](CV_SELECTION_GUIDE.md) | Decision tree for choosing the right CV method |
| [Best Practices](BEST_PRACTICES.md) | Do's and don'ts for medical ML validation |

### Core Features

| Document | Description |
|----------|-------------|
| [Data Leakage Detection](DATA_LEAKAGE_DETECTION.md) | Understanding and detecting 6 types of data leakage |
| [Leakage Detection Implementation](LEAKAGE_DETECTION_IMPLEMENTATION.md) | Technical details of detection algorithms |
| [Practical CV Guide](PRACTICAL_CV_GUIDE.md) | Real-world CV scenarios and solutions |

### Advanced Topics

| Document | Description |
|----------|-------------|
| [Framework Integration Guide](FRAMEWORK_GUIDE.md) | Using TrustCV with PyTorch, TensorFlow, MONAI |
| [Regulatory CV Guidelines](REGULATORY_CV_GUIDELINES.md) | Mapping to FDA/CE MDR documentation requirements |
| [ML Toolbox Comparison](ML_TOOLBOX_CV_COMPARISON.md) | TrustCV vs scikit-learn and other libraries |

### API Reference

| Document | Description |
|----------|-------------|
| [API Reference](API_REFERENCE.md) | Complete API documentation |
| [IID Splitters](iid_splitters.md) | Documentation for IID CV methods |

---

## CV Methods Overview

TrustCV implements **29 cross-validation methods** across 4 categories:

### IID Methods (9)
Standard methods for independent, identically distributed data:
- HoldOut, KFoldMedical, StratifiedKFoldMedical
- RepeatedKFold, LOOCV, LPOCV
- BootstrapValidation, MonteCarloCV, NestedCV

### Grouped Methods (8)
For data with patient/subject groupings:
- GroupKFoldMedical, StratifiedGroupKFold, LeaveOneGroupOut
- LeavePGroupsOut, RepeatedGroupKFold, GroupShuffleSplit
- NestedGroupedCV, HierarchicalCV

### Temporal Methods (8)
For time-series and longitudinal data:
- TimeSeriesSplit, RollingWindowCV, ExpandingWindowCV
- BlockedTimeSeries, PurgedKFoldCV, CombinatorialPurgedCV
- NestedTemporalCV, EmbargoCV

### Spatial Methods (4)
For geographic or imaging data:
- SpatialBlockCV, BufferedSpatialCV
- SpatiotemporalBlockCV, EnvironmentalHealthCV

---

## Data Leakage Detection

TrustCV automatically detects 6 types of data leakage:

1. **Patient Leakage** - Same patient in train and test sets
2. **Temporal Leakage** - Future data used to predict past
3. **Spatial Leakage** - Nearby samples in train and test
4. **Preprocessing Leakage** - Global statistics computed before split
5. **Duplicate Detection** - Exact duplicate samples across sets
6. **Feature-Target Leakage** - Features correlated with target

See [Data Leakage Detection](DATA_LEAKAGE_DETECTION.md) for details.

---

## Quick Links

- **Installation**: `pip install trustcv`
- **Repository**: [github.com/ki-smile/trustcv](https://github.com/ki-smile/trustcv)
- **Notebooks**: See `notebooks/` directory for interactive tutorials
- **Examples**: See `examples/` directory for complete scripts

---

## Building Documentation

To build the Sphinx documentation locally:

```bash
cd docs
pip install -e ..[docs]
make html
```

The generated HTML will be in `docs/_build/html/`.

---

*Part of the TrustCV toolkit by [SMAILE (Stockholm Medical AI and Learning Environments)](https://smile.ki.se) @ Karolinska Institutet*
