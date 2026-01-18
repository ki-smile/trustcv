# trustcv — Trustworthy Cross-Validation Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://github.com/ki-smile/trustcv)



**TrustCV** is a framework-agnostic toolkit for **reliable cross-validation** in safety-critical and regulated settings.
It builds on familiar scikit-learn idioms, but adds:

- Carefully designed cross-validation splitters (starting with IID in v0.1).
- Automatic **data leakage** and **class balance** checks.
- **Clinical/industrial metrics** with confidence intervals.
- Simple **reporting utilities** that support regulatory documentation.

> **Status:** v1.0.0 – Production release with 29 CV methods across IID, Grouped, Temporal, and Spatial categories. Full data leakage detection and reporting utilities for regulatory documentation.

---

## Why TrustCV?

Standard cross-validation is easy to misuse:

- Train/test splits can accidentally **leak information** (e.g., shared patients, timestamps, engineered features).
- Imbalanced datasets can give **overly optimistic** metrics if not stratified or monitored.
- For clinical and industrial applications, we often need **meaningful metrics** and reproducible reports, not just accuracy.

TrustCV addresses these issues by:

- Providing **well-tested IID splitters** with clear semantics.
- Running **leakage and balance checks** alongside your CV.
- Exposing **clinical metrics** and simple **reporting utilities** for audits and regulatory files.

---

## What's in v1.0.0

This release includes **29 cross-validation methods** across four categories:

- **IID splitters** (9 methods):
  `HoldOut`, `KFold`, `StratifiedKFold`, `RepeatedKFold`, `LeaveOneOut`, `LeavePOut`, `BootstrapValidation`, `MonteCarloCV`, `NestedCV`

- **Grouped splitters** (6 methods):
  `GroupKFold`, `StratifiedGroupKFold`, `LeaveOneGroupOut`, `RepeatedGroupKFold`, `NestedGroupedCV`, `HierarchicalGroupKFold`

- **Temporal splitters** (8 methods):
  `TimeSeriesSplit`, `BlockedTimeSeriesSplit`, `RollingWindowCV`, `ExpandingWindowCV`, `PurgedKFold`, `CombinatorialPurgedKFold`, `PurgedGroupTimeSeriesSplit`, `NestedTemporalCV`

- **Spatial splitters** (4 methods):
  `SpatialBlockCV`, `BufferedSpatialCV`, `SpatiotemporalBlockCV`, `EnvironmentalHealthCV`

- **Framework-agnostic runner**:
  `UniversalCVRunner` + `CVResults` for consistent, reusable CV loops across scikit-learn, PyTorch, TensorFlow, MONAI, and JAX.

- **High-level validator**:
  `TrustCVValidator` with automatic method selection and leakage detection.

- **Data integrity checks**:
  `DataLeakageChecker` (6 leakage types), `BalanceChecker`, and `LeakageReport`.

- **Clinical/medical metrics**:
  `ClinicalMetrics` with confidence intervals (sensitivity, specificity, PPV/NPV, etc.).

- **Regulatory documentation support**:
  `RegulatoryReport` for generating documentation that maps to FDA/CE MDR requirements.

---


## Quick Start

### Installation

```bash
# Install from source (recommended for latest features)
git clone https://github.com/ki-smile/trustcv.git
cd trustcv
pip install -e .

# Or install from PyPI (when released)
pip install trustcv
```


## Quickstart – IID CV with TrustCV

Here is a minimal example:

```python
from trustcv import TrustCV  # or TrustCVValidator
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)

# Simple usage
validator = TrustCV(method="stratified_kfold", n_splits=5)
results = validator.validate(model=RandomForestClassifier(), X=X, y=y)
print(results.summary())
```

### Full Example with All Options

```python
from trustcv import TrustCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

# Validates with leakage checks and computes clinical CIs
validator = TrustCV(
    method="stratified_kfold",
    n_splits=5,
    random_state=42,
    check_leakage=True,
    check_balance=True,
)
results = validator.validate(model=model, X=X, y=y)

print(results.summary()) 

# Output:
#=== Trustworthy Cross-Validation Results ===

#Performance Metrics (mean +/- std) (method: bootstrap):
#  accuracy: 0.956 +/- 0.014 [95% CI (bootstrap): 0.946-0.967]
#  roc_auc: 0.989 +/- 0.009 [95% CI (bootstrap): 0.981-0.995]
#  sensitivity: 0.966 +/- 0.030 [95% CI (bootstrap): 0.939-0.986]
#  specificity: 0.939 +/- 0.056 [95% CI (bootstrap): 0.893-0.981]
#  precision: 0.965 +/- 0.031 [95% CI (bootstrap): 0.937-0.989]
#  recall: 0.966 +/- 0.030 [95% CI (bootstrap): 0.939-0.986]
#  f1: 0.965 +/- 0.011 [95% CI (bootstrap): 0.957-0.973]


#Data Integrity Checks:
#  Leakage Check: PASSED
#  Class Balance: PASSED
```

For a higher-level workflow with leakage and balance checks, see the
 [Quickstart: IID CV with TrustCV](https://github.com/ki-smile/trustcv/blob/main/docs/Quickstart%3A%20IID%20Cross-Validation%20with%20TrustCV.md) and  [IID Splitters](https://github.com/ki-smile/trustcv/blob/main/docs/iid_splitters.md) tutorial.

 ### Run the example notebooks

Prefer to learn by running code? From the repo root, open the notebooks in `notebooks/`:

- `notebooks/01_IID_Methods_Showcase.ipynb` – quick tour of the IID splitters and metrics.
- `notebooks/02_Advanced_Workflow_UniversalRunner.ipynb` – end-to-end UniversalCVRunner workflow.
- `notebooks/03_TrustCVValidator_Showcase.ipynb` – TrustCVValidator examples with leakage/balance checks.
- `notebooks/04_TrustCVValidator_IID_Comparison.ipynb` – side-by-side IID method comparison.
- `notebooks/05_CrossValidation_Comparison.ipynb` – comprehensive CV methods comparison.
- Reports generated by the notebooks are saved in `notebooks/reports/` (HTML/PDF).



------

## How TrustCV relates to scikit-learn

**Similarities:**

- Uses familiar scikit-learn idioms: estimators with `fit`/`predict`, splitter objects with `split(X, y)`.
- Works seamlessly with scikit-learn models, pipelines, and metrics.
- IID splitters follow scikit-learn semantics (e.g., `KFold`, `StratifiedKFold`).

**Added value:**

- **Leakage and balance checks**: `DataLeakageChecker` and `BalanceChecker` run alongside your CV.
- **Clinical metrics**: `ClinicalMetrics` computes sensitivity, specificity, PPV/NPV, ROC/PR metrics, and CIs.
- **Structured results**: `CVResults` and `ValidationResult` standardize fold-level outputs.
- **Reporting**: `UniversalRegulatoryReport` turns your evaluation into a reproducible HTML/JSON report.

------

## Framework Support

**Supported frameworks:**
- scikit-learn (native)
- PyTorch (via adapter)
- TensorFlow/Keras (via adapter)
- MONAI (via adapter, for medical imaging)
- JAX/Flax (via adapter)
- XGBoost, LightGBM, CatBoost (via sklearn-compatible API)

------

## Contributors

See [AUTHORS.md](https://github.com/ki-smile/trustcv/blob/main/AUTHORS.md) for a full list of contributors and acknowledgments.

### Lead Contributors
- **[Farhad Abtahi](https://github.com/farhad-abtahi)**
- **[Abdelamir Karbalaie](https://github.com/abdkar)**





### Contributing
We welcome contributions!
- Code contributions
- Medical use case examples
- Documentation improvements
- Bug reports and feature requests

Please see:

- [`CONTRIBUTING.md`](https://github.com/ki-smile/trustcv/blob/main/CONTRIBUTING.md)
- [`CODE_OF_CONDUCT.md`](https://github.com/ki-smile/trustcv/blob/main/CODE_OF_CONDUCT.md)

------



## 3. Quickstart: IID CV with TrustCV – outline

**See file:** `docs/quickstart_iid.md`  
(and a matching notebook: `notebooks/Quickstart_IID_TrustCV.ipynb`)


## Repository Structure

```
trustcv/       # Python package (splitters, validators, metrics, core)
docs/          # Documentation & guides
notebooks/     # Jupyter tutorials
examples/      # Real-world examples
tests/         # Unit & integration tests
website/       # Static site and visualizations
```

## Development

```bash
pip install -e .[dev]
pytest tests/
cd docs && make html
```

## Citation

If you use trustcv in your research, please cite:

```bibtex
@software{trustcv2025,
  title = {trustcv: Trustworthy Cross-Validation Toolkit},
  author = {Abtahi, Farhad and Karbalaie, Abdolamir},
  year = {2025},
  url = {https://github.com/ki-smile/trustcv}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact & Support

- GitHub Issues: https://github.com/ki-smile/trustcv/issues

---

## ⚠️ Disclaimer

This toolkit is for research and educational purposes. Always validate results with domain experts before clinical deployment.

**Regulatory Note**: TrustCV provides documentation templates and structured outputs that can support regulatory submissions, but regulatory compliance depends on the complete device lifecycle and cannot be guaranteed by any single tool. Always consult with regulatory affairs professionals for your specific submission requirements.

---

Advancing Medical AI Through Rigorous Validation