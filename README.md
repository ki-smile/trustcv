# trustcv — Trustworthy Cross-Validation Toolkit  

[![PyPI](https://img.shields.io/pypi/v/trustcv)](https://pypi.org/project/trustcv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://ki-smile.github.io/trustcv/)
[![Website](https://img.shields.io/badge/website-live-brightgreen)](https://ki-smile.github.io/trustcv/)

**Website & Docs:** [https://ki-smile.github.io/trustcv/](https://ki-smile.github.io/trustcv/)

📖 **[Visit the project website](https://ki-smile.github.io/trustcv/)** for interactive demos, tutorials, and the full documentation.

**TrustCV** is a framework-agnostic toolkit for **reliable cross-validation** in safety-critical and regulated settings.
It builds on familiar scikit-learn idioms, but adds:

- Carefully designed IID, grouped, temporal, and spatial cross-validation splitters.
- Automatic **data leakage** and **class balance** checks.
- **Clinical/industrial metrics** with confidence intervals.
- Simple **reporting utilities** that support regulatory documentation.

> **Status:** v1.1.0 – Conservative check statuses, corrected confidence intervals, calibrated leakage diagnostics, and an actionable CV advisor.

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

## What's in v1.1.0

This release includes **29 cross-validation methods** across four categories:

- **IID splitters** (9 methods):
  `HoldOut`, `KFold`, `StratifiedKFold`, `RepeatedKFold`, `LeaveOneOut`, `LeavePOut`, `BootstrapValidation`, `MonteCarloCV`, `NestedCV`

- **Grouped splitters** (6 methods):
  `GroupKFold`, `StratifiedGroupKFold`, `LeaveOneGroupOut`, `RepeatedGroupKFold`, `NestedGroupedCV`, `HierarchicalGroupKFold`

- **Temporal splitters** (8 methods):
  `TimeSeriesSplit`, `BlockedTimeSeriesSplit`, `RollingWindowCV`, `ExpandingWindowCV`, `PurgedKFold`, `CombinatorialPurgedKFold`, `PurgedGroupTimeSeriesSplit`, `NestedTemporalCV`

- **Spatial splitters** (4 methods):
  `SpatialBlockCV`, `BufferedSpatialCV`, `SpatiotemporalBlockCV`, `EnvironmentalHealthCV`

- **Framework-agnostic runner & `split_kwargs`**:
  `UniversalCVRunner` + `CVResults` for consistent, reusable CV loops across scikit-learn, PyTorch, TensorFlow, MONAI, and JAX with support for split-specific arguments (`split_kwargs`).

- **Metric Feasibility Diagnostics System**:
  `check_fold_metric_feasibility()` diagnostic warnings and tables when group-exclusive splitters (e.g., LOGO) have single-class or small validation folds.

- **High-level validator & integrity checks**:
  `TrustCVValidator` with automatic leakage checks, near-duplicate detection, hierarchical group leakage, auto-computed spatial thresholds, and `LeakageDetectionCallback`.

- **Clinical/medical metrics**:
  `ClinicalMetrics` with confidence intervals (sensitivity, specificity, PPV/NPV, etc.).

- **Regulatory documentation support**:
  `RegulatoryReport` for generating documentation that maps to FDA/CE MDR requirements.

---


## Quick Start

### Installation

```bash
pip install trustcv
```

Ask the advisor for a runnable splitter first, then keep all preprocessing inside the model pipeline:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustcv import TrustCV, recommend_cv

X, y = load_breast_cancer(return_X_y=True)
recommendation = recommend_cv(X, y)
print(recommendation.category, recommendation.method, type(recommendation.splitter).__name__)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
result = TrustCV(
    method=recommendation.method,
    n_splits=5,
    random_state=42,
    declare_independent_samples=True,
).validate(model=model, X=X, y=y, cv=recommendation.splitter)
print(result.summary())
```

Real output from v1.1.0:

```text
iid stratified_kfold StratifiedKFoldMedical
=== Trustworthy Cross-Validation Results ===

Performance Metrics (mean +/- std) (method: corrected_t):
  accuracy: 0.974 +/- 0.019 [95% CI (corrected_t): 0.939-1.008]
  roc_auc: 0.995 +/- 0.006 [95% CI (corrected_t): 0.984-1.006]
  sensitivity: 0.992 +/- 0.013 [95% CI (corrected_t): 0.968-1.015]
  specificity: 0.944 +/- 0.059 [95% CI (corrected_t): 0.834-1.053]
  precision: 0.968 +/- 0.032 [95% CI (corrected_t): 0.908-1.029]
  recall: 0.992 +/- 0.013 [95% CI (corrected_t): 0.968-1.015]
  f1: 0.979 +/- 0.014 [95% CI (corrected_t): 0.953-1.006]

Data Integrity Checks:
  Duplicate Samples: PASSED — No exact duplicate feature rows were found.
  Group Leakage: NOT_APPLICABLE — Samples were explicitly declared independent; no groups were supplied.
  Preprocessing Leakage: PASSED — Preprocessing steps are inside an sklearn Pipeline and are refit within each fold.
  External Leakage Detector: PASSED — The external leakage detector found no supported leakage pattern.
  Target Leakage Features: PASSED — No feature crossed the configured univariate target-leakage threshold.
  Permutation Sanity: NOT_CHECKED — Permutation sanity checking is disabled.
  Class Balance: INFO — Minority-to-majority class ratio is 0.594.
  Covariate Shift: INFO — No statistically significant covariate shift was detected.

Overall Status: PASSED
Leakage Check: PASSED
```

## Integrity checks and status semantics

Every `ValidationResult.checks` mapping contains the following keys:

| Check | Status behavior | Meaning |
| --- | --- | --- |
| `duplicate_samples` | `PASSED` or `WARNING` | Exact row hashes; works for NumPy arrays and DataFrames. |
| `group_leakage` | `PASSED`, `FAILED`, `ERROR`, `NOT_CHECKED`, or `NOT_APPLICABLE` | Group overlap can be verified only when groups are supplied, unless independence is explicitly declared. |
| `preprocessing_leakage` | `PASSED` or `NOT_CHECKED` | Passes only when preprocessing is inside an sklearn `Pipeline` before the final estimator. |
| `external_leakage_detector` | `PASSED`, `FAILED`, or `ERROR` | A detector crash is an error, never a silent pass. |
| `target_leakage_features` | `PASSED`, `WARNING`, or `NOT_APPLICABLE` | Flags near-deterministic binary AUC or regression Spearman association. |
| `permutation_sanity` | `PASSED`, `FAILED`, or `NOT_CHECKED` | Opt-in check for shuffled labels scoring above chance inside the CV loop. |
| `class_balance` | `INFO` or `WARNING` | Imbalance describes the data and is never a failed leakage check. |
| `covariate_shift` | `INFO`, `WARNING`, or `NOT_CHECKED` | Bonferroni-corrected KS shift report; shift is not leakage. |

`overall_status` considers the first five leakage-relevant checks and the permutation check when enabled. It is `FAILED` for any `FAILED`/`ERROR`, `PASSED` only when all relevant checks are `PASSED`/`NOT_APPLICABLE`, and `NOT_FULLY_VERIFIED` otherwise.

`leakage_check` remains for compatibility. Use `external_leakage_detected` for non-inverted semantics. The legacy `has_leakage` key is deprecated for one release and is inverted: `external_leakage_detected == (not has_leakage)`.

### What trustcv can and cannot detect

TrustCV can verify exact duplicates, supplied-group separation, whether visible sklearn preprocessing is inside a `Pipeline`, selected leakage patterns in `DataLeakageChecker`, near-deterministic single-feature target encodings, class balance, covariate shift, and—when enabled—whether shuffled labels score above chance inside the CV loop.

TrustCV cannot infer missing patient IDs, timestamps, coordinates, or causal relationships. It cannot detect preprocessing or feature selection already applied to the full dataset before `validate()` receives `X`. The permutation check is not a general leakage detector: with a fixed pre-selected matrix its null scores may remain at chance while the observed score is inflated. A `PASSED` status therefore means every applicable supported check passed; it is not proof that all possible leakage is absent.

### Confidence intervals

`corrected_t` is the default and uses the Nadeau–Bengio correction with actual mean train/test fold sizes. `oof_bootstrap` resamples pooled out-of-fold predictions and defaults to whole-group cluster resampling when groups are supplied; set `ci_cluster=False` only for row-level comparison. Non-partition splitters fall back to `corrected_t` with a warning. Legacy `ci_method="bootstrap"` remains available but warns because bootstrapping a handful of correlated fold scores produces intervals that are too narrow.
### Run the interactive notebooks

Prefer to learn by running code? From the repo root, open the 14 notebooks in `notebooks/`:

- `notebooks/01_IID_Methods_Showcase.ipynb` – Quick tour of IID splitters and metrics.
- `notebooks/02_Advanced_Workflow_UniversalRunner.ipynb` – End-to-end UniversalCVRunner workflow with callbacks.
- `notebooks/03_TrustCVValidator_Showcase.ipynb` – TrustCVValidator examples with leakage/balance checks.
- `notebooks/04_TrustCVValidator_IID_Comparison.ipynb` – Side-by-side IID method comparison.
- `notebooks/05_CrossValidation_Comparison.ipynb` – Comprehensive CV methods comparison across categories.
- `notebooks/06_TrustCV_DeepLearning_Showcase_with_Glossary.ipynb` – Deep learning framework integration (PyTorch & TensorFlow).
- `notebooks/07_trustcv_grouped_cv_medical.ipynb` – Grouped cross-validation for patient/subject-level data.
- `notebooks/08_TrustCV_Structured_Scenarios_S3.ipynb` – Structured validation scenarios for medical AI pipeline.
- `notebooks/09_RealData_UCI_HAR_Grouped_Leakage_NestedCV.ipynb` – UCI HAR sensor dataset grouped leakage analysis & nested CV.
- `notebooks/10_Advanced_Workflow_oob_clinical_metrics.ipynb` – Out-of-fold clinical metrics aggregation & diagnostics.
- `notebooks/11_Spatial_Synthetic_EnvironmentalHealth_TrustCV_SplitKwargs.ipynb` – Spatial & environmental health splitters with `split_kwargs`.
- `notebooks/12_PhysioNet2019_Sepsis_RealClinicalData_TrustCV_cloud_TEMPORAL.ipynb` – PhysioNet 2019 Sepsis clinical benchmark & temporal validation.
- `notebooks/13_LeakageChecker_Injection_Boundary_Validation.ipynb` – Data leakage checker injection boundary validation.
- `notebooks/14_LOGO_metric_feasibility_warning.ipynb` – Leave-One-Group-Out fold metric feasibility diagnostic system.
- `notebooks/14_Trust_Claims_v1.1.ipynb` – End-to-end v1.1.0 trust-claims validation: correct semantics, leakage checks, CI, and advisor workflow.

Reports generated by the notebooks are saved in `notebooks/reports/` (HTML/PDF).



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
