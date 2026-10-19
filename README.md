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

> **Status:** v1.2.2 – New documentation site with a generated API reference and executed examples; Torch/MONAI runners work with stratified splitters; validation refuses splitters that produce no folds; reports and docstrings no longer claim regulatory compliance.

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

## What's in trustcv

trustcv includes **32 cross-validation methods** (the 29 in the paper plus later additions):

- **IID splitters** (9 methods):
  `HoldOut`, `KFold`, `StratifiedKFold`, `RepeatedKFold`, `LeaveOneOut`, `LeavePOut`, `BootstrapValidation`, `MonteCarloCV`, `NestedCV`

- **Grouped splitters** (8 methods):
  `GroupKFold`, `StratifiedGroupKFold`, `LeaveOneGroupOut`, `LeavePGroupsOut`, `RepeatedGroupKFold`, `HierarchicalGroupKFold`, `MultilevelCV`, `NestedGroupedCV`

- **Temporal splitters** (8 methods):
  `TimeSeriesSplit`, `BlockedTimeSeriesSplit`, `RollingWindowCV`, `ExpandingWindowCV`, `PurgedKFold`, `CombinatorialPurgedKFold`, `PurgedGroupTimeSeriesSplit`, `NestedTemporalCV`

- **Spatial splitters** (4 methods):
  `SpatialBlockCV`, `BufferedSpatialCV`, `SpatiotemporalBlockCV`, `EnvironmentalHealthCV`

- **Multilabel splitters** (3 methods, added after the paper):
  `MultilabelStratifiedKFold`, `MultilabelStratifiedGroupKFold`, `MultiLabelGroupSplitter`

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

Real output from v1.2.0:

```text
iid stratified_kfold StratifiedKFoldMedical
=== Trustworthy Cross-Validation Results ===

Performance Metrics (mean +/- std) (method: corrected_t):
  accuracy: 0.974 +/- 0.019 [95% CI (corrected_t): 0.939-1.000]
  roc_auc: 0.995 +/- 0.006 [95% CI (corrected_t): 0.984-1.000]
  sensitivity: 0.992 +/- 0.013 [95% CI (corrected_t): 0.968-1.000]
  specificity: 0.944 +/- 0.059 [95% CI (corrected_t): 0.834-1.000]
  precision: 0.968 +/- 0.032 [95% CI (corrected_t): 0.908-1.000]
  recall: 0.992 +/- 0.013 [95% CI (corrected_t): 0.968-1.000]
  f1: 0.979 +/- 0.014 [95% CI (corrected_t): 0.953-1.000]

Data Integrity Checks:
  Duplicate Samples: PASSED — No exact duplicate feature rows were found.
  Group Leakage: NOT_APPLICABLE — Samples were explicitly declared independent; no groups were supplied.
  Preprocessing Leakage: PASSED — The model is an sklearn Pipeline, so its steps are refit within each fold. Preprocessing applied to X before validate() (feature selection, imputation, scaling, encoding, resampling) is not visible to this check.
  External Leakage Detector: PASSED — The external leakage detector found no supported leakage pattern (5 of 5 scored folds audited).
  Target Leakage Features: PASSED — No feature almost determines the target on its own. 11 feature(s) are very strong single predictors: feature_0 (AUC 0.94, inverse), feature_2 (AUC 0.95, inverse), feature_3 (AUC 0.94, inverse), ...; confirm they are known at prediction time.
  Permutation Sanity: NOT_CHECKED — Permutation sanity checking is disabled.
  Class Balance: INFO — Minority-to-majority class ratio is 0.594.
  Covariate Shift: INFO — No statistically significant covariate shift was detected.

Overall Status: PASSED
Leakage Check: PASSED

Recommendations:
  - Very strong single predictor(s) 'feature_0', 'feature_2', 'feature_3': confirm each is available before the outcome is known; leaked fields (codes or results recorded after diagnosis) often look like this.
```

## Integrity checks and status semantics

Every `ValidationResult.checks` mapping contains the following keys:

| Check | Status behavior | Meaning |
| --- | --- | --- |
| `duplicate_samples` | `PASSED` or `WARNING` | Exact row-value hashes (the DataFrame index is ignored); reports how many test rows have an identical training row in the scored folds. |
| `group_leakage` | `PASSED`, `FAILED`, `WARNING`, `ERROR`, `NOT_CHECKED`, or `NOT_APPLICABLE` | With groups: overlap is checked on every scored fold. Without groups: `WARNING` when rows look clustered (repeated subjects, augmentation, oversampling) or an identifier column repeats, even if independence was declared. |
| `preprocessing_leakage` | `PASSED` or `NOT_CHECKED` | Passes only when preprocessing is inside an sklearn `Pipeline` before the final estimator. |
| `external_leakage_detector` | `PASSED`, `FAILED`, `ERROR`, or `NOT_CHECKED` | Audits the folds that were scored (up to `max_audit_folds` for distance checks). A crash is an error, never a silent pass; multi-input features are `NOT_CHECKED`. |
| `target_leakage_features` | `PASSED`, `WARNING`, `NOT_CHECKED`, or `NOT_APPLICABLE` | `WARNING` names features that almost determine the target. Very strong single predictors (`target_strong_threshold`, default 0.90) are listed without changing the status, because real biomarkers look the same. |
| `permutation_sanity` | `PASSED`, `FAILED`, or `NOT_CHECKED` | Opt-in check for shuffled labels scoring above chance inside the CV loop. |
| `class_balance` | `INFO` or `WARNING` | Imbalance describes the data and is never a failed leakage check. |
| `covariate_shift` | `INFO`, `WARNING`, or `NOT_CHECKED` | Bonferroni-corrected KS shift report; shift is not leakage. `INFO` when group separation was verified, since held-out groups are expected to differ. |

`overall_status` considers the first five leakage-relevant checks and the permutation check when enabled:

| Status | Meaning |
| --- | --- |
| `FAILED` | A relevant check is `FAILED` or `ERROR`. |
| `NEEDS_REVIEW` | A relevant check found signs of possible leakage (`WARNING`) that cannot be proven automatically. A person should confirm them. |
| `NOT_FULLY_VERIFIED` | Nothing suspicious, but some condition could not be checked. |
| `PASSED` | Every relevant check is `PASSED` or `NOT_APPLICABLE`. |

`leakage_check` remains for compatibility. Use `external_leakage_detected` for non-inverted semantics. The legacy `has_leakage` key is deprecated for one release and is inverted: `external_leakage_detected == (not has_leakage)`.

### What trustcv can and cannot detect

TrustCV can verify exact duplicates, supplied-group separation, whether visible sklearn preprocessing is inside a `Pipeline`, selected leakage patterns in `DataLeakageChecker`, near-deterministic single-feature target encodings, class balance, covariate shift, and—when enabled—whether shuffled labels score above chance inside the CV loop.

When no groups are supplied, a scan looks for rows that come in unusually tight clusters (repeated subjects, augmentation, oversampling) and for identifier-like columns. A hit means `NEEDS_REVIEW`, not proof: tight clusters of genuinely independent samples look the same (`check_independence=False` turns the scan off). TrustCV cannot otherwise infer missing patient IDs, timestamps, coordinates, or causal relationships. It cannot detect preprocessing or feature selection already applied to the full dataset before `validate()` receives `X`. The permutation check is not a general leakage detector: with a fixed pre-selected matrix its null scores may remain at chance while the observed score is inflated. A `PASSED` status therefore means every applicable supported check passed; it is not proof that all possible leakage is absent.

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
- `notebooks/15_Trust_Claims_v1.1.ipynb` – End-to-end v1.1.0 trust-claims validation: correct semantics, leakage checks, CI, and advisor workflow.

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

See the [quickstart](https://ki-smile.github.io/trustcv/get-started/quickstart.html) on the documentation site.


## Repository Structure

```
trustcv/       # Python package (splitters, validators, metrics, core, CLI, bundled examples)
docs/          # Documentation site source (MkDocs Material; build with `mkdocs serve`)
notebooks/     # Jupyter tutorials
examples/      # Heavier framework examples (PyTorch, TensorFlow, JAX, MONAI)
tests/         # Unit & integration tests (including executed documentation snippets)
```

## Development

```bash
pip install -e .[dev]
pytest tests/
cd docs && make html
```

## Command Line

`pip install trustcv` ships a `trustcv` command with 7 runnable usage recipes:

```bash
trustcv --help               # Quick start
trustcv about                # Project info, links, citation
trustcv examples             # 7 numbered usage recipes
trustcv examples 1 --run     # Run a recipe
trustcv examples 3 --copy .  # Copy a recipe to edit
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
