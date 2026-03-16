# trustcv — GitHub Copilot Instructions

## Project
**trustcv** (v1.0.7) — Framework-agnostic toolkit for trustworthy cross-validation in safety-critical/medical AI settings. Developed at SMAILE, Karolinska Institutet. Python 3.8+.

- PyPI: https://pypi.org/project/trustcv/
- GitHub: https://github.com/ki-smile/trustcv

## Architecture

```
trustcv/
├── core/           # CVResults, CVSplitter, FrameworkAdapter, UniversalCVRunner
├── splitters/      # 29 CV methods: iid, grouped, temporal, spatial (+multilabel)
├── checkers/       # DataLeakageChecker (8 leakage types), BalanceChecker
├── metrics/        # ClinicalMetrics with confidence intervals
├── frameworks/     # Adapters: pytorch, tensorflow, monai, jax (lazy-imported)
├── reporting/      # RegulatoryReport for FDA/CE MDR documentation
├── datasets/       # Medical dataset loaders and synthetic generators
├── visualization/  # Plotting utilities
└── validators.py   # TrustCVValidator (alias: TrustCV), MedicalValidator
```

**Key flow**: `TrustCVValidator` orchestrates splitters → leakage/balance checks → CV execution (via `UniversalCVRunner`) → clinical metrics → regulatory reports.

All splitters follow the scikit-learn splitter interface (`split(X, y, groups)` yielding train/test indices).

Framework adapters are optional and lazy-imported — core functionality works with sklearn alone.

## Commands

```bash
pip install -e .            # editable install
pip install -e .[dev]       # with dev dependencies
pip install -e .[all]       # all optional frameworks
pytest tests/               # all tests
black trustcv/ --line-length 100
isort trustcv/ --profile black
flake8 trustcv/
mypy trustcv/
```

## Conventions

- **Line length**: 100 (black)
- **Import sorting**: isort with black profile
- **Commits**: conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- **Version**: synchronized across `pyproject.toml`, `setup.py`, and `trustcv/__init__.py`
- Backward-compatible aliases exist with deprecation warnings (e.g., `KFoldMedical` → `KFold`)

## Key APIs

### TrustCVValidator (alias: TrustCV)
```python
from trustcv import TrustCV

validator = TrustCV(
    method='stratified_kfold',  # CV method
    n_splits=5,                 # number of folds
    random_state=42,            # seed
    check_leakage=True,         # leakage detection
    check_balance=True,         # class balance check
    compliance=None,            # 'FDA', 'CE', or None
)

# Cross-validation (keyword-only args)
results = validator.validate(model=model, X=X, y=y, groups=patient_ids)
# results.mean_scores, results.confidence_intervals, results.summary()

# Train + evaluate on held-out test
results = validator.fit_validate(model=model, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
```

### UniversalCVRunner
```python
from trustcv import UniversalCVRunner, StratifiedKFold

runner = UniversalCVRunner(cv_splitter=StratifiedKFold(n_splits=5), framework='auto')
results = runner.run(model=model, data=(X, y), epochs=50)
```

### DataLeakageChecker
```python
from trustcv import DataLeakageChecker

checker = DataLeakageChecker()
report = checker.check(X, y, groups=patient_ids, timestamps=dates)
# report.has_leakage, report.severity, report.leakage_types, report.recommendations

# LeakageDetectionCallback — auto-checks each fold during CV
from trustcv.core.callbacks import LeakageDetectionCallback
leakage_cb = LeakageDetectionCallback(data=(X, y), groups=patient_ids)
```

### Splitter Categories
- **IID** (9): HoldOut, KFold, StratifiedKFold, RepeatedKFold, LOOCV, LPOCV, BootstrapValidation, MonteCarloCV, NestedCV
- **Grouped** (7): GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut, LeavePGroupsOut, RepeatedGroupKFold, NestedGroupedCV, HierarchicalGroupKFold
- **Temporal** (8): TimeSeriesSplit, BlockedTimeSeries, RollingWindowCV, ExpandingWindowCV, PurgedKFoldCV, CombinatorialPurgedCV, PurgedGroupTimeSeriesSplit, NestedTemporalCV
- **Spatial** (4): SpatialBlockCV, BufferedSpatialCV, SpatiotemporalBlockCV, EnvironmentalHealthCV
- **Multilabel** (2): MultilabelStratifiedKFold, MultilabelStratifiedGroupKFold

## Important Patterns
- Always use `groups=patient_ids` when patients have multiple samples to prevent leakage
- `validate()` uses keyword-only arguments: `validator.validate(model=m, X=X, y=y)`
- Method names are flexible: `'stratified_kfold'`, `'stratifiedkfold'`, `'StratifiedKFold'` all work
- Old class names emit deprecation warnings — prefer canonical names (KFold over KFoldMedical)
