# Changelog

All notable changes to trustcv are documented in this file.

## [Unreleased]

### Fixed

- `TrustCV(method="bootstrap", bootstrap_validation_estimator=".632")` (and `".632+"`, or `cv=BootstrapValidation(estimator=...)`) silently reported the plain out-of-bag mean. `validate()` now applies the estimator: each fold's model is also scored on its training rows, and built-in scores in [0, 1] and mse/rmse/mae are corrected (`.632+` uses Efron's relative overfitting rate with a no-information error from shuffled out-of-bag labels). Other metrics keep the out-of-bag mean with a warning; details are in `result.diagnostics["bootstrap"]`.
- `SpatialBlockCV` tested only `n_splits` randomly chosen grid cells, so rows in the other cells were never tested (for example 4 of 9 cells with `n_splits=5`). Whole blocks are now assigned to folds, so every row is tested exactly once and no block is split. `block_size` (previously accepted but ignored) sets square cells on a fixed grid in coordinate units.
- `RegulatoryReport` HTML showed 0 for metrics and the confusion matrix when `calculate_clinical_metrics` was not called, and labelled the cross-validation score "Mean Accuracy" whatever metric it was. Missing values now read "not computed", and the score is labelled with its metric and shown with its interval and method.

## [1.2.2] - 2026-10-19

### Fixed

- `TorchCVRunner` and `MONAICVRunner` called the splitter without labels, so stratified splitters (`StratifiedGroupKFold`, `RepeatedKFold(stratify=True)`, ...) failed with "missing y". They now pass labels from a `(X, y)` tuple, a dataset's `targets`/`labels` attribute, or scalar MONAI `label` values, without loading items.

- `TrustCV.validate()` raises `ValueError` when the splitter produces no folds, instead of returning a result with no scores (with `groups=`, the group check could report PASSED over zero folds).
- `RegulatoryReport` HTML no longer states that the report "complies with" the chosen standard; a scope section says it does not establish compliance, and the approval block is a review and sign-off section. Docstrings no longer call the reports "FDA/CE-compliant".

- Docstrings of `TorchCVRunner`, `KerasCVRunner`, `ProgressLogger`, `RegulatoryComplianceLogger` and the `compliance` parameter of `TrustCVValidator` no longer claim to ensure regulatory compliance; the logger now prints "Validation log saved to ...".

### Added

- `NestedCV`, `NestedGroupedCV` and `NestedTemporalCV` take `n_jobs` (default `-1`, as before) for the inner grid search. It was hard-coded to all cores, which failed where worker processes cannot receive the estimator (for example code run with `python -c`, some notebook and cluster setups).

### Changed

- Documentation moves to MkDocs Material, with a generated API reference, `api-schema.json` and `llms.txt`; documentation code snippets run in CI. Several docstrings that contradicted their signatures are fixed.
- `trustcv.checks.OVERALL_STATUSES` lists the overall status values.

## [1.2.1] - 2026-10-17

### Fixed

- **`KerasCVRunner` selected models on the held-out fold by default.** It added `EarlyStopping(restore_best_weights=True)` and passed the CV test fold as `validation_data`, so each fold's best epoch was chosen on the data that then scored it. The held-out fold is now only scored. By default there is no validation data and no automatic early stopping (train for `epochs`). In a noise benchmark the old default's optimism was small (AUC 0.504 vs 0.491, within run-to-run noise), so expect small changes in reported scores.
- `KerasCVRunner` copies Keras callbacks per fold, so state such as `EarlyStopping.best` no longer carries from one fold to the next.
- `KerasCVRunner.run(..., verbose=0)` raised a duplicate-keyword error.
- Examples no longer select on the held-out fold. The PyTorch example reported the best test-fold AUC over epochs, MONAI the best test-fold accuracy, and TensorFlow early-stopped on the test fold. They now train for fixed epochs and score once. The TensorFlow bootstrap section computes the real AUC instead of a constant 0.5. Bundled recipe 7 drops `EarlyStopping` and fixes its Keras summary, whose error was hidden by a broad `except`.
- The test for duplicate column labels no longer depends on scikit-learn accepting duplicate column names (scikit-learn 1.9 rejects them).

### Added

- `KerasCVRunner.run(inner_validation=0.1)`: early stopping on an inner split of each training fold. It is group-aware with `groups=`, stratified when feasible, and `inner_validation_strategy="last"` handles chronological data. With no `EarlyStopping` passed, one is added on the inner split. Per-fold details are in `result.metadata["model_selection"]`.
- `trustcv.core.inner_split.inner_validation_split`, the framework-agnostic helper behind it.
- trustcv's `EarlyStopping` and `ModelCheckpoint(save_best_only=True)` warn once per CV run when they select on a `val_*` metric. The Torch, MONAI, JAX and Universal runners compute those on the held-out fold.

### Changed

- `KerasCVRunner(validation_split>0)` (Keras's last-fraction split) still works, but warns that it is not group-aware and can split repeated rows. It cannot be combined with `inner_validation`; passing `validation_data` is rejected.

## [1.2.0] - 2026-10-17

### Added

- New overall status `NEEDS_REVIEW`: a leakage-relevant check found signs of possible leakage (`WARNING`) that cannot be proven automatically. It ranks between `FAILED` and `NOT_FULLY_VERIFIED`.
- Independence scan when no groups are supplied: rows that come in unusually tight clusters (repeated subjects, augmentation, oversampling) make `group_leakage` a `WARNING`, also when independence was declared. Turn off with `TrustCV(check_independence=False)`.
- Identifier-column detection for DataFrames (`patient_id`, `PatientID`, `mrn`, ...), with a hint to drop the column and pass it as `groups=`. Repeated identifier values without groups make `group_leakage` a `WARNING`.
- `DataLeakageChecker.check(folds=...)` audits given `(train_idx, test_idx)` splits.
- Strong single predictors (`target_strong_threshold`, default 0.90) are named in the target-leakage message and recommendations without changing its status.
- `TrustCV(max_audit_folds=10)` caps how many scored folds the distance-based detector audits; overlap and exact-duplicate checks always cover every fold. Coverage is in `result.diagnostics["fold_audit"]`.
- The test suite refuses to run against a trustcv other than the checkout under test.

### Changed

- `validate()` audits the folds that were actually scored. Group-overlap, legacy integrity keys and the external detector no longer re-run the splitter or use the detector's own splits. Custom `leakage_checker`s without a `folds` parameter still work and say they used their own splits.
- Hints are actionable and use current API names, for example `TrustCV(method='stratified_group_kfold')` for group overlap. Repeated per-fold hints are deduplicated.
- Target-leakage messages name the features and their scores; inverse associations are shown as such.
- The Pipeline preprocessing message states that preprocessing applied before `validate()` is not visible.
- Covariate shift is `INFO` when group separation was verified on the scored folds.
- Confidence intervals of built-in metrics bounded to [0, 1] are clipped to that range (raw bounds kept in `diagnostics["confidence_intervals"][metric]["clipped_from"]`). Custom scorers are not clipped.
- Behaviour change: results that were `NOT_FULLY_VERIFIED` only because of a `WARNING` are now `NEEDS_REVIEW`, and declared-independent data that looks clustered changes from `PASSED` to `NEEDS_REVIEW`.

### Fixed

- Multi-input (dict/list) features no longer make the detector and the target scan error, which turned `overall_status` into `FAILED`; they are `NOT_CHECKED`.
- The detector's exact-duplicate check hashed the DataFrame index along with the values, so identical rows with different indices were missed. It now hashes values only and counts test rows.

## [1.1.2] - 2026-10-17

### Added

- `ClinicalMetrics.calculate_all(..., groups=...)`: the AUC confidence interval resamples whole patients when group ids are given, so repeated measures no longer give too-narrow intervals.
- `groups` must be 1-D with one entry per observation; a mismatch raises `ValueError` instead of silently resampling a different population.
- Regression tests for the AUC interval (pandas input with a non-default index, cluster bootstrap, single-class input).
- CI job that runs the TensorFlow/Keras tests against `trustcv[tensorflow]`.

### Fixed

- The AUC interval returns `(nan, nan)` instead of raising `IndexError` when no resample contains both classes. The `_auc_ci` docstring no longer claims DeLong; it is a bootstrap.

## [1.1.1] - 2026-10-17

### Added

- `trustcv` command line tool with `about` and `examples`. Seven runnable usage recipes now ship inside the package (`trustcv examples`, `--run`, `--copy`). Examples moved from `examples/` to `trustcv/examples/`; framework-specific scripts stay in `examples/`.

### Fixed

- `ClinicalMetrics.calculate_all` no longer raises `KeyError` when `y_true` is a pandas Series with a non-default index.

### Changed

- `dashboard()` no longer writes `trustcv_dashboard.html` into the current directory by default. Pass `save_path="<file>.html"` to save the dashboard to disk.

## [1.1.0] - 2026-09-21

### Trust and validation semantics

- The default confidence interval method is now **corrected_t**; **oof_bootstrap** adds pooled OOF and group-cluster resampling. Legacy fold-score **bootstrap** remains available with a warning that correlated fold-score intervals are too narrow.
- A reported **PASSED** overall status now requires every applicable leakage-relevant check to be PASSED or NOT_APPLICABLE. Unknown and unverified conditions produce **NOT_FULLY_VERIFIED**.
- Class imbalance is now **INFO** or **WARNING**, never FAILED.
- Added structured **CheckResult** entries, **overall_status**, automatic target-feature scanning, an opt-in permutation CV-loop sanity check, and **recommend_cv**.
- Added **external_leakage_detected**. The inverted legacy **has_leakage** result key is deprecated for one release.
- KS tests now report Bonferroni-corrected covariate shift and never create leakage types.
- Near-duplicate detection now uses train-standardized Euclidean distance calibrated to within-training nearest-neighbour spacing instead of cosine similarity.
- Preprocessing is reported PASSED only for visible sklearn Pipeline steps; preprocessing applied before validate() cannot be detected.
- Legacy leakage keys no longer report a pass when the leakage detector fails or cannot run.

### Differences from the version described in the paper (v1.0.7)

- **Confidence intervals:** v1.0.7 defaulted to bootstrapping a small set of correlated fold scores. v1.1.0 defaults to the Nadeau–Bengio corrected resampled t interval and offers pooled OOF bootstrap intervals.
- **PASSED semantics:** v1.0.7 could report leakage PASSED when evidence was missing or a detector failed. v1.1.0 reports PASSED only when every applicable leakage-relevant structured check passed; incomplete evidence is NOT_FULLY_VERIFIED.
- **Near-duplicate detection:** v1.0.7 used a fixed cosine-similarity threshold that produced false positives in low dimensions. v1.1.0 uses train-standardized Euclidean distance calibrated to within-training nearest-neighbour spacing.

## [1.0.7] - 2026-03-16

### Bug Fixes

- **Fixed severity ranking in DataLeakageChecker** — The `check_cv_splits()` method used Python's `max()` on severity strings (`"critical"`, `"high"`, etc.), which does alphabetical comparison. This caused `max("critical", "high")` to incorrectly return `"high"`. Replaced with explicit rank-based `_worst_severity()` helper using `{"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}`.

- **Fixed temporal leakage detection being too aggressive** — Previously flagged leakage whenever `min_test < max_train` (any date overlap), which triggers even for normal k-fold scenarios. Now only flags when test data appears **before** training starts (`min_test < min_train`) or when >50% of test samples fall within the training period. Added `overlap_fraction` and `test_before_train_fraction` metrics.

- **Replaced bare `except:` clauses** — Bare `except:` in temporal timestamp parsing and feature correlation now catch specific exceptions (`ValueError`, `TypeError`, `OverflowError`) instead of silently swallowing all errors.

### New Features

- **Near-duplicate detection** (`check_near_duplicates()`) — Detects near-duplicate samples between train/test sets using cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`). Configurable similarity threshold (default 0.99). Automatically integrated into `check_cv_splits()`.

- **Hierarchical leakage detection** (`check_hierarchical_leakage()`) — Checks if parent-level groups (e.g., hospitals) leak across train/test sets, even when child-level groups (e.g., patients) are properly separated. Useful for multi-level medical data (Hospital → Department → Patient).

- **Auto-computed spatial threshold** — `check()` and `check_cv_splits()` now auto-compute a spatial proximity threshold (10th percentile of pairwise distances) when coordinates are provided but no explicit threshold is given. Previously spatial checks were silently skipped in the convenience `check()` method.

- **LeakageDetectionCallback** — New callback for `UniversalCVRunner` that automatically checks each fold for data leakage during cross-validation. Reports per-fold leakage findings and aggregates a summary at the end. Usage: `LeakageDetectionCallback(data=(X, y), groups=patient_ids, verbose=1)`.

- **Auto-running DataLeakageChecker in TrustCVValidator** — When `check_leakage=True` (the default), `TrustCVValidator.validate()` now automatically creates and runs a `DataLeakageChecker` internally, without requiring the user to pass one explicitly. The full suite of leakage checks (patient, duplicate, near-duplicate, temporal, feature statistics, spatial, label distribution) runs automatically.

- **Truly comprehensive `comprehensive_check()`** — Previously only ran `check_feature_target_leakage()`. Now also runs the full CV-based `check()` suite and merges results, recommendations, and severity into a single report with `overall_severity`.

### Quality Improvements

- **KS-test for feature statistics** — `_check_feature_statistics()` now runs a Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) on each feature in addition to the existing mean/std comparison. Features with KS p-value < 0.001 are flagged as having significantly different distributions. Reports `ks_suspicious_features` count and `ks_pvalues` list.

- **Chi-squared test for label distribution** — `_check_label_distribution()` now runs a chi-squared contingency test (`scipy.stats.chi2_contingency`) alongside the existing max-difference check. Reports `chi2_pvalue` for statistical significance of distribution differences.

- **Temporal overlap quantification** — `_check_temporal_leakage()` now computes and returns `overlap_fraction` (fraction of test samples with timestamps within the training period) and `test_before_train_fraction` (fraction of test samples before training starts).

### Documentation

- **AI Agent documentation** — Added machine-readable documentation files for AI coding agents:
  - `website/llms.txt` — Concise API reference following the [llms.txt standard](https://llmstxt.org/)
  - `website/llms-full.txt` — Extended reference with all 29 CV methods and full signatures
  - `website/api-schema.json` — Structured JSON API schema for programmatic agents
  - `.github/copilot-instructions.md` — GitHub Copilot repo instructions
  - `.cursorrules` — Cursor IDE project rules
  - `website/ai-agents.html` — Website page documenting AI agent support with setup guides

- **All documentation updated** for v1.0.7 leakage improvements (CLAUDE.md, llms.txt, llms-full.txt, api-schema.json, copilot-instructions.md, cursorrules).

### Tests

- **47 new unit tests** in `tests/test_leakage_improvements.py` covering:
  - Severity ranking correctness (5 tests)
  - Temporal leakage fix (6 tests)
  - Near-duplicate detection (6 tests)
  - Hierarchical leakage (3 tests)
  - Spatial auto-threshold (2 tests)
  - KS-test feature statistics (3 tests)
  - Chi-squared label distribution (2 tests)
  - Comprehensive check integration (2 tests)
  - LeakageDetectionCallback (4 tests)
  - Validator auto-checker (2 tests)
  - LeakageReport dataclass (3 tests)
  - check() convenience method (4 tests)
  - Preprocessing leakage (2 tests)
  - Existing 9 tests continue to pass

## [1.0.6] - Previous Release

Initial public release with 29 CV methods, DataLeakageChecker, BalanceChecker, ClinicalMetrics, UniversalCVRunner, and framework adapters for PyTorch, TensorFlow, MONAI, and JAX.
