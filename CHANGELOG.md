# Changelog

All notable changes to trustcv are documented in this file.

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
