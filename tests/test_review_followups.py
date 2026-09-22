"""Regression tests for the independent review follow-ups."""

import numpy as np
import pytest
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

from trustcv import TrustCV
from trustcv.checkers.leakage import DataLeakageChecker
from trustcv.uncertainty import _corrected_t_interval, _oof_bootstrap_interval


class _FailsAfterPrimaryCV(ClassifierMixin, BaseEstimator):
    fit_calls = 0

    def __init__(self, successful_fits=3):
        self.successful_fits = successful_fits

    def fit(self, X, y):
        type(self).fit_calls += 1
        if type(self).fit_calls > self.successful_fits:
            raise RuntimeError("deliberate permutation fit failure")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def test_oof_bootstrap_multiclass_metrics_use_global_labels():
    """A missing rare class must not change a resample's macro averaging."""
    y_true = np.array([0] * 20 + [1] * 20 + [2])
    y_pred = y_true.copy()
    y_pred[-1] = 0
    groups = np.arange(y_true.size)
    labels = np.unique(y_true)
    seed = 12
    n_bootstrap = 200

    interval, details = _oof_bootstrap_interval(
        "f1_macro",
        y_true=y_true,
        y_pred=y_pred,
        y_score=None,
        groups=groups,
        cluster=True,
        regression=False,
        n_bootstrap=n_bootstrap,
        level=0.95,
        rng=np.random.default_rng(seed),
    )

    rng = np.random.default_rng(seed)
    expected = []
    legacy = []
    missing_rare_class = 0
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(groups, size=groups.size, replace=True)
        sampled_y = y_true[sampled_groups]
        sampled_pred = y_pred[sampled_groups]
        if np.unique(sampled_y).size < 2:
            continue
        missing_rare_class += int(2 not in sampled_y)
        expected.append(
            f1_score(
                sampled_y,
                sampled_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
        legacy.append(
            f1_score(sampled_y, sampled_pred, average="macro", zero_division=0)
        )

    assert missing_rare_class > 0
    assert details["skipped_resamples"] == n_bootstrap - len(expected)
    assert interval == pytest.approx(np.percentile(expected, [2.5, 97.5]))
    assert interval != pytest.approx(np.percentile(legacy, [2.5, 97.5]))


def test_oof_bootstrap_stratifies_imbalanced_ungrouped_classification():
    y_true = np.array([0] * 95 + [1] * 5)
    y_score = np.linspace(0.0, 1.0, y_true.size)

    _, details = _oof_bootstrap_interval(
        "roc_auc",
        y_true=y_true,
        y_pred=None,
        y_score=y_score,
        groups=None,
        cluster=False,
        regression=False,
        n_bootstrap=200,
        level=0.95,
        rng=np.random.default_rng(7),
    )

    assert details["skipped_resamples"] == 0
    assert details["skipped_fraction"] == 0.0


def test_oof_cluster_bootstrap_warns_when_many_resamples_are_skipped():
    groups = np.repeat(np.arange(20), 5)
    y_true = np.repeat(np.array([0] * 19 + [1]), 5)
    y_score = y_true.astype(float)

    with pytest.warns(UserWarning, match="skipped.*10%"):
        _, details = _oof_bootstrap_interval(
            "roc_auc",
            y_true=y_true,
            y_pred=None,
            y_score=y_score,
            groups=groups,
            cluster=True,
            regression=False,
            n_bootstrap=200,
            level=0.95,
            rng=np.random.default_rng(9),
        )

    assert details["skipped_resamples"] > 20
    assert details["skipped_fraction"] > 0.10


def test_permutation_nan_observed_score_is_an_error():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(60, 4))
    y = np.tile([0, 1], 30)
    _FailsAfterPrimaryCV.fit_calls = 0

    result = TrustCV(
        method="stratified_kfold",
        n_splits=3,
        check_leakage=False,
        permutation_check=True,
        n_permutations=3,
        random_state=0,
    ).validate(model=_FailsAfterPrimaryCV(), X=X, y=y)

    check = result.checks["permutation_sanity"]
    assert np.isnan(result.permutation["observed"])
    assert all(np.isnan(score) for score in result.permutation["null_scores"])
    assert np.isnan(result.permutation["p_value"])
    assert check.status == "ERROR"
    assert "NaN" in check.message or "undefined" in check.message.lower()


def test_near_duplicates_calibrate_after_deduplicating_training_rows():
    rng = np.random.default_rng(31)
    unique_train = rng.normal(size=(40, 5))
    X_train = np.vstack([np.repeat(unique_train[:1], 60, axis=0), unique_train])
    X_near = unique_train[1:11] + rng.normal(scale=1e-4, size=(10, 5))
    X_independent = rng.normal(loc=8.0, size=(100, 5))
    checker = DataLeakageChecker(verbose=False)

    near_report = checker.check_cv_splits(X_train, X_near)
    independent_report = checker.check_cv_splits(X_train, X_independent)

    assert "near_duplicate" in near_report.leakage_types
    assert near_report.details["near_duplicate_leakage"]["near_duplicate_count"] == 10
    assert "near_duplicate" not in independent_report.leakage_types


def test_corrected_t_uses_actual_uneven_fold_sizes_and_requires_them():
    scores = np.array([0.61, 0.68, 0.72, 0.66, 0.75])
    train_sizes = np.array([82, 82, 82, 83, 83])
    test_sizes = np.array([21, 21, 21, 20, 20])
    validator = TrustCV(ci_method="corrected_t", ci_level=0.95)

    interval = validator._calculate_confidence_intervals(
        {"test_score": scores},
        train_sizes=train_sizes,
        test_sizes=test_sizes,
    )["score"]

    variance = np.var(scores, ddof=1)
    correction = 1 / scores.size + np.mean(test_sizes) / np.mean(train_sizes)
    margin = stats.t.ppf(0.975, scores.size - 1) * np.sqrt(correction * variance)
    expected = (np.mean(scores) - margin, np.mean(scores) + margin)
    assert interval == pytest.approx(expected)

    with pytest.raises(ValueError, match="train_sizes.*test_sizes"):
        validator._compute_confidence_interval(scores)


def test_corrected_t_docstring_states_repeated_cv_formula_and_sources():
    docstring = _corrected_t_interval.__doc__ or ""

    assert "df = r * k - 1" in docstring
    assert "(1 / (r * k) + n_test / n_train) * s^2" in docstring
    assert "Nadeau & Bengio (2003, Machine Learning 52:239-281)" in docstring
    assert "Bouckaert & Frank (2004, PAKDD)" in docstring


# ---------------------------------------------------------------------------
# Legacy leakage keys when the detector fails or cannot run
# ---------------------------------------------------------------------------

class _CrashingChecker:
    """Leakage checker whose check() always raises."""

    def check(self, *args, **kwargs):
        raise RuntimeError("deliberate crash")


def test_legacy_leakage_keys_on_crashing_checker():
    """When the external leakage detector crashes, legacy keys must not
    silently report a pass (has_leakage=True in v1.0.7 semantics).
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(99)
    X = rng.normal(size=(60, 4))
    y = np.tile([0, 1], 30)

    result = TrustCV(
        method="stratified_kfold",
        n_splits=3,
        check_leakage=True,
        random_state=0,
    ).validate(
        model=LogisticRegression(max_iter=3000),
        X=X,
        y=y,
        leakage_checker=_CrashingChecker(),
    )

    # has_leakage=False means "NOT passed" in legacy semantics
    assert result.leakage_check["has_leakage"] is False
    # external_leakage_detected=None means "unknown / not checked"
    assert result.leakage_check["external_leakage_detected"] is None
    # The structured check must be ERROR
    assert result.checks["external_leakage_detector"].status == "ERROR"
    # Overall status must be FAILED (ERROR ∈ LEAKAGE_RELEVANT_KEYS)
    assert result.overall_status == "FAILED"
    # Summary must not say PASSED on the external leakage detector line
    for line in result.summary().splitlines():
        if "external leakage" in line.lower():
            assert "PASSED" not in line


def test_legacy_leakage_keys_when_no_checker_can_be_built():
    """When check_leakage=True but no checker is available (e.g. import
    fails or construction raises), legacy keys must not report a pass.
    """
    from unittest.mock import patch
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = np.tile([0, 1], 30)

    # Simulate DataLeakageChecker import failing so effective_checker stays None
    with patch(
        "trustcv.validators.build_initial_checks",
        wraps=__import__("trustcv.checks", fromlist=["build_initial_checks"]).build_initial_checks,
    ):
        # Force the auto-creation path to fail by patching the import
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        import builtins

        real_import = builtins.__import__

        def _failing_dlc_import(name, *args, **kwargs):
            if name == "trustcv.checkers.leakage" or (
                len(args) > 2 and args[2] and "DataLeakageChecker" in (args[2] or [])
            ):
                raise ImportError("simulated import failure")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_failing_dlc_import):
            result = TrustCV(
                method="stratified_kfold",
                n_splits=3,
                check_leakage=True,
                random_state=0,
            ).validate(
                model=LogisticRegression(max_iter=3000),
                X=X,
                y=y,
            )

    # has_leakage=False means "NOT passed" in legacy semantics
    assert result.leakage_check["has_leakage"] is False
    # The structured check must be NOT_CHECKED or ERROR
    assert result.checks["external_leakage_detector"].status in {"NOT_CHECKED", "ERROR"}
