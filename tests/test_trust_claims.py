"""
Acceptance tests for the trustcv "trust claims" fixes.

Place this file at tests/test_trust_claims.py.
All tests here FAIL on trustcv v1.0.7 except two guard tests
(test_honest_pipeline_on_noise_gives_chance_level_auc and
test_near_duplicate_check_still_catches_real_near_duplicates), which pass today
and must keep passing. Every test must PASS after the fix.
Do not weaken, skip, or loosen thresholds in these tests to make them pass.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustcv import TrustCV
from trustcv.checkers.leakage import DataLeakageChecker

LEAKAGE_KEYS = (
    "duplicate_samples",
    "group_leakage",
    "preprocessing_leakage",
    "external_leakage_detector",
    "target_leakage_features",
)
VALID_STATUSES = {
    "PASSED", "FAILED", "WARNING", "NOT_CHECKED", "NOT_APPLICABLE", "INFO", "ERROR",
}


# --------------------------------------------------------------------------- helpers
def _lr():
    return LogisticRegression(max_iter=2000)


def _status(res, key):
    assert hasattr(res, "checks"), "ValidationResult must expose a `checks` dict"
    assert key in res.checks, f"missing check '{key}' in results.checks"
    st = res.checks[key].status
    assert st in VALID_STATUSES, f"invalid status {st!r} for {key}"
    return st


def _leaky_noise_data(seed=0):
    """Pure noise; feature selection done on the FULL dataset (the classic leak)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 5000))
    y = rng.integers(0, 2, 100)
    Xs = SelectKBest(k=20).fit_transform(X, y)
    return X, Xs, y


class _OverlappingCV(BaseCrossValidator):
    """Deliberately broken splitter: every test fold is also inside the training set."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        n = len(X)
        for fold in np.array_split(np.arange(n), self.n_splits):
            yield fold

    def split(self, X, y=None, groups=None):
        n = len(X)
        for test in self._iter_test_indices(X, y, groups):
            yield np.arange(n), test  # train = ALL rows (includes test)


class _CrashingChecker:
    def check(self, *args, **kwargs):
        raise RuntimeError("detector crashed")


# ------------------------------------------------ 1. no false "PASSED" on preprocessing leak
def test_upstream_preprocessing_leak_is_never_reported_as_passed():
    _, Xs, y = _leaky_noise_data()
    res = TrustCV(method="stratified_kfold", n_splits=5, random_state=0).validate(
        model=_lr(), X=Xs, y=y
    )
    assert _status(res, "preprocessing_leakage") == "NOT_CHECKED"
    assert res.overall_status != "PASSED"
    s = res.summary()
    assert "Leakage Check: PASSED" not in s
    assert ("NOT_CHECKED" in s) or ("NOT CHECKED" in s)
    assert any("Pipeline" in r for r in res.recommendations), (
        "must recommend wrapping preprocessing in an sklearn Pipeline"
    )


def test_pipeline_with_transformer_marks_preprocessing_checked_and_can_pass():
    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), _lr())
    res = TrustCV(
        method="stratified_kfold", n_splits=5, random_state=0,
        declare_independent_samples=True,
    ).validate(model=model, X=X, y=y)
    assert _status(res, "preprocessing_leakage") == "PASSED"
    assert _status(res, "group_leakage") == "NOT_APPLICABLE"
    assert res.overall_status == "PASSED", res.summary()


def test_honest_pipeline_on_noise_gives_chance_level_auc():
    """Sanity: the correct workflow (selection inside the Pipeline) is not inflated."""
    X, _, y = _leaky_noise_data()
    model = make_pipeline(SelectKBest(k=20), _lr())
    res = TrustCV(method="stratified_kfold", n_splits=5, random_state=0).validate(
        model=model, X=X, y=y
    )
    assert res.mean_scores["roc_auc"] < 0.65


# ------------------------------------------------------------- 2. three-state group check
def test_group_leakage_not_checked_without_groups():
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV(method="stratified_kfold", n_splits=5).validate(model=_lr(), X=X, y=y)
    assert _status(res, "group_leakage") == "NOT_CHECKED"
    assert res.overall_status != "PASSED"


def test_group_leakage_passed_and_failed_when_groups_given():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 5)
    X = rng.normal(size=(200, 5))
    y = np.repeat(rng.integers(0, 2, 40), 5)
    ok = TrustCV(method="patient_grouped_kfold", n_splits=5).validate(
        model=_lr(), X=X, y=y, groups=groups
    )
    assert _status(ok, "group_leakage") == "PASSED"
    bad = TrustCV(method="kfold", n_splits=5, shuffle=True, random_state=0).validate(
        model=_lr(), X=X, y=y, groups=groups
    )
    assert _status(bad, "group_leakage") == "FAILED"
    assert bad.overall_status == "FAILED"


def test_crashing_detector_is_error_not_silent_pass():
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV(method="stratified_kfold", n_splits=5).validate(
        model=_lr(), X=X, y=y, leakage_checker=_CrashingChecker()
    )
    assert _status(res, "external_leakage_detector") == "ERROR"
    assert res.overall_status != "PASSED"


def test_duplicates_are_checked_for_numpy_arrays_too():
    X, y = load_breast_cancer(return_X_y=True)
    X = np.vstack([X, X[:10]])
    y = np.concatenate([y, y[:10]])
    res = TrustCV(method="stratified_kfold", n_splits=5).validate(model=_lr(), X=X, y=y)
    assert _status(res, "duplicate_samples") in {"WARNING", "FAILED"}


# ------------------------------------------------------------ 3. target-leakage features
def test_single_feature_that_encodes_the_label_is_flagged():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    X = rng.normal(size=(300, 10))
    X[:, 3] = y + rng.normal(scale=0.01, size=300)  # leaked label
    res = TrustCV(method="stratified_kfold", n_splits=5).validate(model=_lr(), X=X, y=y)
    assert _status(res, "target_leakage_features") in {"WARNING", "FAILED"}
    assert 3 in res.checks["target_leakage_features"].details["flagged_features"]
    assert res.overall_status != "PASSED"


# ------------------------------------------------------------- 4. permutation sanity check
def test_permutation_check_is_opt_in():
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV(method="stratified_kfold", n_splits=5).validate(model=_lr(), X=X, y=y)
    assert _status(res, "permutation_sanity") == "NOT_CHECKED"


def test_permutation_check_detects_leakage_inside_the_cv_loop():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(150, 5))
    y = rng.integers(0, 2, 150)
    res = TrustCV(
        method="stratified_kfold", permutation_check=True, n_permutations=10, random_state=0
    ).validate(model=KNeighborsClassifier(n_neighbors=1), X=X, y=y, cv=_OverlappingCV())
    assert _status(res, "permutation_sanity") == "FAILED"
    p = res.permutation
    for k in ("metric", "observed", "null_scores", "null_mean", "p_value", "chance_level"):
        assert k in p
    assert p["null_mean"] > p["chance_level"] + 0.1
    assert res.overall_status == "FAILED"


def test_permutation_check_passes_real_signal():
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV(
        method="stratified_kfold", permutation_check=True, n_permutations=20, random_state=0
    ).validate(model=make_pipeline(StandardScaler(), _lr()), X=X, y=y)
    assert _status(res, "permutation_sanity") == "PASSED"
    assert res.permutation["p_value"] < 0.1
    assert abs(res.permutation["null_mean"] - 0.5) < 0.1


# --------------------------------------------------------------- 5. confidence intervals
def _coverage(ci_method, n_seeds=40):
    hits = 0
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        X = rng.normal(size=(120, 50))
        y = rng.integers(0, 2, 120)
        res = TrustCV(
            method="stratified_kfold", n_splits=5, random_state=s,
            ci_method=ci_method, n_bootstrap=500,
        ).validate(model=_lr(), X=X, y=y)
        lo, hi = res.confidence_intervals["roc_auc"]
        hits += lo <= 0.5 <= hi
    return hits / n_seeds


def test_default_ci_method_is_corrected_t():
    assert TrustCV().ci_method == "corrected_t"


@pytest.mark.parametrize("ci_method", ["corrected_t", "oof_bootstrap"])
def test_new_ci_methods_reach_nominal_coverage_on_noise(ci_method):
    # True AUC is 0.5. A 95% CI must contain 0.5 most of the time.
    # The v1.0.7 fold-mean bootstrap covers ~70%.
    assert _coverage(ci_method) >= 0.85


def test_legacy_fold_bootstrap_still_available_but_warns():
    X, y = load_breast_cancer(return_X_y=True)
    with pytest.warns(UserWarning, match="(?i)fold"):
        TrustCV(ci_method="bootstrap").validate(model=_lr(), X=X, y=y)


def test_oof_bootstrap_resamples_whole_groups():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(40, 5))
    X = np.repeat(base, 5, axis=0) + rng.normal(scale=0.01, size=(200, 5))
    y = np.repeat(rng.integers(0, 2, 40), 5)
    groups = np.repeat(np.arange(40), 5)
    common = dict(method="patient_grouped_kfold", n_splits=5, ci_method="oof_bootstrap",
                  n_bootstrap=500, random_state=0)
    res_g = TrustCV(**common).validate(model=_lr(), X=X, y=y, groups=groups)
    lo_g, hi_g = res_g.confidence_intervals["roc_auc"]
    # With groups, the effective sample size is 40 patients, not 200 rows,
    # so the cluster-bootstrap CI must be clearly wider than a row-level one.
    res_r = TrustCV(**common).validate(
        model=_lr(), X=X, y=y, groups=groups, ci_cluster=False
    )
    lo_r, hi_r = res_r.confidence_intervals["roc_auc"]
    assert (hi_g - lo_g) > 1.5 * (hi_r - lo_r)


# ------------------------------------------------------------ 6. semantics / naming
def test_new_leakage_key_is_not_inverted():
    X, y = load_breast_cancer(return_X_y=True)
    res = TrustCV(method="stratified_kfold").validate(model=_lr(), X=X, y=y)
    lc = res.leakage_check
    assert "external_leakage_detected" in lc  # True means leakage WAS found
    if "has_leakage" in lc:  # legacy key kept for one release only
        assert lc["external_leakage_detected"] == (not lc["has_leakage"])


def test_class_imbalance_is_info_not_failure():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 5))
    y = (rng.random(400) < 0.05).astype(int)
    X[y == 1] += 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TrustCV(method="stratified_kfold").validate(model=_lr(), X=X, y=y)
    assert _status(res, "class_balance") in {"INFO", "WARNING"}
    assert "Class Balance: FAILED" not in res.summary()


def test_covariate_shift_is_reported_as_shift_not_leakage():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 5))
    X_test = rng.normal(loc=3.0, size=(200, 5))
    rep = DataLeakageChecker(verbose=False).check_cv_splits(X_train, X_test)
    assert not rep.has_leakage, rep.leakage_types
    assert "covariate_shift" in rep.details
    assert rep.details["covariate_shift"]["n_shifted_features"] >= 4


def test_near_duplicate_check_has_no_false_positives_in_low_dimensions():
    # v1.0.7 flags two INDEPENDENT samples from the same Gaussian as near-duplicates
    # because cosine similarity >= 0.99 happens by chance in 5 dimensions.
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(200, 5))
        b = rng.normal(size=(200, 5))
        rep = DataLeakageChecker(verbose=False).check_cv_splits(a, b)
        assert "near_duplicate" not in rep.leakage_types, f"seed {seed}"


def test_near_duplicate_check_still_catches_real_near_duplicates():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(200, 5))
    b = np.vstack([a[:40] + rng.normal(scale=1e-4, size=(40, 5)), rng.normal(size=(160, 5))])
    rep = DataLeakageChecker(verbose=False).check_cv_splits(a, b)
    assert "near_duplicate" in rep.leakage_types


# ---------------------------------------------------------------- 7. recommend_cv advisor
def _runs_with(rec, X, y, **kw):
    res = TrustCV(method="kfold").validate(model=_lr(), X=X, y=y, cv=rec.splitter, **kw)
    assert np.isfinite(res.mean_scores["accuracy"])


def test_recommend_cv_returns_actionable_result():
    from trustcv import recommend_cv

    X, y = load_breast_cancer(return_X_y=True)
    rec = recommend_cv(X, y)
    for attr in ("category", "method", "splitter", "rationale", "warnings", "code"):
        assert hasattr(rec, attr)
    assert rec.category == "iid"
    assert isinstance(rec.rationale, str) and len(rec.rationale) > 40
    compile(rec.code, "<rec.code>", "exec")  # generated code must be valid Python
    _runs_with(rec, X, y)


def test_recommend_cv_uses_groups_and_stratification():
    from trustcv import recommend_cv

    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(60), 4)
    y = np.repeat((rng.random(60) < 0.2).astype(int), 4)
    X = rng.normal(size=(240, 5))
    rec = recommend_cv(X, y, groups=groups)
    assert rec.category == "grouped"
    assert "group" in type(rec.splitter).__name__.lower()
    assert "stratif" in type(rec.splitter).__name__.lower()
    _runs_with(rec, X, y, groups=groups)


def test_recommend_cv_temporal_plus_groups():
    from trustcv import recommend_cv

    rng = np.random.default_rng(8)
    n = 300
    groups = np.repeat(np.arange(30), 10)
    times = np.tile(np.arange(10), 30) + rng.random(n)
    rec = recommend_cv(rng.normal(size=(n, 4)), rng.integers(0, 2, n),
                       groups=groups, timestamps=times)
    assert rec.category == "temporal"
    assert "group" in rec.rationale.lower()


def test_recommend_cv_warns_about_too_few_groups_and_rare_events():
    from trustcv import recommend_cv

    rng = np.random.default_rng(9)
    groups = np.repeat(np.arange(4), 25)
    y = np.zeros(100, dtype=int)
    y[:3] = 1
    rec = recommend_cv(rng.normal(size=(100, 3)), y, groups=groups)
    text = " ".join(rec.warnings).lower()
    assert "group" in text
    assert ("event" in text) or ("positive" in text) or ("minority" in text)


def test_recommend_cv_spatial():
    from trustcv import recommend_cv

    rng = np.random.default_rng(10)
    coords = rng.uniform(0, 100, size=(200, 2))
    rec = recommend_cv(rng.normal(size=(200, 4)), rng.integers(0, 2, 200), coordinates=coords)
    assert rec.category == "spatial"
