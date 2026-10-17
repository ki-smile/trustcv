"""Leakage scenarios: what the integrity checks catch, and how they explain it."""

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustcv import DataLeakageChecker, TrustCV
from trustcv.checks import (
    _identifier_columns,
    _independence_scan,
    _looks_like_identifier_name,
    covariate_shift_from_report,
)


def _patients(n_pat=60, reps=4, d=6, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    pid = np.repeat(np.arange(n_pat), reps)
    X = rng.normal(size=(n_pat, d))[pid] + rng.normal(0, noise, (n_pat * reps, d))
    y = np.repeat(rng.randint(0, 2, n_pat), reps)
    return X, y, pid


def _pipe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _validate(X, y, tv_kwargs=None, **kwargs):
    tv = TrustCV(n_splits=5, random_state=0, **(tv_kwargs or {}))
    with _quiet():
        return tv.validate(model=kwargs.pop("model", _pipe()), X=X, y=y, **kwargs)


class _quiet:
    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# --- folds that were actually scored --------------------------------------


def test_group_overlap_uses_scored_folds_and_gives_fix_hint():
    X, y, pid = _patients()
    r = _validate(X, y, groups=pid, cv=KFold(5, shuffle=True, random_state=0))
    assert r.checks["group_leakage"].status == "FAILED"
    assert r.checks["external_leakage_detector"].details["audited_scored_folds"] is True
    hints = [h for h in r.recommendations if "stratified_group_kfold" in h]
    assert len(hints) == 1  # validator and detector share one deduplicated hint


class _StatefulSplitter:
    """Leaks groups only on the first call, so re-splitting would hide it."""

    def __init__(self):
        self.calls = 0

    def get_n_splits(self, *a, **k):
        return 2

    def split(self, X, y=None, groups=None):
        self.calls += 1
        n = len(X)
        idx = np.arange(n)
        if self.calls == 1:
            yield idx[: n // 2 + 4], idx[n // 2 :]  # overlapping rows => shared groups
            yield idx[n // 2 :], idx[: n // 2]
        else:
            yield idx[: n // 2], idx[n // 2 :]
            yield idx[n // 2 :], idx[: n // 2]


def test_stateful_splitter_is_audited_once():
    X, y, pid = _patients()
    splitter = _StatefulSplitter()
    r = _validate(X, y, groups=pid, cv=splitter)
    assert r.checks["group_leakage"].status == "FAILED"
    assert r.leakage_check.get("no_patient_leakage") is False


class _ManyFolds:
    """20 folds; only fold 13 puts one group on both sides."""

    def get_n_splits(self, *a, **k):
        return 20

    def split(self, X, y=None, groups=None):
        n = len(X)
        for i in range(20):
            te = np.arange(i, n, 20)
            tr = np.setdiff1d(np.arange(n), te)
            if i == 13:
                tr = np.append(tr, te[0])  # te[0] also in train
            yield tr, te


def test_cheap_overlap_checks_cover_folds_beyond_the_distance_cap():
    X, y, _ = _patients()
    groups = np.arange(len(y))
    r = _validate(X, y, tv_kwargs={"max_audit_folds": 2}, groups=groups, cv=_ManyFolds())
    assert r.checks["group_leakage"].status == "FAILED"
    coverage = r.diagnostics["fold_audit"]
    assert coverage["n_folds_scored"] == 20 and coverage["n_folds_distance_audited"] == 2


def test_multi_input_features_are_not_checked_rather_than_failed():
    rng = np.random.RandomState(0)
    X = {"a": rng.normal(size=(80, 3)), "b": rng.normal(size=(80, 2))}
    y = rng.randint(0, 2, 80)

    class Model:
        def get_params(self, deep=True):
            return {}

        def set_params(self, **k):
            return self

        def fit(self, X, y):
            self.m = LogisticRegression().fit(np.hstack([X["a"], X["b"]]), y)
            return self

        def predict(self, X):
            return self.m.predict(np.hstack([X["a"], X["b"]]))

        def predict_proba(self, X):
            return self.m.predict_proba(np.hstack([X["a"], X["b"]]))

    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True}, model=Model())
    assert r.checks["external_leakage_detector"].status == "NOT_CHECKED"
    assert r.checks["target_leakage_features"].status == "NOT_CHECKED"
    assert r.overall_status != "FAILED"


def test_custom_checker_without_folds_keyword_still_works():
    class LegacyChecker:
        def check(self, X, y=None, groups=None):
            return type("R", (), {"has_leakage": False, "leakage_types": [], "details": {}, "recommendations": []})()

    X, y, _ = _patients()
    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True}, leakage_checker=LegacyChecker())
    check = r.checks["external_leakage_detector"]
    assert check.status == "PASSED" and "own splits" in check.message


def test_exact_duplicates_ignore_dataframe_index():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    test = pd.DataFrame({"a": [1.0, 9.0], "b": [4.0, 9.0]}, index=[100, 101])
    result = DataLeakageChecker(verbose=False)._check_duplicate_samples(df, test)
    assert result["has_leakage"] and result["duplicate_count"] == 1


# --- advisory independence scan --------------------------------------------


def test_false_independence_claim_needs_review_not_passed():
    X, y, _ = _patients()
    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True})
    check = r.checks["group_leakage"]
    assert check.status == "WARNING" and check.details["independence_scan"]["clustered"]
    assert "declared independent, but rows may not be" in check.message
    assert r.overall_status == "NEEDS_REVIEW"  # evidence, not proof: never FAILED
    assert any("groups=<subject id>" in h for h in r.recommendations)

    quiet = _validate(X, y, tv_kwargs={"declare_independent_samples": True, "check_independence": False})
    assert quiet.checks["group_leakage"].status == "NOT_APPLICABLE"
    assert quiet.overall_status == "PASSED"


def test_clean_independent_data_still_passes():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True})
    assert r.overall_status == "PASSED", r.summary()


def test_missing_groups_with_clustered_rows_needs_review():
    X, y, _ = _patients()
    r = _validate(X, y)
    assert r.checks["group_leakage"].status == "WARNING"
    assert r.overall_status == "NEEDS_REVIEW"


@pytest.mark.parametrize(
    "name, X",
    [
        ("breast_cancer", datasets.load_breast_cancer().data),
        ("wine", datasets.load_wine().data),
        ("diabetes", datasets.load_diabetes().data),
        ("correlated", np.random.RandomState(0).multivariate_normal(np.zeros(10), 0.9 + 0.1 * np.eye(10), 800)),
        ("lognormal", np.random.RandomState(0).lognormal(0, 2, (800, 5))),
        ("swiss_roll", datasets.make_swiss_roll(800, random_state=0)[0]),
    ],
)
def test_scan_does_not_flag_independent_data(name, X):
    scan = _independence_scan(X, random_state=0)
    assert scan["applicable"] and not scan["clustered"], (name, scan["flagged_fraction"])


@pytest.mark.parametrize(
    "X",
    [
        np.random.RandomState(0).randint(0, 2, (500, 12)),  # binary features tie naturally
        np.random.RandomState(0).randint(0, 11, (500, 2)),  # coarse integer grid
        np.random.RandomState(0).normal(size=(30, 4)),  # too few rows
        {"a": np.zeros((50, 2))},  # multi-input
    ],
)
def test_scan_declines_data_it_cannot_judge(X):
    assert _independence_scan(X, random_state=0)["applicable"] is False


def test_scan_flags_repeated_subjects_and_oversampling():
    X, _, _ = _patients(n_pat=150, reps=4, d=8, noise=0.1)
    assert _independence_scan(X, random_state=0)["clustered"]
    rng = np.random.RandomState(0)
    base = rng.normal(size=(600, 8))
    i = rng.randint(0, 600, 300)
    j = rng.randint(0, 600, 300)
    from sklearn.neighbors import NearestNeighbors

    nbr = NearestNeighbors(n_neighbors=6).fit(base).kneighbors(base[i])[1][:, 1:]
    partner = base[nbr[np.arange(300), rng.randint(0, 5, 300)]]
    smote = np.vstack([base, base[i] + rng.rand(300, 1) * (partner - base[i])])
    assert _independence_scan(smote, random_state=0)["clustered"]
    assert _independence_scan(X, random_state=0) == _independence_scan(X, random_state=0)


# --- identifiers, target proxies, wording ------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("patient_id", True), ("PatientID", True), ("subjectId", True), ("mrn", True),
        ("record_number", True), ("acid", False), ("lipid", False), ("covid_status", False),
        ("patient_age", False), ("record_length", False), ("humidity", False),
    ],
)
def test_identifier_names(name, expected):
    assert _looks_like_identifier_name(name) is expected


def test_identifier_column_in_features_gets_named_hint():
    X, y, pid = _patients()
    df = pd.DataFrame(X, columns=[f"lab_{i}" for i in range(X.shape[1])])
    df["patient_id"] = pid
    df["constant_id"] = 1  # constants are ignored
    found = _identifier_columns(df)
    assert [c["name"] for c in found] == ["patient_id"] and found[0]["repeated_values"] > 0
    r = _validate(df, y)
    assert any("groups=X['patient_id']" in h for h in r.recommendations)


def test_target_proxy_is_named_and_strong_predictors_are_advisory():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 300)
    df = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    df["discharge_code"] = y + rng.normal(0, 0.1, 300)
    df["inverse_code"] = -(y + rng.normal(0, 0.1, 300))
    df["treatment_given"] = np.where(rng.rand(300) < 0.9, y, 1 - y)
    r = _validate(df, y, tv_kwargs={"declare_independent_samples": True})
    check = r.checks["target_leakage_features"]
    assert check.status == "WARNING"
    assert "discharge_code" in check.message and "inverse_code" in check.message
    assert "treatment_given" in check.message  # tier 2 is listed, not counted as leakage
    assert check.details["flagged_features"] == ["discharge_code", "inverse_code"]

    df2 = df.drop(columns=["discharge_code", "inverse_code"])
    r2 = _validate(df2, y, tv_kwargs={"declare_independent_samples": True})
    strong = r2.checks["target_leakage_features"]
    # Strong predictors are listed but do not change the status: real biomarkers
    # look the same (clean breast-cancer data has 11 such features).
    assert strong.status == "PASSED" and "treatment_given" in strong.message
    assert any("treatment_given" in h for h in r2.recommendations)


def test_pipeline_pass_states_what_it_cannot_see():
    X, y, _ = _patients()
    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True})
    assert "before validate()" in r.checks["preprocessing_leakage"].message


def test_bounded_metric_intervals_are_clipped_but_custom_scorers_are_not():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True})
    for name, (lo, hi) in r.confidence_intervals.items():
        if name in ("accuracy", "roc_auc", "sensitivity", "specificity", "precision", "recall", "f1"):
            assert 0.0 <= lo <= hi <= 1.0, name

    unbounded = make_scorer(lambda yt, yp: float(np.random.RandomState(len(yt)).uniform(0.5, 1.5)))
    r2 = _validate(X, y, tv_kwargs={"declare_independent_samples": True}, scoring={"accuracy": unbounded})
    lo, hi = r2.confidence_intervals["accuracy"]
    assert hi > 1.0  # a custom scorer's name says nothing about its range


def test_covariate_shift_is_info_only_when_groups_were_verified_separate():
    report = type("R", (), {"details": {"covariate_shift": {"n_shifted_features": 3}}})()
    assert covariate_shift_from_report(report).status == "WARNING"
    shifted = covariate_shift_from_report(report, groups_separated=True)
    assert shifted.status == "INFO" and shifted.details["n_shifted_features"] == 3


# --- regressions from the Codex implementation review ------------------------


def test_same_row_in_train_and_test_fails_even_beyond_the_cap_without_groups():
    X, y, _ = _patients(n_pat=200, reps=1)
    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True, "max_audit_folds": 2}, cv=_ManyFolds())
    check = r.checks["duplicate_samples"]
    assert check.status == "FAILED" and check.details["folds_with_index_overlap"] == [13]
    assert r.overall_status == "FAILED"


def _report(leak):
    return type("R", (), {"has_leakage": leak, "leakage_types": ["custom"] if leak else [], "details": {}, "recommendations": []})()


def test_custom_checker_is_still_called_for_multi_input_data():
    calls = []

    class DictChecker:
        def check(self, X, y=None, groups=None):
            calls.append(type(X))
            return _report(True)

    rng = np.random.RandomState(0)
    X = {"a": rng.normal(size=(80, 3))}
    y = rng.randint(0, 2, 80)

    class Model:
        def get_params(self, deep=True):
            return {}

        def set_params(self, **k):
            return self

        def fit(self, X, y):
            self.m = LogisticRegression().fit(X["a"], y)
            return self

        def predict(self, X):
            return self.m.predict(X["a"])

        def predict_proba(self, X):
            return self.m.predict_proba(X["a"])

    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True}, model=Model(), leakage_checker=DictChecker())
    assert calls and r.checks["external_leakage_detector"].status == "FAILED"


@pytest.mark.parametrize("style", ["forwarding", "ignoring"])
def test_kwargs_only_checkers_are_not_sent_folds(style):
    class Checker:
        def _check(self, X, y=None, groups=None):
            return _report(False)

        def check(self, **kwargs):
            if style == "forwarding":
                return self._check(**kwargs)
            return _report(False)

    X, y, _ = _patients()
    r = _validate(X, y, tv_kwargs={"declare_independent_samples": True}, leakage_checker=Checker())
    check = r.checks["external_leakage_detector"]
    assert check.status == "PASSED" and check.details["audited_scored_folds"] is False


def test_retained_folds_are_exactly_the_scored_indices():
    seen = []

    class Recorder(DataLeakageChecker):
        def check(self, X, y=None, groups=None, folds=None, **kw):
            seen.extend(folds)
            return super().check(X, y, groups=groups, folds=folds)

    class Bootstrap:
        def get_n_splits(self, *a, **k):
            return 3

        def split(self, X, y=None, groups=None):
            rng = np.random.RandomState(0)
            n = len(X)
            for _ in range(3):
                tr = rng.randint(0, n, n)
                yield tr, np.setdiff1d(np.arange(n), tr)

    X, y, _ = _patients(n_pat=120, reps=1)
    expected = list(Bootstrap().split(X))
    _validate(X, y, tv_kwargs={"declare_independent_samples": True}, cv=Bootstrap(), leakage_checker=Recorder(verbose=False))
    assert len(seen) == 3
    for (tr, te), (etr, ete) in zip(seen, expected):
        assert np.array_equal(tr, etr) and np.array_equal(te, ete)  # repeats and order kept


def test_duplicate_column_labels_do_not_abort_validation():
    X, y, _ = _patients()
    df = pd.DataFrame(X[:, :4], columns=["a", "b", "patient_id", "patient_id"])
    r = _validate(df, y, tv_kwargs={"declare_independent_samples": True})
    assert r.overall_status in {"PASSED", "NEEDS_REVIEW", "NOT_FULLY_VERIFIED", "FAILED"}
