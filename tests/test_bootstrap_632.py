"""validate() applies Efron & Tibshirani's (1997) .632 / .632+ estimators.

Every expected value is computed here independently, from the bootstrap splits
and a full-data refit, not by calling the library's own formula.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustcv import TrustCV
from trustcv.splitters import BootstrapValidation, SpatialBlockCV


def _reference(model, X, y, splitter, loss):
    """Err1 (eq. 7), apparent error and no-information pieces, computed by hand."""
    total, count = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in splitter.split(X, y):
        pred = clone(model).fit(X[tr], y[tr]).predict(X[te])
        np.add.at(total, te, loss(y[te], pred))
        np.add.at(count, te, 1)
    seen = count > 0
    err1 = np.mean(total[seen] / count[seen])
    full = clone(model).fit(X, y).predict(X)
    return err1, np.mean(loss(y, full)), full


def _validate(model, X, y, estimator, metrics, n=30):
    cv = BootstrapValidation(n_iterations=n, estimator=estimator, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TrustCV(declare_independent_samples=True, metrics=metrics, random_state=0).validate(
            model=model, X=X, y=y, cv=cv
        )
    return result, BootstrapValidation(n_iterations=n, estimator=estimator, random_state=0)


def _zero_one(t, p):
    return (t != p).astype(float)


def test_632_matches_the_reference_estimator():
    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    result, splitter = _validate(model, X, y, ".632", ["accuracy", "roc_auc"])
    err1, err, _ = _reference(model, X, y, splitter, _zero_one)
    expected = 1 - (0.368 * err + 0.632 * err1)
    assert result.mean_scores["accuracy"] == pytest.approx(expected)
    detail = result.diagnostics["bootstrap"]["corrected"]["accuracy"]
    assert detail["err1"] == pytest.approx(err1) and detail["apparent_error"] == pytest.approx(err)
    assert "roc_auc" in result.diagnostics["bootstrap"]["not_corrected"]


def test_632_plus_applies_equation_32_when_oob_error_exceeds_no_information():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = np.tile([0, 1], 15)  # labels unrelated to X: 1-NN overfits completely
    model = KNeighborsClassifier(n_neighbors=1)
    result, splitter = _validate(model, X, y, ".632+", ["accuracy"], n=100)
    err1, err, full = _reference(model, X, y, splitter, _zero_one)
    labels = np.union1d(y, full)
    gamma = sum(np.mean(y == c) * (1 - np.mean(full == c)) for c in labels)
    err1_capped = min(err1, gamma)
    rate = (err1_capped - err) / (gamma - err) if (err1 > err and gamma > err) else 0.0
    expected_err = 0.368 * err + 0.632 * err1 + (err1_capped - err) * 0.368 * 0.632 * rate / (1 - 0.368 * rate)
    assert err1 > gamma  # the capped branch is exercised
    assert result.mean_scores["accuracy"] == pytest.approx(1 - expected_err)
    assert 1 - result.mean_scores["accuracy"] < err1  # not simply the out-of-bag error


def test_regression_mse_matches_the_reference():
    X, y = load_diabetes(return_X_y=True)
    result, splitter = _validate(LinearRegression(), X, y, ".632", ["mse", "r2"], n=20)
    err1, err, _ = _reference(LinearRegression(), X, y, splitter, lambda t, p: (t - p) ** 2)
    assert result.mean_scores["mse"] == pytest.approx(0.368 * err + 0.632 * err1)
    assert "r2" in result.diagnostics["bootstrap"]["not_corrected"]


def test_corrected_metrics_report_no_interval():
    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    result, _ = _validate(model, X, y, ".632", ["accuracy", "roc_auc"], n=10)
    lo, hi = result.confidence_intervals["accuracy"]
    assert np.isnan(lo) and np.isnan(hi)
    assert np.isfinite(result.confidence_intervals["roc_auc"]).all()  # uncorrected metric keeps its interval
    assert "interval_note" in result.diagnostics["bootstrap"]


def test_standard_estimator_is_the_out_of_bag_mean():
    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    result, _ = _validate(model, X, y, "standard", ["accuracy"], n=10)
    assert result.diagnostics["bootstrap"]["corrected"] == {}
    assert result.mean_scores["accuracy"] == pytest.approx(np.mean(result.scores["accuracy"]))


def test_custom_scorers_are_not_corrected_and_warn():
    X, y = load_breast_cancer(return_X_y=True)
    with pytest.warns(UserWarning, match="not corrected"):
        result = TrustCV(method="bootstrap", bootstrap_validation_estimator=".632",
                         bootstrap_validation_iterations=5, random_state=0, declare_independent_samples=True,
                         ).validate(model=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                    X=X, y=y, scoring={"acc": make_scorer(accuracy_score)})
    assert result.diagnostics["bootstrap"]["not_corrected"] == ["acc"]


# --- regressions from the review ----------------------------------------------


def test_rare_class_bootstrap_does_not_crash():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.array([0] * 18 + [1] * 2)
    result, _ = _validate(DummyClassifier(), X, y, ".632", ["accuracy", "roc_auc"], n=30)
    assert "accuracy" in result.mean_scores


def test_nan_from_a_custom_scorer_does_not_crash_ordinary_kfold():
    calls = {"n": 0}

    def flaky(yt, yp):
        calls["n"] += 1
        return float("nan") if calls["n"] == 2 else 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        TrustCV(declare_independent_samples=True).validate(
            model=DummyClassifier(), X=np.random.rand(50, 2), y=np.tile([0, 1], 25),
            cv=KFold(5), scoring={"s": make_scorer(flaky)},
        )


def test_spatial_block_cv_refuses_too_few_blocks():
    coords = np.repeat([[1, 1], [2, 2], [3, 3], [4, 4]], 20, axis=0)
    with pytest.raises(ValueError, match="occupied spatial blocks"):
        list(SpatialBlockCV(n_splits=4, block_size=10).split(np.zeros((80, 1)), coordinates=coords))


def test_compute_error_uses_equation_32():
    bv = BootstrapValidation(estimator=".632+", random_state=0)
    y_true = np.tile([0, 1], 50)
    y_pred_train = np.tile([0, 1], 50)  # gamma = 0.5 for balanced labels and predictions
    err_train, err_test = 0.0, 0.7  # Err1 > gamma, so Err1 is capped at gamma
    got = bv.compute_error([err_train], [err_test], y_true=y_true, y_pred_train=y_pred_train)
    gamma = 1 - np.mean(np.random.RandomState(0).permutation(y_pred_train) == y_true)
    capped = min(err_test, gamma)
    rate = (capped - err_train) / (gamma - err_train)
    expected = 0.368 * err_train + 0.632 * err_test + (capped - err_train) * 0.368 * 0.632 * rate / (1 - 0.368 * rate)
    assert got == pytest.approx(expected)
