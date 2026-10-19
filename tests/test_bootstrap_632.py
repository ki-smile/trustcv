"""The bootstrap .632/.632+ estimators are applied by validate(), not ignored."""

import warnings

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustcv import TrustCV
from trustcv.splitters import BootstrapValidation

X, y = load_breast_cancer(return_X_y=True)


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def _run(estimator, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TrustCV(
            method="bootstrap", bootstrap_validation_estimator=estimator,
            bootstrap_validation_iterations=15, random_state=0, declare_independent_samples=True,
            metrics=kw.pop("metrics", ["accuracy", "roc_auc"]),
        ).validate(model=_model(), X=X, y=y, **kw)


def test_standard_reports_the_out_of_bag_mean():
    r = _run("standard")
    assert r.diagnostics["bootstrap"]["estimator"] == "standard"
    assert r.diagnostics["bootstrap"]["corrected"] == {}


def test_632_combines_out_of_bag_and_training_error():
    r = _run(".632")
    info = r.diagnostics["bootstrap"]["corrected"]["accuracy"]
    expected = 1 - (0.632 * (1 - info["oob_mean"]) + 0.368 * (1 - info["train_mean"]))
    assert r.mean_scores["accuracy"] == pytest.approx(expected)
    assert info["train_mean"] > info["oob_mean"]  # resubstitution is optimistic
    assert r.mean_scores["accuracy"] > info["oob_mean"]  # so .632 lifts the pessimistic OOB


def test_632_plus_weight_is_between_632_and_1():
    r = _run(".632+")
    info = r.diagnostics["bootstrap"]["corrected"]["accuracy"]
    assert 0.632 <= info["weight"] <= 1.0
    lo, hi = r.confidence_intervals["accuracy"]
    assert lo <= r.mean_scores["accuracy"] <= hi


def test_regression_errors_are_corrected_directly():
    Xr, yr = load_diabetes(return_X_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = TrustCV(method="bootstrap", bootstrap_validation_estimator=".632",
                    bootstrap_validation_iterations=10, random_state=0, declare_independent_samples=True,
                    ).validate(model=LinearRegression(), X=Xr, y=yr)
    info = r.diagnostics["bootstrap"]["corrected"]["mse"]
    assert r.mean_scores["mse"] == pytest.approx(0.632 * info["oob_mean"] + 0.368 * info["train_mean"])
    assert "r2" in r.diagnostics["bootstrap"]["not_corrected"]


def test_custom_scorers_are_left_uncorrected_with_a_warning():
    with pytest.warns(UserWarning, match="not corrected"):
        r = TrustCV(method="bootstrap", bootstrap_validation_estimator=".632",
                    bootstrap_validation_iterations=5, random_state=0, declare_independent_samples=True,
                    ).validate(model=_model(), X=X, y=y, scoring={"acc": make_scorer(accuracy_score)})
    assert r.diagnostics["bootstrap"]["not_corrected"] == ["acc"]


def test_bootstrap_splitter_passed_as_cv_is_honoured():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = TrustCV(declare_independent_samples=True, metrics=["accuracy"], random_state=0).validate(
            model=_model(), X=X, y=y, cv=BootstrapValidation(n_iterations=10, estimator=".632", random_state=0)
        )
    assert "accuracy" in r.diagnostics["bootstrap"]["corrected"]
