import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from trustcv.core.runner import UniversalCVRunner
from trustcv.metrics import check_fold_metric_feasibility, oob_clinical_metrics
from trustcv.splitters.grouped import GroupKFoldMedical, LeaveOneGroupOut


def _make_logo_single_class_data():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(6), 2)
    y_by_group = np.array([0, 0, 0, 1, 1, 1])
    y = np.repeat(y_by_group, 2)
    X = np.column_stack(
        [
            y + rng.normal(scale=0.05, size=len(y)),
            groups / groups.max(),
            rng.normal(size=len(y)),
        ]
    )
    return X, y, groups


def test_check_fold_metric_feasibility_identifies_logo_single_class_folds():
    X, y, groups = _make_logo_single_class_data()
    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups=groups))

    diagnostics = check_fold_metric_feasibility(
        y,
        [test_idx for _, test_idx in splits],
        metric_names=["roc_auc", "sensitivity", "specificity"],
        min_test_samples=5,
    )

    assert diagnostics["has_warnings"] is True
    assert len(diagnostics["folds"]) == 6
    assert all(row["is_single_class"] for row in diagnostics["folds"])
    assert all(row["test_fold_size"] == 2 for row in diagnostics["folds"])
    assert all(row["roc_auc_feasible"] is False for row in diagnostics["folds"])
    assert "pooled out-of-fold" in diagnostics["recommendation"]


def test_universal_runner_warns_and_exposes_diagnostics_for_logo_single_class_folds():
    X, y, groups = _make_logo_single_class_data()
    runner = UniversalCVRunner(
        cv_splitter=LeaveOneGroupOut(),
        framework="sklearn",
        verbose=0,
    )

    with pytest.warns(UserWarning, match="pooled out-of-fold prediction aggregation"):
        results = runner.run(
            model=LogisticRegression(),
            data=(X, y),
            groups=groups,
            metrics=["roc_auc", "accuracy"],
        )

    diagnostics = results.diagnostics["metric_feasibility"]
    assert diagnostics["has_warnings"] is True
    assert results.metric_feasibility_warnings
    assert all(row["n_unique_classes"] == 1 for row in diagnostics["folds"])


def test_group_kfold_with_feasible_folds_does_not_warn():
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(6), 4)
    y = np.tile([0, 1], len(groups) // 2)
    X = np.column_stack([y, rng.normal(size=len(y)), groups])

    runner = UniversalCVRunner(
        cv_splitter=GroupKFoldMedical(n_splits=3, shuffle=False),
        framework="sklearn",
        verbose=0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = runner.run(
            model=DummyClassifier(strategy="prior"),
            data=(X, y),
            groups=groups,
            metrics=["roc_auc", "accuracy"],
        )

    feasibility_warnings = [
        warning
        for warning in caught
        if "pooled out-of-fold prediction aggregation" in str(warning.message)
    ]
    assert feasibility_warnings == []
    assert results.diagnostics["metric_feasibility"]["has_warnings"] is False


def test_pooled_oof_metrics_work_when_logo_fold_auc_is_not_feasible():
    X, y, groups = _make_logo_single_class_data()
    runner = UniversalCVRunner(
        cv_splitter=LeaveOneGroupOut(),
        framework="sklearn",
        verbose=0,
    )

    with pytest.warns(UserWarning, match="pooled out-of-fold prediction aggregation"):
        results = runner.run(
            model=LogisticRegression(),
            data=(X, y),
            groups=groups,
            metrics=["roc_auc", "accuracy"],
        )

    pooled = oob_clinical_metrics(results, y)
    assert pooled is not None
    assert "auc_roc" in pooled
    assert 0.0 <= pooled["auc_roc"] <= 1.0
