"""Regression tests for the independent review follow-ups."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from trustcv.uncertainty import _oof_bootstrap_interval


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
