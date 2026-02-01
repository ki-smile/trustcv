import os
import sys
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustcv.splitters.multilabel_group import MultilabelStratifiedGroupKFold


def _make_grouped_multilabel(n_groups: int = 60, n_labels: int = 5, seed: int = 123):
    rng = np.random.RandomState(seed)
    group_sizes = rng.randint(2, 8, size=n_groups)
    base_p = np.linspace(0.05, 0.35, n_labels)

    X, y_rows, groups = [], [], []
    for gid, gsize in enumerate(group_sizes):
        # group-specific label tendencies; keep some rare labels
        p = np.clip(base_p + rng.normal(0, 0.04, size=n_labels) + rng.uniform(-0.05, 0.05), 0.01, 0.8)
        for _ in range(gsize):
            y_rows.append(rng.binomial(1, p))
            groups.append(gid)
            X.append(rng.randn(4))

    X_arr = np.asarray(X)
    y_arr = np.asarray(y_rows, dtype=int)
    groups_arr = np.asarray(groups)
    return X_arr, y_arr, groups_arr


def _mean_abs_prevalence_deviation(cv, X, y, groups):
    global_prev = y.mean(axis=0)
    deviations = []
    for _, test_idx in cv.split(X, y, groups):
        fold_prev = y[test_idx].mean(axis=0)
        deviations.append(np.abs(fold_prev - global_prev))
    return np.mean(deviations)


def test_group_disjoint_and_balance():
    X, y, groups = _make_grouped_multilabel()
    cv = MultilabelStratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(cv.split(X, y, groups))

    # 1) group disjointness
    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)

    # 2) prevalence deviation vs baseline GroupKFold
    baseline = GroupKFold(n_splits=5)
    new_dev = _mean_abs_prevalence_deviation(cv, X, y, groups)
    base_dev = _mean_abs_prevalence_deviation(baseline, X, y, groups)
    assert new_dev <= base_dev + 1e-8

    # 3) fold size balance (max diff <= 20% of mean)
    fold_sizes = [len(test_idx) for _, test_idx in splits]
    mean_size = np.mean(fold_sizes)
    assert (max(fold_sizes) - min(fold_sizes)) <= 0.2 * mean_size + 1e-8
