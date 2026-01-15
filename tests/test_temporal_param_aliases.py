import numpy as np
import pytest

from trustcv.splitters.temporal import (
    ExpandingWindowCV,
    RollingWindowCV,
    BlockedTimeSeries,
    PurgedKFoldCV,
    CombinatorialPurgedCV,
    PurgedGroupTimeSeriesSplit,
)


def _toy_X(n=100):
    return np.arange(n).reshape(-1, 1)


def test_expanding_alias_min_train_size_warns_and_works():
    X = _toy_X(100)
    with pytest.warns(DeprecationWarning):
        cv = ExpandingWindowCV(min_train_size=20, forecast_horizon=10, step_size=10)
    splits = list(cv.split(X))
    assert len(splits) > 0


def test_purged_kfold_alias_embargo_pct_warns_and_works():
    X = _toy_X(100)
    with pytest.warns(DeprecationWarning):
        cv = PurgedKFoldCV(n_splits=5, purge_gap=2, embargo_pct=0.1)
    splits = list(cv.split(X))
    assert len(splits) == 5


def test_combinatorial_aliases_warn_and_work():
    X = _toy_X(60)
    with pytest.warns(DeprecationWarning):
        # strict_order=False to get all C(4,2)=6 combinations
        cv = CombinatorialPurgedCV(n_splits=4, n_test_groups=2, purge_gap=0, embargo_pct=0.0, strict_order=False)
    # Number of combinations C(4,2)=6
    splits = list(cv.split(X))
    assert len(splits) == 6


def test_purged_group_alias_exclude_test_groups_and_timestamps_none():
    # Test that exclude_test_groups alias triggers deprecation warning
    X = _toy_X(100)
    groups = np.array([i // 10 for i in range(100)])  # 10 samples per group
    with pytest.warns(DeprecationWarning):
        # Test the alias triggers warning (using False to avoid empty splits)
        cv = PurgedGroupTimeSeriesSplit(n_splits=3, purge_gap=0, exclude_test_groups=False)
    # timestamps=None should work
    splits = list(cv.split(X, groups=groups, timestamps=None))
    assert len(splits) > 0


def test_blocked_timeseries_timestamps_none():
    X = _toy_X(90)
    cv = BlockedTimeSeries(n_splits=3, block_size=10)
    splits = list(cv.split(X, timestamps=None))
    assert len(splits) == 3

