import numpy as np
import pandas as pd
import pytest

from trustcv import ClinicalMetrics


def _data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, n)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.25, n), 0, 1)
    return y, (p > 0.5).astype(int), p


def test_pandas_non_default_index_matches_numpy():
    y, pred, p = _data()
    idx = np.arange(len(y))[::-1] + 1000  # non-default, non-contiguous index
    cm = ClinicalMetrics()
    from_pd = cm.calculate_all(pd.Series(y, idx), pd.Series(pred, idx), pd.Series(p, idx))
    from_np = cm.calculate_all(y, pred, p)
    assert from_pd["auc_roc_ci"] == from_np["auc_roc_ci"]


def test_cluster_bootstrap_widens_interval_for_repeated_measures():
    rng = np.random.RandomState(1)
    n_patients, reps = 30, 10
    patient = np.repeat(np.arange(n_patients), reps)
    y = np.repeat(rng.randint(0, 2, n_patients), reps)
    # patient-level offset makes repeated rows nearly duplicates of each other
    p = np.repeat(rng.normal(0, 1, n_patients), reps) + 0.8 * y + rng.normal(0, 0.05, len(y))
    cm = ClinicalMetrics()
    lo_r, hi_r = cm._auc_ci(y, p)
    lo_g, hi_g = cm._auc_ci(y, p, groups=patient)
    assert (hi_g - lo_g) > (hi_r - lo_r)


def test_single_class_returns_nan_interval():
    lo, hi = ClinicalMetrics()._auc_ci(np.zeros(20, dtype=int), np.linspace(0, 1, 20))
    assert np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("bad", [[0, 1], [0, 1, 2, 3, 4, 5], [], [[0, 1], [2, 3], [4, 5], [6, 7]]])
def test_misaligned_groups_are_rejected(bad):
    y, p = np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.95, 0.05])
    with pytest.raises(ValueError, match="one entry per observation"):
        ClinicalMetrics().calculate_all(y, (p > 0.5).astype(int), p, groups=np.array(bad))


def test_misaligned_groups_rejected_even_without_probabilities():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="one entry per observation"):
        ClinicalMetrics().calculate_all(y, y, groups=np.array([0, 1]))
