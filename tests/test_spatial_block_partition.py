"""SpatialBlockCV tests every row exactly once and honours block_size."""

import numpy as np
import pytest

from trustcv.splitters import SpatialBlockCV

rng = np.random.default_rng(0)
COORDS = rng.uniform(0, 10, size=(400, 2))
X = np.zeros((400, 1))


def _test_rows(cv, coords=COORDS):
    return [te for _, te in cv.split(X, coordinates=coords)]


@pytest.mark.parametrize("n_splits", [2, 3, 5, 7])
def test_default_grid_tests_every_row_exactly_once(n_splits):
    folds = _test_rows(SpatialBlockCV(n_splits=n_splits, random_state=0))
    assert len(folds) == n_splits
    counts = np.bincount(np.concatenate(folds), minlength=len(X))
    assert np.all(counts == 1)


def test_block_size_sets_cells_and_keeps_each_cell_in_one_fold():
    cv = SpatialBlockCV(n_splits=5, block_size=2.0, random_state=0)
    folds = _test_rows(cv)
    assert np.all(np.bincount(np.concatenate(folds), minlength=len(X)) == 1)
    cell = (COORDS[:, 0] // 2.0).astype(int) * 100 + (COORDS[:, 1] // 2.0).astype(int)
    fold_of = np.empty(len(X), int)
    for k, te in enumerate(folds):
        fold_of[te] = k
    for c in np.unique(cell):
        assert len(set(fold_of[cell == c])) == 1  # a cell is never split across folds
    assert len(np.unique(cell)) == 25


def test_kmeans_blocks_still_partition():
    folds = _test_rows(SpatialBlockCV(n_splits=4, block_shape="kmeans", random_state=0))
    assert np.all(np.bincount(np.concatenate(folds), minlength=len(X)) == 1)


def test_invalid_block_size_is_rejected():
    with pytest.raises(ValueError):
        SpatialBlockCV(block_size=0)
