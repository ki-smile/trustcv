"""
Multilabel stratified k-fold splitter (optional dependency).
"""

from typing import Any, Iterator, Tuple

import numpy as np


class MultilabelStratifiedKFold:
    """
    Thin wrapper around iterative-stratification's MultilabelStratifiedKFold.

    Parameters mirror sklearn's StratifiedKFold for familiarity.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Any = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold as _ISKF
        except ImportError as e:
            raise ImportError(
                "iterative-stratification is required for MultilabelStratifiedKFold. "
                "Install with `pip install iterative-stratification` or use the "
                "`multilabel` extra."
            ) from e
        self._impl = _ISKF(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._impl.get_n_splits(X, y, groups)

    def split(self, X, y, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self._impl.split(X, y, groups)
