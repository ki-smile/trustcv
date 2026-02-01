"""
Multilabel stratified GroupKFold splitter.

Keeps all samples from the same group (e.g., patient) together while
greedily balancing multilabel prevalence and fold sizes.
"""

from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_random_state


class MultilabelStratifiedGroupKFold(_BaseKFold):
    """
    Group-aware multilabel stratified K-Fold cross-validator.

    Parameters mirror sklearn's KFold plus weighting terms for the
    greedy assignment heuristic.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle groups before sorting for tie-breaking.
    random_state : int or None, default=None
        Random seed used when ``shuffle=True``.
    alpha : float, default=1.0
        Weight for label balance cost.
    beta : float, default=1.0
        Weight for fold size balance cost.
    eps : float, default=1e-9
        Small constant to avoid division by zero in weight computation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Any = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-9,
    ):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _validate_input(
        self, y: np.ndarray, groups: np.ndarray, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if groups is None:
            raise ValueError("groups must be provided for MultilabelStratifiedGroupKFold.")

        groups_arr = np.asarray(groups)
        if len(groups_arr) != n_samples:
            raise ValueError(
                f"groups length mismatch: expected {n_samples}, got {len(groups_arr)}"
            )

        y_arr = np.asarray(y)
        if y_arr.ndim != 2:
            raise ValueError("y must be a 2D multilabel indicator matrix.")
        unique_vals = np.unique(y_arr)
        if not set(unique_vals).issubset({0, 1}):
            raise ValueError("y must contain only binary indicators {0,1}.")

        return y_arr, groups_arr

    def _prepare_groups(
        self, y: np.ndarray, groups: np.ndarray
    ) -> Tuple[List[Any], Dict[Any, np.ndarray], np.ndarray, np.ndarray]:
        """Build group mapping and summary statistics."""
        unique_groups = np.unique(groups)
        if len(unique_groups) < self.n_splits:
            raise ValueError(
                f"Cannot have number of splits={self.n_splits} greater than number of groups={len(unique_groups)}"
            )

        group_indices: Dict[Any, np.ndarray] = {}
        group_sizes = np.zeros(len(unique_groups), dtype=int)
        group_label_counts = np.zeros((len(unique_groups), y.shape[1]), dtype=float)

        for i, g in enumerate(unique_groups):
            idx = np.where(groups == g)[0]
            group_indices[g] = idx
            group_sizes[i] = len(idx)
            group_label_counts[i] = y[idx].sum(axis=0)

        return list(unique_groups), group_indices, group_sizes, group_label_counts

    def _order_groups(
        self,
        group_ids: Sequence[Any],
        group_sizes: np.ndarray,
        group_label_counts: np.ndarray,
        weights: np.ndarray,
    ) -> List[int]:
        """Order groups by rarity mass, then size, with optional RNG tie-breaker."""
        rarity_scores = (weights * group_label_counts).sum(axis=1)
        rng = check_random_state(self.random_state) if self.shuffle else None
        tie_breaker = (
            rng.permutation(len(group_ids)) if self.shuffle else np.arange(len(group_ids))
        )

        # Primary: rarity score (desc), Secondary: group size (desc), Tertiary: random or index
        order = np.lexsort((tie_breaker, -group_sizes, -rarity_scores))
        return order.tolist()

    def split(self, X, y, groups) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        X_arr = np.asarray(X) if not hasattr(X, "iloc") else X
        n_samples = len(X_arr)

        y_arr, groups_arr = self._validate_input(y, groups, n_samples)
        group_ids, group_indices, group_sizes, group_label_counts = self._prepare_groups(
            y_arr, groups_arr
        )

        total_counts = y_arr.sum(axis=0).astype(float)
        desired_counts = total_counts / self.n_splits
        desired_fold_size = n_samples / self.n_splits
        weights = 1.0 / (total_counts + self.eps)

        fold_label_counts = np.zeros((self.n_splits, y_arr.shape[1]), dtype=float)
        fold_sizes = np.zeros(self.n_splits, dtype=int)
        fold_indices: List[List[int]] = [[] for _ in range(self.n_splits)]
        unused_folds = set(range(self.n_splits))

        for idx in self._order_groups(group_ids, group_sizes, group_label_counts, weights):
            g = group_ids[idx]
            g_size = group_sizes[idx]
            g_labels = group_label_counts[idx]

            candidate_folds = list(unused_folds) if unused_folds else list(range(self.n_splits))

            costs = []
            for fold in candidate_folds:
                new_counts = fold_label_counts[fold] + g_labels
                new_size = fold_sizes[fold] + g_size
                current_label_cost = np.sum(weights * (fold_label_counts[fold] - desired_counts) ** 2)
                current_size_cost = ((fold_sizes[fold] - desired_fold_size) / desired_fold_size) ** 2
                label_cost = np.sum(weights * (new_counts - desired_counts) ** 2) - current_label_cost
                size_cost = ((new_size - desired_fold_size) / desired_fold_size) ** 2 - current_size_cost
                total_cost = self.alpha * label_cost + self.beta * size_cost
                costs.append(total_cost)

            costs_arr = np.asarray(costs)
            best_cost = costs_arr.min()
            candidate_cost_idx = np.where(np.isclose(costs_arr, best_cost))[0]
            candidate_fold_ids = [candidate_folds[i] for i in candidate_cost_idx]

            if len(candidate_fold_ids) > 1:
                # tie-breaker: smallest fold size, then lowest index
                candidate_sizes = np.array([fold_sizes[f] for f in candidate_fold_ids])
                best_size = candidate_sizes.min()
                candidate_fold_ids = [
                    f for f, s in zip(candidate_fold_ids, candidate_sizes) if s == best_size
                ]

            best_fold = int(min(candidate_fold_ids))

            fold_label_counts[best_fold] += g_labels
            fold_sizes[best_fold] += g_size
            fold_indices[best_fold].extend(group_indices[g].tolist())
            if best_fold in unused_folds:
                unused_folds.remove(best_fold)

        all_indices = np.arange(n_samples)
        for fold in range(self.n_splits):
            test_idx = np.array(fold_indices[fold], dtype=int)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            train_idx = all_indices[train_mask]
            yield train_idx, test_idx
