"""Permutation-based CV-loop sanity checks."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score, roc_auc_score


def _slice_rows(values: Any, indices: np.ndarray) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return values[indices]


def _score_estimator(estimator: Any, X_test: Any, y_test: np.ndarray, regression: bool) -> float:
    if regression:
        return float(r2_score(y_test, estimator.predict(X_test)))

    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(X_test))
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X_test))
    else:
        raise TypeError("Permutation AUC requires predict_proba or decision_function.")

    labels = np.unique(y_test)
    if labels.size < 2:
        return float("nan")
    if scores.ndim == 2 and scores.shape[1] == 2:
        scores = scores[:, 1]
    if scores.ndim == 1:
        return float(roc_auc_score(y_test, scores))
    return float(roc_auc_score(y_test, scores, multi_class="ovr", average="macro"))


def _cv_primary_score(
    model: Any,
    X: Any,
    y: np.ndarray,
    splitter: Any,
    groups: Optional[np.ndarray],
    regression: bool,
) -> float:
    fold_scores = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        estimator = clone(model)
        X_train = _slice_rows(X, train_idx)
        X_test = _slice_rows(X, test_idx)
        y_train = y[train_idx]
        y_test = y[test_idx]
        estimator.fit(X_train, y_train)
        score = _score_estimator(estimator, X_test, y_test, regression)
        if np.isfinite(score):
            fold_scores.append(score)
    return float(np.mean(fold_scores)) if fold_scores else float("nan")


def _shuffle_labels(
    y: np.ndarray,
    rng: np.random.Generator,
    groups: Optional[np.ndarray],
) -> np.ndarray:
    if groups is None:
        return rng.permutation(y)

    unique_groups = np.unique(groups)
    donor_groups = rng.permutation(unique_groups)
    shuffled = np.empty_like(y)
    for receiving, donor in zip(unique_groups, donor_groups):
        receiving_idx = np.flatnonzero(groups == receiving)
        donor_values = y[groups == donor]
        if donor_values.size == receiving_idx.size:
            shuffled[receiving_idx] = donor_values
        elif np.all(donor_values == donor_values[0]):
            shuffled[receiving_idx] = donor_values[0]
        else:
            shuffled[receiving_idx] = rng.choice(
                donor_values, size=receiving_idx.size, replace=True
            )
    return shuffled


def _run_permutation_sanity(
    *,
    model: Any,
    X: Any,
    y: Any,
    splitter: Any,
    groups: Any,
    n_permutations: int,
    random_state: Optional[int],
    regression: bool,
) -> Dict[str, Any]:
    """Run a shuffled-label sanity check of the fitted CV loop.

    The check can reveal a CV loop that scores shuffled labels above chance,
    such as a splitter that places test rows in training. It cannot detect
    preprocessing or feature selection completed before this function receives
    the fixed feature matrix.
    """
    y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    if y_array.ndim != 1:
        raise ValueError("Permutation sanity currently supports one-dimensional targets.")
    groups_array = (
        None
        if groups is None
        else (groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups))
    )
    rng = np.random.default_rng(random_state)
    observed = _cv_primary_score(model, X, y_array, splitter, groups_array, regression)
    null_scores = []
    for _ in range(max(int(n_permutations), 1)):
        shuffled_y = _shuffle_labels(y_array, rng, groups_array)
        null_scores.append(
            _cv_primary_score(model, X, shuffled_y, splitter, groups_array, regression)
        )
    null_array = np.asarray(null_scores, dtype=float)
    finite_null = null_array[np.isfinite(null_array)]
    null_mean = float(np.mean(finite_null)) if finite_null.size else float("nan")
    p_value = float(
        (1 + np.count_nonzero(finite_null >= observed)) / (1 + finite_null.size)
    )
    return {
        "metric": "r2" if regression else "roc_auc",
        "observed": float(observed),
        "null_scores": [float(value) for value in null_array],
        "null_mean": null_mean,
        "p_value": p_value,
        "chance_level": 0.0 if regression else 0.5,
    }
