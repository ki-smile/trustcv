"""Inner validation split inside a CV training fold.

Model selection (early stopping, best-epoch checkpoints) must use data from
the training fold only; the held-out fold scores the selected model.
"""

from typing import Any, Optional, Tuple

import numpy as np


def inner_validation_split(
    train_idx: Any,
    y: Any = None,
    groups: Any = None,
    *,
    fraction: float = 0.1,
    strategy: str = "auto",
    random_state: Optional[int] = None,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
    """Split a training fold into inner-train and inner-validation indices.

    Parameters
    ----------
    train_idx : array-like of int
        Indices of the outer training fold. May contain repeats (bootstrap);
        a repeated row always stays on one side, and multiplicity and order
        are preserved.
    y : array-like, optional
        Full target array (indexed by ``train_idx``); enables stratification
        for classification-like targets.
    groups : array-like, optional
        Full group array; when given, no group is on both sides.
    fraction : float
        Share of the training fold (by unique rows, or by groups) used for
        inner validation; must be in (0, 1).
    strategy : {"auto", "last"}
        ``"auto"``: group split if ``groups``, else stratified when feasible,
        else random. ``"last"``: the last ``fraction`` of ``train_idx`` in its
        given order, for chronologically ordered data; with ``groups``, earlier
        rows of a group that reaches the validation tail are left out of
        inner training (purged), so validation never precedes training.
    random_state : int, optional
        Seed for the random strategies.

    Returns
    -------
    (split, reason)
        ``split`` is ``(inner_train_idx, inner_val_idx)`` or ``None`` when no
        valid split exists; ``reason`` names the strategy used or why it failed.
    """
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction!r}")
    if strategy not in ("auto", "last"):
        raise ValueError(f"strategy must be 'auto' or 'last', got {strategy!r}")
    train_idx = np.asarray(train_idx, dtype=int)
    # first occurrence order of each unique row
    _, first = np.unique(train_idx, return_index=True)
    unique_rows = train_idx[np.sort(first)]
    if unique_rows.size < 4:
        return None, "fewer than 4 unique training rows"

    y_arr = None if y is None else (y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y))
    g_arr = None if groups is None else (groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups))

    purged = unique_rows[:0]
    if strategy == "last":
        n_val = int(round(fraction * unique_rows.size))
        val_rows = unique_rows[unique_rows.size - n_val :] if n_val else unique_rows[:0]
        reason = "last"
        if g_arr is not None:
            # purge earlier rows of groups that reach the tail: no group on both
            # sides, and validation still comes after training in time
            head = unique_rows[: unique_rows.size - n_val]
            purged = head[np.isin(g_arr[head], g_arr[val_rows])]
            if purged.size:
                reason = f"last (purged {purged.size} earlier row(s) of {np.unique(g_arr[purged]).size} group(s))"
    elif g_arr is not None:
        from sklearn.model_selection import GroupShuffleSplit

        row_groups = g_arr[unique_rows]
        if np.unique(row_groups).size < 2:
            return None, "fewer than 2 groups in the training fold"
        try:
            _, val_pos = next(
                GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_state).split(
                    unique_rows, groups=row_groups
                )
            )
        except ValueError as exc:  # e.g. the fraction leaves no group for inner train
            return None, f"group split infeasible ({exc})"
        val_rows = unique_rows[val_pos]
        reason = "group"
    else:
        rng = np.random.RandomState(random_state)
        n_val = max(int(round(fraction * unique_rows.size)), 1)
        val_rows, reason = None, "random"
        if y_arr is not None and y_arr.ndim == 1 and _classification_like(y_arr[unique_rows]):
            val_rows = _stratified_pick(unique_rows, y_arr[unique_rows], n_val, rng)
            reason = "stratified" if val_rows is not None else "random (stratification infeasible)"
        if val_rows is None:
            val_rows = rng.permutation(unique_rows)[:n_val]

    in_val = np.isin(train_idx, val_rows)
    keep = ~in_val & ~np.isin(train_idx, purged)
    inner_val, inner_train = train_idx[in_val], train_idx[keep]
    if np.unique(inner_val).size < 2 or np.unique(inner_train).size < 2:
        return None, f"{reason} split leaves fewer than 2 rows on one side"
    if y_arr is not None and y_arr.ndim == 1 and _classification_like(y_arr[unique_rows]):
        classes = np.unique(y_arr[unique_rows])
        if np.unique(y_arr[inner_val]).size < 2 or np.unique(y_arr[inner_train]).size < classes.size:
            return None, f"{reason} split leaves a class missing from inner train or a single class in inner validation"
    return (inner_train, inner_val), reason


def _classification_like(y: np.ndarray) -> bool:
    """Class labels: non-float values, or whole-number floats with few levels."""
    if not np.issubdtype(y.dtype, np.floating):
        return True
    finite = y[np.isfinite(y)]
    return bool(finite.size) and np.all(finite == np.round(finite)) and np.unique(finite).size <= 20


def _stratified_pick(rows: np.ndarray, labels: np.ndarray, n_val: int, rng: np.random.RandomState) -> Optional[np.ndarray]:
    """One row per class at least, proportional otherwise; None if impossible."""
    classes, counts = np.unique(labels, return_counts=True)
    if n_val < classes.size or np.any(counts < 2):
        return None
    alloc = np.maximum(1, np.floor(counts / counts.sum() * n_val).astype(int))
    alloc = np.minimum(alloc, counts - 1)  # keep every class in inner train
    picked = [rng.choice(rows[labels == c], a, replace=False) for c, a in zip(classes, alloc)]
    return np.concatenate(picked)
