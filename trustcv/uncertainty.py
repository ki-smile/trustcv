"""Confidence intervals for cross-validation estimates."""

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _corrected_t_interval(
    values: Iterable[float],
    *,
    train_sizes: Optional[Iterable[int]] = None,
    test_sizes: Optional[Iterable[int]] = None,
    level: float,
) -> Tuple[float, float]:
    """Return the corrected resampled-t interval for cross-validation scores.

    For repeated ``k``-fold CV with ``r`` repeats, this uses
    ``df = r * k - 1`` and estimated mean-score variance
    ``(1 / (r * k) + n_test / n_train) * s^2``, where the size ratio is
    calculated from the actual mean test and train fold sizes.

    References: Nadeau & Bengio (2003, Machine Learning 52:239-281) and
    Bouckaert & Frank (2004, PAKDD).
    """
    if train_sizes is None or test_sizes is None:
        raise ValueError(
            "corrected_t requires actual train_sizes and test_sizes for every fold."
        )
    values_array = np.asarray(list(values), dtype=float)
    train_sizes_array = np.asarray(list(train_sizes), dtype=float)
    test_sizes_array = np.asarray(list(test_sizes), dtype=float)
    if not (
        values_array.size == train_sizes_array.size == test_sizes_array.size
    ):
        raise ValueError(
            "values, train_sizes, and test_sizes must have the same length."
        )
    finite = np.isfinite(values_array)
    values_array = values_array[finite]
    train_sizes_array = train_sizes_array[finite]
    test_sizes_array = test_sizes_array[finite]
    if values_array.size <= 1:
        return (float("nan"), float("nan"))
    if (
        not np.all(np.isfinite(train_sizes_array))
        or not np.all(np.isfinite(test_sizes_array))
        or np.any(train_sizes_array <= 0)
        or np.any(test_sizes_array <= 0)
    ):
        raise ValueError("train_sizes and test_sizes must be finite and positive.")
    train_mean = float(np.mean(train_sizes_array))
    test_mean = float(np.mean(test_sizes_array))
    sample_variance = float(np.var(values_array, ddof=1))
    correction = 1.0 / values_array.size + test_mean / train_mean
    standard_error = float(np.sqrt(correction * sample_variance))
    quantile = float(stats.t.ppf((1.0 + level) / 2.0, values_array.size - 1))
    mean = float(np.mean(values_array))
    return (mean - quantile * standard_error, mean + quantile * standard_error)


def _fold_bootstrap_interval(
    values: Iterable[float],
    *,
    n_bootstrap: int,
    level: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values_array = np.asarray(list(values), dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size <= 1:
        return (float("nan"), float("nan"))
    boot = np.empty(max(int(n_bootstrap), 1), dtype=float)
    for index in range(boot.size):
        sample = rng.integers(0, values_array.size, size=values_array.size)
        boot[index] = float(np.mean(values_array[sample]))
    alpha = 1.0 - level
    low, high = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return (float(low), float(high))


def _metric_from_predictions(
    metric: str,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_score: Optional[np.ndarray],
    *,
    labels: Optional[np.ndarray] = None,
    pos_label: Any = None,
) -> float:
    if metric in {"roc_auc", "auc"}:
        if y_score is None:
            raise ValueError("ROC-AUC requires out-of-fold scores.")
        if y_score.ndim == 1:
            return float(roc_auc_score(y_true, y_score))
        return float(
            roc_auc_score(
                y_true,
                y_score,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    if metric in {"r2", "r2_score"}:
        return float(r2_score(y_true, y_pred))
    if metric in {"mse", "mean_squared_error"}:
        return float(mean_squared_error(y_true, y_pred))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric in {"mae", "mean_absolute_error"}:
        return float(mean_absolute_error(y_true, y_pred))
    if metric in {"explained_variance", "evs"}:
        return float(explained_variance_score(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric in {"f1", "f1_score"}:
        return float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=pos_label,
                zero_division=0,
            )
        )
    if metric == "f1_macro":
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    if metric == "f1_micro":
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="micro",
                zero_division=0,
            )
        )
    if metric == "precision":
        return float(
            precision_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=pos_label,
                zero_division=0,
            )
        )
    if metric == "recall":
        return float(
            recall_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=pos_label,
                zero_division=0,
            )
        )
    if metric in {"sensitivity", "tpr", "recall_pos"}:
        return float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    if metric in {"specificity", "tnr"}:
        return float(recall_score(y_true, y_pred, pos_label=labels[0], zero_division=0))
    raise ValueError(f"OOF bootstrap does not support metric {metric!r}.")


def _is_partition(test_indices: Iterable[np.ndarray], n_samples: int) -> bool:
    indices = np.concatenate([np.asarray(value, dtype=int) for value in test_indices])
    if indices.size != n_samples:
        return False
    counts = np.bincount(indices, minlength=n_samples)
    return counts.size == n_samples and bool(np.all(counts == 1))


def _oof_bootstrap_interval(
    metric: str,
    *,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_score: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    cluster: bool,
    regression: bool,
    n_bootstrap: int,
    level: float,
    rng: np.random.Generator,
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    """Bootstrap a metric over pooled out-of-fold predictions.

    Ungrouped classification resamples are stratified within each class so
    every replicate retains the full target label set. Grouped classification
    keeps whole-group cluster resampling; undefined replicates are skipped,
    their fraction is returned in the diagnostics, and a ``UserWarning`` is
    emitted when more than 10% of the requested replicates are skipped.
    Regression uses ordinary row or cluster bootstrap resampling.
    """
    estimates = []
    skipped = 0
    n_samples = y_true.shape[0]
    n_resamples = max(int(n_bootstrap), 1)
    labels = None if regression else np.unique(y_true)
    pos_label = None if labels is None or labels.size == 0 else labels[-1]
    unique_groups = np.unique(groups) if cluster and groups is not None else None
    for _ in range(n_resamples):
        if unique_groups is not None:
            sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
            sample_indices = np.concatenate(
                [np.flatnonzero(groups == group) for group in sampled_groups]
            )
        elif not regression:
            sample_indices = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero(y_true == label),
                        size=np.count_nonzero(y_true == label),
                        replace=True,
                    )
                    for label in labels
                ]
            )
        else:
            sample_indices = rng.integers(0, n_samples, size=n_samples)
        sampled_y = y_true[sample_indices]
        if not regression and np.unique(sampled_y).size < 2:
            skipped += 1
            continue
        sampled_pred = None if y_pred is None else y_pred[sample_indices]
        sampled_score = None if y_score is None else y_score[sample_indices]
        try:
            estimate = _metric_from_predictions(
                metric,
                sampled_y,
                sampled_pred,
                sampled_score,
                labels=labels,
                pos_label=pos_label,
            )
        except (ValueError, TypeError):
            skipped += 1
            continue
        if np.isfinite(estimate):
            estimates.append(estimate)
        else:
            skipped += 1
    skipped_fraction = skipped / n_resamples
    details = {
        "skipped_resamples": skipped,
        "skipped_fraction": skipped_fraction,
    }
    if unique_groups is not None and skipped_fraction > 0.10:
        warnings.warn(
            "OOF cluster bootstrap skipped more than 10% of resamples because "
            "the metric was undefined for their sampled class composition "
            f"({skipped}/{n_resamples}, {skipped_fraction:.1%}).",
            UserWarning,
            stacklevel=2,
        )
    if not estimates:
        return (float("nan"), float("nan")), details
    alpha = 1.0 - level
    low, high = np.percentile(
        np.asarray(estimates), [alpha / 2 * 100, (1 - alpha / 2) * 100]
    )
    return (float(low), float(high)), details
