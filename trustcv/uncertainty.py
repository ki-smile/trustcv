"""Confidence intervals for cross-validation estimates."""

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
    train_sizes: Iterable[int],
    test_sizes: Iterable[int],
    level: float,
) -> Tuple[float, float]:
    values_array = np.asarray(list(values), dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size <= 1:
        return (float("nan"), float("nan"))
    train_mean = float(np.mean(list(train_sizes)))
    test_mean = float(np.mean(list(test_sizes)))
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
) -> Tuple[Tuple[float, float], Dict[str, int]]:
    estimates = []
    skipped = 0
    n_samples = y_true.shape[0]
    labels = None if regression else np.unique(y_true)
    pos_label = None if labels is None or labels.size == 0 else labels[-1]
    unique_groups = np.unique(groups) if cluster and groups is not None else None
    for _ in range(max(int(n_bootstrap), 1)):
        if unique_groups is not None:
            sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
            sample_indices = np.concatenate(
                [np.flatnonzero(groups == group) for group in sampled_groups]
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
    if not estimates:
        return (float("nan"), float("nan")), {"skipped_resamples": skipped}
    alpha = 1.0 - level
    low, high = np.percentile(
        np.asarray(estimates), [alpha / 2 * 100, (1 - alpha / 2) * 100]
    )
    return (float(low), float(high)), {"skipped_resamples": skipped}
