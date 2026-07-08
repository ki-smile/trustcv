"""Metric feasibility diagnostics for cross-validation folds."""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


POOLED_OOF_RECOMMENDATION = (
    "Some validation folds are small or contain only one outcome class; "
    "fold-wise AUC may be undefined or unstable. Consider pooled out-of-fold "
    "prediction aggregation."
)


_AUC_METRICS = {
    "auc",
    "roc_auc",
    "auc_roc",
    "roc_auc_score",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "roc_auc_ovr_macro",
    "average_precision",
    "auprc",
    "pr_auc",
}
_SENSITIVITY_METRICS = {"sensitivity", "tpr", "recall_pos", "recall", "recall_score"}
_SPECIFICITY_METRICS = {"specificity", "tnr", "recall_neg"}


def _slice_target(y_true: Sequence, indices: np.ndarray) -> np.ndarray:
    if hasattr(y_true, "iloc"):
        return np.asarray(y_true.iloc[indices])
    return np.asarray(y_true)[indices]


def _class_counts(values: np.ndarray) -> Dict[Any, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {cls.item() if hasattr(cls, "item") else cls: int(count) for cls, count in zip(unique, counts)}


def metric_names_need_class_feasibility(metric_names: Optional[Iterable[str]]) -> bool:
    """Return True when requested metrics include class-dependent fold metrics."""

    if metric_names is None:
        return True
    normalized = {str(metric).strip().lower() for metric in metric_names if metric is not None}
    return bool(normalized & (_AUC_METRICS | _SENSITIVITY_METRICS | _SPECIFICITY_METRICS))


def check_fold_metric_feasibility(
    y_true: Sequence,
    test_indices_by_fold: Iterable[Sequence[int]],
    metric_names: Optional[Iterable[str]] = None,
    min_test_samples: int = 5,
    require_both_classes_for_auc: bool = True,
) -> Dict[str, Any]:
    """
    Diagnose whether fold-wise classification metrics are feasible.

    Parameters
    ----------
    y_true : array-like
        Full target vector aligned with the original dataset.
    test_indices_by_fold : iterable of index arrays
        Validation/test indices for each fold.
    metric_names : iterable of str, optional
        Metrics requested by the user or produced by the workflow.
    min_test_samples : int, default=5
        Test folds smaller than this threshold are flagged as unstable.
    require_both_classes_for_auc : bool, default=True
        If True, ROC-AUC is marked infeasible for single-class validation folds.

    Returns
    -------
    dict
        Structured diagnostics with per-fold feasibility rows and warning text.
    """

    relevant_metrics_requested = metric_names_need_class_feasibility(metric_names)
    metric_names_list = list(metric_names) if metric_names is not None else None
    fold_rows: List[Dict[str, Any]] = []
    warning_messages: List[str] = []

    for fold_idx, test_idx in enumerate(test_indices_by_fold):
        test_idx_arr = np.asarray(test_idx, dtype=int)
        y_fold = _slice_target(y_true, test_idx_arr)
        if y_fold.ndim > 1 and y_fold.shape[1] == 1:
            y_fold = y_fold.ravel()

        test_fold_size = int(len(test_idx_arr))
        if y_fold.ndim != 1:
            row = {
                "fold": int(fold_idx),
                "fold_display": int(fold_idx + 1),
                "test_fold_size": test_fold_size,
                "n_unique_classes": None,
                "class_counts": {},
                "roc_auc_feasible": False,
                "sensitivity_feasible": False,
                "specificity_feasible": False,
                "is_small_fold": test_fold_size < min_test_samples,
                "is_single_class": False,
                "warning": (
                    "Fold has a multi-output target; binary fold-wise AUC, "
                    "sensitivity, and specificity feasibility cannot be inferred."
                ),
                "recommendation": POOLED_OOF_RECOMMENDATION,
            }
            fold_rows.append(row)
            warning_messages.append(row["warning"])
            continue

        counts = _class_counts(y_fold)
        n_unique_classes = int(len(counts))
        is_small_fold = test_fold_size < int(min_test_samples)
        is_single_class = n_unique_classes < 2
        roc_auc_feasible = (n_unique_classes >= 2) if require_both_classes_for_auc else True

        if n_unique_classes == 2:
            classes = list(counts.keys())
            negative_count = counts[classes[0]]
            positive_count = counts[classes[1]]
            sensitivity_feasible = positive_count > 0
            specificity_feasible = negative_count > 0
        else:
            sensitivity_feasible = n_unique_classes >= 2
            specificity_feasible = n_unique_classes >= 2

        reasons = []
        if is_small_fold:
            reasons.append(f"test fold has {test_fold_size} samples (< {min_test_samples})")
        if is_single_class:
            reasons.append("test fold contains only one outcome class")

        warning = ""
        recommendation = ""
        if reasons:
            warning = (
                f"Fold {fold_idx + 1}: {', '.join(reasons)}; fold-wise AUC, "
                "sensitivity, or specificity may be undefined or unstable."
            )
            recommendation = POOLED_OOF_RECOMMENDATION
            warning_messages.append(warning)

        fold_rows.append(
            {
                "fold": int(fold_idx),
                "fold_display": int(fold_idx + 1),
                "test_fold_size": test_fold_size,
                "n_unique_classes": n_unique_classes,
                "class_counts": counts,
                "roc_auc_feasible": bool(roc_auc_feasible),
                "sensitivity_feasible": bool(sensitivity_feasible),
                "specificity_feasible": bool(specificity_feasible),
                "is_small_fold": bool(is_small_fold),
                "is_single_class": bool(is_single_class),
                "warning": warning,
                "recommendation": recommendation,
            }
        )

    has_warnings = relevant_metrics_requested and bool(warning_messages)
    return {
        "folds": fold_rows,
        "warnings": warning_messages if relevant_metrics_requested else [],
        "has_warnings": bool(has_warnings),
        "recommendation": POOLED_OOF_RECOMMENDATION if has_warnings else "",
        "metric_names": metric_names_list,
        "min_test_samples": int(min_test_samples),
        "require_both_classes_for_auc": bool(require_both_classes_for_auc),
    }


def emit_metric_feasibility_warning(diagnostics: Dict[str, Any], stacklevel: int = 2) -> None:
    """Emit the standard pooled-OOF recommendation when diagnostics are problematic."""

    if diagnostics and diagnostics.get("has_warnings"):
        import warnings

        warnings.warn(POOLED_OOF_RECOMMENDATION, UserWarning, stacklevel=stacklevel)
