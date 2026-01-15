"""
Medical-specific metrics and evaluation tools

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import numpy as np

from .clinical import ClinicalMetrics, calculate_clinical_significance, calculate_nnt
from .medical_metrics import (
    clinical_utility_score,
)
from .medical_metrics import likelihood_ratio_negative as lr_negative
from .medical_metrics import likelihood_ratio_positive as lr_positive
from .medical_metrics import negative_predictive_value as npv
from .medical_metrics import (
    net_benefit,
)
from .medical_metrics import positive_predictive_value as ppv
from .medical_metrics import sensitivity as sensitivity_score
from .medical_metrics import specificity as specificity_score
from .medical_metrics import (
    youden_index,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..core.base import CVResults


def oob_clinical_metrics(
    results: "CVResults",
    y: Sequence,
    *,
    clinical: Optional[ClinicalMetrics] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute clinical metrics (with confidence intervals) from out-of-bag predictions.

    Parameters
    ----------
    results : CVResults
        Output from UniversalCVRunner or other trustcv cross-validation workflow.
        Must include `indices` and `scores` with stored predictions/probabilities.
    y : array-like
        Full target vector aligned with the original dataset.
    clinical : ClinicalMetrics, optional
        Custom ClinicalMetrics instance; one is created if omitted.

    Returns
    -------
    dict or None
        Dictionary of metrics from ClinicalMetrics.calculate_all(), or None if
        predictions were unavailable in the results.
    """

    if results is None or not getattr(results, "indices", None):
        return None

    def _slice_target(target: Sequence, idx: np.ndarray) -> np.ndarray:
        if hasattr(target, "iloc"):
            return np.asarray(target.iloc[idx])
        return np.asarray(target)[idx]

    y_true_chunks = []
    y_pred_chunks = []
    y_proba_chunks = []

    scores_iter = results.scores or []
    for (train_idx, test_idx), fold_scores in zip(results.indices, scores_iter):
        if test_idx is None or fold_scores is None:
            continue
        y_true_chunks.append(_slice_target(y, test_idx))

        preds = fold_scores.get("predictions")
        if preds is None:
            preds = fold_scores.get("y_pred")
        if preds is not None:
            y_pred_chunks.append(np.asarray(preds).ravel())

        probas = fold_scores.get("probabilities")
        if probas is None:
            probas = fold_scores.get("y_proba")
        if probas is not None:
            arr = np.asarray(probas)
            if arr.ndim == 2 and arr.shape[1] > 1:
                arr = arr[:, 1]
            y_proba_chunks.append(arr.ravel())

    if not y_pred_chunks:
        return None

    y_true_all = np.concatenate(y_true_chunks)
    y_pred_all = np.concatenate(y_pred_chunks)
    y_proba_all = np.concatenate(y_proba_chunks) if y_proba_chunks else None

    metrics_calc = clinical or ClinicalMetrics()
    return metrics_calc.calculate_all(
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_proba=y_proba_all,
    )


__all__ = [
    "ClinicalMetrics",
    "calculate_nnt",
    "calculate_clinical_significance",
    "sensitivity_score",
    "specificity_score",
    "ppv",
    "npv",
    "lr_positive",
    "lr_negative",
    "youden_index",
    "net_benefit",
    "clinical_utility_score",
    "oob_clinical_metrics",
]
