"""
Medical-specific metrics for cross-validation

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
Includes sensitivity, specificity, PPV, NPV, and clinical utility scores
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sensitivity(y_true, y_pred, pos_label=1):
    """
    Calculate sensitivity (recall/true positive rate)

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        Sensitivity score
    """
    return recall_score(y_true, y_pred, pos_label=pos_label)


def specificity(y_true, y_pred, pos_label=1):
    """
    Calculate specificity (true negative rate, TNR) for binary classification.

    For robustness across label types, this computes recall of the negative class
    rather than relying on numeric label arithmetic.
    """
    # Determine negative label
    import numpy as _np

    uniq = _np.unique(_np.concatenate([_np.asarray(y_true).ravel(), _np.asarray(y_pred).ravel()]))
    if uniq.size >= 2 and pos_label in uniq:
        neg_candidates = [lab for lab in uniq if lab != pos_label]
        neg_label = neg_candidates[0]
    else:
        # Fallback to 0/1 convention
        neg_label = 0 if pos_label != 0 else 1
    return recall_score(y_true, y_pred, pos_label=neg_label)


def positive_predictive_value(y_true, y_pred, pos_label=1):
    """
    Calculate positive predictive value (precision)

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        PPV score
    """
    return precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)


def negative_predictive_value(y_true, y_pred, pos_label=1):
    """
    Calculate negative predictive value

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        NPV score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1 - pos_label, pos_label]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def likelihood_ratio_positive(y_true, y_pred, pos_label=1):
    """
    Calculate positive likelihood ratio (LR+)

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        LR+ score
    """
    sens = sensitivity(y_true, y_pred, pos_label)
    spec = specificity(y_true, y_pred, pos_label)
    return sens / (1 - spec) if (1 - spec) > 0 else np.inf


def likelihood_ratio_negative(y_true, y_pred, pos_label=1):
    """
    Calculate negative likelihood ratio (LR-)

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        LR- score
    """
    sens = sensitivity(y_true, y_pred, pos_label)
    spec = specificity(y_true, y_pred, pos_label)
    return (1 - sens) / spec if spec > 0 else np.inf


def youden_index(y_true, y_pred, pos_label=1):
    """
    Calculate Youden's J statistic (Sensitivity + Specificity - 1)

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        Youden's J statistic
    """
    sens = sensitivity(y_true, y_pred, pos_label)
    spec = specificity(y_true, y_pred, pos_label)
    return sens + spec - 1


def net_benefit(y_true, y_pred, threshold=0.5, pos_label=1):
    """
    Calculate net benefit for clinical decision analysis

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted probabilities or labels
    threshold : float
        Decision threshold
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        Net benefit score
    """
    # Convert probabilities to binary predictions if needed
    if hasattr(y_pred, "max") and y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred_binary, labels=[1 - pos_label, pos_label]
    ).ravel()

    # Net benefit calculation
    n = len(y_true)
    prob_threshold = threshold / (1 - threshold)
    net_benefit = (tp - fp * prob_threshold) / n

    return net_benefit


def clinical_utility_score(y_true, y_pred, cost_fp=1, cost_fn=1, pos_label=1):
    """
    Calculate clinical utility considering misclassification costs

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    cost_fp : float
        Cost of false positive
    cost_fn : float
        Cost of false negative
    pos_label : int/str
        Label of positive class

    Returns
    -------
    float
        Clinical utility score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1 - pos_label, pos_label]).ravel()

    # Utility = Benefit - Cost
    utility = tp - (cost_fp * fp + cost_fn * fn)

    # Normalize by total samples
    return utility / len(y_true)


def comprehensive_medical_metrics(
    y_true, y_pred, y_proba=None, pos_label=1, threshold=0.5, cost_fp=1, cost_fn=1
):
    """
    Calculate comprehensive medical metrics

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities (for ROC-AUC, PR-AUC)
    pos_label : int/str
        Label of positive class
    threshold : float
        Decision threshold for net benefit
    cost_fp, cost_fn : float
        Misclassification costs

    Returns
    -------
    dict
        Dictionary of medical metrics
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["sensitivity"] = sensitivity(y_true, y_pred, pos_label)
    metrics["specificity"] = specificity(y_true, y_pred, pos_label)
    metrics["ppv"] = positive_predictive_value(y_true, y_pred, pos_label)
    metrics["npv"] = negative_predictive_value(y_true, y_pred, pos_label)
    metrics["f1_score"] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    # Advanced metrics
    metrics["lr_positive"] = likelihood_ratio_positive(y_true, y_pred, pos_label)
    metrics["lr_negative"] = likelihood_ratio_negative(y_true, y_pred, pos_label)
    metrics["youden_j"] = youden_index(y_true, y_pred, pos_label)

    # Clinical metrics
    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
            metrics["auc_pr"] = average_precision_score(y_true, y_proba)
            metrics["net_benefit"] = net_benefit(y_true, y_proba, threshold, pos_label)
        except ValueError as e:
            warnings.warn(f"Could not calculate probability-based metrics: {e}")
            metrics["auc_roc"] = np.nan
            metrics["auc_pr"] = np.nan
            metrics["net_benefit"] = np.nan
    else:
        metrics["auc_roc"] = np.nan
        metrics["auc_pr"] = np.nan
        metrics["net_benefit"] = net_benefit(y_true, y_pred, threshold, pos_label)

    metrics["clinical_utility"] = clinical_utility_score(
        y_true, y_pred, cost_fp, cost_fn, pos_label
    )

    return metrics


def medical_classification_report(
    y_true, y_pred, y_proba=None, target_names=None, threshold=0.5, cost_fp=1, cost_fn=1
):
    """
    Generate comprehensive medical classification report

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities
    target_names : list, optional
        Names of target classes
    threshold : float
        Decision threshold
    cost_fp, cost_fn : float
        Misclassification costs

    Returns
    -------
    str
        Formatted report
    """
    # Get metrics for each class if multiclass
    unique_labels = np.unique(y_true)

    if len(unique_labels) == 2:
        # Binary classification
        pos_label = unique_labels[1]
        metrics = comprehensive_medical_metrics(
            y_true, y_pred, y_proba, pos_label, threshold, cost_fp, cost_fn
        )

        report = "Medical Classification Report\n"
        report += "=" * 50 + "\n\n"

        # Basic performance
        report += "Basic Performance Metrics:\n"
        report += f"  Accuracy:     {metrics['accuracy']:.3f}\n"
        report += f"  Sensitivity:  {metrics['sensitivity']:.3f}\n"
        report += f"  Specificity:  {metrics['specificity']:.3f}\n"
        report += f"  PPV:          {metrics['ppv']:.3f}\n"
        report += f"  NPV:          {metrics['npv']:.3f}\n"
        report += f"  F1-Score:     {metrics['f1_score']:.3f}\n\n"

        # Advanced metrics
        report += "Advanced Metrics:\n"
        if np.isfinite(metrics["lr_positive"]):
            report += f"  LR+:          {metrics['lr_positive']:.3f}\n"
        else:
            report += f"  LR+:          ∞\n"
        if np.isfinite(metrics["lr_negative"]):
            report += f"  LR-:          {metrics['lr_negative']:.3f}\n"
        else:
            report += f"  LR-:          ∞\n"
        report += f"  Youden's J:   {metrics['youden_j']:.3f}\n\n"

        # Clinical metrics
        report += "Clinical Utility:\n"
        if not np.isnan(metrics["auc_roc"]):
            report += f"  AUC-ROC:      {metrics['auc_roc']:.3f}\n"
        if not np.isnan(metrics["auc_pr"]):
            report += f"  AUC-PR:       {metrics['auc_pr']:.3f}\n"
        if not np.isnan(metrics["net_benefit"]):
            report += f"  Net Benefit:  {metrics['net_benefit']:.3f}\n"
        report += f"  Clinical Utility: {metrics['clinical_utility']:.3f}\n"

        return report

    else:
        # Multiclass - use standard classification report
        return classification_report(y_true, y_pred, target_names=target_names)
