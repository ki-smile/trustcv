"""
Clinical metrics for medical machine learning evaluation

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team

Includes metrics required for regulatory submissions and clinical practice.
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class ClinicalMetrics:
    """
    Comprehensive clinical metrics calculator

    Calculates:
    - Sensitivity (TPR, Recall)
    - Specificity (TNR)
    - PPV (Positive Predictive Value, Precision)
    - NPV (Negative Predictive Value)
    - NNT (Number Needed to Treat)
    - NNS (Number Needed to Screen)
    - Likelihood Ratios
    - Diagnostic Odds Ratio
    - Youden's Index
    - All with confidence intervals

    Examples
    --------
    >>> metrics = ClinicalMetrics()
    >>> results = metrics.calculate_all(y_true, y_pred, y_proba)
    >>> print(results.summary())
    """

    def __init__(self, confidence_level=0.95, prevalence=None):
        """
        Parameters
        ----------
        confidence_level : float
            Confidence level for intervals (default 0.95)
        prevalence : float, optional
            Disease prevalence if different from dataset
        """
        self.confidence_level = confidence_level
        self.prevalence = prevalence

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Calculate all clinical metrics

        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_proba : array-like, optional
            Predicted probabilities
        sample_weight : array-like, optional
            Sample weights

        Returns
        -------
        dict
            Dictionary containing all metrics with confidence intervals
        """
        # Get confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight=sample_weight).ravel()

        # Calculate base metrics
        metrics = {}

        # Sensitivity (True Positive Rate, Recall)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["sensitivity"] = sensitivity
        metrics["sensitivity_ci"] = self._proportion_ci(tp, tp + fn)

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["specificity"] = specificity
        metrics["specificity_ci"] = self._proportion_ci(tn, tn + fp)

        # PPV (Positive Predictive Value, Precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["ppv"] = ppv
        metrics["ppv_ci"] = self._proportion_ci(tp, tp + fp)

        # NPV (Negative Predictive Value)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics["npv"] = npv
        metrics["npv_ci"] = self._proportion_ci(tn, tn + fn)

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        metrics["accuracy"] = accuracy
        metrics["accuracy_ci"] = self._proportion_ci(tp + tn, tp + tn + fp + fn)

        # F1 Score
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        metrics["f1_score"] = f1

        # Youden's Index
        metrics["youdens_index"] = sensitivity + specificity - 1

        # Likelihood Ratios
        metrics["lr_positive"] = sensitivity / (1 - specificity) if specificity < 1 else np.inf
        metrics["lr_negative"] = (1 - sensitivity) / specificity if specificity > 0 else np.inf

        # Diagnostic Odds Ratio
        if fp > 0 and fn > 0:
            metrics["diagnostic_odds_ratio"] = (tp * tn) / (fp * fn)
        else:
            metrics["diagnostic_odds_ratio"] = np.inf
        metrics["diagnostic_odds_ratio_ci"] = self._diagnostic_odds_ci(tp, tn, fp, fn)

        # NNT and NNS
        metrics["nnt"] = calculate_nnt(sensitivity, specificity, self.prevalence)
        metrics["nns"] = calculate_nns(sensitivity, self.prevalence)

        # AUC if probabilities provided
        if y_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba, sample_weight=sample_weight)
            metrics["auc_roc_ci"] = self._auc_ci(y_true, y_proba)

            # Average Precision (AP)
            metrics["average_precision"] = average_precision_score(
                y_true, y_proba, sample_weight=sample_weight
            )

            # Optimal threshold using Youden's Index
            fpr, tpr, thresholds = roc_curve(y_true, y_proba, sample_weight=sample_weight)
            youden_idx = tpr - fpr
            optimal_idx = np.argmax(youden_idx)
            metrics["optimal_threshold"] = thresholds[optimal_idx]
            metrics["optimal_sensitivity"] = tpr[optimal_idx]
            metrics["optimal_specificity"] = 1 - fpr[optimal_idx]

        # Add confusion matrix
        metrics["confusion_matrix"] = {
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

        # Calculate clinical significance
        metrics["clinical_significance"] = self._assess_clinical_significance(metrics)

        return metrics

    def _proportion_ci(self, successes: int, trials: int) -> Tuple[float, float]:
        """
        Calculate confidence interval for proportion using Wilson score

        Parameters
        ----------
        successes : int
            Number of successes
        trials : int
            Total number of trials

        Returns
        -------
        tuple
            (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 0.0)

        p = successes / trials
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def _auc_ci(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
        """
        Calculate confidence interval for AUC using DeLong method

        Parameters
        ----------
        y_true : array-like
            True labels
        y_proba : array-like
            Predicted probabilities

        Returns
        -------
        tuple
            (lower_bound, upper_bound)
        """
        # Simplified bootstrap CI for AUC
        n_bootstraps = 1000
        rng = np.random.RandomState(42)
        bootstrapped_scores = []

        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_proba[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.sort(bootstrapped_scores)
        alpha = 1 - self.confidence_level
        lower = sorted_scores[int(alpha / 2 * len(sorted_scores))]
        upper = sorted_scores[int((1 - alpha / 2) * len(sorted_scores))]

        return (lower, upper)

    def _assess_clinical_significance(self, metrics: Dict) -> Dict:
        """
        Assess clinical significance of results

        Parameters
        ----------
        metrics : dict
            Calculated metrics

        Returns
        -------
        dict
            Clinical significance assessment
        """
        assessment = {
            "screening_suitable": False,
            "diagnostic_suitable": False,
            "risk_stratification_suitable": False,
            "recommendations": [],
        }

        # Screening criteria (high sensitivity)
        if metrics["sensitivity"] > 0.95:
            assessment["screening_suitable"] = True
            assessment["recommendations"].append(
                "High sensitivity (>95%) - suitable for screening applications"
            )
        elif metrics["sensitivity"] > 0.90:
            assessment["recommendations"].append(
                "Good sensitivity (>90%) - consider for screening with confirmation"
            )

        # Diagnostic criteria (balanced)
        if metrics["sensitivity"] > 0.80 and metrics["specificity"] > 0.80:
            assessment["diagnostic_suitable"] = True
            assessment["recommendations"].append(
                "Balanced performance - suitable for diagnostic applications"
            )

        # Risk stratification (high specificity)
        if metrics["specificity"] > 0.95:
            assessment["risk_stratification_suitable"] = True
            assessment["recommendations"].append(
                "High specificity (>95%) - suitable for confirming positive cases"
            )

        # Check for poor performance
        if metrics["sensitivity"] < 0.70:
            assessment["recommendations"].append(
                "⚠️ Low sensitivity (<70%) - many false negatives expected"
            )
        if metrics["specificity"] < 0.70:
            assessment["recommendations"].append(
                "⚠️ Low specificity (<70%) - many false positives expected"
            )

        # NNT interpretation
        if "nnt" in metrics and metrics["nnt"] is not None:
            if metrics["nnt"] < 10:
                assessment["recommendations"].append(
                    f"Excellent NNT ({metrics['nnt']:.1f}) - highly efficient intervention"
                )
            elif metrics["nnt"] < 50:
                assessment["recommendations"].append(
                    f"Good NNT ({metrics['nnt']:.1f}) - reasonable intervention efficiency"
                )
            else:
                assessment["recommendations"].append(
                    f"High NNT ({metrics['nnt']:.1f}) - consider cost-effectiveness"
                )

        return assessment

    def format_report(self, metrics: Dict) -> str:
        """
        Format metrics into clinical report

        Parameters
        ----------
        metrics : dict
            Calculated metrics

        Returns
        -------
        str
            Formatted report
        """
        report = "=" * 60 + "\n"
        report += "CLINICAL PERFORMANCE METRICS REPORT\n"
        report += "=" * 60 + "\n\n"

        # Primary Metrics
        report += "PRIMARY DIAGNOSTIC METRICS\n"
        report += "-" * 30 + "\n"
        report += f"Sensitivity:  {metrics['sensitivity']:.1%} "
        report += f"[{metrics['sensitivity_ci'][0]:.1%}, {metrics['sensitivity_ci'][1]:.1%}]\n"
        report += f"Specificity:  {metrics['specificity']:.1%} "
        report += f"[{metrics['specificity_ci'][0]:.1%}, {metrics['specificity_ci'][1]:.1%}]\n"
        report += f"PPV:          {metrics['ppv']:.1%} "
        report += f"[{metrics['ppv_ci'][0]:.1%}, {metrics['ppv_ci'][1]:.1%}]\n"
        report += f"NPV:          {metrics['npv']:.1%} "
        report += f"[{metrics['npv_ci'][0]:.1%}, {metrics['npv_ci'][1]:.1%}]\n"

        # Performance Metrics
        report += "\nPERFORMANCE METRICS\n"
        report += "-" * 30 + "\n"
        report += f"Accuracy:         {metrics['accuracy']:.1%}\n"
        report += f"Accuracy CI:      [{metrics['accuracy_ci'][0]:.1%}, {metrics['accuracy_ci'][1]:.1%}]\n"
        report += f"F1 Score:         {metrics['f1_score']:.3f}\n"
        if "auc_roc" in metrics:
            report += f"AUC-ROC:          {metrics['auc_roc']:.3f} "
            report += f"[{metrics['auc_roc_ci'][0]:.3f}, {metrics['auc_roc_ci'][1]:.3f}]\n"
        if "average_precision" in metrics:
            report += f"Average Precision: {metrics['average_precision']:.3f}\n"
        if "optimal_threshold" in metrics:
            report += f"Optimal Threshold (Youden): {metrics['optimal_threshold']:.3f}\n"
            report += f"  ↳ Sensitivity @ Threshold: {metrics['optimal_sensitivity']:.1%}\n"
            report += f"  ↳ Specificity @ Threshold: {metrics['optimal_specificity']:.1%}\n"

        # Clinical Utility
        report += "\nCLINICAL UTILITY\n"
        report += "-" * 30 + "\n"
        report += f"Youden's Index:     {metrics['youdens_index']:.3f}\n"
        report += f"LR+:                {metrics['lr_positive']:.2f}\n"
        report += f"LR-:                {metrics['lr_negative']:.2f}\n"
        report += f"Diagnostic OR:      {metrics['diagnostic_odds_ratio']:.2f}\n"
        if metrics["nnt"]:
            report += f"NNT:                {metrics['nnt']:.1f}\n"
        else:
            report += "NNT:                n/a\n"
        if metrics["nns"]:
            report += f"NNS:                {metrics['nns']:.1f}\n"
        else:
            report += "NNS:                n/a\n"
        report += f"LR+ interpretation:  {self._interpret_lr_positive(metrics['lr_positive'])}\n"
        report += f"LR- interpretation:  {self._interpret_lr_negative(metrics['lr_negative'])}\n"
        if metrics.get("diagnostic_odds_ratio_ci"):
            lo, hi = metrics["diagnostic_odds_ratio_ci"]
            report += f"Diagnostic OR CI:   [{lo:.1f}, {hi:.1f}]\n"

        # Confusion Matrix
        report += "\nCONFUSION MATRIX\n"
        report += "-" * 30 + "\n"
        cm = metrics["confusion_matrix"]
        report += f"True Positives:     {cm['true_positives']}\n"
        report += f"True Negatives:     {cm['true_negatives']}\n"
        report += f"False Positives:    {cm['false_positives']}\n"
        report += f"False Negatives:    {cm['false_negatives']}\n"

        # Clinical Significance
        report += "\nCLINICAL SIGNIFICANCE\n"
        report += "-" * 30 + "\n"
        sig = metrics["clinical_significance"]
        report += f"Screening Suitable:     {'Yes' if sig['screening_suitable'] else 'No'}\n"
        report += f"Diagnostic Suitable:    {'Yes' if sig['diagnostic_suitable'] else 'No'}\n"
        report += (
            f"Risk Stratification:    {'Yes' if sig['risk_stratification_suitable'] else 'No'}\n"
        )

        report += "\nRECOMMENDATIONS\n"
        report += "-" * 30 + "\n"
        for rec in sig["recommendations"]:
            report += f"• {rec}\n"

        return report

    def _diagnostic_odds_ci(
        self, tp: int, tn: int, fp: int, fn: int
    ) -> Optional[Tuple[float, float]]:
        """
        Approximate CI for diagnostic odds ratio using log transformation.
        """
        import numpy as _np

        cells = [tp, tn, fp, fn]
        if any(v < 0 for v in cells):
            return None

        # Apply continuity correction if needed
        corrected = [_v if _v > 0 else 0.5 for _v in cells]
        tp_c, tn_c, fp_c, fn_c = corrected
        if fp_c == 0 or fn_c == 0:
            return None

        dor = (tp_c * tn_c) / (fp_c * fn_c)
        if dor <= 0:
            return None

        log_dor = _np.log(dor)
        se = _np.sqrt(1 / tp_c + 1 / tn_c + 1 / fp_c + 1 / fn_c)
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        lo = _np.exp(log_dor - z * se)
        hi = _np.exp(log_dor + z * se)
        return (float(lo), float(hi))

    @staticmethod
    def _interpret_lr_positive(value: float) -> str:
        if value >= 10:
            return "Strong increase in post-test probability"
        if value >= 5:
            return "Moderate increase in post-test probability"
        if value >= 2:
            return "Small increase in post-test probability"
        return "Minimal change"

    @staticmethod
    def _interpret_lr_negative(value: float) -> str:
        if value <= 0.1:
            return "Strong decrease in post-test probability"
        if value <= 0.2:
            return "Moderate decrease in post-test probability"
        if value <= 0.5:
            return "Small decrease in post-test probability"
        return "Minimal change"


def calculate_nnt(
    sensitivity: float, specificity: float, prevalence: Optional[float] = None
) -> Optional[float]:
    """
    Calculate Number Needed to Treat

    Parameters
    ----------
    sensitivity : float
        Test sensitivity
    specificity : float
        Test specificity
    prevalence : float, optional
        Disease prevalence

    Returns
    -------
    float or None
        NNT value
    """
    if prevalence is None:
        return None

    # Absolute risk reduction
    arr = prevalence * sensitivity - prevalence * (1 - specificity)

    if arr > 0:
        return 1 / arr
    else:
        return None


def calculate_nns(sensitivity: float, prevalence: Optional[float] = None) -> Optional[float]:
    """
    Calculate Number Needed to Screen

    Parameters
    ----------
    sensitivity : float
        Test sensitivity
    prevalence : float, optional
        Disease prevalence

    Returns
    -------
    float or None
        NNS value
    """
    if prevalence is None or sensitivity == 0:
        return None

    return 1 / (prevalence * sensitivity)


def calculate_clinical_significance(
    metric_value: float, metric_type: str, application: str = "diagnostic"
) -> str:
    """
    Assess clinical significance of a metric value

    Parameters
    ----------
    metric_value : float
        The metric value
    metric_type : str
        Type of metric ('sensitivity', 'specificity', 'auc', etc.)
    application : str
        Clinical application ('screening', 'diagnostic', 'prognostic')

    Returns
    -------
    str
        Clinical significance assessment
    """
    thresholds = {
        "screening": {
            "sensitivity": {"excellent": 0.95, "good": 0.90, "acceptable": 0.85},
            "specificity": {"excellent": 0.90, "good": 0.85, "acceptable": 0.80},
        },
        "diagnostic": {
            "sensitivity": {"excellent": 0.90, "good": 0.85, "acceptable": 0.80},
            "specificity": {"excellent": 0.95, "good": 0.90, "acceptable": 0.85},
        },
        "prognostic": {"auc": {"excellent": 0.90, "good": 0.80, "acceptable": 0.70}},
    }

    if application not in thresholds:
        return "Unknown application type"

    if metric_type not in thresholds[application]:
        return "Metric not applicable for this application"

    levels = thresholds[application][metric_type]

    if metric_value >= levels["excellent"]:
        return "Excellent"
    elif metric_value >= levels["good"]:
        return "Good"
    elif metric_value >= levels["acceptable"]:
        return "Acceptable"
    else:
        return "Poor - consider model improvement"
