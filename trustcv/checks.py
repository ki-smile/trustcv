"""Structured integrity checks for TrustCV validation results."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


CHECK_KEYS = (
    "duplicate_samples",
    "group_leakage",
    "preprocessing_leakage",
    "external_leakage_detector",
    "target_leakage_features",
    "permutation_sanity",
    "class_balance",
    "covariate_shift",
)

LEAKAGE_RELEVANT_KEYS = (
    "duplicate_samples",
    "group_leakage",
    "preprocessing_leakage",
    "external_leakage_detector",
    "target_leakage_features",
)

VALID_CHECK_STATUSES = {
    "PASSED",
    "FAILED",
    "WARNING",
    "NOT_CHECKED",
    "NOT_APPLICABLE",
    "INFO",
    "ERROR",
}


@dataclass
class CheckResult:
    """Result of one TrustCV integrity check.

    Parameters
    ----------
    name : str
        Stable machine-readable check name.
    status : str
        One of ``PASSED``, ``FAILED``, ``WARNING``, ``NOT_CHECKED``,
        ``NOT_APPLICABLE``, ``INFO``, or ``ERROR``.
    message : str
        Honest one-line description of what was or was not established.
    details : dict, optional
        Supporting values for programmatic inspection.

    Notes
    -----
    A ``PASSED`` result covers only the condition named by this check. It
    cannot establish that no unmeasured source of leakage exists.
    """

    name: str
    status: str
    message: str
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in VALID_CHECK_STATUSES:
            raise ValueError(f"Invalid check status: {self.status!r}")


def default_checks() -> Dict[str, CheckResult]:
    """Create the complete internal check map."""
    messages = {
        "duplicate_samples": "Duplicate-sample checking was not run.",
        "group_leakage": "Group leakage was not checked.",
        "preprocessing_leakage": "Preprocessing placement was not checked.",
        "external_leakage_detector": "The external leakage detector was not run.",
        "target_leakage_features": "Target-leakage feature scanning was not run.",
        "permutation_sanity": "Permutation sanity checking is disabled.",
        "class_balance": "Class balance was not checked.",
        "covariate_shift": "Covariate-shift reporting is unavailable.",
    }
    return {
        name: CheckResult(name=name, status="NOT_CHECKED", message=messages[name])
        for name in CHECK_KEYS
    }


def ensure_complete_checks(
    checks: Optional[Dict[str, CheckResult]],
) -> Dict[str, CheckResult]:
    """Return a copy containing every required check key."""
    complete = default_checks()
    if checks:
        complete.update(checks)
    return complete


def overall_status(checks: Dict[str, CheckResult]) -> str:
    """Compute the conservative overall verification status."""
    relevant = list(LEAKAGE_RELEVANT_KEYS)
    permutation = checks.get("permutation_sanity")
    if permutation is not None and permutation.status != "NOT_CHECKED":
        relevant.append("permutation_sanity")
    statuses = [checks[name].status for name in relevant]
    if any(status in {"FAILED", "ERROR"} for status in statuses):
        return "FAILED"
    if all(status in {"PASSED", "NOT_APPLICABLE"} for status in statuses):
        return "PASSED"
    return "NOT_FULLY_VERIFIED"


def _duplicate_samples_check(X: Any, enabled: bool) -> CheckResult:
    if not enabled:
        return CheckResult(
            "duplicate_samples", "NOT_CHECKED", "Duplicate-sample checking is disabled."
        )
    if isinstance(X, dict) or (
        isinstance(X, (list, tuple)) and not hasattr(X, "shape")
    ):
        return CheckResult(
            "duplicate_samples",
            "NOT_CHECKED",
            "Duplicate samples cannot be checked automatically for multi-input data.",
        )
    try:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        hashes = pd.util.hash_pandas_object(frame, index=False)
        duplicate_count = int(hashes.duplicated(keep="first").sum())
    except Exception as exc:
        return CheckResult(
            "duplicate_samples",
            "ERROR",
            f"Duplicate-sample checking failed: {exc}",
            {"error": str(exc)},
        )
    if duplicate_count:
        return CheckResult(
            "duplicate_samples",
            "WARNING",
            f"Found {duplicate_count} exact duplicate rows in the supplied features.",
            {"duplicate_count": duplicate_count},
        )
    return CheckResult(
        "duplicate_samples",
        "PASSED",
        "No exact duplicate feature rows were found.",
        {"duplicate_count": 0},
    )


def _group_leakage_check(
    X: Any,
    y: Any,
    groups: Any,
    splitter: Any,
    enabled: bool,
    declare_independent_samples: bool,
) -> CheckResult:
    if not enabled:
        return CheckResult("group_leakage", "NOT_CHECKED", "Group leakage checking is disabled.")
    if groups is None:
        if declare_independent_samples:
            return CheckResult(
                "group_leakage",
                "NOT_APPLICABLE",
                "Samples were explicitly declared independent; no groups were supplied.",
            )
        return CheckResult(
            "group_leakage",
            "NOT_CHECKED",
            "No groups or patient_ids were supplied, so group leakage cannot be checked.",
        )
    groups_arr = groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups)
    try:
        for train_idx, test_idx in splitter.split(X, y, groups_arr):
            train_groups = set(np.unique(groups_arr[train_idx]))
            test_groups = set(np.unique(groups_arr[test_idx]))
            overlap = train_groups.intersection(test_groups)
            if overlap:
                return CheckResult(
                    "group_leakage",
                    "FAILED",
                    "Group or patient IDs overlap between training and test folds.",
                    {"overlap_count": len(overlap)},
                )
    except Exception as exc:
        return CheckResult(
            "group_leakage",
            "ERROR",
            f"Group leakage checking failed while iterating the splitter: {exc}",
            {"error": str(exc)},
        )
    return CheckResult(
        "group_leakage",
        "PASSED",
        "No group or patient ID appears in both training and test within a fold.",
    )


def _preprocessing_check(model: Any) -> CheckResult:
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline) and len(model.steps) > 1:
            return CheckResult(
                "preprocessing_leakage",
                "PASSED",
                "Preprocessing steps are inside an sklearn Pipeline and are refit within each fold.",
            )
    except Exception:
        pass
    return CheckResult(
        "preprocessing_leakage",
        "NOT_CHECKED",
        "Cannot verify that preprocessing (scaling, imputation, feature selection) was fit only on training folds. Wrap preprocessing in an sklearn Pipeline.",
    )


def _class_balance_check(y: Any, enabled: bool, is_regression: bool) -> CheckResult:
    if not enabled:
        return CheckResult("class_balance", "NOT_CHECKED", "Class-balance checking is disabled.")
    if is_regression:
        return CheckResult(
            "class_balance", "NOT_APPLICABLE", "Class balance is not applicable to regression."
        )
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    if y_arr.ndim != 1:
        return CheckResult(
            "class_balance", "INFO", "Class-balance status is informational for multilabel targets."
        )
    _, counts = np.unique(y_arr, return_counts=True)
    if counts.size < 2:
        return CheckResult(
            "class_balance", "WARNING", "Only one class is present in the supplied target."
        )
    ratio = float(counts.min() / counts.max())
    status = "WARNING" if ratio < 0.10 else "INFO"
    message = (
        f"Minority-to-majority class ratio is {ratio:.3f}; severe imbalance may destabilize folds."
        if status == "WARNING"
        else f"Minority-to-majority class ratio is {ratio:.3f}."
    )
    return CheckResult("class_balance", status, message, {"minority_majority_ratio": ratio})


def build_initial_checks(
    *,
    X: Any,
    y: Any,
    model: Any,
    groups: Any,
    splitter: Any,
    check_leakage: bool,
    check_balance: bool,
    declare_independent_samples: bool,
    is_regression: bool,
) -> Dict[str, CheckResult]:
    """Build checks available without an external leakage report."""
    checks = default_checks()
    checks["duplicate_samples"] = _duplicate_samples_check(X, check_leakage)
    checks["group_leakage"] = _group_leakage_check(
        X, y, groups, splitter, check_leakage, declare_independent_samples
    )
    checks["preprocessing_leakage"] = _preprocessing_check(model)
    checks["class_balance"] = _class_balance_check(y, check_balance, is_regression)
    return checks


def covariate_shift_from_report(report: Any) -> CheckResult:
    """Translate an optional leakage-report shift section into a check result."""
    details = getattr(report, "details", {}) or {}
    shift = details.get("covariate_shift")
    if shift is None:
        return CheckResult(
            "covariate_shift", "NOT_CHECKED", "Covariate-shift reporting is unavailable."
        )
    n_shifted = int(shift.get("n_shifted_features", 0))
    status = "WARNING" if n_shifted else "INFO"
    message = (
        f"Covariate shift was detected in {n_shifted} feature(s)."
        if n_shifted
        else "No statistically significant covariate shift was detected."
    )
    return CheckResult("covariate_shift", status, message, dict(shift))
