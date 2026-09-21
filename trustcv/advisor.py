"""Cross-validation design advisor."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from sklearn.utils.multiclass import type_of_target

from .metrics import check_fold_metric_feasibility
from .splitters import (
    GroupKFoldMedical,
    KFoldMedical,
    PurgedGroupTimeSeriesSplit,
    RepeatedKFold,
    SpatialBlockCV,
    StratifiedGroupKFold,
    StratifiedKFoldMedical,
    TimeSeriesSplit,
)


@dataclass
class CVRecommendation:
    """Actionable recommendation for a cross-validation design.

    Parameters
    ----------
    category : {"iid", "grouped", "temporal", "spatial"}
        Dependence structure driving the recommendation.
    method : str or None
        TrustCV method string when one maps directly to the design.
    splitter : object
        Instantiated splitter accepted by ``TrustCV.validate(cv=...)``.
    rationale : str
        Plain-language reason for the choice.
    warnings : list of str
        Data limitations or fold-feasibility concerns.
    code : str
        Valid Python example that constructs the splitter.

    Notes
    -----
    The recommendation uses only supplied arrays and simple target/group
    diagnostics. It cannot infer unprovided patient IDs, timestamps, spatial
    dependence, causal structure, or preprocessing leakage.
    """

    category: str
    method: Optional[str]
    splitter: Any
    rationale: str
    warnings: List[str] = field(default_factory=list)
    code: str = ""


def _classification_target(y: Any) -> bool:
    target = type_of_target(np.asarray(y))
    return target in {"binary", "multiclass"}


def _effective_group_splits(groups: np.ndarray, requested: int) -> int:
    return max(2, min(int(requested), int(np.unique(groups).size)))


def _classification_warnings(
    X: Any,
    y: np.ndarray,
    splitter: Any,
    groups: Optional[np.ndarray],
) -> List[str]:
    warnings: List[str] = []
    _, counts = np.unique(y, return_counts=True)
    if counts.size > 1 and int(counts.min()) < 10:
        warnings.append(
            f"Only {int(counts.min())} minority-class events are available; fold estimates may be unstable."
        )
    try:
        splits = list(splitter.split(X, y, groups))
        diagnostics = check_fold_metric_feasibility(
            y,
            [test for _, test in splits],
            metric_names=["roc_auc"],
        )
        if diagnostics.get("has_warnings"):
            warnings.append(
                "Some proposed folds may have zero positive/minority events; inspect fold feasibility before training."
            )
    except Exception:
        if counts.size > 1 and int(counts.min()) < getattr(splitter, "n_splits", 5):
            warnings.append(
                "The minority event count is smaller than the fold count, so some folds may have zero positives."
            )
    return warnings


def recommend_cv(
    X: Any,
    y: Any,
    *,
    groups: Any = None,
    timestamps: Any = None,
    coordinates: Any = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> CVRecommendation:
    """Recommend a runnable CV splitter from supplied dependence metadata.

    Parameters
    ----------
    X, y : array-like
        Feature matrix and target.
    groups : array-like, optional
        Subject, patient, site, or other repeated-unit identifiers.
    timestamps : array-like, optional
        Observation times. Their presence takes precedence over spatial and
        grouped recommendations.
    coordinates : array-like, optional
        Two-dimensional spatial coordinates.
    n_splits : int, default=5
        Requested fold count.
    random_state : int, default=42
        Seed for splitters that shuffle.

    Returns
    -------
    CVRecommendation
        Category, method, runnable splitter, rationale, warnings, and code.

    Notes
    -----
    The advisor can respond only to metadata supplied here. It cannot discover
    hidden repeated subjects, temporal ordering, spatial autocorrelation, or
    leakage from preprocessing performed before the call.
    """
    y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    groups_array = (
        None
        if groups is None
        else (groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups))
    )
    classification = _classification_target(y_array)
    warnings: List[str] = []

    if groups_array is not None:
        unique_groups = int(np.unique(groups_array).size)
        if unique_groups < 2 * int(n_splits):
            warnings.append(
                f"Only {unique_groups} unique groups are available, fewer than 2 × n_splits; group-fold estimates may be unstable."
            )

    if timestamps is not None:
        if groups_array is not None:
            splitter = PurgedGroupTimeSeriesSplit(
                n_splits=n_splits, group_exclusive=True
            )
            rationale = (
                "Timestamps make temporal ordering the primary constraint, and the supplied groups require patient-aware exclusion; use purged group time-series folds after sorting rows by time."
            )
            code = (
                "from trustcv import PurgedGroupTimeSeriesSplit\n"
                f"splitter = PurgedGroupTimeSeriesSplit(n_splits={n_splits}, group_exclusive=True)\n"
                "# Sort X, y, and groups by timestamps before TrustCV.validate(cv=splitter).\n"
            )
        else:
            splitter = TimeSeriesSplit(n_splits=n_splits)
            rationale = (
                "Timestamps make temporal ordering the primary constraint; use forward time-series folds and sort all rows by time before validation."
            )
            warnings.append(
                "No groups were supplied. If the same subject appears repeatedly, supply group IDs and use group-aware temporal validation."
            )
            code = (
                "from trustcv import TimeSeriesSplit\n"
                f"splitter = TimeSeriesSplit(n_splits={n_splits})\n"
                "# Sort X and y by timestamps before TrustCV.validate(cv=splitter).\n"
            )
        warnings.append(
            "TrustCV.validate has no timestamps argument; sort data by time or pass a preconfigured splitter via cv=."
        )
        rec = CVRecommendation("temporal", "temporal", splitter, rationale, warnings, code)
    elif coordinates is not None:
        splitter = SpatialBlockCV(
            n_splits=n_splits,
            random_state=random_state,
            coordinates=coordinates,
        )
        rationale = (
            "Supplied coordinates imply spatial dependence, so spatial blocks should separate geographic regions instead of treating rows as IID."
        )
        code = (
            "from trustcv import SpatialBlockCV\n"
            f"splitter = SpatialBlockCV(n_splits={n_splits}, random_state={random_state}, coordinates=coordinates)\n"
        )
        rec = CVRecommendation("spatial", None, splitter, rationale, warnings, code)
    elif groups_array is not None and np.unique(groups_array).size < len(groups_array):
        effective_splits = _effective_group_splits(groups_array, n_splits)
        if classification:
            splitter = StratifiedGroupKFold(
                n_splits=effective_splits,
                shuffle=True,
                random_state=random_state,
            )
            method = "stratified_grouped_kfold"
            rationale = (
                "Repeated group IDs show that rows are not independent; stratified group folds keep each subject together while preserving class prevalence as far as possible."
            )
            code = (
                "from trustcv import StratifiedGroupKFold\n"
                f"splitter = StratifiedGroupKFold(n_splits={effective_splits}, shuffle=True, random_state={random_state})\n"
            )
        else:
            splitter = GroupKFoldMedical(
                n_splits=effective_splits,
                shuffle=True,
                random_state=random_state,
            )
            method = "patient_grouped_kfold"
            rationale = (
                "Repeated group IDs show that rows are not independent; grouped folds keep every subject entirely in training or validation."
            )
            warnings.append(
                "Regression with groups cannot be stratified reliably by this advisor; inspect target distributions across group folds."
            )
            code = (
                "from trustcv import GroupKFoldMedical\n"
                f"splitter = GroupKFoldMedical(n_splits={effective_splits}, shuffle=True, random_state={random_state})\n"
            )
        rec = CVRecommendation("grouped", method, splitter, rationale, warnings, code)
    else:
        if classification and len(y_array) < 200:
            splitter = RepeatedKFold(
                n_splits=n_splits,
                n_repeats=5,
                random_state=random_state,
                stratify=True,
            )
            method = "repeated_kfold"
            rationale = (
                "No dependence metadata or repeated IDs were supplied, so an IID design is appropriate; repeated stratified folds reduce split sensitivity for this small classification dataset."
            )
            code = (
                "from trustcv import RepeatedKFold\n"
                f"splitter = RepeatedKFold(n_splits={n_splits}, n_repeats=5, random_state={random_state}, stratify=True)\n"
            )
        elif classification:
            splitter = StratifiedKFoldMedical(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            method = "stratified_kfold"
            rationale = (
                "No temporal, spatial, or repeated-group structure was supplied, so IID stratified folds are appropriate for preserving classification prevalence."
            )
            code = (
                "from trustcv import StratifiedKFoldMedical\n"
                f"splitter = StratifiedKFoldMedical(n_splits={n_splits}, shuffle=True, random_state={random_state})\n"
            )
        else:
            splitter = KFoldMedical(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            method = "kfold"
            rationale = (
                "No temporal, spatial, or repeated-group structure was supplied, so shuffled IID k-fold validation is appropriate for this regression target."
            )
            code = (
                "from trustcv import KFoldMedical\n"
                f"splitter = KFoldMedical(n_splits={n_splits}, shuffle=True, random_state={random_state})\n"
            )
        rec = CVRecommendation("iid", method, splitter, rationale, warnings, code)

    if classification:
        rec.warnings.extend(
            warning
            for warning in _classification_warnings(
                X, y_array, rec.splitter, groups_array
            )
            if warning not in rec.warnings
        )
    return rec
