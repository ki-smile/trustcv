"""Structured integrity checks for TrustCV validation results."""

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    """Compute the conservative overall verification status.

    ``FAILED``: a leakage-relevant check failed or errored.
    ``NEEDS_REVIEW``: a check found signs of possible leakage (``WARNING``)
    that cannot be proven automatically; a person should confirm them.
    ``NOT_FULLY_VERIFIED``: nothing suspicious, but some condition could not
    be checked. ``PASSED``: every relevant check passed or did not apply.
    """
    relevant = list(LEAKAGE_RELEVANT_KEYS)
    permutation = checks.get("permutation_sanity")
    if permutation is not None and permutation.status != "NOT_CHECKED":
        relevant.append("permutation_sanity")
    statuses = [checks[name].status for name in relevant]
    if any(status in {"FAILED", "ERROR"} for status in statuses):
        return "FAILED"
    if any(status == "WARNING" for status in statuses):
        return "NEEDS_REVIEW"
    if all(status in {"PASSED", "NOT_APPLICABLE"} for status in statuses):
        return "PASSED"
    return "NOT_FULLY_VERIFIED"


GROUP_OVERLAP_HINT = (
    "The same group/patient IDs appear in training and test folds, so scores include "
    "memorised subjects. Use TrustCV(method='stratified_group_kfold') (or 'group_kfold') "
    "with groups=."
)


def _is_multi_input(X: Any) -> bool:
    return isinstance(X, Mapping) or (isinstance(X, (list, tuple)) and not hasattr(X, "shape"))


def _row_hashes(X: Any) -> Optional[np.ndarray]:
    """Value-only row hashes (the DataFrame index is ignored); None if unsupported."""
    if _is_multi_input(X):
        return None
    try:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        return pd.util.hash_pandas_object(frame, index=False).to_numpy()
    except Exception:
        return None


class FoldAudit:
    """Overlap evidence collected from the folds that were actually scored.

    Cheap checks (group overlap and exact duplicate rows across train/test)
    run on every fold. Only up to ``max_retained`` folds are kept for the
    distance-based detector, chosen evenly when the fold count is known.
    """

    def __init__(self, X: Any, groups: Any = None, *, max_retained: int = 10, n_folds_hint: Optional[int] = None):
        self.groups = None if groups is None else (
            groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups)
        )
        hashes = _row_hashes(X)
        self.dup_hashes = None
        if hashes is not None:
            dup_mask = pd.Series(hashes).duplicated(keep=False).to_numpy()
            self.dup_mask = dup_mask
            self.hashes = hashes
            self.dup_hashes = hashes if dup_mask.any() else None
        self.max_retained = max(int(max_retained), 1)
        if n_folds_hint and n_folds_hint > self.max_retained:
            self._keep = set(np.linspace(0, n_folds_hint - 1, self.max_retained).round().astype(int).tolist())
        else:
            self._keep = None
        self.n_folds = 0
        self.group_overlap_folds: List[int] = []
        self.max_group_overlap = 0
        self.cross_split_duplicate_rows = 0
        self.duplicate_folds: List[int] = []
        self.index_overlap_rows = 0
        self.index_overlap_folds: List[int] = []
        self.retained: List[tuple] = []
        self.retained_fold_indices: List[int] = []
        self.error: Optional[str] = None

    def add(self, train_idx: Any, test_idx: Any) -> None:
        i = self.n_folds
        self.n_folds += 1
        tr = np.asarray(train_idx, dtype=int)
        te = np.asarray(test_idx, dtype=int)
        try:
            shared = np.intersect1d(tr, te)
            if shared.size:
                # The same sample is used for training and testing: a proven split error.
                self.index_overlap_folds.append(i)
                self.index_overlap_rows += int(shared.size)
            if self.groups is not None:
                overlap = np.intersect1d(self.groups[tr], self.groups[te])
                if overlap.size:
                    self.group_overlap_folds.append(i)
                    self.max_group_overlap = max(self.max_group_overlap, int(overlap.size))
            if self.dup_hashes is not None:
                te_dup = te[self.dup_mask[te]]
                if te_dup.size:
                    tr_dup = np.unique(self.hashes[tr[self.dup_mask[tr]]])
                    n_cross = int(np.isin(self.hashes[te_dup], tr_dup).sum())
                    if n_cross:
                        self.duplicate_folds.append(i)
                        self.cross_split_duplicate_rows += n_cross
        except Exception as exc:  # pragma: no cover - defensive
            self.error = str(exc)
        keep = (i in self._keep) if self._keep is not None else (len(self.retained) < self.max_retained)
        if keep:
            # copy unchanged: custom splitters may reuse index buffers, and bootstrap
            # multiplicities are part of what was scored
            self.retained.append((tr.copy(), te.copy()))
            self.retained_fold_indices.append(i)

    def coverage(self) -> Dict[str, Any]:
        return {
            "n_folds_scored": self.n_folds,
            "n_folds_distance_audited": len(self.retained),
            "distance_audited_fold_indices": list(self.retained_fold_indices),
        }


_ID_TOKENS = {"id", "ids", "uid", "uuid", "guid", "mrn", "pid", "sid"}
_ID_ONLY_TOKENS = {
    "patient", "subject", "record", "accession", "encounter", "visit",
    "number", "no", "nr", "key", "idx", "index",
}


def _name_tokens(name: Any) -> List[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return [t for t in re.split(r"[^A-Za-z0-9]+", text.lower()) if t]


def _looks_like_identifier_name(name: Any) -> bool:
    tokens = _name_tokens(name)
    if not tokens:
        return False
    return any(t in _ID_TOKENS for t in tokens) or all(t in _ID_ONLY_TOKENS for t in tokens)


def _identifier_columns(X: Any) -> List[Dict[str, Any]]:
    """DataFrame columns whose *name* marks them as identifiers.

    Name-based advice only: an identifier inside the features lets a model
    memorise subjects. Constant and boolean columns are ignored.
    """
    if not isinstance(X, pd.DataFrame):
        return []
    found = []
    for i, col in enumerate(X.columns):
        series = X.iloc[:, i]  # positional: column labels may repeat
        if pd.api.types.is_bool_dtype(series) or not _looks_like_identifier_name(col):
            continue
        n_unique = int(series.nunique(dropna=True))
        if n_unique <= 1:
            continue
        counts = series.value_counts(dropna=True)
        found.append(
            {
                "name": str(col),
                "index": i,
                "n_unique": n_unique,
                "repeated_values": int((counts > 1).sum()),
            }
        )
    return found


def _independence_scan(X: Any, *, random_state: Optional[int] = 0, max_query: int = 2000) -> Dict[str, Any]:
    """Scan for rows that come in tight clusters (repeated subjects,
    augmentation or oversampling).

    For independent points with local intrinsic dimension m, the scaled log
    ratios ``j * m * ln(d_{j+1} / d_j)`` of successive neighbour distances
    behave like independent Exp(1) draws. A row is flagged when the largest
    of them, over j = 1..J, exceeds the 99th percentile of the maximum of J
    Exp(1) draws. Only complete rows and columns with many distinct values
    are used. Tight clusters of genuinely independent samples look the
    same, so a positive result is a reason for review (``WARNING`` /
    ``NEEDS_REVIEW``), never proof (``FAILED``).
    """
    if _is_multi_input(X):
        return {"applicable": False, "reason": "multi-input data"}
    try:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        values = frame.select_dtypes("number").to_numpy(dtype=float)
    except Exception as exc:
        return {"applicable": False, "reason": f"features could not be read as numbers ({exc})"}
    if values.ndim != 2:
        return {"applicable": False, "reason": "features are not a 2-D table"}
    values = values[np.isfinite(values).all(axis=1)]
    n = int(values.shape[0])
    if n < 40:
        return {"applicable": False, "reason": "fewer than 40 complete rows"}
    min_levels = min(50, n // 2)
    keep = [len(np.unique(col)) >= min_levels for col in values.T]
    values = values[:, keep]
    if values.shape[1] < 2:
        return {"applicable": False, "reason": f"fewer than 2 columns with at least {min_levels} distinct values"}
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    J = min(30, n // 4 - 1)
    K = 2 * (J + 1)
    rng = np.random.RandomState(random_state)
    query = rng.choice(n, min(n, max_query), replace=False)
    from sklearn.neighbors import NearestNeighbors

    dist = NearestNeighbors(n_neighbors=min(K + 1, n)).fit(values).kneighbors(values[query])[0][:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        dim = float(np.log(2) / np.log(np.median(dist[:, K - 1] / dist[:, J])))
        if not np.isfinite(dim) or dim <= 0:
            return {"applicable": False, "reason": "local dimension could not be estimated"}
        ratios = dist[:, 1 : J + 1] / dist[:, :J]
        scores = np.where(dist[:, :J] > 0, np.arange(1, J + 1) * dim * np.log(ratios), np.nan)
    threshold = float(-np.log(1 - 0.99 ** (1 / J)))
    flagged = np.nan_to_num(np.nanmax(np.where(np.isnan(scores), -np.inf, scores), axis=1), nan=-np.inf) > threshold
    fraction = float(flagged.mean())
    return {
        "applicable": True,
        "flagged_fraction": fraction,
        "clustered": fraction >= 0.10,
        "n_rows_used": n,
        "n_query_rows": int(query.size),
        "n_columns_used": int(values.shape[1]),
        "window": J,
        "local_dimension": dim,
        "random_state": random_state,
    }


def _duplicate_samples_check(X: Any, enabled: bool, audit: Optional[FoldAudit] = None) -> CheckResult:
    if not enabled:
        return CheckResult(
            "duplicate_samples", "NOT_CHECKED", "Duplicate-sample checking is disabled."
        )
    if audit is not None and audit.index_overlap_folds:
        return CheckResult(
            "duplicate_samples",
            "FAILED",
            f"{audit.index_overlap_rows} sample(s) were used for both training and testing in "
            f"{len(audit.index_overlap_folds)} of {audit.n_folds} scored folds: the splitter puts "
            "the same row on both sides.",
            {
                "index_overlap_rows": audit.index_overlap_rows,
                "folds_with_index_overlap": list(audit.index_overlap_folds),
            },
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
        details = {"duplicate_count": duplicate_count}
        message = f"Found {duplicate_count} exact duplicate rows in the supplied features."
        if audit is not None and audit.dup_hashes is not None:
            details["cross_split_test_rows"] = audit.cross_split_duplicate_rows
            details["folds_with_cross_split_duplicates"] = len(audit.duplicate_folds)
            if audit.cross_split_duplicate_rows:
                message += (
                    f" {audit.cross_split_duplicate_rows} test row(s) across "
                    f"{len(audit.duplicate_folds)} of {audit.n_folds} scored folds have an "
                    "identical row in training."
                )
            else:
                message += " No copy ever fell on both sides of a scored split."
        return CheckResult("duplicate_samples", "WARNING", message, details)
    return CheckResult(
        "duplicate_samples",
        "PASSED",
        "No exact duplicate feature rows were found.",
        {"duplicate_count": 0},
    )


def _dependence_evidence(scan: Optional[Dict[str, Any]], id_columns: Optional[List[Dict[str, Any]]]) -> str:
    """Describe signs that rows may not be independent ('' when there are none)."""
    notes = ""
    if scan and scan.get("applicable") and scan.get("clustered"):
        notes += (
            f" {scan['flagged_fraction']:.0%} of rows have unusually close neighbours, "
            "which is typical of repeated subjects, augmentation or oversampling."
        )
    repeated = [c["name"] for c in (id_columns or []) if c.get("repeated_values")]
    if repeated:
        notes += f" Column(s) {', '.join(repeated[:3])} look like identifiers with repeated values."
    return notes


def _group_leakage_check(
    X: Any,
    y: Any,
    groups: Any,
    splitter: Any,
    enabled: bool,
    declare_independent_samples: bool,
    audit: Optional[FoldAudit] = None,
    independence_scan: Optional[Dict[str, Any]] = None,
    identifier_columns: Optional[List[Dict[str, Any]]] = None,
) -> CheckResult:
    if not enabled:
        return CheckResult("group_leakage", "NOT_CHECKED", "Group leakage checking is disabled.")
    details: Dict[str, Any] = {}
    if independence_scan is not None:
        details["independence_scan"] = independence_scan
    if identifier_columns:
        details["identifier_columns"] = identifier_columns
    if groups is None:
        evidence = _dependence_evidence(independence_scan, identifier_columns)
        if evidence:
            # Signs of dependence cannot be proven automatically: needs a person to confirm.
            claim = (
                "Samples were declared independent, but rows may not be:"
                if declare_independent_samples
                else "No groups were supplied, and rows may not be independent:"
            )
            return CheckResult(
                "group_leakage",
                "WARNING",
                claim + evidence + " If rows share a subject, pass groups=.",
                details,
            )
        notes = ""
        if declare_independent_samples:
            return CheckResult(
                "group_leakage",
                "NOT_APPLICABLE",
                "Samples were explicitly declared independent; no groups were supplied." + notes,
                details,
            )
        return CheckResult(
            "group_leakage",
            "NOT_CHECKED",
            "No groups or patient_ids were supplied, so group leakage cannot be checked." + notes,
            details,
        )
    if audit is None:
        # Backward-compatible path: audit the splitter's folds directly.
        audit = FoldAudit(None, groups)
        try:
            groups_arr = audit.groups
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The groups parameter is ignored by KFold",
                    category=UserWarning,
                )
                for train_idx, test_idx in splitter.split(X, y, groups_arr):
                    audit.add(train_idx, test_idx)
        except Exception as exc:
            return CheckResult(
                "group_leakage",
                "ERROR",
                f"Group leakage checking failed while iterating the splitter: {exc}",
                {"error": str(exc)},
            )
    if audit.error:
        return CheckResult(
            "group_leakage", "ERROR", f"Group leakage checking failed: {audit.error}", {"error": audit.error}
        )
    details.update(
        {
            "n_folds_scored": audit.n_folds,
            "folds_with_overlap": len(audit.group_overlap_folds),
            "overlap_count": audit.max_group_overlap,
        }
    )
    if audit.group_overlap_folds:
        return CheckResult(
            "group_leakage",
            "FAILED",
            f"Group or patient IDs overlap between training and test in "
            f"{len(audit.group_overlap_folds)} of {audit.n_folds} scored folds "
            f"(up to {audit.max_group_overlap} shared groups in one fold).",
            details,
        )
    return CheckResult(
        "group_leakage",
        "PASSED",
        f"No group or patient ID appears in both training and test in any of the {audit.n_folds} scored folds.",
        details,
    )


def _preprocessing_check(model: Any) -> CheckResult:
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline) and len(model.steps) > 1:
            return CheckResult(
                "preprocessing_leakage",
                "PASSED",
                "The model is an sklearn Pipeline, so its steps are refit within each fold. "
                "Preprocessing applied to X before validate() (feature selection, imputation, "
                "scaling, encoding, resampling) is not visible to this check.",
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
    audit: Optional[FoldAudit] = None,
    independence_scan: Optional[Dict[str, Any]] = None,
    identifier_columns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, CheckResult]:
    """Build checks available without an external leakage report.

    When ``audit`` is given, overlap evidence comes from the folds that were
    actually scored instead of re-running the splitter.
    """
    checks = default_checks()
    checks["duplicate_samples"] = _duplicate_samples_check(X, check_leakage, audit)
    checks["group_leakage"] = _group_leakage_check(
        X,
        y,
        groups,
        splitter,
        check_leakage,
        declare_independent_samples,
        audit=audit,
        independence_scan=independence_scan,
        identifier_columns=identifier_columns,
    )
    checks["preprocessing_leakage"] = _preprocessing_check(model)
    checks["class_balance"] = _class_balance_check(y, check_balance, is_regression)
    return checks


def covariate_shift_from_report(report: Any, groups_separated: bool = False) -> CheckResult:
    """Translate an optional leakage-report shift section into a check result.

    ``groups_separated`` means group separation was verified on the scored
    folds; shift is then reported as INFO because held-out groups are expected
    to differ, but the evidence is kept.
    """
    details = getattr(report, "details", {}) or {}
    shift = details.get("covariate_shift")
    if shift is None:
        return CheckResult(
            "covariate_shift", "NOT_CHECKED", "Covariate-shift reporting is unavailable."
        )
    n_shifted = int(shift.get("n_shifted_features", 0))
    status = "WARNING" if n_shifted and not groups_separated else "INFO"
    if not n_shifted:
        message = "No statistically significant covariate shift was detected."
    elif groups_separated:
        message = (
            f"Covariate shift was detected in {n_shifted} feature(s); this may reflect "
            "differences between held-out groups, which were verified to be separated."
        )
    else:
        message = f"Covariate shift was detected in {n_shifted} feature(s)."
    return CheckResult("covariate_shift", status, message, dict(shift))
