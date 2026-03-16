"""
Data Leakage Detection for Medical Machine Learning

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team

Common leakage sources in medical data:
1. Patient data in both train and test sets
2. Temporal leakage (future information in training)
3. Data preprocessing on full dataset
4. Feature engineering using test information
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Severity ranking helpers
# ---------------------------------------------------------------------------
_SEVERITY_RANK: Dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def _worst_severity(a: str, b: str) -> str:
    """Return the more severe of two severity strings.

    Uses an explicit rank dictionary so that comparisons are correct
    (alphabetical ``max()`` would wrongly rank *"high"* above *"critical"*).
    """
    if _SEVERITY_RANK.get(a, 0) >= _SEVERITY_RANK.get(b, 0):
        return a
    return b


@dataclass
class LeakageReport:
    """Detailed leakage detection report"""

    has_leakage: bool
    leakage_types: List[str]
    severity: str  # 'none', 'low', 'medium', 'high', 'critical'
    details: Dict[str, Any]
    recommendations: List[str]

    @property
    def summary(self) -> str:
        """Human-readable one-line/paragraph summary."""
        return str(self)

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary form for programmatic use/serialization."""
        return {
            "has_leakage": self.has_leakage,
            "leakage_types": list(self.leakage_types),
            "severity": self.severity,
            "details": self.details,
            "recommendations": list(self.recommendations),
        }

    def __str__(self) -> str:
        if not self.has_leakage:
            return "✅ No data leakage detected"

        report = f"⚠️ Data Leakage Detected (Severity: {self.severity})\n"
        report += f"Types: {', '.join(self.leakage_types)}\n"
        report += "Recommendations:\n"
        for rec in self.recommendations:
            report += f"  • {rec}\n"
        return report


class DataLeakageChecker:
    """
    Comprehensive data leakage detection for ML

    Checks for:
    - Patient-level leakage
    - Temporal leakage
    - Feature leakage
    - Preprocessing leakage
    - Duplicate samples
    - Near-duplicate samples (cosine similarity)
    - Hierarchical group leakage

    Examples
    --------
    >>> checker = DataLeakageChecker()
    >>> report = checker.check_cv_splits(X_train, X_test, y_train, y_test, patient_ids)
    >>> if report.has_leakage:
    >>>     print(report)
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Auto-compute spatial threshold helper
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_spatial_threshold(
        coordinates: Union[np.ndarray, "pd.DataFrame"],
        max_sample: int = 500,
        percentile: float = 10.0,
        random_state: int = 42,
    ) -> float:
        """Compute a spatial threshold as the given percentile of pairwise
        distances from a random sample of the coordinates."""
        from scipy.spatial.distance import pdist

        coords = np.asarray(coordinates)
        n = coords.shape[0]
        if n > max_sample:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n, max_sample, replace=False)
            coords = coords[idx]
        dists = pdist(coords, metric="euclidean")
        if dists.size == 0:
            return 0.0
        return float(np.percentile(dists, percentile))

    # ------------------------------------------------------------------
    # check() — convenience CV-based wrapper
    # ------------------------------------------------------------------
    def check(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps: Optional[Union[np.ndarray, pd.Series]] = None,
        coordinates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        n_splits: int = 5,
        random_state: Optional[int] = 42,
    ) -> "LeakageReport":
        """
        Convenience wrapper to check leakage via CV-style splits.

        If groups are provided, uses GroupKFold; otherwise uses StratifiedKFold
        when labels are available, else KFold. For time series tasks, prefer
        calling `check_cv_splits` with your explicit train/test partitions.

        When *coordinates* are provided but no explicit spatial threshold is
        given, the threshold is auto-computed as the 10th percentile of
        pairwise distances from a random sample.

        Returns a LeakageReport aggregated over folds.
        """
        from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

        # Choose splitter
        if groups is not None:
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X, y, groups)
        elif y is not None:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_iter = splitter.split(X, y)
        else:
            splitter = KFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_iter = splitter.split(X)

        # Auto-compute spatial threshold when coordinates are provided
        spatial_threshold: Optional[float] = None
        if coordinates is not None:
            try:
                spatial_threshold = self._auto_spatial_threshold(
                    coordinates, random_state=random_state or 42
                )
            except Exception:  # scipy may be missing
                spatial_threshold = None

        # Accumulate per-fold reports
        fold_reports: List[LeakageReport] = []
        all_types: set = set()
        recs: set = set()
        worst = "none"

        X_df = X
        y_arr = y
        gps = groups
        ts = timestamps
        coords = coordinates

        for tr, te in split_iter:
            X_tr = (
                X_df[tr] if isinstance(X_df, np.ndarray) else X_df.iloc[tr]
            )
            X_te = (
                X_df[te] if isinstance(X_df, np.ndarray) else X_df.iloc[te]
            )
            y_tr = (
                None
                if y_arr is None
                else (
                    y_arr[tr]
                    if isinstance(y_arr, np.ndarray)
                    else y_arr.iloc[tr]
                )
            )
            y_te = (
                None
                if y_arr is None
                else (
                    y_arr[te]
                    if isinstance(y_arr, np.ndarray)
                    else y_arr.iloc[te]
                )
            )
            gp_tr = (
                None
                if gps is None
                else (
                    gps[tr]
                    if isinstance(gps, np.ndarray)
                    else gps.iloc[tr]
                )
            )
            gp_te = (
                None
                if gps is None
                else (
                    gps[te]
                    if isinstance(gps, np.ndarray)
                    else gps.iloc[te]
                )
            )
            ts_tr = (
                None
                if ts is None
                else (
                    ts[tr] if isinstance(ts, np.ndarray) else ts.iloc[tr]
                )
            )
            ts_te = (
                None
                if ts is None
                else (
                    ts[te] if isinstance(ts, np.ndarray) else ts.iloc[te]
                )
            )
            cr_tr: Optional[np.ndarray] = None
            cr_te: Optional[np.ndarray] = None
            if coords is not None:
                if isinstance(coords, np.ndarray):
                    cr_tr, cr_te = coords[tr], coords[te]
                else:
                    cr_tr, cr_te = coords.iloc[tr], coords.iloc[te]

            rpt = self.check_cv_splits(
                X_tr,
                X_te,
                y_tr,
                y_te,
                gp_tr,
                gp_te,
                ts_tr,
                ts_te,
                cr_tr,
                cr_te,
                spatial_threshold=spatial_threshold,
            )
            fold_reports.append(rpt)
            all_types.update(rpt.leakage_types)
            recs.update(rpt.recommendations)
            if (
                _SEVERITY_RANK.get(rpt.severity, 0)
                > _SEVERITY_RANK.get(worst, 0)
            ):
                worst = rpt.severity

        return LeakageReport(
            has_leakage=any(fr.has_leakage for fr in fold_reports),
            leakage_types=sorted(all_types),
            severity=worst,
            details={
                "folds": [
                    {
                        "fold_index": i,
                        "leakage_types": fr.leakage_types,
                        "severity": fr.severity,
                        "details": fr.details,
                    }
                    for i, fr in enumerate(fold_reports)
                ]
            },
            recommendations=sorted(recs),
        )

    # ------------------------------------------------------------------
    # check_cv_splits() — main per-split checker
    # ------------------------------------------------------------------
    def check_cv_splits(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        patient_ids_train: Optional[Union[np.ndarray, pd.Series]] = None,
        patient_ids_test: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps_train: Optional[Union[np.ndarray, pd.Series]] = None,
        timestamps_test: Optional[Union[np.ndarray, pd.Series]] = None,
        coordinates_train: Optional[
            Union[np.ndarray, pd.DataFrame]
        ] = None,
        coordinates_test: Optional[
            Union[np.ndarray, pd.DataFrame]
        ] = None,
        spatial_threshold: Optional[float] = None,
    ) -> LeakageReport:
        """
        Check for data leakage between train and test sets

        Parameters
        ----------
        X_train, X_test : array-like
            Training and test features
        y_train, y_test : array-like, optional
            Training and test labels
        patient_ids_train, patient_ids_test : array-like, optional
            Patient identifiers
        timestamps_train, timestamps_test : array-like, optional
            Temporal information
        coordinates_train, coordinates_test : array-like, optional
            Spatial coordinates
        spatial_threshold : float, optional
            Distance threshold for spatial proximity check

        Returns
        -------
        LeakageReport
            Detailed leakage analysis
        """
        leakage_types: List[str] = []
        details: Dict[str, Any] = {}
        recommendations: List[str] = []
        severity = "none"

        # Check 1: Patient-level leakage
        if (
            patient_ids_train is not None
            and patient_ids_test is not None
        ):
            patient_leakage = self._check_patient_leakage(
                patient_ids_train, patient_ids_test
            )
            if patient_leakage["has_leakage"]:
                leakage_types.append("patient")
                details["patient_leakage"] = patient_leakage
                recommendations.append(
                    "Use PatientGroupKFold to ensure "
                    "patient-level separation"
                )
                severity = "critical"

        # Check 2: Duplicate samples
        duplicate_leakage = self._check_duplicate_samples(
            X_train, X_test
        )
        if duplicate_leakage["has_leakage"]:
            leakage_types.append("duplicate")
            details["duplicate_leakage"] = duplicate_leakage
            recommendations.append(
                "Remove duplicate samples before splitting"
            )
            severity = _worst_severity(severity, "high")

        # Check 3: Temporal leakage
        if (
            timestamps_train is not None
            and timestamps_test is not None
        ):
            temporal_leakage = self._check_temporal_leakage(
                timestamps_train, timestamps_test
            )
            if temporal_leakage["has_leakage"]:
                leakage_types.append("temporal")
                details["temporal_leakage"] = temporal_leakage
                recommendations.append(
                    "Use TemporalClinical splitter for time-series data"
                )
                severity = _worst_severity(severity, "high")

        # Check 4: Feature statistics leakage (with KS-test)
        feature_leakage = self._check_feature_statistics(
            X_train, X_test
        )
        if feature_leakage["suspicious"]:
            leakage_types.append("feature_statistics")
            details["feature_leakage"] = feature_leakage
            recommendations.append(
                "Check if preprocessing was done before "
                "train-test split"
            )
            severity = _worst_severity(severity, "medium")

        # Check 5: Spatial proximity
        if (
            coordinates_train is not None
            and coordinates_test is not None
        ):
            # Auto-compute threshold if not provided
            effective_threshold = spatial_threshold
            if effective_threshold is None:
                try:
                    combined = np.vstack(
                        [
                            np.asarray(coordinates_train),
                            np.asarray(coordinates_test),
                        ]
                    )
                    effective_threshold = self._auto_spatial_threshold(
                        combined
                    )
                except Exception:
                    effective_threshold = None

            if effective_threshold is not None:
                spatial_result = self.spatial_check(
                    coordinates_train,
                    coordinates_test,
                    effective_threshold,
                )
                details["spatial_proximity"] = spatial_result
                if spatial_result.get("near_fraction", 0.0) > 0.0:
                    leakage_types.append("spatial_proximity")
                    recommendations.append(
                        f"Spatial proximity detected: "
                        f"{spatial_result.get('near_fraction', 0.0):.1%}"
                        f" of test near train "
                        f"(<{effective_threshold})."
                    )
                    sev = "low"
                    frac = spatial_result.get("near_fraction", 0.0)
                    if frac >= 0.2:
                        sev = "medium"
                    severity = _worst_severity(severity, sev)

        # Check 6: Near-duplicate samples (cosine similarity)
        near_dup = self.check_near_duplicates(X_train, X_test)
        if near_dup["has_leakage"]:
            leakage_types.append("near_duplicate")
            details["near_duplicate_leakage"] = near_dup
            recommendations.append(
                "Investigate near-duplicate samples across "
                "train/test (cosine similarity "
                f">= {near_dup['similarity_threshold']})"
            )
            severity = _worst_severity(severity, "high")

        # Check 7: Label distribution
        if y_train is not None and y_test is not None:
            label_check = self._check_label_distribution(
                y_train, y_test
            )
            if label_check["suspicious"]:
                details["label_distribution"] = label_check
                recommendations.append(
                    "Consider using stratified splitting "
                    "for balanced class distribution"
                )
                severity = _worst_severity(severity, "low")

        return LeakageReport(
            has_leakage=len(leakage_types) > 0,
            leakage_types=leakage_types,
            severity=severity,
            details=details,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Individual check methods
    # ------------------------------------------------------------------

    def _check_patient_leakage(
        self,
        patient_ids_train: Union[np.ndarray, pd.Series],
        patient_ids_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Check if same patients appear in train and test"""
        train_patients = set(patient_ids_train)
        test_patients = set(patient_ids_test)

        overlap = train_patients.intersection(test_patients)

        result: Dict[str, Any] = {
            "has_leakage": len(overlap) > 0,
            "overlapping_patients": list(overlap)[:10],  # Show first 10
            "overlap_count": len(overlap),
            "train_unique": len(train_patients),
            "test_unique": len(test_patients),
            "overlap_percentage": (
                len(overlap)
                / len(train_patients.union(test_patients))
                * 100
            ),
        }

        if result["has_leakage"] and self.verbose:
            warnings.warn(
                f"CRITICAL: {result['overlap_count']} patients "
                f"({result['overlap_percentage']:.1f}%) appear in "
                f"both train and test sets!"
            )

        return result

    def spatial_check(
        self,
        coordinates_train: Union[np.ndarray, pd.DataFrame],
        coordinates_test: Union[np.ndarray, pd.DataFrame],
        threshold: float,
    ) -> Dict[str, Any]:
        """
        Simple spatial proximity flag between train and test sets.

        Computes the minimum distance from each test point to any train
        point and reports the fraction within the given threshold, along
        with mean/min of those minimum distances.

        Parameters
        ----------
        coordinates_train : array-like (n_train, 2)
        coordinates_test : array-like (n_test, 2)
        threshold : float
            Distance threshold under which a test point is considered
            "near" a train point.

        Returns
        -------
        Dict
            {
              'near_fraction': float,
              'mean_min_distance': float,
              'min_min_distance': float,
              'threshold': float
            }
        """
        try:
            from scipy.spatial.distance import cdist as _cdist

            tr = np.asarray(coordinates_train)
            te = np.asarray(coordinates_test)
            if (
                tr.ndim != 2
                or te.ndim != 2
                or tr.shape[1] != 2
                or te.shape[1] != 2
            ):
                return {
                    "error": "coordinates must be (n,2)",
                    "near_fraction": 0.0,
                    "mean_min_distance": np.nan,
                    "min_min_distance": np.nan,
                    "threshold": threshold,
                }
            if tr.size == 0 or te.size == 0:
                return {
                    "near_fraction": 0.0,
                    "mean_min_distance": np.nan,
                    "min_min_distance": np.nan,
                    "threshold": threshold,
                }
            dists = _cdist(te, tr, metric="euclidean")
            min_d = (
                dists.min(axis=1) if dists.size else np.array([])
            )
            near_frac = (
                float((min_d < threshold).mean())
                if min_d.size
                else 0.0
            )
            return {
                "near_fraction": near_frac,
                "mean_min_distance": (
                    float(np.nanmean(min_d))
                    if min_d.size
                    else float("nan")
                ),
                "min_min_distance": (
                    float(np.nanmin(min_d))
                    if min_d.size
                    else float("nan")
                ),
                "threshold": float(threshold),
            }
        except Exception as e:
            return {
                "error": str(e),
                "near_fraction": 0.0,
                "mean_min_distance": None,
                "min_min_distance": None,
                "threshold": threshold,
            }

    def _check_duplicate_samples(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Check for duplicate samples between train and test"""
        # Convert to DataFrame for easier handling
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)

        # Create hash of each row
        train_hashes = pd.util.hash_pandas_object(X_train)
        test_hashes = pd.util.hash_pandas_object(X_test)

        # Find overlaps
        common_hashes = set(train_hashes).intersection(set(test_hashes))

        result: Dict[str, Any] = {
            "has_leakage": len(common_hashes) > 0,
            "duplicate_count": len(common_hashes),
            "duplicate_percentage": (
                len(common_hashes) / len(test_hashes) * 100
            ),
        }

        if result["has_leakage"] and self.verbose:
            warnings.warn(
                f"Found {result['duplicate_count']} duplicate samples "
                f"({result['duplicate_percentage']:.1f}%) between "
                f"train and test!"
            )

        return result

    def _check_temporal_leakage(
        self,
        timestamps_train: Union[np.ndarray, pd.Series],
        timestamps_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Check if test data comes before training data (future leakage)"""
        # Convert to datetime if needed
        if isinstance(timestamps_train, (np.ndarray, list)):
            timestamps_train = pd.Series(timestamps_train)
        if isinstance(timestamps_test, (np.ndarray, list)):
            timestamps_test = pd.Series(timestamps_test)

        try:
            timestamps_train = pd.to_datetime(timestamps_train)
            timestamps_test = pd.to_datetime(timestamps_test)
        except (ValueError, TypeError, OverflowError):
            return {
                "has_leakage": False,
                "error": "Could not parse timestamps",
            }

        min_train = timestamps_train.min()
        max_train = timestamps_train.max()
        min_test = timestamps_test.min()
        max_test = timestamps_test.max()

        # Compute overlap fraction: fraction of test samples whose
        # timestamps fall within [min_train, max_train].
        test_in_train_range = (
            (timestamps_test >= min_train) & (timestamps_test <= max_train)
        )
        overlap_fraction = float(test_in_train_range.mean())

        # Fraction of test samples that appear before min_train
        test_before_train_mask = timestamps_test < min_train
        test_before_train_fraction = float(
            test_before_train_mask.mean()
        )

        test_before_train = bool(min_test < min_train)

        # Flag as leakage when test data appears before training starts
        # OR when a majority of test samples fall within the training
        # period (overlap_fraction > 0.5).
        has_leakage = test_before_train or overlap_fraction > 0.5

        result: Dict[str, Any] = {
            "has_leakage": has_leakage,
            "train_period": f"{min_train} to {max_train}",
            "test_period": f"{min_test} to {max_test}",
            "overlap_exists": bool(min_test < max_train),
            "test_before_train": test_before_train,
            "overlap_fraction": overlap_fraction,
            "test_before_train_fraction": test_before_train_fraction,
        }

        if test_before_train and self.verbose:
            warnings.warn(
                "CRITICAL: Test data contains dates before training "
                "data! This causes temporal leakage."
            )

        return result

    def _check_feature_statistics(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Check if feature statistics are suspiciously similar.

        In addition to the existing near-zero diff check, runs a
        Kolmogorov-Smirnov test on each feature (when scipy is
        available).  A feature is KS-suspicious if its p-value < 0.001
        (very different distributions).
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Calculate statistics
        train_mean = np.mean(X_train, axis=0)
        test_mean = np.mean(X_test, axis=0)
        train_std = np.std(X_train, axis=0)
        test_std = np.std(X_test, axis=0)

        # Near-zero diff check (original logic)
        mean_diff = np.abs(train_mean - test_mean)
        std_diff = np.abs(train_std - test_std)
        suspicious_features = np.where(
            (mean_diff < 1e-10) & (std_diff < 1e-10) & (train_std > 0)
        )[0]

        # KS-test on each feature (optional scipy dependency)
        ks_suspicious_count = 0
        ks_pvalues: Optional[List[float]] = None
        try:
            from scipy.stats import ks_2samp

            n_features = X_train.shape[1]
            ks_pvalues = []
            for i in range(n_features):
                _, p = ks_2samp(X_train[:, i], X_test[:, i])
                ks_pvalues.append(p)
                if p < 0.001:
                    ks_suspicious_count += 1
        except ImportError:
            pass

        result: Dict[str, Any] = {
            "suspicious": (
                len(suspicious_features) > X_train.shape[1] * 0.5
            ),
            "suspicious_features": int(len(suspicious_features)),
            "total_features": int(X_train.shape[1]),
            "likely_normalized_together": len(suspicious_features) > 0,
            "ks_suspicious_features": ks_suspicious_count,
        }
        if ks_pvalues is not None:
            result["ks_pvalues"] = ks_pvalues

        if result["suspicious"] and self.verbose:
            warnings.warn(
                f"{result['suspicious_features']} features have "
                "identical statistics. Data might have been "
                "preprocessed before splitting!"
            )

        return result

    def _check_label_distribution(
        self,
        y_train: Union[np.ndarray, pd.Series],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Any]:
        """Check if label distributions are significantly different.

        Also runs a chi-squared test when scipy is available.
        """
        unique_train, counts_train = np.unique(
            y_train, return_counts=True
        )
        unique_test, counts_test = np.unique(
            y_test, return_counts=True
        )

        train_dist = counts_train / len(y_train)
        test_dist = counts_test / len(y_test)

        # Calculate distribution difference
        max_diff: float = 0.0
        for i, label in enumerate(unique_train):
            if label in unique_test:
                idx_test = np.where(unique_test == label)[0][0]
                diff = abs(float(train_dist[i] - test_dist[idx_test]))
                max_diff = max(max_diff, diff)

        # Chi-squared test (optional scipy)
        chi2_pvalue: Optional[float] = None
        try:
            from scipy.stats import chi2_contingency

            # Build contingency table aligning labels
            all_labels = sorted(
                set(unique_train.tolist()) | set(unique_test.tolist())
            )
            train_counts_aligned = []
            test_counts_aligned = []
            for lbl in all_labels:
                tr_idx = np.where(unique_train == lbl)[0]
                te_idx = np.where(unique_test == lbl)[0]
                train_counts_aligned.append(
                    int(counts_train[tr_idx[0]]) if len(tr_idx) else 0
                )
                test_counts_aligned.append(
                    int(counts_test[te_idx[0]]) if len(te_idx) else 0
                )
            table = np.array(
                [train_counts_aligned, test_counts_aligned]
            )
            # Ensure no zero-sum columns
            nonzero_cols = table.sum(axis=0) > 0
            table = table[:, nonzero_cols]
            if table.shape[1] >= 2:
                _, pval, _, _ = chi2_contingency(table)
                chi2_pvalue = float(pval)
        except ImportError:
            pass

        result: Dict[str, Any] = {
            "suspicious": max_diff > 0.2,
            "max_difference": max_diff,
            "train_distribution": dict(
                zip(unique_train.tolist(), train_dist.tolist())
            ),
            "test_distribution": dict(
                zip(unique_test.tolist(), test_dist.tolist())
            ),
        }
        if chi2_pvalue is not None:
            result["chi2_pvalue"] = chi2_pvalue

        if result["suspicious"] and self.verbose:
            warnings.warn(
                f"Large class distribution difference "
                f"({max_diff:.1%}). "
                "Consider using stratified splitting."
            )

        return result

    # ------------------------------------------------------------------
    # Near-duplicate detection (P1 #5)
    # ------------------------------------------------------------------
    def check_near_duplicates(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        similarity_threshold: float = 0.99,
    ) -> Dict[str, Any]:
        """Detect near-duplicate samples across train/test via cosine
        similarity.

        Features are standardized with training-set statistics before
        cosine similarity is computed. This avoids false positives on
        mixed-scale tabular data where raw feature magnitudes dominate
        vector direction.

        Parameters
        ----------
        X_train, X_test : array-like
            Feature matrices.
        similarity_threshold : float
            Cosine similarity threshold above which a pair is
            considered a near-duplicate. Default 0.99.

        Returns
        -------
        dict
            ``has_leakage``, ``near_duplicate_count``,
            ``near_duplicate_percentage``, ``similarity_threshold``.
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return {
                "has_leakage": False,
                "near_duplicate_count": 0,
                "near_duplicate_percentage": 0.0,
                "similarity_threshold": similarity_threshold,
                "error": "sklearn not available for cosine similarity",
            }

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        try:
            X_train = np.asarray(X_train, dtype=float)
            X_test = np.asarray(X_test, dtype=float)

            train_mean = X_train.mean(axis=0)
            train_std = X_train.std(axis=0)
            train_std[train_std == 0] = 1.0

            X_train_normalized = (X_train - train_mean) / train_std
            X_test_normalized = (X_test - train_mean) / train_std

            sim = cosine_similarity(X_test_normalized, X_train_normalized)
            near_dup_mask = sim.max(axis=1) >= similarity_threshold
            count = int(near_dup_mask.sum())
            pct = float(count / len(X_test) * 100) if len(X_test) else 0.0
        except Exception:
            return {
                "has_leakage": False,
                "near_duplicate_count": 0,
                "near_duplicate_percentage": 0.0,
                "similarity_threshold": similarity_threshold,
                "error": "Could not compute cosine similarity",
            }

        result: Dict[str, Any] = {
            "has_leakage": count > 0,
            "near_duplicate_count": count,
            "near_duplicate_percentage": pct,
            "similarity_threshold": similarity_threshold,
        }

        if result["has_leakage"] and self.verbose:
            warnings.warn(
                f"Found {count} near-duplicate test samples "
                f"({pct:.1f}%) with cosine similarity "
                f">= {similarity_threshold}."
            )

        return result

    # ------------------------------------------------------------------
    # Hierarchical leakage detection (P1 #6)
    # ------------------------------------------------------------------
    def check_hierarchical_leakage(
        self,
        groups_train: Union[np.ndarray, pd.Series, List],
        groups_test: Union[np.ndarray, pd.Series, List],
        parent_groups_train: Union[np.ndarray, pd.Series, List],
        parent_groups_test: Union[np.ndarray, pd.Series, List],
    ) -> Dict[str, Any]:
        """Check for hierarchical group leakage (e.g. hospital ->
        patient).

        If any *parent* group appears in both training and test sets
        this constitutes hierarchical leakage, even when child groups
        are properly separated.

        Parameters
        ----------
        groups_train, groups_test : array-like
            Child-level group identifiers.
        parent_groups_train, parent_groups_test : array-like
            Parent-level group identifiers (same length as child
            groups).

        Returns
        -------
        dict
            ``has_leakage``, ``overlapping_parents``,
            ``overlap_count``.
        """
        parent_train_set = set(np.asarray(parent_groups_train))
        parent_test_set = set(np.asarray(parent_groups_test))
        overlapping = parent_train_set.intersection(parent_test_set)

        result: Dict[str, Any] = {
            "has_leakage": len(overlapping) > 0,
            "overlapping_parents": sorted(
                str(p) for p in list(overlapping)[:20]
            ),
            "overlap_count": len(overlapping),
        }

        if result["has_leakage"] and self.verbose:
            warnings.warn(
                f"Hierarchical leakage: {len(overlapping)} parent "
                f"group(s) appear in both train and test."
            )

        return result

    # ------------------------------------------------------------------
    # Preprocessing leakage
    # ------------------------------------------------------------------
    def check_preprocessing_leakage(
        self,
        X_original: Union[np.ndarray, pd.DataFrame],
        X_processed: Union[np.ndarray, pd.DataFrame],
        split_indices: Tuple[np.ndarray, np.ndarray],
    ) -> bool:
        """
        Check if preprocessing was done before or after splitting

        Parameters
        ----------
        X_original : array-like
            Original data before preprocessing
        X_processed : array-like
            Data after preprocessing
        split_indices : tuple
            (train_indices, test_indices)

        Returns
        -------
        bool
            True if leakage detected
        """
        train_idx, test_idx = split_indices

        # Check if normalization used global statistics
        if isinstance(X_processed, pd.DataFrame):
            X_processed = X_processed.values

        global_mean = np.mean(X_processed, axis=0)
        train_mean = np.mean(X_processed[train_idx], axis=0)
        test_mean = np.mean(X_processed[test_idx], axis=0)

        # If train mean equals global mean, likely preprocessed together
        mean_similarity = bool(
            np.allclose(train_mean, global_mean, rtol=1e-10)
        )

        if mean_similarity and self.verbose:
            warnings.warn(
                "Preprocessing appears to use global statistics. "
                "This causes data leakage! Fit preprocessing "
                "only on training data."
            )

        return mean_similarity

    # ------------------------------------------------------------------
    # Feature-target leakage
    # ------------------------------------------------------------------
    def check_feature_target_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Check if any features are too correlated with target
        (potential leakage)

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        threshold : float
            Correlation threshold above which to flag as suspicious

        Returns
        -------
        dict
            Leakage detection results
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = [
                f"feature_{i}" for i in range(X.shape[1])
            ]

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        # Calculate correlations
        correlations: List[float] = []
        suspicious_features: List[Dict[str, Any]] = []

        for i in range(X_array.shape[1]):
            try:
                from scipy.stats import pearsonr

                corr, _ = pearsonr(X_array[:, i], y_array)
                correlations.append(abs(corr))

                if abs(corr) > threshold:
                    suspicious_features.append(
                        {
                            "index": i,
                            "name": feature_names[i],
                            "correlation": corr,
                        }
                    )
            except (ValueError, TypeError):
                # For categorical, use mutual information
                from sklearn.feature_selection import (
                    mutual_info_classif,
                )

                mi = mutual_info_classif(
                    X_array[:, i : i + 1],
                    y_array,
                    random_state=42,
                )[0]
                normalized_mi = min(mi, 1.0)
                correlations.append(normalized_mi)

                if normalized_mi > threshold:
                    suspicious_features.append(
                        {
                            "index": i,
                            "name": feature_names[i],
                            "mutual_info": normalized_mi,
                        }
                    )

        result: Dict[str, Any] = {
            "has_leakage": len(suspicious_features) > 0,
            "suspicious_features": suspicious_features,
            "max_correlation": (
                max(correlations) if correlations else 0
            ),
            "num_suspicious": len(suspicious_features),
        }

        if result["has_leakage"] and self.verbose:
            warnings.warn(
                f"Found {len(suspicious_features)} features with "
                f"suspiciously high correlation to target "
                f"(>{threshold}). Possible target leakage!"
            )
            for feat in suspicious_features[:3]:
                print(
                    f"  - {feat['name']}: "
                    f"{feat.get('correlation', feat.get('mutual_info', 0)):.3f}"
                )

        return result

    # ------------------------------------------------------------------
    # comprehensive_check() — truly comprehensive
    # ------------------------------------------------------------------
    def comprehensive_check(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        feature_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Run all available leakage checks

        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target variable
        groups : array-like, optional
            Group identifiers (e.g., patient IDs)
        timestamps : array-like, optional
            Temporal information
        coordinates : array-like, optional
            Spatial coordinates
        feature_threshold : float
            Threshold for feature-target correlation

        Returns
        -------
        dict
            Comprehensive leakage report with recommendations
        """
        report: Dict[str, Any] = {
            "feature_leakage": None,
            "recommendations": [],
        }

        # Check feature-target leakage
        feature_result = self.check_feature_target_leakage(
            X, y, threshold=feature_threshold
        )
        # Convert suspicious_features to indices for backward compat
        feature_result["suspicious_features"] = [
            f["index"]
            for f in feature_result.get("suspicious_features", [])
        ]
        report["feature_leakage"] = feature_result

        if feature_result["has_leakage"]:
            report["recommendations"].append(
                f"Remove or investigate "
                f"{feature_result['num_suspicious']} suspicious "
                "features with high target correlation"
            )

        # Additional checks based on available data
        if groups is not None:
            report["recommendations"].append(
                "Use grouped cross-validation to prevent "
                "patient-level leakage"
            )

        if timestamps is not None:
            report["recommendations"].append(
                "Use temporal cross-validation to prevent "
                "future information leakage"
            )

        if coordinates is not None:
            report["recommendations"].append(
                "Use spatial cross-validation to handle "
                "spatial autocorrelation"
            )

        # Run the CV-based check() suite and merge results
        cv_report = self.check(
            X,
            y,
            groups=groups,
            timestamps=timestamps,
            coordinates=coordinates,
        )
        report["cv_leakage_report"] = cv_report.to_dict()

        # Merge CV recommendations (avoid duplicates)
        existing_recs = set(report["recommendations"])
        for rec in cv_report.recommendations:
            if rec not in existing_recs:
                report["recommendations"].append(rec)
                existing_recs.add(rec)

        # Merge severity
        report["overall_severity"] = _worst_severity(
            cv_report.severity,
            "high" if feature_result["has_leakage"] else "none",
        )

        return report
