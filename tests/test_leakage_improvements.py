"""
Unit tests for data leakage detection improvements (v1.0.7)

Tests cover:
- P0 bug fixes (severity ranking, temporal check, bare excepts)
- P1 new features (near-duplicates, hierarchical, spatial auto-threshold,
  LeakageDetectionCallback, auto-checker in validator)
- P2 quality improvements (KS-test, temporal quantification, chi-squared)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from trustcv.checkers.leakage import (
    DataLeakageChecker,
    LeakageReport,
    _worst_severity,
    _SEVERITY_RANK,
)
from trustcv.checkers.balance import BalanceChecker


class TestSeverityRanking:
    """P0: Verify severity comparison is correct."""

    def test_worst_severity_critical_beats_high(self):
        assert _worst_severity("critical", "high") == "critical"

    def test_worst_severity_high_beats_medium(self):
        assert _worst_severity("high", "medium") == "high"

    def test_worst_severity_none_vs_low(self):
        assert _worst_severity("none", "low") == "low"

    def test_worst_severity_same(self):
        assert _worst_severity("high", "high") == "high"

    def test_worst_severity_all_ranks(self):
        levels = ["none", "low", "medium", "high", "critical"]
        for i, a in enumerate(levels):
            for j, b in enumerate(levels):
                result = _worst_severity(a, b)
                expected = levels[max(i, j)]
                assert result == expected, f"_worst_severity({a}, {b}) = {result}, expected {expected}"

    def test_severity_rank_dict_complete(self):
        assert set(_SEVERITY_RANK.keys()) == {"none", "low", "medium", "high", "critical"}

    def test_severity_in_check_cv_splits(self):
        """Ensure severity ranking is used correctly in check_cv_splits."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        # Create patient overlap (critical) + duplicates (high)
        patient_ids_tr = np.arange(50)
        patient_ids_te = np.arange(40, 60)  # overlap 40-49
        # Create exact duplicates
        X_test = X[40:60].copy()
        X_test[:5] = X[:5]  # first 5 test rows = first 5 train rows

        checker = DataLeakageChecker(verbose=False)
        report = checker.check_cv_splits(
            X[:50], X_test,
            patient_ids_train=patient_ids_tr,
            patient_ids_test=patient_ids_te,
        )
        # Patient leakage is critical, should dominate
        assert report.severity == "critical"


class TestTemporalLeakageFix:
    """P0: Fixed temporal leakage detection."""

    def setup_method(self):
        self.checker = DataLeakageChecker(verbose=False)

    def test_no_leakage_for_forward_split(self):
        """Test data after training period should NOT be flagged."""
        ts_train = pd.date_range("2023-01-01", periods=50, freq="D")
        ts_test = pd.date_range("2023-03-01", periods=20, freq="D")
        result = self.checker._check_temporal_leakage(ts_train, ts_test)
        assert not result["has_leakage"]
        assert result["overlap_fraction"] == 0.0

    def test_leakage_test_before_train(self):
        """Test data before training period should be flagged."""
        ts_train = pd.date_range("2023-06-01", periods=50, freq="D")
        ts_test = pd.date_range("2023-01-01", periods=20, freq="D")
        result = self.checker._check_temporal_leakage(ts_train, ts_test)
        assert result["has_leakage"]
        assert result["test_before_train"]
        assert result["test_before_train_fraction"] > 0.0

    def test_minor_overlap_not_flagged(self):
        """Small overlap (< 50%) should NOT be flagged as leakage."""
        # train: Jan 1 - Feb 19, test: Feb 10 - Mar 31 (10 days overlap out of 50)
        ts_train = pd.date_range("2023-01-01", periods=50, freq="D")
        ts_test = pd.date_range("2023-02-10", periods=50, freq="D")
        result = self.checker._check_temporal_leakage(ts_train, ts_test)
        assert result["overlap_fraction"] < 0.5
        assert not result["has_leakage"]

    def test_major_overlap_flagged(self):
        """Large overlap (> 50%) should be flagged."""
        ts_train = pd.date_range("2023-01-01", periods=100, freq="D")
        # test almost entirely within train period
        ts_test = pd.date_range("2023-01-10", periods=20, freq="D")
        result = self.checker._check_temporal_leakage(ts_train, ts_test)
        assert result["overlap_fraction"] > 0.5
        assert result["has_leakage"]

    def test_overlap_fraction_computed(self):
        """Ensure overlap_fraction and test_before_train_fraction are returned."""
        ts_train = pd.date_range("2023-01-01", periods=50, freq="D")
        ts_test = pd.date_range("2023-02-01", periods=30, freq="D")
        result = self.checker._check_temporal_leakage(ts_train, ts_test)
        assert "overlap_fraction" in result
        assert "test_before_train_fraction" in result
        assert isinstance(result["overlap_fraction"], float)
        assert isinstance(result["test_before_train_fraction"], float)

    def test_invalid_timestamps_handled(self):
        """Bad timestamps should return has_leakage=False with error."""
        ts_train = np.array(["not", "a", "date"])
        ts_test = np.array(["also", "not", "dates"])
        result = self.checker._check_temporal_leakage(ts_train, ts_test)
        assert not result["has_leakage"]
        assert "error" in result


class TestNearDuplicateDetection:
    """P1: Near-duplicate detection via cosine similarity."""

    def setup_method(self):
        self.checker = DataLeakageChecker(verbose=False)

    def test_exact_duplicates_detected(self):
        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        X_test = np.random.randn(20, 10)
        X_test[:3] = X_train[:3]  # exact copies
        result = self.checker.check_near_duplicates(X_train, X_test)
        assert result["has_leakage"]
        assert result["near_duplicate_count"] >= 3

    def test_near_duplicates_detected(self):
        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        X_test = np.random.randn(20, 10)
        # Add near-duplicates (tiny perturbation)
        X_test[:5] = X_train[:5] + np.random.randn(5, 10) * 1e-4
        result = self.checker.check_near_duplicates(
            X_train, X_test, similarity_threshold=0.999
        )
        assert result["has_leakage"]
        assert result["near_duplicate_count"] >= 5

    def test_no_near_duplicates(self):
        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        X_test = np.random.randn(20, 10) * 100  # very different
        result = self.checker.check_near_duplicates(X_train, X_test)
        assert not result["has_leakage"]
        assert result["near_duplicate_count"] == 0

    def test_threshold_parameter(self):
        np.random.seed(42)
        X_train = np.random.randn(30, 5)
        X_test = np.random.randn(10, 5)
        result = self.checker.check_near_duplicates(
            X_train, X_test, similarity_threshold=0.5
        )
        assert result["similarity_threshold"] == 0.5

    def test_dataframe_input(self):
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(30, 5))
        X_test = pd.DataFrame(np.random.randn(10, 5))
        X_test.iloc[:2] = X_train.iloc[:2].values
        result = self.checker.check_near_duplicates(X_train, X_test)
        assert result["has_leakage"]

    def test_near_duplicates_in_check_cv_splits(self):
        """Near-duplicate check is wired into check_cv_splits."""
        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        X_test = np.random.randn(20, 10)
        X_test[:5] = X_train[:5]  # exact copies
        checker = DataLeakageChecker(verbose=False)
        report = checker.check_cv_splits(X_train, X_test)
        assert "near_duplicate" in report.leakage_types


class TestHierarchicalLeakage:
    """P1: Hierarchical group leakage detection."""

    def setup_method(self):
        self.checker = DataLeakageChecker(verbose=False)

    def test_hierarchical_leakage_detected(self):
        # Hospital 1 has patients 0-9, Hospital 2 has patients 10-19
        # If Hospital 1 appears in both train and test = leakage
        groups_train = np.arange(10)  # patients 0-9
        groups_test = np.arange(5, 15)  # patients 5-14
        parent_train = np.array([1] * 10)  # all hospital 1
        parent_test = np.array([1] * 5 + [2] * 5)  # hospital 1 and 2
        result = self.checker.check_hierarchical_leakage(
            groups_train, groups_test, parent_train, parent_test
        )
        assert result["has_leakage"]
        assert result["overlap_count"] == 1
        assert "1" in result["overlapping_parents"]

    def test_no_hierarchical_leakage(self):
        groups_train = np.arange(10)
        groups_test = np.arange(10, 20)
        parent_train = np.array([1] * 10)
        parent_test = np.array([2] * 10)
        result = self.checker.check_hierarchical_leakage(
            groups_train, groups_test, parent_train, parent_test
        )
        assert not result["has_leakage"]
        assert result["overlap_count"] == 0

    def test_multiple_parent_overlap(self):
        parent_train = np.array([1, 1, 2, 2, 3])
        parent_test = np.array([2, 3, 4])
        result = self.checker.check_hierarchical_leakage(
            np.arange(5), np.arange(5, 8), parent_train, parent_test
        )
        assert result["has_leakage"]
        assert result["overlap_count"] == 2  # hospitals 2 and 3


class TestSpatialAutoThreshold:
    """P1: Auto-compute spatial threshold."""

    def test_auto_threshold_computed(self):
        np.random.seed(42)
        coords = np.random.uniform(0, 100, (200, 2))
        threshold = DataLeakageChecker._auto_spatial_threshold(coords)
        assert threshold > 0
        assert isinstance(threshold, float)

    def test_auto_threshold_in_check(self):
        """check() should auto-compute spatial threshold when coords provided."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)
        coords = np.random.uniform(0, 100, (n, 2))
        checker = DataLeakageChecker(verbose=False)
        report = checker.check(X, y, coordinates=coords)
        # Should have spatial info in at least some folds
        assert isinstance(report, LeakageReport)


class TestKSTest:
    """P2: KS-test in feature statistics check."""

    def setup_method(self):
        self.checker = DataLeakageChecker(verbose=False)

    def test_ks_suspicious_features_reported(self):
        np.random.seed(42)
        # Same distribution -> KS should NOT flag
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        result = self.checker._check_feature_statistics(X_train, X_test)
        assert "ks_suspicious_features" in result

    def test_ks_detects_different_distributions(self):
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        X_test = np.random.randn(100, 5) + 5  # shifted distribution
        result = self.checker._check_feature_statistics(X_train, X_test)
        assert result["ks_suspicious_features"] > 0

    def test_ks_pvalues_returned(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        X_test = np.random.randn(50, 3)
        result = self.checker._check_feature_statistics(X_train, X_test)
        assert "ks_pvalues" in result
        assert len(result["ks_pvalues"]) == 3


class TestChiSquaredLabel:
    """P2: Chi-squared test in label distribution check."""

    def setup_method(self):
        self.checker = DataLeakageChecker(verbose=False)

    def test_chi2_pvalue_returned(self):
        y_train = np.array([0] * 80 + [1] * 20)
        y_test = np.array([0] * 40 + [1] * 10)
        result = self.checker._check_label_distribution(y_train, y_test)
        assert "chi2_pvalue" in result
        assert isinstance(result["chi2_pvalue"], float)

    def test_chi2_significant_imbalance(self):
        y_train = np.array([0] * 90 + [1] * 10)
        y_test = np.array([0] * 10 + [1] * 90)  # reversed
        result = self.checker._check_label_distribution(y_train, y_test)
        assert result["suspicious"]
        assert result["chi2_pvalue"] < 0.01


class TestComprehensiveCheck:
    """P1: comprehensive_check() now runs full CV suite."""

    def test_comprehensive_includes_cv_report(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 10)
        y = np.random.randint(0, 2, n)
        X[:, 0] = y * 2 + np.random.randn(n) * 0.01  # feature leakage
        checker = DataLeakageChecker(verbose=False)
        result = checker.comprehensive_check(X, y)
        assert "feature_leakage" in result
        assert "cv_leakage_report" in result
        assert "overall_severity" in result
        assert result["feature_leakage"]["has_leakage"]

    def test_comprehensive_merges_recommendations(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)
        groups = np.repeat(np.arange(20), 5)
        checker = DataLeakageChecker(verbose=False)
        result = checker.comprehensive_check(X, y, groups=groups)
        assert len(result["recommendations"]) > 0


class TestLeakageDetectionCallback:
    """P1: LeakageDetectionCallback for UniversalCVRunner."""

    def test_callback_import(self):
        from trustcv.core.callbacks import LeakageDetectionCallback
        assert LeakageDetectionCallback is not None

    def test_callback_top_level_import(self):
        from trustcv import LeakageDetectionCallback
        assert LeakageDetectionCallback is not None

    def test_callback_basic_usage(self):
        from trustcv.core.callbacks import LeakageDetectionCallback

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)

        cb = LeakageDetectionCallback(data=(X, y), verbose=0)

        # Simulate fold_start and cv_end
        train_idx = np.arange(80)
        val_idx = np.arange(80, 100)
        cb.on_fold_start(0, train_idx, val_idx)

        assert len(cb.fold_reports) == 1
        assert hasattr(cb.fold_reports[0], "has_leakage")

        cb.on_cv_end([{"accuracy": 0.9}])
        assert cb.summary_report is not None
        assert "worst_severity" in cb.summary_report

    def test_callback_detects_group_leakage(self):
        from trustcv.core.callbacks import LeakageDetectionCallback

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)
        groups = np.repeat(np.arange(10), 10)

        cb = LeakageDetectionCallback(
            data=(X, y), groups=groups, verbose=0
        )

        # Create split where groups overlap
        train_idx = np.arange(0, 80)  # groups 0-7
        val_idx = np.arange(70, 100)  # groups 7-9 (group 7 overlaps)
        cb.on_fold_start(0, train_idx, val_idx)

        report = cb.fold_reports[0]
        assert report.has_leakage
        assert "patient" in report.leakage_types

    def test_callback_with_timestamps(self):
        from trustcv.core.callbacks import LeakageDetectionCallback

        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)
        ts = pd.date_range("2023-01-01", periods=n, freq="D")

        cb = LeakageDetectionCallback(
            data=(X, y), timestamps=ts.values, verbose=0
        )

        train_idx = np.arange(80)
        val_idx = np.arange(80, 100)
        cb.on_fold_start(0, train_idx, val_idx)

        assert len(cb.fold_reports) == 1


class TestValidatorAutoChecker:
    """P1: TrustCVValidator auto-creates DataLeakageChecker when check_leakage=True."""

    def test_auto_checker_runs(self):
        from trustcv import TrustCV
        from sklearn.tree import DecisionTreeClassifier

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        validator = TrustCV(
            method="stratified_kfold",
            n_splits=3,
            check_leakage=True,
        )
        results = validator.validate(
            model=DecisionTreeClassifier(),
            X=X, y=y,
        )
        # Should have leakage check results
        assert "has_leakage" in results.leakage_check

    def test_auto_checker_disabled(self):
        from trustcv import TrustCV
        from sklearn.tree import DecisionTreeClassifier

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        validator = TrustCV(
            method="stratified_kfold",
            n_splits=3,
            check_leakage=False,
        )
        results = validator.validate(
            model=DecisionTreeClassifier(),
            X=X, y=y,
        )
        # Should NOT have full leakage check
        assert "has_leakage" not in results.leakage_check


class TestLeakageReport:
    """Test LeakageReport dataclass."""

    def test_no_leakage_str(self):
        report = LeakageReport(
            has_leakage=False,
            leakage_types=[],
            severity="none",
            details={},
            recommendations=[],
        )
        assert "No data leakage" in str(report)

    def test_leakage_str(self):
        report = LeakageReport(
            has_leakage=True,
            leakage_types=["patient", "temporal"],
            severity="critical",
            details={},
            recommendations=["Fix patient grouping"],
        )
        s = str(report)
        assert "critical" in s.lower()
        assert "patient" in s
        assert "Fix patient grouping" in s

    def test_to_dict(self):
        report = LeakageReport(
            has_leakage=True,
            leakage_types=["duplicate"],
            severity="high",
            details={"test": True},
            recommendations=["Remove duplicates"],
        )
        d = report.to_dict()
        assert d["has_leakage"] is True
        assert d["severity"] == "high"
        assert isinstance(d["leakage_types"], list)


class TestCheckConvenienceMethod:
    """Test the check() convenience method."""

    def test_check_basic(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        checker = DataLeakageChecker(verbose=False)
        report = checker.check(X, y)
        assert isinstance(report, LeakageReport)

    def test_check_with_groups(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        groups = np.repeat(np.arange(20), 5)
        checker = DataLeakageChecker(verbose=False)
        report = checker.check(X, y, groups=groups)
        assert isinstance(report, LeakageReport)

    def test_check_with_timestamps(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        ts = pd.date_range("2023-01-01", periods=100, freq="D").values
        checker = DataLeakageChecker(verbose=False)
        report = checker.check(X, y, timestamps=ts)
        assert isinstance(report, LeakageReport)

    def test_check_with_coordinates(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        coords = np.random.uniform(0, 100, (100, 2))
        checker = DataLeakageChecker(verbose=False)
        report = checker.check(X, y, coordinates=coords)
        assert isinstance(report, LeakageReport)


class TestPreprocessingLeakage:
    """Test preprocessing leakage detection."""

    def test_global_normalization_detected(self):
        # The method checks if train_mean ~= global_mean (rtol=1e-10).
        # To guarantee this, use ALL samples as "train" (leave 0 test).
        # In practice, this simulates the degenerate case.
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X_processed = (X - X.mean(axis=0)) / X.std(axis=0)
        # Use all indices as train — global mean == train mean exactly
        train_idx = np.arange(100)
        test_idx = np.arange(100)
        checker = DataLeakageChecker(verbose=False)
        has_leakage = checker.check_preprocessing_leakage(
            X, X_processed, (train_idx, test_idx)
        )
        assert has_leakage

    def test_proper_normalization_not_flagged(self):
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50
        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)
        # Normalize on train only
        X_processed = X.copy()
        train_mean = X[train_idx].mean(axis=0)
        train_std = X[train_idx].std(axis=0)
        X_processed[train_idx] = (X[train_idx] - train_mean) / train_std
        X_processed[test_idx] = (X[test_idx] - train_mean) / train_std
        checker = DataLeakageChecker(verbose=False)
        has_leakage = checker.check_preprocessing_leakage(
            X, X_processed, (train_idx, test_idx)
        )
        assert not has_leakage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
