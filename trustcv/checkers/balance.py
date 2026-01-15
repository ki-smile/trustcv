"""
Balance checker for medical datasets

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class BalanceChecker:
    """
    Checks for class imbalance and data distribution issues

    Critical for medical datasets where disease prevalence
    can be extremely low (< 1% positive cases)
    """

    def __init__(self, threshold: float = 0.1):
        """
        Parameters
        ----------
        threshold : float
            Imbalance threshold for warnings (default 10% difference)
        """
        self.threshold = threshold
        self.report = {}

    def check_class_balance(
        self, y: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Check class distribution in dataset

        Parameters
        ----------
        y : array-like
            Target labels
        groups : array-like, optional
            Group identifiers (e.g., patient IDs)

        Returns
        -------
        dict
            Balance report with statistics and warnings
        """
        y = np.array(y)
        unique_classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)

        # Calculate class distribution
        class_dist = {
            str(cls): {"count": int(count), "percentage": float(count / n_samples * 100)}
            for cls, count in zip(unique_classes, counts)
        }

        # Calculate imbalance ratio
        min_class = counts.min()
        max_class = counts.max()
        imbalance_ratio = max_class / min_class if min_class > 0 else np.inf

        # Check for severe imbalance
        minority_pct = min_class / n_samples

        report = {
            "n_classes": len(unique_classes),
            "class_distribution": class_dist,
            "imbalance_ratio": float(imbalance_ratio),
            "minority_percentage": float(minority_pct * 100),
            "warnings": [],
        }

        # Generate warnings
        if imbalance_ratio > 10:
            report["warnings"].append(
                f"SEVERE IMBALANCE: Ratio {imbalance_ratio:.1f}:1. "
                "Consider stratified splitting or resampling."
            )
        elif imbalance_ratio > 5:
            report["warnings"].append(
                f"Moderate imbalance: Ratio {imbalance_ratio:.1f}:1. "
                "Use stratified cross-validation."
            )

        if minority_pct < 0.01:
            report["warnings"].append(
                f"Rare event: Minority class < 1% ({minority_pct*100:.2f}%). "
                "Consider specialized techniques (SMOTE, cost-sensitive learning)."
            )

        # Check group-level balance if provided
        if groups is not None:
            report["group_analysis"] = self._check_group_balance(y, groups)

        self.report = report
        return report

    def _check_group_balance(self, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
        """
        Check balance within groups (e.g., per patient)

        Parameters
        ----------
        y : array-like
            Target labels
        groups : array-like
            Group identifiers

        Returns
        -------
        dict
            Group-level balance analysis
        """
        groups = np.array(groups)
        unique_groups = np.unique(groups)

        group_stats = []
        for group in unique_groups:
            group_mask = groups == group
            group_y = y[group_mask]

            if len(np.unique(group_y)) > 1:
                # Mixed labels in group
                unique_classes, counts = np.unique(group_y, return_counts=True)
                majority_class = unique_classes[np.argmax(counts)]
                group_stats.append(
                    {
                        "group": group,
                        "n_samples": len(group_y),
                        "n_classes": len(unique_classes),
                        "majority_class": int(majority_class),
                        "mixed": True,
                    }
                )
            else:
                # Single class in group
                group_stats.append(
                    {
                        "group": group,
                        "n_samples": len(group_y),
                        "n_classes": 1,
                        "majority_class": int(group_y[0]),
                        "mixed": False,
                    }
                )

        # Analyze group patterns
        mixed_groups = sum(1 for g in group_stats if g["mixed"])
        single_class_groups = len(group_stats) - mixed_groups

        analysis = {
            "n_groups": len(unique_groups),
            "mixed_label_groups": mixed_groups,
            "single_label_groups": single_class_groups,
            "warnings": [],
        }

        if mixed_groups > 0:
            analysis["warnings"].append(
                f"{mixed_groups} groups have mixed labels. "
                "This may affect patient-level predictions."
            )

        return analysis

    def check_feature_distribution(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check feature distributions for potential issues

        Parameters
        ----------
        X : array-like
            Feature matrix
        feature_names : list, optional
            Names of features

        Returns
        -------
        dict
            Feature distribution analysis
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        feature_stats = []
        warnings = []

        for i, name in enumerate(feature_names):
            feature = X[:, i]

            stats = {
                "name": name,
                "mean": float(np.mean(feature)),
                "std": float(np.std(feature)),
                "min": float(np.min(feature)),
                "max": float(np.max(feature)),
                "n_unique": len(np.unique(feature)),
                "n_missing": int(np.isnan(feature).sum()),
                "skewness": float(self._calculate_skewness(feature)),
            }

            # Check for issues
            if stats["std"] == 0:
                warnings.append(f"{name}: Zero variance (constant feature)")
            elif stats["n_unique"] == 1:
                warnings.append(f"{name}: Only one unique value")
            elif stats["n_missing"] > n_samples * 0.2:
                warnings.append(
                    f"{name}: High missing rate ({stats['n_missing']/n_samples*100:.1f}%)"
                )
            elif abs(stats["skewness"]) > 2:
                warnings.append(f"{name}: Highly skewed (skewness={stats['skewness']:.2f})")

            feature_stats.append(stats)

        return {"n_features": n_features, "feature_statistics": feature_stats, "warnings": warnings}

    def _calculate_skewness(self, x: np.ndarray) -> float:
        """Calculate skewness of a distribution"""
        x = x[~np.isnan(x)]  # Remove NaN values
        if len(x) < 3:
            return 0.0

        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            return 0.0

        skewness = np.mean(((x - mean) / std) ** 3)
        return skewness

    def check_cv_balance(self, X, y, cv_splitter, groups=None):
        """
        Check balance across CV folds

        Parameters
        ----------
        X : array-like
            Features
        y : array-like
            Labels
        cv_splitter : cross-validation object
            CV splitter to test
        groups : array-like, optional
            Group labels

        Returns
        -------
        dict
            CV balance analysis
        """
        y = np.array(y)
        fold_stats = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups)):
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Calculate distributions
            train_dist = np.bincount(y_train) / len(y_train)
            test_dist = np.bincount(y_test) / len(y_test)

            # Ensure same length
            max_len = max(len(train_dist), len(test_dist))
            train_dist = np.pad(train_dist, (0, max_len - len(train_dist)))
            test_dist = np.pad(test_dist, (0, max_len - len(test_dist)))

            # Calculate distribution difference
            dist_diff = np.abs(train_dist - test_dist).max()

            fold_stats.append(
                {
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "train_distribution": train_dist.tolist(),
                    "test_distribution": test_dist.tolist(),
                    "max_distribution_diff": float(dist_diff),
                }
            )

        # Check for issues
        warnings = []
        max_diff = max(f["max_distribution_diff"] for f in fold_stats)

        if max_diff > self.threshold:
            warnings.append(
                f"Large distribution difference between train/test: {max_diff*100:.1f}%. "
                "Consider stratified splitting."
            )

        size_variation = np.std([f["test_size"] for f in fold_stats])
        if size_variation > np.mean([f["test_size"] for f in fold_stats]) * 0.1:
            warnings.append(
                "Large variation in fold sizes. " "This may affect performance estimates."
            )

        return {
            "n_folds": len(fold_stats),
            "fold_statistics": fold_stats,
            "max_distribution_difference": float(max_diff),
            "warnings": warnings,
        }

    def generate_report(self) -> str:
        """
        Generate human-readable balance report

        Returns
        -------
        str
            Formatted report
        """
        if not self.report:
            return "No balance check performed yet."

        lines = ["=" * 50, "BALANCE CHECK REPORT", "=" * 50]

        # Class balance
        if "class_distribution" in self.report:
            lines.append("\nCLASS BALANCE:")
            for cls, stats in self.report["class_distribution"].items():
                lines.append(
                    f"  Class {cls}: {stats['count']} samples ({stats['percentage']:.1f}%)"
                )
            lines.append(f"  Imbalance ratio: {self.report['imbalance_ratio']:.2f}:1")

        # Warnings
        if self.report.get("warnings"):
            lines.append("\nWARNINGS:")
            for warning in self.report["warnings"]:
                lines.append(f"  ⚠️  {warning}")

        # Group analysis
        if "group_analysis" in self.report:
            ga = self.report["group_analysis"]
            lines.append(f"\nGROUP ANALYSIS:")
            lines.append(f"  Total groups: {ga['n_groups']}")
            lines.append(f"  Mixed-label groups: {ga['mixed_label_groups']}")
            lines.append(f"  Single-label groups: {ga['single_label_groups']}")

        return "\n".join(lines)
