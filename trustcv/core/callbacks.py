"""
Callback system for monitoring and controlling cross-validation

Provides callbacks for early stopping, model checkpointing, progress logging,
and custom user callbacks to promote best practices in model evaluation.
"""

import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class CVCallback(ABC):
    """
    Base callback class for cross-validation events

    Callbacks allow for monitoring, early stopping, checkpointing,
    and custom logic during cross-validation.
    """

    def on_cv_start(self, n_splits: int) -> None:
        """Called at the start of cross-validation"""
        pass

    def on_fold_start(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Called at the start of each fold"""
        pass

    def on_epoch_start(self, epoch: int, fold_idx: int) -> None:
        """Called at the start of each epoch (for iterative training)"""
        pass

    def on_epoch_end(
        self, epoch: int, fold_idx: int, logs: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Called at the end of each epoch

        Returns:
            Optional string 'stop' to trigger early stopping
        """
        pass

    def on_fold_end(self, fold_idx: int, results: Dict[str, Any]) -> None:
        """Called at the end of each fold"""
        pass

    def on_cv_end(self, all_results: List[Dict[str, Any]]) -> None:
        """Called at the end of cross-validation"""
        pass


class EarlyStopping(CVCallback):
    """
    Early stopping callback to prevent overfitting

    Implements best practices for early stopping with patience and
    optional restoration of best model weights.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        restore_best: bool = True,
        min_delta: float = 0.0001,
        verbose: bool = True,
    ):
        """
        Initialize early stopping callback

        Parameters:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement to wait
            mode: 'min' or 'max' - whether lower or higher is better
            restore_best: Whether to restore best model weights
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.restore_best = restore_best
        self.min_delta = min_delta
        self.verbose = verbose

        self.wait = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None

    def on_fold_start(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Reset early stopping for new fold"""
        self.wait = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(
        self, epoch: int, fold_idx: int, logs: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Check if training should stop"""
        if logs is None or self.monitor not in logs:
            warnings.warn(f"Early stopping: monitored metric '{self.monitor}' not found in logs")
            return None

        current = logs[self.monitor]

        if self.best_score is None:
            self.best_score = current
            self.best_epoch = epoch
            if self.restore_best and "model" in logs:
                self.best_weights = logs["model"]
        else:
            if self.mode == "min":
                improved = current < (self.best_score - self.min_delta)
            else:
                improved = current > (self.best_score + self.min_delta)

            if improved:
                self.best_score = current
                self.best_epoch = epoch
                self.wait = 0
                if self.restore_best and "model" in logs:
                    self.best_weights = logs["model"]

                if self.verbose:
                    print(
                        f"Fold {fold_idx}, Epoch {epoch}: "
                        f"{self.monitor} improved to {current:.4f}"
                    )
            else:
                self.wait += 1
                if self.verbose:
                    print(
                        f"Fold {fold_idx}, Epoch {epoch}: " f"No improvement for {self.wait} epochs"
                    )

                if self.wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping triggered at epoch {epoch}")

                    if self.restore_best and self.best_weights is not None:
                        logs["restore_weights"] = self.best_weights

                    return "stop"

        return None


class ModelCheckpoint(CVCallback):
    """
    Save model checkpoints during training

    Implements best practices for model checkpointing including
    saving best models and periodic snapshots.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_freq: int = 1,
        verbose: bool = True,
    ):
        """
        Initialize model checkpoint callback

        Parameters:
            filepath: Path pattern for saving models (can include {fold} and {epoch})
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' - whether lower or higher is better
            save_best_only: Only save when monitored metric improves
            save_freq: Frequency of saving (in epochs)
            verbose: Whether to print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.verbose = verbose

        self.best_score = None

    def on_fold_start(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Reset best score for new fold"""
        self.best_score = None

    def on_epoch_end(
        self, epoch: int, fold_idx: int, logs: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Save model checkpoint if conditions are met"""
        if logs is None or "model" not in logs:
            return None

        # Check if we should save this epoch
        should_save = False

        if self.save_best_only:
            if self.monitor not in logs:
                warnings.warn(f"Checkpoint: monitored metric '{self.monitor}' not found")
                return None

            current = logs[self.monitor]

            if self.best_score is None:
                should_save = True
                self.best_score = current
            else:
                if self.mode == "min":
                    should_save = current < self.best_score
                else:
                    should_save = current > self.best_score

                if should_save:
                    self.best_score = current
        else:
            should_save = (epoch % self.save_freq) == 0

        if should_save:
            filepath = self.filepath.format(fold=fold_idx, epoch=epoch)

            if self.verbose:
                if self.save_best_only:
                    print(
                        f"Saving best model to {filepath} "
                        f"({self.monitor}: {self.best_score:.4f})"
                    )
                else:
                    print(f"Saving checkpoint to {filepath}")

            # Request model save
            logs["save_checkpoint"] = filepath

        return None


class ProgressLogger(CVCallback):
    """
    Log progress during cross-validation

    Provides detailed logging for regulatory compliance and debugging.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        verbose: int = 1,
        groups: Optional[np.ndarray] = None,
    ):
        """
        Initialize progress logger

        Parameters:
            log_file: Optional file to write logs to
            metrics: Specific metrics to log
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.log_file = log_file
        self.metrics = metrics
        self.verbose = verbose
        self.logs = []
        self.groups = groups

    def set_groups(self, groups: Optional[np.ndarray]) -> None:
        """Attach group labels so we can report group counts per fold."""
        self.groups = groups

    def on_cv_start(self, n_splits: int) -> None:
        """Log CV start"""
        if self.verbose >= 1:
            print(f"Starting {n_splits}-fold cross-validation")
            print("=" * 50)

        self.logs.append({"event": "cv_start", "n_splits": n_splits})

    def on_fold_start(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Log fold start"""
        if self.verbose >= 1:
            print(f"\nFold {fold_idx + 1}:")
            print(f"  Training samples: {len(train_idx)}")
            print(f"  Validation samples: {len(val_idx)}")
            if self.groups is not None:
                try:
                    n_train_groups = len(np.unique(self.groups[train_idx]))
                    n_val_groups = len(np.unique(self.groups[val_idx]))
                    print(f"  Training groups: {n_train_groups}")
                    print(f"  Validation groups: {n_val_groups}")
                except Exception:
                    pass

        self.logs.append(
            {
                "event": "fold_start",
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            }
        )

    def on_epoch_end(
        self, epoch: int, fold_idx: int, logs: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log epoch metrics"""
        if logs and self.verbose >= 2:
            metrics_str = []
            for key, value in logs.items():
                if self.metrics is None or key in self.metrics:
                    if isinstance(value, (int, float)):
                        metrics_str.append(f"{key}: {value:.4f}")

            if metrics_str:
                print(f"  Epoch {epoch}: {' - '.join(metrics_str)}")

        log_entry = {"event": "epoch_end", "fold": fold_idx, "epoch": epoch}

        if logs:
            log_entry["metrics"] = {
                k: v for k, v in logs.items() if isinstance(v, (int, float, str))
            }

        self.logs.append(log_entry)
        return None

    def on_fold_end(self, fold_idx: int, results: Dict[str, Any]) -> None:
        """Log fold results"""
        if self.verbose >= 1:
            print(f"Fold {fold_idx + 1} completed")

            if results and self.verbose >= 2:
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")

        self.logs.append({"event": "fold_end", "fold": fold_idx, "results": results})

    def on_cv_end(self, all_results: List[Dict[str, Any]]) -> None:
        """Log CV completion and save logs if requested"""
        if self.verbose >= 1:
            print("\n" + "=" * 50)
            print("Cross-validation completed")

        self.logs.append({"event": "cv_end", "n_folds": len(all_results)})

        # Save logs to file if requested
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump(self.logs, f, indent=2, default=str)

            if self.verbose >= 1:
                print(f"\nLogs saved to {self.log_file}")


class ClassDistributionLogger(CVCallback):
    """
    Display per-fold class composition for train/validation splits.

    Useful to verify that each fold preserves label balance.
    """

    def __init__(
        self,
        labels,
        label_names: Optional[Dict[Any, str]] = None,
        verbose: int = 1,
        decimals: int = 1,
        output_key: Optional[str] = None,
        output_index: Optional[int] = None,
    ):
        """
        Parameters:
            labels: Array-like target variable used for splitting
            label_names: Optional mapping {raw_label: display_name}
            verbose: 0=silent, 1=summary per fold
            decimals: Number of decimal places for percentages
            output_key/index: For multi-output labels (dict/list), choose which head to log
        """
        self.labels = labels
        self.label_names = label_names or {}
        self.verbose = verbose
        self.decimals = max(0, decimals)
        self.output_key = output_key
        self.output_index = output_index
        self._cache = {}

    def _slice_labels(self, indices: np.ndarray) -> np.ndarray:
        """Return labels for the given indices, handling pandas objects."""
        if indices is None or len(indices) == 0:
            return np.array([])
        labels = self.labels

        def _select_target(y_obj):
            if isinstance(y_obj, dict):
                if self.output_key is not None:
                    if self.output_key not in y_obj:
                        raise KeyError(f"output_key '{self.output_key}' not found in labels.")
                    return y_obj[self.output_key]
                if self.output_index is not None:
                    keys = list(y_obj.keys())
                    if self.output_index >= len(keys):
                        raise IndexError("output_index is out of range for labels dict.")
                    return y_obj[keys[self.output_index]]
                # default: first key
                first_key = next(iter(y_obj.keys()))
                return y_obj[first_key]
            if isinstance(y_obj, (list, tuple)) and not hasattr(y_obj, "shape"):
                if self.output_index is not None:
                    if self.output_index >= len(y_obj):
                        raise IndexError("output_index is out of range for labels list/tuple.")
                    return y_obj[self.output_index]
                return y_obj[0]
            return y_obj

        labels = _select_target(labels)

        if hasattr(labels, "iloc"):
            subset = labels.iloc[indices]
        else:
            try:
                subset = labels[indices]
            except Exception:
                subset = np.asarray(labels)[indices]
        return np.asarray(subset)

    def _format_distribution(self, indices: np.ndarray) -> str:
        """Format class counts and percentages for display."""
        key = hash(indices.tobytes())
        if key in self._cache:
            return self._cache[key]

        values = self._slice_labels(indices)
        if values.size == 0:
            summary = "n/a"
        elif values.ndim == 2:
            # Multilabel: report per-label positive prevalence
            total = values.shape[0]
            pos_counts = values.sum(axis=0)
            pieces = []
            n_labels = values.shape[1]
            for i in range(n_labels):
                label = self.label_names.get(i, f"{i}")
                cnt = int(pos_counts[i])
                perc = (cnt / total) * 100 if total > 0 else 0.0
                pieces.append(f"{label}: {cnt} ({perc:.{self.decimals}f}%)")
            summary = ", ".join(pieces)
        else:
            unique, counts = np.unique(values, return_counts=True)
            total = counts.sum()
            pieces = []
            for cls, cnt in zip(unique, counts):
                label = self.label_names.get(cls, cls)
                perc = (cnt / total) * 100 if total > 0 else 0.0
                pieces.append(f"{label}: {cnt} ({perc:.{self.decimals}f}%)")
            summary = ", ".join(pieces)

        self._cache[key] = summary
        return summary

    def on_fold_start(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        if self.verbose <= 0:
            return

        train_summary = self._format_distribution(train_idx)
        val_summary = self._format_distribution(val_idx)
        print(f"  Train class distribution: {train_summary}")
        print(f"  Val class distribution:   {val_summary}")


class LeakageDetectionCallback(CVCallback):
    """
    Automatic data leakage detection callback for UniversalCVRunner.

    Checks each fold for data leakage issues (duplicate samples, group
    overlap, near-duplicates) and reports a summary at the end of
    cross-validation.

    Parameters
    ----------
    data : tuple
        Full dataset as (X, y) or (X, y, groups)
    groups : array-like, optional
        Group identifiers (e.g., patient IDs)
    timestamps : array-like, optional
        Temporal information
    coordinates : array-like, optional
        Spatial coordinates
    verbose : int
        Verbosity level (0=silent, 1=summary, 2=per-fold)
    """

    def __init__(
        self,
        data,
        groups=None,
        timestamps=None,
        coordinates=None,
        verbose: int = 1,
    ):
        self.data = data
        self.X = data[0]
        self.y = data[1] if len(data) > 1 else None
        self.groups = groups
        self.timestamps = timestamps
        self.coordinates = coordinates
        self.verbose = verbose
        self.fold_reports: List[Any] = []
        self.summary_report: Optional[Dict[str, Any]] = None

    def _get_checker(self):
        """Lazy-import DataLeakageChecker to avoid circular deps."""
        from ..checkers.leakage import DataLeakageChecker

        return DataLeakageChecker(verbose=False)

    def _safe_slice(self, arr, idx):
        """Slice an array-like by indices, returning None if arr is None."""
        if arr is None:
            return None
        arr = np.asarray(arr)
        return arr[idx]

    def on_fold_start(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> None:
        """Run leakage checks on the current fold split."""
        checker = self._get_checker()

        X_train = self.X[train_idx]
        X_val = self.X[val_idx]
        y_train = (
            self._safe_slice(self.y, train_idx)
            if self.y is not None
            else None
        )
        y_val = (
            self._safe_slice(self.y, val_idx)
            if self.y is not None
            else None
        )

        report = checker.check_cv_splits(
            X_train=X_train,
            X_test=X_val,
            y_train=y_train,
            y_test=y_val,
            patient_ids_train=self._safe_slice(
                self.groups, train_idx
            ),
            patient_ids_test=self._safe_slice(
                self.groups, val_idx
            ),
            timestamps_train=self._safe_slice(
                self.timestamps, train_idx
            ),
            timestamps_test=self._safe_slice(
                self.timestamps, val_idx
            ),
            coordinates_train=self._safe_slice(
                self.coordinates, train_idx
            ),
            coordinates_test=self._safe_slice(
                self.coordinates, val_idx
            ),
        )

        self.fold_reports.append(report)

        if self.verbose >= 2 and report.has_leakage:
            print(
                f"  [LeakageDetection] Fold {fold_idx}: "
                f"leakage detected — severity={report.severity}, "
                f"types={report.leakage_types}"
            )

    def on_cv_end(self, all_results: List[Dict[str, Any]]) -> None:
        """Aggregate fold reports and print a summary."""
        folds_with_leakage = [
            i
            for i, r in enumerate(self.fold_reports)
            if r.has_leakage
        ]
        all_types: set = set()
        severity_order = [
            "none",
            "low",
            "medium",
            "high",
            "critical",
        ]
        worst_severity = "none"

        for report in self.fold_reports:
            if report.has_leakage:
                all_types.update(report.leakage_types)
                if severity_order.index(
                    report.severity
                ) > severity_order.index(worst_severity):
                    worst_severity = report.severity

        self.summary_report = {
            "folds_with_leakage": folds_with_leakage,
            "n_folds_with_leakage": len(folds_with_leakage),
            "n_folds_total": len(self.fold_reports),
            "worst_severity": worst_severity,
            "all_leakage_types": sorted(all_types),
        }

        if self.verbose >= 1:
            n_leak = len(folds_with_leakage)
            n_total = len(self.fold_reports)
            if n_leak == 0:
                print(
                    "[LeakageDetection] No data leakage "
                    f"detected across {n_total} folds."
                )
            else:
                print(
                    f"[LeakageDetection] Leakage found in "
                    f"{n_leak}/{n_total} folds | "
                    f"worst severity: {worst_severity} | "
                    f"types: {sorted(all_types)}"
                )


class RegulatoryComplianceLogger(CVCallback):
    """
    Specialized logger for regulatory compliance

    Ensures all necessary information for FDA/CE MDR compliance
    is properly logged and documented.
    """

    def __init__(
        self,
        output_dir: str,
        study_name: str,
        include_data_characteristics: bool = True,
        include_model_details: bool = True,
    ):
        """
        Initialize regulatory compliance logger

        Parameters:
            output_dir: Directory for compliance logs
            study_name: Name of the study/experiment
            include_data_characteristics: Log data distribution info
            include_model_details: Log model architecture details
        """
        self.output_dir = Path(output_dir)
        self.study_name = study_name
        self.include_data_characteristics = include_data_characteristics
        self.include_model_details = include_model_details

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compliance_log = {"study_name": study_name, "cv_method": None, "folds": []}

    def on_cv_start(self, n_splits: int) -> None:
        """Document CV methodology"""
        import datetime

        self.compliance_log["start_time"] = datetime.datetime.now().isoformat()
        self.compliance_log["n_splits"] = n_splits

    def on_fold_start(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
        """Document data split for each fold"""
        fold_log = {
            "fold_idx": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_val_ratio": len(train_idx) / len(val_idx),
        }

        self.compliance_log["folds"].append(fold_log)

    def on_fold_end(self, fold_idx: int, results: Dict[str, Any]) -> None:
        """Document fold results"""
        if fold_idx < len(self.compliance_log["folds"]):
            self.compliance_log["folds"][fold_idx]["results"] = results

    def on_cv_end(self, all_results: List[Dict[str, Any]]) -> None:
        """Generate compliance report"""
        import datetime

        self.compliance_log["end_time"] = datetime.datetime.now().isoformat()

        # Save detailed log
        log_path = self.output_dir / f"{self.study_name}_compliance_log.json"
        with open(log_path, "w") as f:
            json.dump(self.compliance_log, f, indent=2, default=str)

        print(f"\nRegulatory compliance log saved to {log_path}")
