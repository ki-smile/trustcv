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
        self, log_file: Optional[str] = None, metrics: Optional[List[str]] = None, verbose: int = 1
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
    ):
        """
        Parameters:
            labels: Array-like target variable used for splitting
            label_names: Optional mapping {raw_label: display_name}
            verbose: 0=silent, 1=summary per fold
            decimals: Number of decimal places for percentages
        """
        self.labels = labels
        self.label_names = label_names or {}
        self.verbose = verbose
        self.decimals = max(0, decimals)
        self._cache = {}

    def _slice_labels(self, indices: np.ndarray) -> np.ndarray:
        """Return labels for the given indices, handling pandas objects."""
        if indices is None or len(indices) == 0:
            return np.array([])
        if hasattr(self.labels, "iloc"):
            subset = self.labels.iloc[indices]
        else:
            try:
                subset = self.labels[indices]
            except Exception:
                subset = np.asarray(self.labels)[indices]
        return np.asarray(subset)

    def _format_distribution(self, indices: np.ndarray) -> str:
        """Format class counts and percentages for display."""
        key = hash(indices.tobytes())
        if key in self._cache:
            return self._cache[key]

        values = self._slice_labels(indices)
        if values.size == 0:
            summary = "n/a"
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
