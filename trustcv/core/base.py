"""
Base classes for framework-agnostic cross-validation

These abstractions allow trustcv to work with any ML framework while
promoting best practices in model evaluation and regulatory compliance.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CVResults:
    """
    Standardized results container for cross-validation

    Attributes:
        scores: Per-fold metric dicts, e.g., [{"score": 0.91, ...}, ...]
        models: Trained models from each fold (optional)
        predictions: Discrete predictions per fold (if available)
        probabilities: Prediction probabilities per fold (classification)
        indices: Train/test indices for each fold
        metadata: Additional info (framework, n_splits, cv_method, ...)
    """

    scores: List[Dict[str, Any]]
    models: Optional[List[Any]] = None
    predictions: Optional[List[np.ndarray]] = None
    probabilities: Optional[List[np.ndarray]] = None
    indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----- helpers -----
    def _to_float_list(self, value) -> List[float]:
        import numpy as _np

        if isinstance(value, dict):
            for key in ("folds", "values", "fold_scores"):
                if key in value:
                    try:
                        arr = _np.asarray(value[key], dtype=float)
                        return arr.ravel().tolist()
                    except Exception:
                        return []
            mean = value.get("mean") if isinstance(value, dict) else None
            if _np.isscalar(mean):
                return [float(mean)]
            return []
        if _np.isscalar(value):
            try:
                return [float(value)]
            except Exception:
                return []
        try:
            arr = _np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return [float(arr)]
            if arr.ndim == 1:
                return arr.tolist()
            return []
        except Exception:
            return []

    def _collect_metric_series(self) -> Dict[str, Dict[str, List[float]]]:
        """Collect per-metric values across folds along with weights when available."""
        series: Dict[str, Dict[str, List[float]]] = {}
        weights = self.metadata.get("fold_sizes")
        use_weights = isinstance(weights, (list, tuple)) and len(weights) == len(self.scores or [])
        for fold_idx, fold in enumerate(self.scores or []):
            weight = None
            if use_weights:
                try:
                    weight = float(weights[fold_idx])
                except Exception:
                    weight = None
            if not isinstance(fold, dict):
                continue
            for k, v in fold.items():
                # Skip large arrays like predictions/probabilities
                if k.lower() in ("predictions", "probabilities", "y_pred", "y_proba"):
                    continue
                vals = self._to_float_list(v)
                if not vals:
                    continue
                entry = series.setdefault(k, {"values": [], "weights": []})
                entry["values"].extend(vals)
                if weight is not None:
                    entry["weights"].extend([weight] * len(vals))
                else:
                    entry["weights"].extend([None] * len(vals))
        return series

    def _metric_stats(self, name: str) -> Tuple[float, float]:
        import numpy as _np

        series = self._collect_metric_series()
        entry = series.get(name)
        if not entry:
            return _np.nan, _np.nan
        vals_list = entry.get("values", [])
        if len(vals_list) == 0:
            return _np.nan, _np.nan
        vals_arr = _np.asarray(vals_list, dtype=float)
        weights_raw = entry.get("weights", [])
        numeric_mask = [isinstance(w, (int, float)) for w in weights_raw]
        has_weights = bool(weights_raw) and all(numeric_mask) and len(weights_raw) == len(vals_arr)
        if has_weights:
            weights_arr = _np.asarray(weights_raw, dtype=float)
            total_weight = _np.sum(weights_arr)
            if total_weight > 0:
                mean = float(_np.average(vals_arr, weights=weights_arr))
                variance = _np.average((vals_arr - mean) ** 2, weights=weights_arr)
                std = float(_np.sqrt(variance))
                return mean, std
        mean = float(_np.mean(vals_arr))
        std = float(_np.std(vals_arr, ddof=1)) if len(vals_arr) > 1 else 0.0
        return mean, std

    # ----- public API -----
    @property
    def metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregated metrics per name: {name: {"mean": float, "std": float}}.
        """
        out: Dict[str, Dict[str, float]] = {}
        series = self._collect_metric_series()
        for name in series.keys():
            m, s = self._metric_stats(name)
            if m == m:  # not NaN
                out[name] = {"mean": m, "std": s}
        return out

    @property
    def mean_score(self) -> Dict[str, float]:
        """
        Mean value per metric: {name: mean}.
        Backward-compat for callers expecting a mapping.
        """
        return {name: vals["mean"] for name, vals in self.metrics.items()}

    @property
    def std_scores(self) -> Dict[str, float]:
        """
        Std value per metric: {name: std}.
        """
        return {name: vals["std"] for name, vals in self.metrics.items()}

    def summary(self) -> str:
        import numpy as _np

        lines = ["Cross-Validation Results Summary:"]
        primary = self.metadata.get("primary_metric", "score")
        m, s = self._metric_stats(primary)
        if _np.isfinite(m):
            lines.append(f"  {primary}: {m:.4f} (+/- {s:.4f})")
        else:
            lines.append(f"  {primary}: n/a")

        for name in sorted(self.metrics.keys()):
            if name == primary:
                continue
            mm, ss = self._metric_stats(name)
            if _np.isfinite(mm):
                lines.append(f"  {name}: {mm:.4f} (+/- {ss:.4f})")
        return "\n".join(lines)


class CVSplitter(ABC):
    """
    Abstract base class for all CV splitters

    This class defines the interface that all cross-validation splitters
    must implement, ensuring compatibility across different frameworks.
    """

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Get the number of splits/folds

        Parameters:
            X: Features array (optional for some splitters)
            y: Target array (optional for some splitters)
            groups: Group labels for grouped CV (optional)

        Returns:
            Number of cross-validation folds
        """
        pass

    @abstractmethod
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for cross-validation

        Parameters:
            X: Features array
            y: Target array (optional)
            groups: Group labels for grouped CV (optional)

        Yields:
            train_indices: Indices for training set
            test_indices: Indices for test/validation set
        """
        pass

    def validate_split(
        self, train_idx: np.ndarray, test_idx: np.ndarray, groups: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:
        """
        Validate that a split meets best practices

        Parameters:
            train_idx: Training indices
            test_idx: Test indices
            groups: Group labels (optional)

        Returns:
            Dictionary with validation results
        """
        validations = {
            "no_overlap": len(np.intersect1d(train_idx, test_idx)) == 0,
            "train_not_empty": len(train_idx) > 0,
            "test_not_empty": len(test_idx) > 0,
        }

        if groups is not None:
            train_groups = np.unique(groups[train_idx])
            test_groups = np.unique(groups[test_idx])
            validations["no_group_leakage"] = len(np.intersect1d(train_groups, test_groups)) == 0

        return validations


class FrameworkAdapter(ABC):
    """
    Abstract adapter for framework-specific implementations

    This class provides the interface for adapting trustcv's splitting
    strategies to different ML frameworks' data handling and training loops.
    """

    def __init__(self, **kwargs):
        """
        Initialize adapter with framework-specific parameters

        Parameters:
            **kwargs: Framework-specific configuration
        """
        self.config = kwargs

    @abstractmethod
    def create_data_splits(
        self, data: Any, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> Tuple[Any, Any]:
        """
        Create framework-specific data loaders/datasets from indices

        Parameters:
            data: Original dataset (format depends on framework)
            train_idx: Indices for training data
            val_idx: Indices for validation data

        Returns:
            train_data: Framework-specific training data structure
            val_data: Framework-specific validation data structure
        """
        pass

    @abstractmethod
    def train_epoch(
        self, model: Any, train_data: Any, optimizer: Optional[Any] = None, **kwargs
    ) -> Dict[str, float]:
        """
        Train model for one epoch

        Parameters:
            model: Framework-specific model
            train_data: Training data in framework-specific format
            optimizer: Optimizer (optional, framework-specific)
            **kwargs: Additional training parameters

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def evaluate(
        self, model: Any, val_data: Any, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data

        Parameters:
            model: Framework-specific model
            val_data: Validation data in framework-specific format
            metrics: List of metrics to compute (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    def clone_model(self, model: Any) -> Any:
        """
        Create a fresh copy of the model

        Parameters:
            model: Original model

        Returns:
            Cloned model with reset weights
        """
        # Default implementation - frameworks can override
        warnings.warn(
            "Using default model cloning which may not work for all frameworks. "
            "Consider implementing framework-specific cloning.",
            UserWarning,
        )
        return model

    def get_predictions(self, model: Any, data: Any) -> np.ndarray:
        """
        Get predictions from model

        Parameters:
            model: Trained model
            data: Data to predict on

        Returns:
            Predictions as numpy array
        """
        raise NotImplementedError("Prediction extraction must be implemented by framework adapter")

    def save_model(self, model: Any, path: str) -> None:
        """
        Save model to disk

        Parameters:
            model: Model to save
            path: Path to save model
        """
        raise NotImplementedError("Model saving must be implemented by framework adapter")

    def load_model(self, path: str) -> Any:
        """
        Load model from disk

        Parameters:
            path: Path to load model from

        Returns:
            Loaded model
        """
        raise NotImplementedError("Model loading must be implemented by framework adapter")


class SklearnAdapter(FrameworkAdapter):
    """
    Adapter for scikit-learn models (backward compatibility)
    """

    def create_data_splits(
        self, data: Tuple[np.ndarray, np.ndarray], train_idx: np.ndarray, val_idx: np.ndarray
    ) -> Tuple[Any, Any]:
        """Create train/validation splits for sklearn.

        Accepts data as (X, y) or (X, y, groups); extra items are ignored.
        """
        if isinstance(data, tuple):
            if len(data) >= 2:
                X, y = data[0], data[1]
            else:
                raise ValueError("data must be a tuple (X, y) or (X, y, groups)")
        else:
            raise ValueError("data must be a tuple (X, y) for sklearn adapter")

        # Robust row slicing: for pandas objects use .iloc with positional indices,
        # for numpy arrays fall back to standard indexing.
        def _slice_rows(arr, idx):
            try:
                if hasattr(arr, "iloc"):
                    return arr.iloc[idx]
                return arr[idx]
            except Exception:
                # Last-resort conversion
                import numpy as _np

                a = _np.asarray(arr)
                return a[idx]

        X_tr, y_tr = _slice_rows(X, train_idx), _slice_rows(y, train_idx)
        X_te, y_te = _slice_rows(X, val_idx), _slice_rows(y, val_idx)
        return (X_tr, y_tr), (X_te, y_te)

    def train_epoch(
        self,
        model: Any,
        train_data: Tuple[np.ndarray, np.ndarray],
        optimizer: None = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Train sklearn model (single fit call)"""
        X_train, y_train = train_data
        model.fit(X_train, y_train, **kwargs)

        # Return training score if model supports it
        train_metrics = {}
        if hasattr(model, "score"):
            train_metrics["train_score"] = model.score(X_train, y_train)

        return train_metrics

    def evaluate(
        self,
        model: Any,
        val_data: Tuple[np.ndarray, np.ndarray],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate sklearn model"""
        X_val, y_val = val_data

        eval_metrics: Dict[str, float] = {}

        # Default score (sklearn's estimator.score)
        if hasattr(model, "score"):
            try:
                eval_metrics["score"] = float(model.score(X_val, y_val))
            except Exception:
                pass

        # Get predictions
        y_pred = None
        if hasattr(model, "predict"):
            try:
                y_pred = model.predict(X_val)
                eval_metrics["predictions"] = y_pred
            except Exception:
                y_pred = None

        # Get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_val)
                eval_metrics["probabilities"] = y_proba
            except Exception:
                y_proba = None

        # Compute common classification metrics when labels are provided
        try:
            import numpy as _np
            from sklearn.metrics import (
                accuracy_score,
                balanced_accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            if y_pred is not None:
                # Some regressors return floats; only compute cls metrics for discrete labels
                # Treat binary or integer classes as classification
                if (
                    _np.issubdtype(_np.asarray(y_val).dtype, _np.integer)
                    or _np.array_equal(_np.unique(y_val), [0, 1])
                    or len(_np.unique(y_val)) <= 10
                ):
                    eval_metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
                    # F1 (binary only by default)
                    try:
                        eval_metrics["f1"] = float(f1_score(y_val, y_pred))
                    except Exception:
                        pass
                    try:
                        eval_metrics["precision"] = float(precision_score(y_val, y_pred))
                    except Exception:
                        pass
                    try:
                        eval_metrics["recall"] = float(recall_score(y_val, y_pred))
                    except Exception:
                        pass
                    try:
                        eval_metrics["balanced_accuracy"] = float(
                            balanced_accuracy_score(y_val, y_pred)
                        )
                    except Exception:
                        pass

            # AUC if probabilities available for binary classification
            if y_proba is not None:
                ys = _np.asarray(y_proba)
                try:
                    if ys.ndim == 2 and ys.shape[1] > 1:
                        ys = ys[:, 1]
                    eval_metrics["roc_auc"] = float(roc_auc_score(y_val, ys))
                except Exception:
                    pass
        except Exception:
            # Metrics import failure or non-applicable
            pass

        return eval_metrics

    def clone_model(self, model: Any) -> Any:
        """Clone sklearn model"""
        from sklearn.base import clone

        return clone(model)

    def get_predictions(self, model: Any, data: Any) -> np.ndarray:
        """Get predictions from sklearn model"""
        if isinstance(data, tuple):
            X, _ = data
        else:
            X = data

        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        elif hasattr(model, "predict"):
            return model.predict(X)
        else:
            raise ValueError("Model doesn't have predict or predict_proba method")

    def save_model(self, model: Any, path: str) -> None:
        """Save sklearn model using joblib"""
        import joblib

        joblib.dump(model, path)

    def load_model(self, path: str) -> Any:
        """Load sklearn model using joblib"""
        import joblib

        return joblib.load(path)
