"""
Universal cross-validation runner for all frameworks

Provides a unified interface for running cross-validation across
different ML frameworks while maintaining best practices and
regulatory compliance.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .base import CVResults, FrameworkAdapter, SklearnAdapter
from .callbacks import CVCallback, ProgressLogger


class UniversalCVRunner:
    """
    Framework-agnostic cross-validation runner

    Automatically detects the framework and runs appropriate cross-validation
    while ensuring best practices in model evaluation.
    """

    # Mapping of framework names to their adapters
    FRAMEWORK_ADAPTERS = {
        "sklearn": "SklearnAdapter",
        "pytorch": "PyTorchAdapter",
        "tensorflow": "TensorFlowAdapter",
        "keras": "TensorFlowAdapter",
        "monai": "MONAIAdapter",
        "jax": "JAXAdapter",
        "xgboost": "XGBoostAdapter",
        "lightgbm": "LightGBMAdapter",
        "catboost": "CatBoostAdapter",
    }

    def __init__(
        self,
        cv_splitter: Any,
        framework: str = "auto",
        adapter: Optional[FrameworkAdapter] = None,
        verbose: int = 1,
    ):
        """
        Initialize universal CV runner

        Parameters:
            cv_splitter: Cross-validation splitter from trustcv
            framework: Framework name or 'auto' for auto-detection
            adapter: Custom adapter (overrides framework parameter)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.cv_splitter = cv_splitter
        self.framework = framework
        self.adapter = adapter
        self.verbose = verbose

        # Add default progress logger if verbose
        self.default_callbacks = []
        if verbose > 0:
            self.default_callbacks.append(ProgressLogger(verbose=verbose))

    def detect_framework(self, model: Any) -> str:
        """
        Auto-detect framework from model type

        Parameters:
            model: Model object or class

        Returns:
            Framework name
        """
        model_type = str(type(model))
        model_module = model.__module__ if hasattr(model, "__module__") else ""

        # Check module name first (more reliable)
        if "sklearn" in model_module:
            return "sklearn"
        elif "torch" in model_module or "pytorch" in model_module:
            return "pytorch"
        elif "tensorflow" in model_module or "keras" in model_module:
            return "tensorflow"
        elif "monai" in model_module:
            return "monai"
        elif "jax" in model_module or "flax" in model_module:
            return "jax"
        elif "xgboost" in model_module:
            return "xgboost"
        elif "lightgbm" in model_module:
            return "lightgbm"
        elif "catboost" in model_module:
            return "catboost"

        # Fallback to type string checking
        if "sklearn" in model_type:
            return "sklearn"
        elif "torch" in model_type:
            return "pytorch"
        elif "tensorflow" in model_type or "keras" in model_type:
            return "tensorflow"
        elif "monai" in model_type:
            return "monai"
        elif "jax" in model_type:
            return "jax"

        # Default to sklearn if cannot detect
        warnings.warn(
            f"Could not detect framework for model type {model_type}. "
            "Defaulting to sklearn adapter. Specify framework explicitly if needed."
        )
        return "sklearn"

    def get_adapter(self, framework: str) -> FrameworkAdapter:
        """
        Get or create adapter for specified framework

        Parameters:
            framework: Framework name

        Returns:
            Framework adapter instance
        """
        if framework not in self.FRAMEWORK_ADAPTERS:
            raise ValueError(
                f"Unknown framework: {framework}. "
                f"Supported frameworks: {list(self.FRAMEWORK_ADAPTERS.keys())}"
            )

        adapter_name = self.FRAMEWORK_ADAPTERS[framework]

        # Try to import the adapter
        if framework == "sklearn":
            return SklearnAdapter()
        else:
            try:
                from ..frameworks import get_adapter

                adapter_class = get_adapter(framework)
                return adapter_class()
            except (ImportError, ValueError) as e:
                raise ImportError(
                    f"Could not load adapter for {framework}. "
                    f"Make sure {framework} is installed. Error: {e}"
                )

    def run(
        self,
        model: Union[Any, Callable],
        data: Any,
        epochs: Optional[int] = None,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        metrics: Optional[List[str]] = None,
        callbacks: Optional[List[CVCallback]] = None,
        groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> CVResults:
        """
        Run cross-validation with automatic framework detection

        Parameters:
            model: Model instance or function that returns model
            data: Data in framework-appropriate format
            epochs: Number of epochs (for neural networks)
            optimizer: Optimizer (framework-specific)
            loss_fn: Loss function (framework-specific)
            metrics: List of metrics to compute
            callbacks: List of callbacks
            groups: Group labels for grouped CV
            **kwargs: Additional framework-specific parameters

        Returns:
            CVResults object with scores and models
        """
        # Combine default and user callbacks
        all_callbacks = self.default_callbacks + (callbacks or [])

        # Detect or set framework
        if self.adapter is None:
            if self.framework == "auto":
                # Get first model instance for detection
                if callable(model):
                    test_model = model()
                    self.framework = self.detect_framework(test_model)
                    del test_model  # Clean up
                else:
                    self.framework = self.detect_framework(model)

            self.adapter = self.get_adapter(self.framework)

        if self.verbose >= 1:
            print(f"Using {self.framework} adapter for cross-validation")

        # Prepare results storage
        all_scores = []
        all_models = []
        all_predictions = []
        all_probabilities = []
        fold_sizes = []
        all_indices = []

        # Get number of samples and optional groups from data tuple
        groups_from_data = None
        X_data = None
        y_data = None
        if isinstance(data, tuple):
            if len(data) == 2:
                X, y = data
            elif len(data) == 3:
                X, y, groups_from_data = data
            else:
                raise ValueError("data tuple must be (X, y) or (X, y, groups)")
            n_samples = len(X)
            X_data = X
            y_data = y
            # If caller did not pass groups kwarg, use groups from data tuple
            if groups is None and groups_from_data is not None:
                groups = groups_from_data
        elif hasattr(data, "__len__"):
            n_samples = len(data)
        else:
            # For generators or tf.data.Dataset
            n_samples = None

        # Get number of splits
        n_splits = self.cv_splitter.get_n_splits()

        # Trigger CV start callbacks
        for callback in all_callbacks:
            if hasattr(callback, "on_cv_start"):
                callback.on_cv_start(n_splits)

        # Cross-validation loop
        fold_idx = 0
        split_source = range(n_samples) if n_samples is not None else data
        if y_data is not None:
            split_iter = self.cv_splitter.split(
                split_source,
                y=y_data,
                groups=groups,
                **kwargs,
            )
        else:
            split_iter = self.cv_splitter.split(
                split_source,
                groups=groups,
                **kwargs,
            )

        for split_indices in split_iter:

            train_idx, val_idx = split_indices

            # Trigger fold start callbacks
            for callback in all_callbacks:
                if hasattr(callback, "on_fold_start"):
                    callback.on_fold_start(fold_idx, train_idx, val_idx)

            # Get or create model for this fold
            if callable(model):
                fold_model = model()
            else:
                fold_model = self.adapter.clone_model(model)

            # Create data splits (support (X,y) or (X,y,groups) tuples)
            data_for_adapter = data
            if isinstance(data, tuple) and len(data) >= 2:
                # Always pass only (X, y) to framework adapters by default
                data_for_adapter = (X, y)
            train_data, val_data = self.adapter.create_data_splits(
                data_for_adapter, train_idx, val_idx
            )

            # Framework-specific training
            if self.framework in ["pytorch", "tensorflow", "monai", "jax"]:
                # Neural network training with epochs
                if epochs is None:
                    epochs = 10
                    warnings.warn(f"No epochs specified for {self.framework}. Using default: 10")

                fold_history = []
                for epoch in range(epochs):
                    # Trigger epoch start callbacks
                    for callback in all_callbacks:
                        if hasattr(callback, "on_epoch_start"):
                            callback.on_epoch_start(epoch, fold_idx)

                    # Train epoch
                    train_metrics = self.adapter.train_epoch(
                        fold_model, train_data, optimizer, loss_fn, **kwargs
                    )

                    # Evaluate
                    val_metrics = self.adapter.evaluate(fold_model, val_data, loss_fn, metrics)

                    # Combine metrics
                    epoch_logs = {**train_metrics, **val_metrics}
                    fold_history.append(epoch_logs)

                    # Trigger epoch end callbacks
                    stop_training = False
                    for callback in all_callbacks:
                        if hasattr(callback, "on_epoch_end"):
                            result = callback.on_epoch_end(epoch, fold_idx, epoch_logs)
                            if result == "stop":
                                stop_training = True
                                break

                    if stop_training:
                        break

                # Final evaluation
                final_metrics = self.adapter.evaluate(fold_model, val_data, loss_fn, metrics)

            else:
                # Traditional ML training (single fit)
                train_metrics = self.adapter.train_epoch(fold_model, train_data, **kwargs)
                final_metrics = self.adapter.evaluate(fold_model, val_data, metrics=metrics)

            # Ensure predictions/probabilities are captured for downstream use
            predictions = final_metrics.get("predictions")
            if predictions is None:
                try:
                    predictions = self.adapter.get_predictions(fold_model, val_data)
                    final_metrics["predictions"] = predictions
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"Could not get predictions: {e}")
            if predictions is not None:
                all_predictions.append(predictions)

            probabilities = final_metrics.get("probabilities")
            if probabilities is not None:
                all_probabilities.append(probabilities)

            # Store results
            all_scores.append(final_metrics)
            all_models.append(fold_model)
            all_indices.append((train_idx, val_idx))
            fold_sizes.append(int(len(val_idx)))

            # Trigger fold end callbacks
            for callback in all_callbacks:
                if hasattr(callback, "on_fold_end"):
                    callback.on_fold_end(fold_idx, final_metrics)

            fold_idx += 1

        # Trigger CV end callbacks
        for callback in all_callbacks:
            if hasattr(callback, "on_cv_end"):
                callback.on_cv_end(all_scores)

        # Create results object
        results = CVResults(
            scores=all_scores,
            models=all_models,
            predictions=all_predictions if all_predictions else None,
            probabilities=all_probabilities if all_probabilities else None,
            indices=all_indices,
            metadata={
                "framework": self.framework,
                "n_splits": n_splits,
                "cv_method": self.cv_splitter.__class__.__name__,
                "fold_sizes": fold_sizes,
            },
        )

        if epochs is not None:
            results.metadata["epochs"] = epochs

        # Print summary if verbose
        if self.verbose >= 1:
            print("\n" + results.summary())

        return results

    def run_with_hyperparameter_tuning(
        self,
        model_fn: Callable,
        param_grid: Dict[str, List],
        data: Any,
        scoring: str = "accuracy",
        n_trials: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run cross-validation with hyperparameter tuning

        Parameters:
            model_fn: Function that takes params and returns model
            param_grid: Dictionary of parameter names and values to try
            data: Training data
            scoring: Scoring metric to optimize
            n_trials: Number of trials for random search
            **kwargs: Additional arguments for run()

        Returns:
            Dictionary with best parameters and results
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install it with: pip install optuna"
            )

        best_score = -np.inf
        best_params = None
        best_results = None

        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in param_grid.items():
                if isinstance(param_values, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(
                            param_name, param_values[0], param_values[1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_values[0], param_values[1]
                        )

            # Create model with sampled parameters
            model = model_fn(**params)

            # Run CV
            results = self.run(model, data, **kwargs)

            # Get score
            mean_scores = results.mean_score
            score = mean_scores.get(scoring, mean_scores.get("val_loss", 0))

            # Store if best
            nonlocal best_score, best_params, best_results
            if score > best_score:
                best_score = score
                best_params = params
                best_results = results

            return score

        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=self.verbose > 0)

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_results": best_results,
            "study": study,
        }
