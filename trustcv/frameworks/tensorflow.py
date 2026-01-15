"""
TensorFlow/Keras integration for trustcv

Provides seamless integration with TensorFlow/Keras models and tf.data
pipelines while maintaining trustcv's best practices.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.base import CVResults, FrameworkAdapter


class TensorFlowAdapter(FrameworkAdapter):
    """
    Adapter for TensorFlow/Keras models and datasets

    Handles TensorFlow-specific data loading, training, and evaluation
    while promoting best practices in ML.
    """

    def __init__(
        self,
        batch_size: int = 32,
        buffer_size: int = 1000,
        prefetch: int = 2,
        cache: bool = False,
        shuffle_train: bool = True,
        mixed_precision: bool = False,
    ):
        """
        Initialize TensorFlow adapter

        Parameters:
            batch_size: Batch size for datasets
            buffer_size: Buffer size for shuffling
            prefetch: Number of batches to prefetch
            cache: Whether to cache dataset in memory
            shuffle_train: Shuffle training data
            mixed_precision: Use mixed precision training
        """
        super().__init__(
            batch_size=batch_size,
            buffer_size=buffer_size,
            prefetch=prefetch,
            cache=cache,
            shuffle_train=shuffle_train,
        )

        try:
            import tensorflow as tf

            self.tf = tf
            self.keras = tf.keras

            # Configure mixed precision if requested
            if mixed_precision:
                from tensorflow.keras import mixed_precision

                policy = mixed_precision.Policy("mixed_float16")
                mixed_precision.set_global_policy(policy)

        except ImportError:
            raise ImportError(
                "TensorFlow is required for TensorFlowAdapter. "
                "Install it with: pip install tensorflow"
            )

    def create_data_splits(
        self, data: Any, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> Tuple[Any, Any]:
        """
        Create TensorFlow datasets from indices

        Parameters:
            data: tf.data.Dataset, tuple of (X, y) arrays, or data dict
            train_idx: Training indices
            val_idx: Validation indices

        Returns:
            train_dataset: tf.data.Dataset for training
            val_dataset: tf.data.Dataset for validation
        """
        # Handle different data formats
        if isinstance(data, tuple):
            X, y = data

            # Convert to TensorFlow tensors if needed
            if not isinstance(X, self.tf.Tensor):
                X = self.tf.constant(X, dtype=self.tf.float32)
            if not isinstance(y, self.tf.Tensor):
                y = self.tf.constant(y, dtype=self.tf.float32)

            # Create datasets from tensor slices
            train_dataset = self.tf.data.Dataset.from_tensor_slices(
                (self.tf.gather(X, train_idx), self.tf.gather(y, train_idx))
            )
            val_dataset = self.tf.data.Dataset.from_tensor_slices(
                (self.tf.gather(X, val_idx), self.tf.gather(y, val_idx))
            )

        elif isinstance(data, self.tf.data.Dataset):
            # Handle tf.data.Dataset using index-based filtering
            # Convert indices to tensors for efficient lookup
            train_idx_tensor = self.tf.constant(train_idx, dtype=self.tf.int64)
            val_idx_tensor = self.tf.constant(val_idx, dtype=self.tf.int64)

            # Create lookup tables for O(1) membership checking
            train_table = self.tf.lookup.StaticHashTable(
                self.tf.lookup.KeyValueTensorInitializer(
                    train_idx_tensor,
                    self.tf.ones_like(train_idx_tensor, dtype=self.tf.int64)
                ),
                default_value=0
            )
            val_table = self.tf.lookup.StaticHashTable(
                self.tf.lookup.KeyValueTensorInitializer(
                    val_idx_tensor,
                    self.tf.ones_like(val_idx_tensor, dtype=self.tf.int64)
                ),
                default_value=0
            )

            # Filter datasets lazily without loading all data into memory
            indexed_data = data.enumerate()

            train_dataset = indexed_data.filter(
                lambda idx, _: train_table.lookup(idx) == 1
            ).map(lambda _, item: item)

            val_dataset = indexed_data.filter(
                lambda idx, _: val_table.lookup(idx) == 1
            ).map(lambda _, item: item)

        else:
            raise ValueError("Data must be a tuple of (X, y) or a tf.data.Dataset")

        # Apply data pipeline optimizations
        if self.config["shuffle_train"]:
            train_dataset = train_dataset.shuffle(self.config["buffer_size"])

        train_dataset = train_dataset.batch(self.config["batch_size"])
        val_dataset = val_dataset.batch(self.config["batch_size"])

        if self.config["cache"]:
            train_dataset = train_dataset.cache()
            val_dataset = val_dataset.cache()

        train_dataset = train_dataset.prefetch(self.config["prefetch"])
        val_dataset = val_dataset.prefetch(self.config["prefetch"])

        return train_dataset, val_dataset

    def train_epoch(
        self,
        model: Any,
        train_data: Any,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        metrics: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Train TensorFlow/Keras model for one epoch

        Parameters:
            model: Keras model
            train_data: tf.data.Dataset
            optimizer: Optimizer (used if model not compiled)
            loss_fn: Loss function (used if model not compiled)
            metrics: List of metrics
            **kwargs: Additional fit parameters

        Returns:
            Dictionary of training metrics
        """
        # Check if model is compiled
        if not model.compiled:
            if optimizer is None:
                optimizer = self.keras.optimizers.Adam()
            if loss_fn is None:
                loss_fn = self.keras.losses.BinaryCrossentropy()
            if metrics is None:
                metrics = ["accuracy"]

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        # Train for one epoch
        history = model.fit(train_data, epochs=1, verbose=0, **kwargs)

        # Extract metrics
        train_metrics = {}
        for key, value in history.history.items():
            if not key.startswith("val_"):
                train_metrics[f"train_{key}"] = value[0]

        return train_metrics

    def evaluate(
        self, model: Any, val_data: Any, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate TensorFlow/Keras model

        Parameters:
            model: Keras model
            val_data: tf.data.Dataset
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate model
        results = model.evaluate(val_data, verbose=0, return_dict=True)

        # Format metrics
        eval_metrics = {}
        for key, value in results.items():
            eval_metrics[f"val_{key}"] = value

        # Get predictions if needed
        predictions = model.predict(val_data, verbose=0)
        eval_metrics["predictions"] = predictions

        return eval_metrics

    def clone_model(self, model: Any) -> Any:
        """
        Clone Keras model with fresh weights

        Parameters:
            model: Original model

        Returns:
            Cloned model with reinitialized weights
        """
        # Clone model architecture
        model_config = model.get_config()
        cloned = self.keras.Model.from_config(model_config)

        # Compile with same settings if original was compiled
        if model.compiled:
            cloned.compile(
                optimizer=model.optimizer.__class__(**model.optimizer.get_config()),
                loss=model.loss,
                metrics=model.compiled_metrics._user_metrics,
            )

        return cloned

    def get_predictions(self, model: Any, data: Any) -> np.ndarray:
        """
        Get predictions from Keras model

        Parameters:
            model: Trained model
            data: Data to predict on

        Returns:
            Predictions as numpy array
        """
        if isinstance(data, tuple):
            X, _ = data
            predictions = model.predict(X, verbose=0)
        elif isinstance(data, self.tf.data.Dataset):
            predictions = model.predict(data, verbose=0)
        else:
            predictions = model.predict(data, verbose=0)

        return predictions

    def save_model(self, model: Any, path: str) -> None:
        """Save Keras model"""
        model.save(path)

    def load_model(self, path: str) -> Any:
        """Load Keras model"""
        return self.keras.models.load_model(path)


class KerasCVRunner:
    """
    High-level cross-validation runner for Keras models

    Simplifies running cross-validation with Keras models while
    ensuring best practices and regulatory compliance.
    """

    def __init__(
        self,
        model_fn: Callable,
        cv_splitter: Any,
        adapter: Optional[TensorFlowAdapter] = None,
        compile_kwargs: Optional[Dict] = None,
        store_models: bool = False,
    ):
        """
        Initialize Keras CV runner

        Parameters:
            model_fn: Function that returns a new model instance
            cv_splitter: Cross-validation splitter from trustcv
            adapter: TensorFlow adapter (creates default if None)
            compile_kwargs: Arguments for model.compile()
            store_models: Whether to store trained models (can use significant memory)
        """
        self.model_fn = model_fn
        self.cv_splitter = cv_splitter
        self.adapter = adapter or TensorFlowAdapter()
        self.compile_kwargs = compile_kwargs or {}
        self.store_models = store_models

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        validation_split: float = 0.0,
        callbacks: Optional[List[Any]] = None,
        groups: Optional[np.ndarray] = None,
        **fit_kwargs,
    ) -> CVResults:
        """
        Run cross-validation with Keras model

        Parameters:
            X: Feature array
            y: Target array
            epochs: Number of training epochs per fold
            validation_split: Validation split within training (0.0 to use CV split)
            callbacks: List of callbacks (both Keras and trustcv)
            groups: Group labels for grouped CV
            **fit_kwargs: Additional arguments for model.fit()

        Returns:
            CVResults object with scores and models
        """
        callbacks = callbacks or []
        all_scores = []
        all_models = []
        all_predictions = []
        all_indices = []
        all_histories = []

        n_samples = len(X)
        n_splits = self.cv_splitter.get_n_splits()

        # Separate Keras callbacks from trustcv callbacks
        keras_callbacks = []
        cv_callbacks = []

        for callback in callbacks:
            if hasattr(callback, "set_model"):  # Keras callback
                keras_callbacks.append(callback)
            else:  # trustcv callback
                cv_callbacks.append(callback)

        # Trigger CV start callbacks
        for callback in cv_callbacks:
            callback.on_cv_start(n_splits)

        # Cross-validation loop
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv_splitter.split(X, y, groups=groups)
        ):
            print(f"\nFold {fold_idx + 1}/{n_splits}")
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

            # Trigger fold start callbacks
            for callback in cv_callbacks:
                callback.on_fold_start(fold_idx, train_idx, val_idx)

            # Create new model for this fold
            model = self.model_fn()

            # Compile model if kwargs provided
            if self.compile_kwargs:
                model.compile(**self.compile_kwargs)

            # Create data splits
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Add early stopping if not already present
            early_stop_exists = any(
                isinstance(cb, self.adapter.keras.callbacks.EarlyStopping) for cb in keras_callbacks
            )

            fold_callbacks = keras_callbacks.copy()
            if not early_stop_exists:
                fold_callbacks.append(
                    self.adapter.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    )
                )

            # Train model
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if validation_split == 0.0 else None,
                validation_split=validation_split if validation_split > 0.0 else 0.0,
                epochs=epochs,
                callbacks=fold_callbacks,
                verbose=1,
                **fit_kwargs,
            )

            # Evaluate on validation set
            val_results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)

            # Get predictions
            val_predictions = model.predict(X_val, verbose=0)

            # Format results
            fold_results = {f"val_{k}": v for k, v in val_results.items()}
            fold_results["predictions"] = val_predictions

            # Store results
            all_scores.append(fold_results)
            if self.store_models:
                all_models.append(model)
            all_predictions.append(val_predictions)
            all_indices.append((train_idx, val_idx))
            all_histories.append(history.history)

            # Trigger fold end callbacks
            for callback in cv_callbacks:
                callback.on_fold_end(fold_idx, fold_results)

            # Memory cleanup between folds
            if not self.store_models:
                del model
            import gc
            gc.collect()
            # Clear Keras session to free memory
            self.adapter.keras.backend.clear_session()

        # Trigger CV end callbacks
        for callback in cv_callbacks:
            callback.on_cv_end(all_scores)

        # Return results
        return CVResults(
            scores=all_scores,
            models=all_models if self.store_models else None,
            predictions=all_predictions,
            indices=all_indices,
            metadata={"framework": "tensorflow", "epochs": epochs, "histories": all_histories},
        )
