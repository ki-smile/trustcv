"""
JAX/Flax integration for trustcv

Provides seamless integration with JAX/Flax models while maintaining
trustcv's best practices for cross-validation.

JAX uses a functional programming paradigm where:
- Models are stateless functions
- State (parameters) is managed explicitly
- Randomness uses PRNG keys
- JIT compilation provides performance benefits

Requirements:
    pip install jax jaxlib flax optax
"""

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.base import CVResults, FrameworkAdapter
from ..core.selection import (
    best_weights_to_restore,
    fold_selection_split,
    run_epoch_callbacks,
    save_checkpoints,
)


class JAXAdapter(FrameworkAdapter):
    """
    Adapter for JAX/Flax models

    Handles JAX-specific data loading, training, and evaluation
    while promoting best practices in ML.

    JAX/Flax uses a functional paradigm where model parameters are
    explicitly managed. This adapter works with Flax's TrainState
    or raw parameter pytrees.
    """

    def __init__(
        self,
        batch_size: int = 32,
        shuffle_train: bool = True,
        seed: int = 0,
        use_jit: bool = True,
    ):
        """
        Initialize JAX adapter

        Parameters:
            batch_size: Batch size for training
            shuffle_train: Shuffle training data each epoch
            seed: Random seed for JAX PRNG
            use_jit: Whether to JIT compile train/eval functions
        """
        super().__init__(
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            seed=seed,
            use_jit=use_jit,
        )

        try:
            import jax
            import jax.numpy as jnp

            self.jax = jax
            self.jnp = jnp
            self.key = jax.random.PRNGKey(seed)

        except ImportError:
            raise ImportError(
                "JAX is required for JAXAdapter. "
                "Install it with: pip install jax jaxlib"
            )

        try:
            import flax
            import flax.linen as nn
            from flax.training import train_state

            self.flax = flax
            self.nn = nn
            self.train_state = train_state
        except ImportError:
            self.flax = None
            self.nn = None
            self.train_state = None
            warnings.warn(
                "Flax not installed. Some features may be limited. "
                "Install with: pip install flax"
            )

        try:
            import optax

            self.optax = optax
        except ImportError:
            self.optax = None
            warnings.warn(
                "Optax not installed. You'll need to provide custom optimizers. "
                "Install with: pip install optax"
            )

    def _get_key(self) -> Any:
        """Get a new PRNG key and update internal state."""
        self.key, subkey = self.jax.random.split(self.key)
        return subkey

    def _batch_generator(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> Any:
        """
        Generate batches from data.

        Parameters:
            X: Feature array
            y: Target array
            shuffle: Whether to shuffle data

        Yields:
            Batches of (X_batch, y_batch)
        """
        n_samples = len(X)
        batch_size = self.config["batch_size"]

        if shuffle:
            key = self._get_key()
            perm = self.jax.random.permutation(key, n_samples)
            X = X[perm]
            y = y[perm]

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            yield self.jnp.array(X[i:end]), self.jnp.array(y[i:end])

    def create_data_splits(
        self, data: Any, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Create train/validation splits from indices

        Parameters:
            data: Tuple of (X, y) arrays
            train_idx: Training indices
            val_idx: Validation indices

        Returns:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
        """
        if isinstance(data, tuple):
            X, y = data
        else:
            raise ValueError("Data must be a tuple of (X, y) for JAX adapter")

        # Convert indices to numpy if needed
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        return (X_train, y_train), (X_val, y_val)

    def train_epoch(
        self,
        model: Any,
        train_data: Tuple[np.ndarray, np.ndarray],
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        state: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train JAX/Flax model for one epoch

        Parameters:
            model: Flax module or callable (params, x) -> logits
            train_data: Tuple of (X, y) arrays
            optimizer: Optax optimizer (required if state is None)
            loss_fn: Loss function (params, x, y) -> loss
            state: Flax TrainState (optional, created if None)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training metrics and updated state
        """
        X_train, y_train = train_data

        # Create or use existing state
        if state is None:
            if self.train_state is None:
                raise ImportError("Flax is required for automatic state management")
            if optimizer is None:
                if self.optax is None:
                    raise ImportError("Optax is required for default optimizer")
                optimizer = self.optax.adam(1e-3)

            # Initialize model if it's a Flax module
            if hasattr(model, "init"):
                key = self._get_key()
                dummy_input = self.jnp.ones((1,) + X_train.shape[1:])
                variables = model.init(key, dummy_input)
                params = variables.get("params", variables)
            else:
                raise ValueError(
                    "Model must be a Flax module with init method, "
                    "or provide a pre-initialized state"
                )

            state = self.train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=optimizer
            )

        # Define train step with RNG handling for dropout
        def make_train_step(use_jit):
            def train_step(state, batch_x, batch_y, rng_key):
                dropout_key = self.jax.random.fold_in(rng_key, state.step)

                def loss_wrapper(params):
                    # Pass rngs for stochastic layers like Dropout
                    logits = state.apply_fn(
                        {"params": params},
                        batch_x,
                        training=True,
                        rngs={"dropout": dropout_key}
                    )
                    one_hot = self.jax.nn.one_hot(batch_y, logits.shape[-1])
                    return self.optax.softmax_cross_entropy(logits, one_hot).mean()

                if loss_fn is not None:
                    # Use custom loss function
                    def custom_loss_wrapper(params):
                        return loss_fn(params, batch_x, batch_y)
                    loss, grads = self.jax.value_and_grad(custom_loss_wrapper)(state.params)
                else:
                    loss, grads = self.jax.value_and_grad(loss_wrapper)(state.params)

                state = state.apply_gradients(grads=grads)
                return state, loss

            if use_jit:
                return self.jax.jit(train_step)
            return train_step

        train_step = make_train_step(self.config["use_jit"])

        # Training loop over batches
        total_loss = 0.0
        n_batches = 0
        train_rng = self._get_key()

        for batch_x, batch_y in self._batch_generator(
            X_train, y_train, shuffle=self.config["shuffle_train"]
        ):
            train_rng, step_rng = self.jax.random.split(train_rng)
            state, loss = train_step(state, batch_x, batch_y, step_rng)
            total_loss += float(loss)
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        return {
            "train_loss": avg_loss,
            "state": state,  # Return updated state
        }

    def evaluate(
        self,
        model: Any,
        val_data: Tuple[np.ndarray, np.ndarray],
        state: Optional[Any] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate JAX/Flax model

        Parameters:
            model: Flax module or state with apply_fn
            val_data: Tuple of (X, y) arrays
            state: Flax TrainState containing params
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation metrics
        """
        X_val, y_val = val_data

        # Get apply function and params
        if state is not None:
            apply_fn = state.apply_fn
            params = state.params
        elif hasattr(model, "apply"):
            apply_fn = model.apply
            # Need params from somewhere
            raise ValueError("Must provide state with params for evaluation")
        else:
            raise ValueError("Model must have apply method or provide state")

        # Compute predictions (training=False disables dropout)
        def make_predict_fn(use_jit):
            def predict_batch(params, x):
                return apply_fn({"params": params}, x, training=False)
            if use_jit:
                return self.jax.jit(predict_batch)
            return predict_batch

        predict_batch = make_predict_fn(self.config["use_jit"])

        # Predict in batches to avoid memory issues
        all_logits = []
        for batch_x, _ in self._batch_generator(X_val, y_val, shuffle=False):
            logits = predict_batch(params, batch_x)
            all_logits.append(np.array(logits))

        all_logits = np.vstack(all_logits)
        predictions = np.argmax(all_logits, axis=-1)
        probabilities = self._softmax(all_logits)

        # Compute metrics
        y_val_np = np.asarray(y_val)
        accuracy = float(np.mean(predictions == y_val_np))

        # Compute loss
        one_hot = np.eye(all_logits.shape[-1])[y_val_np]
        loss = float(-np.mean(np.sum(one_hot * np.log(probabilities + 1e-8), axis=-1)))

        eval_metrics = {
            "val_loss": loss,
            "val_acc": accuracy,
            "predictions": predictions,
            "probabilities": probabilities,
            "targets": y_val_np,
        }

        return eval_metrics

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax in numpy."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def clone_model(self, model: Any) -> Any:
        """
        Clone JAX/Flax model

        For Flax, models are stateless - just return the same module.
        Parameters are reinitialized during training.

        Parameters:
            model: Flax module

        Returns:
            Same module (Flax modules are stateless)
        """
        # Flax modules are stateless, so we just return the same instance
        # Parameters will be reinitialized in train_epoch
        return model

    def get_predictions(
        self, model: Any, data: Any, state: Optional[Any] = None
    ) -> np.ndarray:
        """
        Get predictions from JAX/Flax model

        Parameters:
            model: Flax module
            data: Data to predict on (tuple or array)
            state: Flax TrainState containing params

        Returns:
            Predictions as numpy array
        """
        if isinstance(data, tuple):
            X, _ = data
        else:
            X = data

        if state is None:
            raise ValueError("Must provide state with trained params")

        @self.jax.jit if self.config["use_jit"] else lambda x: x
        def predict(params, x):
            return state.apply_fn({"params": params}, x)

        # Predict in batches
        all_preds = []
        batch_size = self.config["batch_size"]
        for i in range(0, len(X), batch_size):
            batch = self.jnp.array(X[i : i + batch_size])
            logits = predict(state.params, batch)
            probs = self.jax.nn.softmax(logits)
            all_preds.append(np.array(probs))

        return np.vstack(all_preds)

    def save_model(self, state: Any, path: str) -> None:
        """
        Save Flax TrainState

        Parameters:
            state: Flax TrainState to save
            path: Path to save to
        """
        try:
            from flax.training import checkpoints

            # Orbax needs absolute paths
            checkpoints.save_checkpoint(os.path.abspath(path), state, step=0, overwrite=True)
        except ImportError:
            # Fallback to pickle
            import pickle

            with open(path, "wb") as f:
                pickle.dump({"params": state.params, "step": state.step}, f)

    def load_model(self, path: str, state_template: Any) -> Any:
        """
        Load Flax TrainState

        Parameters:
            path: Path to load from
            state_template: Template TrainState with correct structure

        Returns:
            Restored TrainState
        """
        try:
            from flax.training import checkpoints

            return checkpoints.restore_checkpoint(os.path.abspath(path), state_template)
        except ImportError:
            import pickle

            with open(path, "rb") as f:
                data = pickle.load(f)
            return state_template.replace(params=data["params"])


class JAXCVRunner:
    """
    High-level cross-validation runner for JAX/Flax models

    Simplifies running cross-validation with JAX/Flax models while
    ensuring best practices.

    Example:
        ```python
        import flax.linen as nn
        from trustcv.frameworks.jax import JAXCVRunner, JAXAdapter
        from trustcv.splitters import StratifiedKFold

        class MLP(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(64)(x)
                x = nn.relu(x)
                x = nn.Dense(2)(x)
                return x

        runner = JAXCVRunner(
            model_fn=MLP,
            cv_splitter=StratifiedKFold(n_splits=5)
        )

        results = runner.run(X, y, epochs=10)
        ```
    """

    def __init__(
        self,
        model_fn: Callable,
        cv_splitter: Any,
        adapter: Optional[JAXAdapter] = None,
        store_models: bool = False,
    ):
        """
        Initialize JAX CV runner

        Parameters:
            model_fn: Function/class that returns a new Flax module
            cv_splitter: Cross-validation splitter from trustcv
            adapter: JAX adapter (creates default if None)
            store_models: Whether to store trained states (can use memory)
        """
        self.model_fn = model_fn
        self.cv_splitter = cv_splitter
        self.adapter = adapter or JAXAdapter()
        self.store_models = store_models

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        callbacks: Optional[List[Any]] = None,
        groups: Optional[np.ndarray] = None,
        inner_validation: float = 0.0,
        inner_validation_strategy: str = "auto",
        inner_random_state: Optional[int] = 0,
        report_heldout_per_epoch: bool = False,
    ) -> CVResults:
        """
        Run cross-validation with JAX/Flax model

        Parameters:
            X: Feature array
            y: Target array
            epochs: Number of training epochs per fold
            optimizer: Optax optimizer (default: Adam with lr=1e-3)
            loss_fn: Custom loss function
            callbacks: List of trustcv callbacks
            groups: Group labels for grouped CV
            inner_validation: Fraction of each training fold set aside for model
                selection. ``val_*`` metrics for EarlyStopping and ModelCheckpoint
                are computed on it; the held-out fold is scored once, after
                training. 0 trains for the full epochs with no ``val_*`` metrics.
                Restoring best weights restores the TrainState; non-parameter
                collections (BatchNorm statistics) are not handled.
            inner_validation_strategy: "auto" (group, then stratified, then random) or
                "last" (the last part of each training fold, for time-ordered data).
            inner_random_state: Seed for the inner split (offset by the fold index).
            report_heldout_per_epoch: Also record held-out loss per epoch in
                ``metadata["heldout_curves"]``, for plotting only. Callbacks never see it.

        Returns:
            CVResults object with scores and optionally states
        """
        if optimizer is None and self.adapter.optax is not None:
            optimizer = self.adapter.optax.adam(1e-3)

        callbacks = callbacks or []
        all_scores = []
        all_states = []
        all_predictions = []
        all_indices = []
        model_selection = []
        heldout_curves = []
        warned = {}

        n_samples = len(X)
        n_splits = self.cv_splitter.get_n_splits()

        # Trigger CV start callbacks
        for callback in callbacks:
            if hasattr(callback, "on_cv_start"):
                callback.on_cv_start(n_splits)

        # Cross-validation loop
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv_splitter.split(X, y, groups=groups)
        ):
            print(f"\nFold {fold_idx + 1}/{n_splits}")
            print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

            # Trigger fold start callbacks
            for callback in callbacks:
                if hasattr(callback, "on_fold_start"):
                    callback.on_fold_start(fold_idx, train_idx, val_idx)

            # Model selection uses only rows of the training fold
            fit_idx, select_idx, selection = fold_selection_split(
                train_idx, y, groups,
                inner_validation=inner_validation, strategy=inner_validation_strategy,
                random_state=inner_random_state, fold_idx=fold_idx, warned=warned,
            )
            model = self.model_fn()
            train_data, val_data = self.adapter.create_data_splits((X, y), fit_idx, val_idx)
            select_data = None
            if select_idx is not None:
                select_data = (X[select_idx], y[select_idx])

            state = None
            curve = []
            epochs_trained = 0
            for epoch in range(epochs):
                # Trigger epoch start callbacks
                for callback in callbacks:
                    if hasattr(callback, "on_epoch_start"):
                        callback.on_epoch_start(epoch, fold_idx)

                train_result = self.adapter.train_epoch(
                    model,
                    train_data,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    state=state,
                )
                state = train_result["state"]
                train_loss = train_result["train_loss"]
                epochs_trained = epoch + 1

                # A TrainState is immutable, so the state itself is a snapshot
                logs = {"train_loss": train_loss, "model": model, "snapshot": lambda: state}
                if select_data is not None:
                    select_metrics = self.adapter.evaluate(model, select_data, state=state)
                    logs.update({k: v for k, v in select_metrics.items() if k.startswith("val_")})
                if report_heldout_per_epoch:
                    heldout = self.adapter.evaluate(model, val_data, state=state)
                    curve.append({"epoch": epoch, "loss": heldout["val_loss"], "acc": heldout["val_acc"]})

                if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                    extra = f" - val_loss: {logs['val_loss']:.4f}" if "val_loss" in logs else ""
                    print(f"  Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f}{extra}")

                stop_training = run_epoch_callbacks(callbacks, epoch, fold_idx, logs)
                save_checkpoints(logs, lambda path: self.adapter.save_model(state, path))
                if stop_training:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            best = best_weights_to_restore(callbacks)
            if best is not None:
                state = best
            selection["epochs_trained"] = epochs_trained
            selection["restored_best_weights"] = best is not None
            model_selection.append(selection)
            if report_heldout_per_epoch:
                heldout_curves.append(curve)

            # Score the held-out fold once
            final_metrics = self.adapter.evaluate(model, val_data, state=state)

            # Store results
            all_scores.append(final_metrics)
            if self.store_models:
                all_states.append(state)
            if "predictions" in final_metrics:
                all_predictions.append(final_metrics["predictions"])
            all_indices.append((train_idx, val_idx))

            # Trigger fold end callbacks
            for callback in callbacks:
                if hasattr(callback, "on_fold_end"):
                    callback.on_fold_end(fold_idx, final_metrics)

            # Memory cleanup
            if not self.store_models:
                del state
            import gc

            gc.collect()

        # Trigger CV end callbacks
        for callback in callbacks:
            if hasattr(callback, "on_cv_end"):
                callback.on_cv_end(all_scores)

        return CVResults(
            scores=all_scores,
            models=all_states if self.store_models else None,
            predictions=all_predictions if all_predictions else None,
            indices=all_indices,
            metadata={
                "framework": "jax",
                "epochs": epochs,
                "model_selection": model_selection,
                **({"heldout_curves": heldout_curves} if report_heldout_per_epoch else {}),
            },
        )
