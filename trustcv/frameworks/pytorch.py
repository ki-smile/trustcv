"""
PyTorch integration for trustcv

Provides seamless integration with PyTorch datasets, DataLoaders,
and training loops while maintaining trustcv's best practices.
"""

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
from ._labels import split_labels

try:
    import torch
except ImportError as e:
    raise ImportError(
        "PyTorch is required for trustcv.frameworks.pytorch. Install it with: pip install torch"
    ) from e


class PyTorchAdapter(FrameworkAdapter):
    """
    Adapter for PyTorch models and datasets

    Handles PyTorch-specific data loading, training, and evaluation
    while promoting best practices in ML.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        drop_last: bool = False,
        device: str = "auto",
    ):
        """
        Initialize PyTorch adapter

        Parameters:
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster GPU transfer
            shuffle_train: Shuffle training data
            drop_last: Drop last incomplete batch
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
            drop_last=drop_last,
        )
        self.torch = torch

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def create_data_splits(
        self, data: Any, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> Tuple[Any, Any]:
        """
        Create PyTorch DataLoaders from indices

        Parameters:
            data: PyTorch Dataset or tuple of (X, y) tensors/arrays
            train_idx: Training indices
            val_idx: Validation indices

        Returns:
            train_loader: PyTorch DataLoader for training
            val_loader: PyTorch DataLoader for validation
        """
        from torch.utils.data import DataLoader, Subset, TensorDataset

        # Handle different data formats
        if isinstance(data, tuple):
            X, y = data
            if not isinstance(X, self.torch.Tensor):
                X = self.torch.FloatTensor(X)
            if not isinstance(y, self.torch.Tensor):
                y = self.torch.LongTensor(y) if y.dtype == np.int64 else self.torch.FloatTensor(y)

            dataset = TensorDataset(X, y)
        else:
            # Assume it's already a PyTorch Dataset
            dataset = data

        # Create subsets
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        # Create DataLoaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config["batch_size"],
            shuffle=self.config["shuffle_train"],
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"] and str(self.device) != "cpu",
            drop_last=self.config["drop_last"],
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"] and str(self.device) != "cpu",
            drop_last=False,
        )

        return train_loader, val_loader

    def train_epoch(
        self,
        model: Any,
        train_data: Any,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train PyTorch model for one epoch

        Parameters:
            model: PyTorch model
            train_data: PyTorch DataLoader
            optimizer: PyTorch optimizer
            loss_fn: Loss function
            scheduler: Learning rate scheduler (optional)

        Returns:
            Dictionary of training metrics
        """
        if optimizer is None:
            raise ValueError("Optimizer is required for PyTorch training")
        if loss_fn is None:
            loss_fn = self.torch.nn.CrossEntropyLoss()

        model.train()
        model = model.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_data):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                # Handle more complex batch structures
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None

            inputs = inputs.to(self.device)
            if targets is not None:
                targets = targets.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Calculate loss
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                loss = outputs  # Assume model returns loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Calculate accuracy for classification
            if targets is not None and hasattr(outputs, "dim") and outputs.dim() > 1:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                if targets.dim() == outputs.dim():
                    # One-hot encoded targets
                    _, targets = targets.max(1)
                correct += predicted.eq(targets).sum().item()

        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        metrics = {"train_loss": total_loss / len(train_data)}

        if total > 0:
            metrics["train_acc"] = correct / total

        return metrics

    def evaluate(
        self,
        model: Any,
        val_data: Any,
        loss_fn: Optional[Any] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate PyTorch model

        Parameters:
            model: PyTorch model
            val_data: PyTorch DataLoader
            loss_fn: Loss function
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        model = model.to(self.device)

        if loss_fn is None:
            loss_fn = self.torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with self.torch.no_grad():
            for batch in val_data:
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else None

                inputs = inputs.to(self.device)
                if targets is not None:
                    targets = targets.to(self.device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                if targets is not None:
                    loss = loss_fn(outputs, targets)
                    total_loss += loss.item()

                # Store predictions
                if hasattr(outputs, "dim") and outputs.dim() > 1:
                    probs = self.torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)

                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

                    if targets is not None:
                        if targets.dim() == outputs.dim():
                            # One-hot encoded targets
                            _, targets_idx = targets.max(1)
                            all_targets.extend(targets_idx.cpu().numpy())
                        else:
                            all_targets.extend(targets.cpu().numpy())

                        total += targets.size(0)
                        correct += (
                            predicted.eq(targets if targets.dim() == 1 else targets_idx)
                            .sum()
                            .item()
                        )

        eval_metrics = {"val_loss": total_loss / len(val_data) if len(val_data) > 0 else 0.0}

        if total > 0:
            eval_metrics["val_acc"] = correct / total

        if all_preds:
            eval_metrics["predictions"] = np.array(all_preds)

        if all_probs:
            eval_metrics["probabilities"] = np.array(all_probs)

        if all_targets:
            eval_metrics["targets"] = np.array(all_targets)

        return eval_metrics

    def clone_model(self, model: Any) -> Any:
        """
        Clone PyTorch model with fresh weights

        Parameters:
            model: Original model

        Returns:
            Cloned model with reinitialized weights
        """
        import copy

        # Create a deep copy of the model
        cloned = copy.deepcopy(model)

        # Reinitialize weights
        def init_weights(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        cloned.apply(init_weights)

        return cloned

    def get_predictions(self, model: Any, data: Any) -> np.ndarray:
        """
        Get predictions from PyTorch model

        Parameters:
            model: Trained model
            data: Data to predict on (DataLoader or tensor)

        Returns:
            Predictions as numpy array
        """
        from torch.utils.data import DataLoader

        model.eval()
        model = model.to(self.device)

        # Convert to DataLoader if necessary
        if not isinstance(data, DataLoader):
            if isinstance(data, tuple):
                X, _ = data
            else:
                X = data

            if not isinstance(X, self.torch.Tensor):
                X = self.torch.FloatTensor(X)

            data = DataLoader(X, batch_size=self.config["batch_size"], shuffle=False)

        all_preds = []

        with self.torch.no_grad():
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)
                outputs = model(inputs)

                if outputs.dim() > 1:
                    # Classification - get probabilities
                    probs = self.torch.softmax(outputs, dim=1)
                    all_preds.append(probs.cpu().numpy())
                else:
                    # Regression
                    all_preds.append(outputs.cpu().numpy())

        return np.vstack(all_preds)

    def save_model(self, model: Any, path: str) -> None:
        """Save PyTorch model"""
        self.torch.save({"model_state_dict": model.state_dict(), "model_config": str(model)}, path)

    def load_model(self, path: str, model_class: Optional[Any] = None) -> Any:
        """Load PyTorch model"""
        checkpoint = self.torch.load(path, map_location=self.device)

        if model_class is None:
            raise ValueError(
                "model_class is required to instantiate the model. "
                "Pass the model class or an instance of the model."
            )

        if hasattr(model_class, "__call__"):
            model = model_class()
        else:
            model = model_class

        model.load_state_dict(checkpoint["model_state_dict"])
        return model


class TorchCVRunner:
    """
    High-level cross-validation runner for PyTorch models

    Runs cross-validation with PyTorch models: a fresh model per fold,
    group-aware splitting and per-fold results that can support documentation.
    """

    def __init__(
        self,
        model_fn: Callable,
        cv_splitter: Any,
        adapter: Optional[PyTorchAdapter] = None,
        store_models: bool = False,
    ):
        """
        Initialize PyTorch CV runner

        Parameters:
            model_fn: Function that returns a new model instance
            cv_splitter: Cross-validation splitter from trustcv
            adapter: PyTorch adapter (creates default if None)
            store_models: Whether to store trained models (can use significant memory)
        """
        self.model_fn = model_fn
        self.cv_splitter = cv_splitter
        self.adapter = adapter or PyTorchAdapter()
        self.store_models = store_models

    def run(
        self,
        dataset: Any,
        epochs: int = 10,
        optimizer_fn: Optional[Callable] = None,
        loss_fn: Optional[Any] = None,
        scheduler_fn: Optional[Callable] = None,
        callbacks: Optional[List[Any]] = None,
        groups: Optional[np.ndarray] = None,
        inner_validation: float = 0.0,
        inner_validation_strategy: str = "auto",
        inner_random_state: Optional[int] = 0,
        report_heldout_per_epoch: bool = False,
    ) -> CVResults:
        """
        Run cross-validation with PyTorch model

        Parameters:
            dataset: PyTorch Dataset or tuple of (X, y)
            epochs: Number of training epochs per fold
            optimizer_fn: Function that takes model and returns optimizer
            loss_fn: Loss function
            scheduler_fn: Function that takes optimizer and returns scheduler
            callbacks: List of callbacks
            groups: Group labels for grouped CV
            inner_validation: Fraction of each training fold set aside for model
                selection. Per-epoch ``val_*`` metrics (for EarlyStopping and
                ModelCheckpoint) are computed on it; the held-out fold is only
                scored once, after training. 0 trains for the full epochs with
                no ``val_*`` metrics.
            inner_validation_strategy: "auto" (group, then stratified, then
                random), "group", "stratified" or "random".
            inner_random_state: Seed for the inner split (offset by the fold index).
            report_heldout_per_epoch: Also record held-out loss per epoch in
                ``metadata["heldout_curves"]``, for plotting only. Callbacks never see it.

        Returns:
            CVResults object with scores and models
        """
        if optimizer_fn is None:
            # Default optimizer
            optimizer_fn = lambda m: self.adapter.torch.optim.Adam(m.parameters())

        callbacks = callbacks or []
        all_scores = []
        all_models = []
        all_predictions = []
        all_indices = []
        model_selection = []
        heldout_curves = []
        warned = {}

        # Get dataset size for splitting
        if isinstance(dataset, tuple):
            X, y = dataset
            n_samples = len(X)
        else:
            n_samples = len(dataset)
        labels = split_labels(dataset)

        # Trigger CV start callbacks
        n_splits = self.cv_splitter.get_n_splits()
        for callback in callbacks:
            callback.on_cv_start(n_splits)

        # Cross-validation loop
        for fold_idx, (train_idx, val_idx) in enumerate(
            # labels let stratified splitters work; None for splitters that ignore them
            self.cv_splitter.split(np.arange(n_samples), labels, groups=groups)
        ):
            # Trigger fold start callbacks
            for callback in callbacks:
                callback.on_fold_start(fold_idx, train_idx, val_idx)

            # Create new model for this fold
            model = self.model_fn()
            optimizer = optimizer_fn(model)
            scheduler = scheduler_fn(optimizer) if scheduler_fn else None

            # Model selection uses only rows of the training fold
            fit_idx, select_idx, selection = fold_selection_split(
                train_idx, labels, groups,
                inner_validation=inner_validation, strategy=inner_validation_strategy,
                random_state=inner_random_state, fold_idx=fold_idx, warned=warned,
            )
            train_loader, val_loader = self.adapter.create_data_splits(dataset, fit_idx, val_idx)
            select_loader = None
            if select_idx is not None:
                select_loader = self.adapter.create_data_splits(dataset, select_idx, select_idx)[1]

            def snapshot():
                return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            curve = []
            epochs_trained = 0
            for epoch in range(epochs):
                # Trigger epoch start callbacks
                for callback in callbacks:
                    callback.on_epoch_start(epoch, fold_idx)

                train_metrics = self.adapter.train_epoch(
                    model, train_loader, optimizer, loss_fn, scheduler
                )
                epochs_trained = epoch + 1
                logs = {**train_metrics, "model": model, "snapshot": snapshot}
                if select_loader is not None:
                    select_metrics = self.adapter.evaluate(model, select_loader, loss_fn)
                    logs.update({k: v for k, v in select_metrics.items() if k.startswith("val_")})
                if report_heldout_per_epoch:
                    heldout = self.adapter.evaluate(model, val_loader, loss_fn)
                    curve.append({"epoch": epoch, "loss": heldout["val_loss"], "acc": heldout.get("val_acc")})

                stop_training = run_epoch_callbacks(callbacks, epoch, fold_idx, logs)
                save_checkpoints(logs, lambda path: self.adapter.save_model(model, path))
                if stop_training:
                    print(f"Early stopping at epoch {epoch}")
                    break

            best = best_weights_to_restore(callbacks)
            if best is not None:
                model.load_state_dict(best)
            selection["epochs_trained"] = epochs_trained
            selection["restored_best_weights"] = best is not None
            model_selection.append(selection)
            if report_heldout_per_epoch:
                heldout_curves.append(curve)

            # Score the held-out fold once
            final_metrics = self.adapter.evaluate(model, val_loader, loss_fn)

            # Store results
            all_scores.append(final_metrics)
            if self.store_models:
                all_models.append(model)
            if "predictions" in final_metrics:
                all_predictions.append(final_metrics["predictions"])
            all_indices.append((train_idx, val_idx))

            # Trigger fold end callbacks
            for callback in callbacks:
                callback.on_fold_end(fold_idx, final_metrics)

            # Memory cleanup between folds
            if not self.store_models:
                del model
                del optimizer
                if scheduler is not None:
                    del scheduler
            del train_loader, val_loader, select_loader
            import gc
            gc.collect()
            # Clear CUDA cache if available
            if self.adapter.torch.cuda.is_available():
                self.adapter.torch.cuda.empty_cache()

        # Trigger CV end callbacks
        for callback in callbacks:
            callback.on_cv_end(all_scores)

        # Return results
        return CVResults(
            scores=all_scores,
            models=all_models if self.store_models else None,
            predictions=all_predictions if all_predictions else None,
            indices=all_indices,
            metadata={
                "framework": "pytorch",
                "epochs": epochs,
                "model_selection": model_selection,
                **({"heldout_curves": heldout_curves} if report_heldout_per_epoch else {}),
            },
        )
