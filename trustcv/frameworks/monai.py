"""
MONAI (Medical Open Network for AI) integration for trustcv

Provides specialized support for medical imaging workflows with MONAI,
including 3D medical image handling, transforms, and medical-specific metrics.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..core.base import CVResults
from .pytorch import PyTorchAdapter


class MONAIAdapter(PyTorchAdapter):
    """
    Adapter for MONAI medical imaging workflows

    Extends PyTorchAdapter with MONAI-specific features for medical imaging,
    including support for 3D volumes, medical transforms, and clinical metrics.
    """

    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 2,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        cache_rate: float = 0.0,
        device: str = "auto",
        roi_size: Optional[Tuple[int, ...]] = None,
        sw_batch_size: int = 4,
    ):
        """
        Initialize MONAI adapter

        Parameters:
            batch_size: Batch size for DataLoaders (smaller for 3D volumes)
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster GPU transfer
            shuffle_train: Shuffle training data
            cache_rate: Percentage of data to cache (0.0 to 1.0)
            device: Device to use ('auto', 'cpu', 'cuda')
            roi_size: Region of interest size for sliding window inference
            sw_batch_size: Batch size for sliding window inference
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
            device=device,
        )

        self.cache_rate = cache_rate
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size

        try:
            import monai

            self.monai = monai

            # Set MONAI config
            monai.config.print_config()

        except ImportError:
            raise ImportError(
                "MONAI is required for MONAIAdapter. " "Install it with: pip install monai"
            )

    def create_data_splits(
        self,
        data: Any,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train_transforms: Optional[Any] = None,
        val_transforms: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Create MONAI DataLoaders with medical imaging support

        Parameters:
            data: MONAI Dataset, list of dictionaries, or file paths
            train_idx: Training indices
            val_idx: Validation indices
            train_transforms: MONAI transforms for training
            val_transforms: MONAI transforms for validation

        Returns:
            train_loader: MONAI DataLoader for training
            val_loader: MONAI DataLoader for validation
        """
        from monai.data import (
            CacheDataset,
            DataLoader,
            Dataset,
            SmartCacheDataset,
            list_data_collate,
        )

        # Handle different data formats
        if isinstance(data, list):
            # List of dictionaries (MONAI format)
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
        elif hasattr(data, "__getitem__"):
            # Dataset-like object
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
        else:
            raise ValueError("Data must be a list of dictionaries or a MONAI Dataset")

        # Create MONAI datasets with caching if specified
        if self.cache_rate > 0:
            # Use CacheDataset for better performance
            train_dataset = CacheDataset(
                data=train_data,
                transform=train_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.config["num_workers"],
            )
            val_dataset = CacheDataset(
                data=val_data,
                transform=val_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.config["num_workers"],
            )
        else:
            # Standard datasets
            train_dataset = Dataset(data=train_data, transform=train_transforms)
            val_dataset = Dataset(data=val_data, transform=val_transforms)

        # Create DataLoaders with medical imaging optimizations
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=self.config["shuffle_train"],
            num_workers=self.config["num_workers"],
            collate_fn=list_data_collate,
            pin_memory=self.config["pin_memory"] and str(self.device) != "cpu",
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=list_data_collate,
            pin_memory=self.config["pin_memory"] and str(self.device) != "cpu",
        )

        return train_loader, val_loader

    def train_epoch(
        self,
        model: Any,
        train_data: Any,
        optimizer: Optional[Any] = None,
        loss_fn: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        inferer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train MONAI model for one epoch with medical imaging support

        Parameters:
            model: MONAI/PyTorch model
            train_data: MONAI DataLoader
            optimizer: Optimizer
            loss_fn: Loss function (e.g., DiceLoss, DiceCELoss)
            scheduler: Learning rate scheduler
            inferer: MONAI inferer for sliding window inference

        Returns:
            Dictionary of training metrics
        """
        if optimizer is None:
            raise ValueError("Optimizer is required for training")

        if loss_fn is None:
            # Default to DiceCELoss for segmentation
            from monai.losses import DiceCELoss

            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

        model.train()
        model = model.to(self.device)

        epoch_loss = 0
        step = 0

        # Optional: Use AMP for mixed precision training
        scaler = self.torch.cuda.amp.GradScaler() if self.torch.cuda.is_available() else None

        for batch_data in train_data:
            step += 1
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            optimizer.zero_grad()

            if scaler is not None:
                # Mixed precision training
                with self.torch.cuda.amp.autocast():
                    if inferer is not None:
                        outputs = inferer(inputs, model)
                    else:
                        outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                if inferer is not None:
                    outputs = inferer(inputs, model)
                else:
                    outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        return {"train_loss": epoch_loss / step, "learning_rate": optimizer.param_groups[0]["lr"]}

    def evaluate(
        self,
        model: Any,
        val_data: Any,
        loss_fn: Optional[Any] = None,
        metrics: Optional[List[Any]] = None,
        inferer: Optional[Any] = None,
        post_transforms: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate MONAI model with medical imaging metrics

        Parameters:
            model: MONAI/PyTorch model
            val_data: MONAI DataLoader
            loss_fn: Loss function
            metrics: List of MONAI metrics (DiceMetric, HausdorffDistance, etc.)
            inferer: MONAI inferer for sliding window inference
            post_transforms: Post-processing transforms

        Returns:
            Dictionary of evaluation metrics
        """
        from monai.inferers import sliding_window_inference
        from monai.metrics import DiceMetric, HausdorffDistanceMetric

        model.eval()
        model = model.to(self.device)

        if loss_fn is None:
            from monai.losses import DiceCELoss

            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

        # Default metrics if not provided
        if metrics is None:
            metrics = [
                DiceMetric(include_background=False, reduction="mean"),
                HausdorffDistanceMetric(include_background=False, reduction="mean"),
            ]

        epoch_loss = 0
        step = 0

        # Reset metrics
        for metric in metrics:
            if hasattr(metric, "reset"):
                metric.reset()

        with self.torch.no_grad():
            for batch_data in val_data:
                step += 1
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                # Inference
                if inferer is not None:
                    outputs = inferer(inputs, model)
                elif self.roi_size is not None:
                    # Use sliding window inference for large volumes
                    outputs = sliding_window_inference(
                        inputs, self.roi_size, self.sw_batch_size, model
                    )
                else:
                    outputs = model(inputs)

                # Calculate loss
                loss = loss_fn(outputs, labels)
                epoch_loss += loss.item()

                # Apply post-processing if provided
                if post_transforms is not None:
                    outputs = post_transforms(outputs)

                # Update metrics
                for metric in metrics:
                    metric(y_pred=outputs, y=labels)

        # Aggregate metrics
        eval_metrics = {"val_loss": epoch_loss / step}

        for i, metric in enumerate(metrics):
            metric_name = metric.__class__.__name__.lower().replace("metric", "")
            if hasattr(metric, "aggregate"):
                metric_value = metric.aggregate().item()
            else:
                metric_value = metric.compute().item() if hasattr(metric, "compute") else 0.0
            eval_metrics[f"val_{metric_name}"] = metric_value

        return eval_metrics

    def create_default_transforms(
        self, spatial_size: Tuple[int, ...], mode: str = "train", modality: str = "ct"
    ) -> Any:
        """
        Create default MONAI transforms for common medical imaging tasks

        Parameters:
            spatial_size: Target spatial size for images
            mode: 'train' or 'val' - determines augmentation level
            modality: Image modality ('ct', 'mri', 'ultrasound')

        Returns:
            MONAI Compose transform
        """
        from monai.transforms import (
            AddChanneld,
            Compose,
            CropForegroundd,
            EnsureTyped,
            LoadImaged,
            Orientationd,
            RandAffined,
            RandCropByPosNegLabeld,
            RandFlipd,
            RandRotate90d,
            RandShiftIntensityd,
            Resized,
            ScaleIntensityRanged,
            Spacingd,
            ToTensord,
        )

        # Base transforms
        transforms = [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]

        # Modality-specific intensity normalization
        if modality == "ct":
            transforms.append(
                ScaleIntensityRanged(
                    keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
                )
            )
        elif modality == "mri":
            transforms.append(
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True
                )
            )

        # Spatial transforms
        transforms.extend(
            [
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Resized(keys=["image", "label"], spatial_size=spatial_size),
            ]
        )

        # Training augmentations
        if mode == "train":
            transforms.extend(
                [
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=spatial_size,
                        pos=1,
                        neg=1,
                        num_samples=4,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                ]
            )

        transforms.append(EnsureTyped(keys=["image", "label"]))

        return Compose(transforms)


class MONAICVRunner:
    """
    High-level cross-validation runner for MONAI medical imaging models

    Simplifies running cross-validation with MONAI models while ensuring
    best practices for medical imaging validation.
    """

    def __init__(
        self,
        model_fn: Callable,
        cv_splitter: Any,
        adapter: Optional[MONAIAdapter] = None,
        store_models: bool = False,
    ):
        """
        Initialize MONAI CV runner

        Parameters:
            model_fn: Function that returns a new model instance
            cv_splitter: Cross-validation splitter from trustcv
            adapter: MONAI adapter (creates default if None)
            store_models: Whether to store trained models (can use significant memory for 3D models)
        """
        self.model_fn = model_fn
        self.cv_splitter = cv_splitter
        self.adapter = adapter or MONAIAdapter()
        self.store_models = store_models

    def run(
        self,
        data_dicts: List[Dict],
        epochs: int = 100,
        train_transforms: Optional[Any] = None,
        val_transforms: Optional[Any] = None,
        optimizer_fn: Optional[Callable] = None,
        loss_fn: Optional[Any] = None,
        metrics: Optional[List[Any]] = None,
        scheduler_fn: Optional[Callable] = None,
        callbacks: Optional[List[Any]] = None,
        groups: Optional[np.ndarray] = None,
        inferer: Optional[Any] = None,
    ) -> CVResults:
        """
        Run cross-validation with MONAI model

        Parameters:
            data_dicts: List of dictionaries with 'image' and 'label' keys
            epochs: Number of training epochs per fold
            train_transforms: MONAI transforms for training
            val_transforms: MONAI transforms for validation
            optimizer_fn: Function that takes model and returns optimizer
            loss_fn: MONAI loss function
            metrics: List of MONAI metrics
            scheduler_fn: Function that takes optimizer and returns scheduler
            callbacks: List of callbacks
            groups: Group labels for grouped CV (e.g., patient IDs)
            inferer: MONAI inferer for sliding window inference

        Returns:
            CVResults object with scores and models
        """
        from monai.losses import DiceCELoss
        from monai.metrics import DiceMetric

        # Default optimizer
        if optimizer_fn is None:
            optimizer_fn = lambda m: self.adapter.torch.optim.AdamW(
                m.parameters(), lr=1e-4, weight_decay=1e-5
            )

        # Default loss for segmentation
        if loss_fn is None:
            loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

        # Default metrics
        if metrics is None:
            metrics = [DiceMetric(include_background=False, reduction="mean")]

        # Default transforms if not provided
        if train_transforms is None or val_transforms is None:
            # Infer spatial size from first image
            import nibabel as nib

            first_image = nib.load(data_dicts[0]["image"]).shape[:3]
            spatial_size = tuple(min(s, 128) for s in first_image)  # Cap at 128

            if train_transforms is None:
                train_transforms = self.adapter.create_default_transforms(
                    spatial_size, mode="train"
                )
            if val_transforms is None:
                val_transforms = self.adapter.create_default_transforms(spatial_size, mode="val")

        callbacks = callbacks or []
        all_scores = []
        all_models = []
        all_indices = []

        n_samples = len(data_dicts)
        n_splits = self.cv_splitter.get_n_splits()

        # Trigger CV start callbacks
        for callback in callbacks:
            callback.on_cv_start(n_splits)

        # Cross-validation loop
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.cv_splitter.split(range(n_samples), groups=groups)
        ):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
            print(f"{'='*50}")

            # Trigger fold start callbacks
            for callback in callbacks:
                callback.on_fold_start(fold_idx, train_idx, val_idx)

            # Create new model for this fold
            model = self.model_fn()
            optimizer = optimizer_fn(model)
            scheduler = scheduler_fn(optimizer) if scheduler_fn else None

            # Create data loaders
            train_loader, val_loader = self.adapter.create_data_splits(
                data_dicts, train_idx, val_idx, train_transforms, val_transforms
            )

            # Best model tracking
            best_metric = -1
            best_metric_epoch = -1

            # Training loop
            for epoch in range(epochs):
                # Trigger epoch start callbacks
                for callback in callbacks:
                    callback.on_epoch_start(epoch, fold_idx)

                # Train epoch
                train_metrics = self.adapter.train_epoch(
                    model, train_loader, optimizer, loss_fn, scheduler, inferer
                )

                # Evaluate every 5 epochs to save time
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    val_metrics = self.adapter.evaluate(
                        model, val_loader, loss_fn, metrics, inferer
                    )

                    # Track best model
                    if val_metrics.get("val_dice", 0) > best_metric:
                        best_metric = val_metrics.get("val_dice", 0)
                        best_metric_epoch = epoch

                    # Print progress
                    print(f"Epoch {epoch + 1}/{epochs}")
                    print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                    print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                    print(f"  Val Dice: {val_metrics.get('val_dice', 0):.4f}")
                    print(f"  Best Dice: {best_metric:.4f} (epoch {best_metric_epoch + 1})")

                    # Prepare logs for callbacks
                    logs = {**train_metrics, **val_metrics, "model": model}

                    # Trigger epoch end callbacks
                    stop_training = False
                    for callback in callbacks:
                        result = callback.on_epoch_end(epoch, fold_idx, logs)
                        if result == "stop":
                            stop_training = True
                            break

                    if stop_training:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Final evaluation
            final_metrics = self.adapter.evaluate(model, val_loader, loss_fn, metrics, inferer)

            print(f"\nFold {fold_idx + 1} Final Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

            # Store results
            all_scores.append(final_metrics)
            if self.store_models:
                all_models.append(model)
            all_indices.append((train_idx, val_idx))

            # Trigger fold end callbacks
            for callback in callbacks:
                callback.on_fold_end(fold_idx, final_metrics)

            # Memory cleanup between folds (critical for 3D medical imaging)
            if not self.store_models:
                del model
                del optimizer
                if scheduler is not None:
                    del scheduler
            del train_loader, val_loader
            import gc
            gc.collect()
            # Clear CUDA cache - essential for 3D volumes
            if self.adapter.torch.cuda.is_available():
                self.adapter.torch.cuda.empty_cache()

        # Trigger CV end callbacks
        for callback in callbacks:
            callback.on_cv_end(all_scores)

        # Return results
        return CVResults(
            scores=all_scores,
            models=all_models if self.store_models else None,
            predictions=None,  # Predictions for 3D volumes would be too large
            indices=all_indices,
            metadata={"framework": "monai", "epochs": epochs, "n_folds": n_splits},
        )
