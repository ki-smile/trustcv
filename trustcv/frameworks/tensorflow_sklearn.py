"""
Lightweight Keras <-> sklearn wrappers for TrustCV.

TensorFlow is imported lazily so it stays an optional dependency.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import inspect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required to use the Keras wrappers. "
            "Install with `pip install tensorflow`."
        ) from e
    return tf


def _softmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / np.sum(exp, axis=1, keepdims=True)


class KerasBaseWrap(BaseEstimator):
    """
    Base wrapper that handles model creation, compilation, and raw prediction.
    """

    def __init__(
        self,
        build_fn: Callable[..., Any],
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 0,
        build_kwargs: Optional[Dict[str, Any]] = None,
        compile_kwargs: Optional[Dict[str, Any]] = None,
        task: Optional[str] = "auto",
        threshold: float = 0.5,
        proba_mode: str = "auto",
        multilabel: Optional[bool] = None,
        output_transform: Optional[Callable[[Any], Any]] = None,
        output_key: Optional[str] = None,
        output_index: Optional[int] = None,
        allow_dataset: bool = True,
        accept_dict_inputs: bool = True,
        squeeze_regression: bool = True,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = 42,
    ):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.build_kwargs = build_kwargs
        self.compile_kwargs = compile_kwargs
        self.task = task or "auto"
        self.threshold = threshold
        self.proba_mode = proba_mode
        self.multilabel = multilabel
        self.output_transform = output_transform
        self.output_key = output_key
        self.output_index = output_index
        self.allow_dataset = allow_dataset
        self.accept_dict_inputs = accept_dict_inputs
        self.squeeze_regression = squeeze_regression
        # Avoid replacing an explicitly passed (even empty) dict so that sklearn.clone
        # sees the exact same object and does not error out.
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.random_state = random_state

        # runtime attributes
        self.model_ = None
        self.history_ = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.label_encoding_: Optional[str] = None  # "sparse" | "onehot" | "multilabel" | "continuous" | None
        self.task_: Optional[str] = None

    # ---- helpers ----
    def _is_dataset(self, X: Any) -> bool:
        try:
            tf = _require_tensorflow()
        except ImportError:
            return False
        dataset_types = (getattr(tf.data, "Dataset", ()), getattr(tf.data, "DatasetSpec", ()))
        return isinstance(X, dataset_types)

    def _peek_dataset(self, ds: Any):
        """Return one element from a tf.data.Dataset without consuming future iterations."""
        try:
            iterator = iter(ds)
            first = next(iterator)
            return first
        except Exception:
            return None

    def _shape_without_batch(self, arr: Any):
        shp = getattr(arr, "shape", None)
        if shp is None:
            try:
                shp = np.asarray(arr).shape
            except Exception:
                return None
        if len(shp) == 0:
            return (1,)
        if len(shp) == 1:
            return (1,)
        return tuple(shp[1:])

    def _infer_input_shape(self, X: Any) -> Any:
        if self._is_dataset(X):
            # Do not infer from dataset; user can pass explicit shapes via build_kwargs.
            return None
        if isinstance(X, dict):
            shapes = {}
            for k, v in X.items():
                shapes[k] = self._shape_without_batch(v)
            return shapes
        if isinstance(X, (list, tuple)) and len(X) > 0 and not hasattr(X, "shape"):
            return [self._shape_without_batch(v) for v in X]
        if hasattr(X, "shape"):
            return self._shape_without_batch(X)
        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return self._shape_without_batch(arr)

    def _ensure_min_dims(self, arr: Any):
        """Ensure arrays have a sample dimension; leave datasets/dicts unchanged."""
        if self._is_dataset(arr):
            return arr
        if hasattr(arr, "shape"):
            if arr.ndim == 0:
                return np.reshape(arr, (1, 1))
            if arr.ndim == 1:
                return np.reshape(arr, (-1, 1))
            return arr
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _coerce_features(self, X: Any):
        """
        Convert user-provided features into shapes Keras can consume.
        Dict/list inputs are passed through unless coercion is required for 1D arrays.
        """
        if self._is_dataset(X):
            if not self.allow_dataset:
                raise ValueError("tf.data.Dataset inputs are not allowed when allow_dataset=False.")
            return X
        if isinstance(X, dict):
            if not self.accept_dict_inputs:
                raise ValueError("Dict inputs are disabled when accept_dict_inputs=False.")
            return {k: self._ensure_min_dims(v) for k, v in X.items()}
        if isinstance(X, (list, tuple)) and len(X) > 0 and not hasattr(X, "shape"):
            return [self._ensure_min_dims(v) for v in X]
        return self._ensure_min_dims(X.to_numpy() if hasattr(X, "to_numpy") else X)

    def _coerce_labels(self, y: Any):
        if y is None:
            return None
        if isinstance(y, dict):
            return {k: self._coerce_labels(v) for k, v in y.items()}
        if isinstance(y, (list, tuple)) and not hasattr(y, "shape"):
            return [self._coerce_labels(v) for v in y]
        if self._is_dataset(y):
            return y
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        if y_arr.ndim == 0:
            y_arr = y_arr.reshape(1)
        return y_arr

    def _get_sample_length(self, branch: Any) -> Optional[int]:
        if branch is None:
            return None
        if self._is_dataset(branch):
            return None
        if hasattr(branch, "shape") and len(branch.shape) > 0:
            return branch.shape[0]
        try:
            return len(branch)
        except Exception:
            return None

    def _check_length_match(self, X, y):
        """Ensure every feature branch has the same sample dimension as y."""
        if self._is_dataset(X):
            return
        lengths = []
        if isinstance(X, dict):
            lengths.extend([self._get_sample_length(v) for v in X.values()])
        elif isinstance(X, (list, tuple)) and len(X) > 0 and not hasattr(X, "shape"):
            lengths.extend([self._get_sample_length(v) for v in X])
        else:
            lengths.append(self._get_sample_length(X))
        if isinstance(y, dict):
            lengths.extend([self._get_sample_length(v) for v in y.values()])
        elif isinstance(y, (list, tuple)) and not hasattr(y, "shape"):
            lengths.extend([self._get_sample_length(v) for v in y])
        else:
            lengths.append(self._get_sample_length(y))
        lengths_clean = [l for l in lengths if l is not None]
        if len(set(lengths_clean)) > 1:
            raise ValueError(
                "Input branches must share the same number of samples. "
                f"Lengths found: {lengths_clean}"
            )

    def _select_y_head(self, y: Any):
        if not isinstance(y, (dict, list, tuple)):
            return y
        if isinstance(y, dict):
            if self.output_key is None and self.output_index is None:
                raise ValueError("y is a dict; set output_key or output_index to select the target head.")
            if self.output_key is not None:
                if self.output_key not in y:
                    raise KeyError(f"output_key '{self.output_key}' not found in y.")
                return y[self.output_key]
            keys = list(y.keys())
            if self.output_index is None or self.output_index >= len(keys):
                raise IndexError("output_index is out of range for y dict.")
            return y[keys[self.output_index]]
        # list/tuple
        if self.output_index is None:
            raise ValueError("y is a list/tuple; set output_index to select the target head.")
        if self.output_index >= len(y):
            raise IndexError("output_index is out of range for y list/tuple.")
        return y[self.output_index]

    def _infer_task_and_labels(self, y: np.ndarray) -> Tuple[str, int, str]:
        """
        Infer task, return (task, n_classes, label_encoding).
        label_encoding in {"sparse", "onehot", "multilabel", "continuous"}
        """
        y_arr = np.asarray(y)
        task = self.task if self.task not in (None, "auto") else None
        if self.multilabel and task is None:
            task = "multilabel"

        if task == "regression":
            return "regression", None, "continuous"

        if y_arr.ndim == 2 and y_arr.shape[1] > 1:
            row_sums = np.sum(y_arr, axis=1)
            unique_values = np.unique(y_arr)
            if task is None:
                if np.allclose(row_sums, 1.0):
                    task = "multiclass"
                elif np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
                    task = "multilabel"
                else:
                    task = "multilabel"
            if task == "multiclass" and np.allclose(row_sums, 1.0):
                return "multiclass", y_arr.shape[1], "onehot"
            if task == "multilabel":
                return "multilabel", y_arr.shape[1], "multilabel"
            # fallback
            return "multiclass", y_arr.shape[1], "onehot"

        if task is None:
            if np.issubdtype(y_arr.dtype, np.floating) and len(np.unique(y_arr)) > 20:
                task = "regression"
            else:
                uniq = np.unique(y_arr)
                task = "binary" if len(uniq) == 2 else "multiclass"

        if task == "regression":
            return "regression", None, "continuous"

        uniq = np.unique(y_arr)
        n_classes = len(uniq)
        if task == "multilabel":
            # treat as single-column multilabel? fallback to binary/multiclass
            task = "binary" if n_classes == 2 else "multiclass"
        return task, n_classes, "sparse"

    def _build_model(self, input_shape: Any, n_classes: Optional[int]):
        _require_tensorflow()
        kwargs = dict(self.build_kwargs or {})
        try:
            sig = inspect.signature(self.build_fn)
            params = sig.parameters
            call_kwargs = dict(kwargs)
            if input_shape is not None and "input_shape" in params:
                call_kwargs["input_shape"] = input_shape
            if n_classes is not None and "n_classes" in params:
                call_kwargs["n_classes"] = n_classes
            return self.build_fn(**call_kwargs)
        except TypeError:
            try:
                return self.build_fn(**kwargs)
            except TypeError as e:
                raise TypeError(
                    "build_fn must accept input_shape and n_classes or be callable with build_kwargs."
                ) from e

    def _ensure_compiled(self, model, task: str, label_encoding: Optional[str], n_classes: Optional[int]):
        tf = _require_tensorflow()
        has_optimizer = getattr(model, "optimizer", None) is not None
        if has_optimizer:
            return
        outputs = getattr(model, "outputs", None)
        n_outputs = len(outputs) if outputs is not None else 1
        if task == "regression":
            loss = "mse"
            metrics = []
        elif task == "multilabel":
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        elif task == "binary":
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            loss = "categorical_crossentropy" if label_encoding == "onehot" else "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
        defaults = dict(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
        defaults.update(self.compile_kwargs or {})
        metrics_val = defaults.get("metrics")
        if n_outputs > 1 and isinstance(metrics_val, (list, tuple)):
            if len(metrics_val) == 0:
                metrics_val = None
            elif len(metrics_val) == 1:
                metrics_val = [metrics_val[0]] * n_outputs
            defaults["metrics"] = metrics_val
        model.compile(**defaults)

    def _fit_with_metrics_fallback(self, model, *fit_args, **fit_kwargs):
        """
        Keras (2.16) raises when a multi-output model was compiled with a flat/empty
        metrics list. If that happens, recompile with metrics broadcast or disabled
        and retry the fit so TrustCV validation can proceed.
        """
        try:
            return model.fit(*fit_args, **fit_kwargs)
        except ValueError as e:
            msg = str(e)
            if "metrics" not in msg or "as many entries as the model has outputs" not in msg:
                raise

            outputs = getattr(model, "outputs", None)
            n_outputs = len(outputs) if outputs is not None else None

            opt = getattr(model, "optimizer", None)
            if opt is not None:
                try:
                    opt = opt.__class__.from_config(opt.get_config())
                except Exception:
                    # fallback to same instance if cloning fails
                    pass

            loss_val = getattr(model, "loss", None)
            if isinstance(loss_val, (list, tuple)):
                loss_val = list(loss_val)

            compile_args: Dict[str, Any] = {"optimizer": opt, "loss": loss_val}
            if getattr(model, "loss_weights", None) is not None:
                compile_args["loss_weights"] = model.loss_weights

            if self.compile_kwargs:
                for k, v in self.compile_kwargs.items():
                    if k == "metrics":
                        continue
                    compile_args.setdefault(k, v)

            metrics_kw = None
            if self.compile_kwargs and "metrics" in self.compile_kwargs:
                mkw = self.compile_kwargs["metrics"]
                if isinstance(mkw, (list, tuple)) and n_outputs and len(mkw) == 1:
                    metrics_kw = [mkw[0]] * n_outputs
                elif isinstance(mkw, dict):
                    metrics_kw = mkw
            if metrics_kw is None and n_outputs and n_outputs > 1:
                # Give each output an explicit empty metric list to satisfy length checks.
                output_names = getattr(model, "output_names", None)
                if output_names and len(output_names) == n_outputs:
                    metrics_kw = {name: [] for name in output_names}
                else:
                    metrics_kw = [[] for _ in range(n_outputs)]
            compile_args["metrics"] = metrics_kw

            model.compile(**compile_args)
            try:
                return model.fit(*fit_args, **fit_kwargs)
            except ValueError as e2:
                # Surface the original message with guidance if it still fails.
                if "metrics" in str(e2) and "as many entries as the model has outputs" in str(e2):
                    raise ValueError(
                        "Keras model metrics still mismatch after fallback recompilation. "
                        "Pass per-output metrics via `compile_kwargs={'metrics': ...}`."
                    ) from e2
                raise

    def _merge_fit_kwargs(self, runtime_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Combine default fit_kwargs passed at init with kwargs supplied at call time."""
        merged = dict(self.fit_kwargs or {})
        merged.update(runtime_kwargs or {})
        return merged

    def _maybe_set_seed(self):
        """Set deterministic seeds when random_state is provided."""
        if self.random_state is None:
            return
        try:
            tf = _require_tensorflow()
            try:
                tf.keras.utils.set_random_seed(int(self.random_state))
            except Exception:
                tf.random.set_seed(int(self.random_state))
        except Exception:
            # Fallback to numpy seed if TF seeding fails
            try:
                np.random.seed(int(self.random_state))
            except Exception:
                pass

    # ---- raw predict ----
    def _predict_raw(self, X):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X_arr = self._coerce_features(X)
        return self.model_.predict(X_arr, batch_size=self.batch_size, verbose=0)

    # ---- output selection ----
    def _select_output(self, raw_pred: Any) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Apply optional output_transform and select a head if the model returns multiple outputs.
        Returns (selected_output_as_array, optional_pred_from_transform)
        """
        extra_pred = None
        if self.output_transform is not None:
            transformed = self.output_transform(raw_pred)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                raw_pred, extra_pred = transformed
            else:
                raw_pred = transformed
        # Handle dict outputs
        if isinstance(raw_pred, dict):
            if self.output_key is None and self.output_index is None:
                raise ValueError("Model returned a dict; set output_key or output_index to select an output.")
            if self.output_key is not None:
                if self.output_key not in raw_pred:
                    raise KeyError(f"output_key '{self.output_key}' not found in model outputs.")
                raw_pred = raw_pred[self.output_key]
            else:
                keys = list(raw_pred.keys())
                if self.output_index is None or self.output_index >= len(keys):
                    raise IndexError("output_index is out of range for model outputs.")
                raw_pred = raw_pred[keys[self.output_index]]
        elif isinstance(raw_pred, (list, tuple)):
            if self.output_index is None:
                raise ValueError("Model returned multiple outputs; set output_index to select one.")
            if self.output_index >= len(raw_pred):
                raise IndexError("output_index is out of range for model outputs.")
            raw_pred = raw_pred[self.output_index]
        return np.asarray(raw_pred), extra_pred


class KerasClassifierWrap(KerasBaseWrap, ClassifierMixin):
    """
    Classification-friendly wrapper with probability + label handling.
    """

    def fit(self, X, y=None, **fit_kwargs):
        tf = _require_tensorflow()
        self._maybe_set_seed()
        X_arr = self._coerce_features(X)
        y_coerced = self._coerce_labels(y)
        fit_kwargs = self._merge_fit_kwargs(fit_kwargs)

        if not self._is_dataset(X_arr):
            if y_coerced is None:
                raise ValueError("y is required for non-dataset inputs.")
            self._check_length_match(X_arr, y_coerced)
            target_for_infer = self._select_y_head(y_coerced) if isinstance(y_coerced, (dict, list, tuple)) else y_coerced
        else:
            # Dataset path: attempt to peek first element for inference
            peek = self._peek_dataset(X_arr)
            target_for_infer = None
            if peek is not None and isinstance(peek, (tuple, list)) and len(peek) >= 2:
                target_for_infer = peek[1]
            elif y_coerced is not None:
                target_for_infer = y_coerced

        if target_for_infer is None and self.task in (None, "auto"):
            raise ValueError(
                "Could not infer task from data. Provide task explicitly or include labels in the dataset."
            )

        task, n_classes, label_enc = self._infer_task_and_labels(
            target_for_infer if target_for_infer is not None else np.asarray([])
        )
        if task == "regression":
            raise ValueError("KerasClassifierWrap received regression target. Use KerasRegressorWrap.")
        if n_classes is None or (isinstance(n_classes, int) and n_classes < 1):
            n_classes = 1

        # classes_ mapping
        if target_for_infer is None:
            self.classes_ = np.arange(n_classes)
        elif label_enc in {"multilabel", "onehot"}:
            self.classes_ = np.arange(n_classes)
        else:
            arr = np.asarray(target_for_infer)
            self.classes_ = np.unique(arr)
        self.n_classes_ = n_classes
        self.label_encoding_ = label_enc
        self.task_ = task

        input_shape = self._infer_input_shape(
            X_arr
            if not self._is_dataset(X_arr)
            else (self._peek_dataset(X_arr)[0] if self._peek_dataset(X_arr) else X_arr)
        )
        self.model_ = self._build_model(input_shape, n_classes)
        self._ensure_compiled(self.model_, task, label_enc, n_classes if n_classes is not None else 1)

        if self._is_dataset(X_arr):
            self.history_ = self._fit_with_metrics_fallback(
                self.model_,
                X_arr,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                **fit_kwargs,
            )
        else:
            self.history_ = self._fit_with_metrics_fallback(
                self.model_,
                X_arr,
                y_coerced,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                **fit_kwargs,
            )
        return self

    def _normalize_proba(self, selected: np.ndarray) -> np.ndarray:
        proba = np.asarray(selected)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)

        if self.task_ == "multilabel":
            if self.proba_mode == "sigmoid" or (
                self.proba_mode == "auto" and (proba.min() < 0.0 or proba.max() > 1.0)
            ):
                proba = 1.0 / (1.0 + np.exp(-proba))
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            return proba

        if self.task_ == "binary":
            if self.proba_mode == "sigmoid" or (
                self.proba_mode == "auto" and (proba.min() < 0.0 or proba.max() > 1.0)
            ):
                proba = 1.0 / (1.0 + np.exp(-proba))
            if proba.shape[1] == 1:
                pos = proba[:, 0]
                proba = np.stack([1.0 - pos, pos], axis=1)
            elif proba.shape[1] != 2:
                raise ValueError("Binary prediction output must have 1 or 2 columns.")
            return proba

        # multiclass
        if self.proba_mode == "softmax":
            proba = _softmax(proba)
        elif self.proba_mode == "auto":
            row_sums = proba.sum(axis=1, keepdims=True)
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                proba = _softmax(proba)
        return proba

    def predict_proba(self, X):
        if self.task_ == "regression":
            raise NotImplementedError("predict_proba is not defined for regression tasks.")
        raw = self._predict_raw(X)
        selected, _ = self._select_output(raw)
        return self._normalize_proba(selected)

    def predict(self, X):
        raw = self._predict_raw(X)
        selected, transformed_pred = self._select_output(raw)

        if transformed_pred is not None:
            return transformed_pred

        if self.task_ == "regression":
            pred = np.asarray(selected)
            if self.squeeze_regression and pred.ndim > 1 and pred.shape[1] == 1:
                pred = pred.ravel()
            return pred

        proba = self._normalize_proba(selected)
        if self.task_ == "multilabel":
            return (proba >= self.threshold).astype(int)
        if self.task_ == "binary":
            labels = (proba[:, 1] >= self.threshold).astype(int)
            return self.classes_[labels]
        labels = np.argmax(proba, axis=1)
        return self.classes_[labels]


class KerasRegressorWrap(KerasBaseWrap, RegressorMixin):
    """Regression wrapper."""

    def fit(self, X, y=None, **fit_kwargs):
        tf = _require_tensorflow()
        self._maybe_set_seed()
        X_arr = self._coerce_features(X)
        y_arr = self._coerce_labels(y)
        fit_kwargs = self._merge_fit_kwargs(fit_kwargs)

        if not self._is_dataset(X_arr):
            if y_arr is None:
                raise ValueError("y is required for regression fit when X is not a dataset.")
            self._check_length_match(X_arr, y_arr)

        task = self.task or "regression"
        self.task_ = "regression"
        self.label_encoding_ = "continuous"
        self.n_classes_ = None

        if self._is_dataset(X_arr):
            peek = self._peek_dataset(X_arr)
            input_shape = self._infer_input_shape(peek[0] if isinstance(peek, (tuple, list)) else X_arr)
        else:
            input_shape = self._infer_input_shape(X_arr)

        self.model_ = self._build_model(input_shape, n_classes=1)
        self._ensure_compiled(self.model_, task, "continuous", 1)

        if self._is_dataset(X_arr):
            self.history_ = self._fit_with_metrics_fallback(
                self.model_,
                X_arr,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                **fit_kwargs,
            )
        else:
            self.history_ = self._fit_with_metrics_fallback(
                self.model_,
                X_arr,
                y_arr,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                **fit_kwargs,
            )
        return self

    def predict(self, X):
        raw = self._predict_raw(X)
        selected, _ = self._select_output(raw)
        pred = np.asarray(selected)
        if self.squeeze_regression and pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.ravel()
        return pred


# Backward-compat aliases
KerasSkWrap = KerasClassifierWrap
