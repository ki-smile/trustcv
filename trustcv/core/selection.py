"""Model selection inside a CV training fold, shared by the deep-learning runners.

The held-out fold of each split is only scored. Early stopping, best-weight
restoration and checkpoints use an inner selection split of the training
fold, or nothing at all.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .inner_split import inner_validation_split


def fold_selection_split(
    train_idx: Any,
    labels: Any = None,
    groups: Any = None,
    *,
    inner_validation: float = 0.0,
    strategy: str = "auto",
    random_state: Optional[int] = 0,
    fold_idx: int = 0,
    warned: Optional[Dict[str, bool]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Split a training fold into rows to fit on and rows to select on.

    Returns ``(fit_idx, select_idx, info)``. ``select_idx`` is None when
    ``inner_validation`` is 0 or no valid split exists (warned once per run
    through the shared ``warned`` dict).
    """
    train_idx = np.asarray(train_idx, dtype=int)
    if not inner_validation:
        return train_idx, None, {"strategy": "none"}
    split, reason = inner_validation_split(
        train_idx,
        labels,
        groups,
        fraction=inner_validation,
        strategy=strategy,
        random_state=None if random_state is None else random_state + fold_idx,
    )
    if split is None:
        if warned is not None and not warned.get("fallback"):
            warnings.warn(
                f"No inner validation split in fold {fold_idx + 1} ({reason}); training for the "
                "full epochs without model selection.",
                UserWarning,
                stacklevel=3,
            )
            warned["fallback"] = True
        return train_idx, None, {"strategy": "none", "reason": reason}
    fit_idx, select_idx = split
    return fit_idx, select_idx, {
        "strategy": f"inner_validation:{reason}",
        "n_inner_train": int(len(fit_idx)),
        "n_inner_val": int(len(select_idx)),
    }


def run_epoch_callbacks(callbacks: List[Any], epoch: int, fold_idx: int, logs: Dict[str, Any]) -> bool:
    """Call every callback's ``on_epoch_end`` and return whether to stop.

    All callbacks run before stopping, so a checkpoint or logger after an
    early stopper still sees the epoch. Checkpoint requests accumulate in
    ``logs["checkpoints"]``.
    """
    logs.setdefault("checkpoints", [])
    stop = False
    for callback in callbacks:
        if hasattr(callback, "on_epoch_end") and callback.on_epoch_end(epoch, fold_idx, logs) == "stop":
            stop = True
    return stop


def save_checkpoints(logs: Dict[str, Any], save_fn: Any) -> None:
    """Write the checkpoints the callbacks requested this epoch with ``save_fn(path)``."""
    for path in logs.get("checkpoints", []):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        save_fn(path)


def best_weights_to_restore(callbacks: List[Any]) -> Any:
    """Weights kept by an EarlyStopping with ``restore_best=True``, or None."""
    for callback in callbacks:
        if getattr(callback, "restore_best", False) and getattr(callback, "best_weights", None) is not None:
            return callback.best_weights
    return None
