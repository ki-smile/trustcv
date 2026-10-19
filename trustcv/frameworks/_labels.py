"""Labels for CV splitting in the deep-learning runners (no framework imports)."""

from numbers import Number
from typing import Any, Optional

import numpy as np


def split_labels(data: Any) -> Optional[np.ndarray]:
    """Return per-sample labels for ``splitter.split(X, y, groups)``, or None.

    - ``(X, y)`` tuple: ``y``
    - Dataset with a ``targets`` or ``labels`` attribute: that attribute
      (items are never loaded, which could read every image from disk)
    - list of dicts (MONAI style): the ``"label"`` values, only when every one
      is a scalar number; paths to segmentation masks give None
    """
    if isinstance(data, tuple) and len(data) == 2:
        y = data[1]
        return None if y is None else np.asarray(y.cpu() if hasattr(y, "cpu") else y)
    for attr in ("targets", "labels"):
        values = getattr(data, attr, None)
        if values is not None and not callable(values):
            return np.asarray(values)
    if isinstance(data, list) and data and all(isinstance(d, dict) for d in data):
        labels = [d.get("label") for d in data]
        if all(isinstance(v, Number) for v in labels):
            return np.asarray(labels)
    return None
