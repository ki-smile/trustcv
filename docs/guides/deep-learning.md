---
description: Cross-validate PyTorch, TensorFlow/Keras, MONAI and JAX models with trustcv runners, and keep early stopping and checkpoints off the held-out fold.
---

# Deep learning

trustcv's splitters produce index arrays, so any of them can drive a deep-learning training loop. The runners in `trustcv.frameworks` and `trustcv.core.runner` do that loop for you: for each fold they build a fresh model, train it on the training indices and score it on the held-out indices.

| Runner | Framework | Install |
| --- | --- | --- |
| `UniversalCVRunner` | Detects scikit-learn, PyTorch, Keras, MONAI or JAX from the model | `pip install trustcv` plus your framework |
| `TorchCVRunner` | PyTorch | `pip install "trustcv[pytorch]"` |
| `KerasCVRunner` | TensorFlow/Keras | `pip install "trustcv[tensorflow]"` |
| `MONAICVRunner` | MONAI (medical imaging) | `pip install "trustcv[monai]"` |
| `JAXCVRunner` | JAX/Flax | `pip install "trustcv[jax]"` |

The runners report performance. They do not run the leakage checks of `TrustCV.validate()`, so their `overall_status` stays `NOT_FULLY_VERIFIED`. To check the split itself, run `DataLeakageChecker.check(folds=...)` on the folds, or validate a scikit-learn baseline with `TrustCV` on the same splitter and groups.

## Model selection must not use the held-out fold

Early stopping, best-epoch checkpoints and learning-rate schedules that react to a validation metric all choose the model. If that metric is computed on the CV test fold, the model is chosen on the data that then scores it, and the reported score is optimistic.

Every runner computes per-epoch `val_*` metrics only on an inner split of the training fold, and only when you pass `inner_validation=` (for example `0.1`). The held-out fold is scored once, after training.

| `inner_validation` | Training | Per-epoch `val_*` metrics | Early stopping and best checkpoints |
| --- | --- | --- | --- |
| `0` (default) | All rows of the training fold, for the full `epochs` | None | Nothing to monitor; trustcv's callbacks warn once |
| `0.1` | 90% of the training fold | The other 10%, grouped by patient when `groups=` is given | On the inner split |

`EarlyStopping(restore_best=True)` restores the best epoch's weights before the held-out fold is scored. `ModelCheckpoint` writes files with the adapter's `save_model`. Both refuse a `monitor` that starts with `heldout`. A fixed schedule such as `StepLR` or `CosineAnnealingLR` is fine with or without an inner split.

To plot held-out learning curves, pass `report_heldout_per_epoch=True`; they go to `results.metadata["heldout_curves"]`, which callbacks never see.

!!! note "Changed in 1.3.0"
    Before 1.3.0, `TorchCVRunner`, `MONAICVRunner`, `JAXCVRunner` and `UniversalCVRunner` passed held-out `val_*` metrics to callbacks, so `EarlyStopping` stopped on the test fold, `restore_best` restored nothing, and `ModelCheckpoint` saved nothing. Code that relied on those metrics now gets a warning; pass `inner_validation=` to select on an inner split. The final score keys (`val_loss`, `val_acc`, ...) are unchanged.

!!! note "Changed in 1.2.1"
    Before 1.2.1, `KerasCVRunner` added `EarlyStopping(restore_best_weights=True)` and passed the held-out fold as `validation_data`. It now only scores the held-out fold, and by default trains for `epochs` with no early stopping. Expect small changes in reported Keras scores.

## Example data

The examples below share this data: 60 patients with 4 rows each. Every example splits by patient with `StratifiedGroupKFold`, so a patient is never in both training and test.

```python
import numpy as np

rng = np.random.default_rng(0)
patient_ids = np.repeat(np.arange(60), 4)
X = (rng.normal(size=(60, 8))[patient_ids] + rng.normal(0, 0.3, (240, 8))).astype("float32")
y = np.repeat(rng.integers(0, 2, 60), 4)
```

## UniversalCVRunner

`UniversalCVRunner` takes a model (or a function that returns one) and `data=(X, y)`, and detects the framework. Extra arrays that only the splitter needs, such as `timestamps` or `coordinates`, go in `split_kwargs=`.

```python
from sklearn.linear_model import LogisticRegression
from trustcv import StratifiedGroupKFold
from trustcv.core.runner import UniversalCVRunner

runner = UniversalCVRunner(cv_splitter=StratifiedGroupKFold(n_splits=3, random_state=0), verbose=0)
results = runner.run(model=LogisticRegression(max_iter=1000), data=(X, y), groups=patient_ids)
print(f"{results.mean_score['roc_auc']:.3f}")
print(results.overall_status)  # NOT_FULLY_VERIFIED: the runner does not run leakage checks
```

`results.mean_score` is a dictionary with one mean per metric. Per-fold values are in `results.scores` and the train/test indices in `results.indices`.

## PyTorch

`TorchCVRunner` takes a model factory and a dataset: an `(X, y)` tuple of tensors, or a `torch.utils.data.Dataset`. Here it stops early on an inner split and restores the best epoch before scoring the held-out fold.

```python
import torch
import torch.nn as nn
from trustcv import StratifiedGroupKFold
from trustcv.core.callbacks import EarlyStopping
from trustcv.frameworks.pytorch import TorchCVRunner


def make_net():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))


runner = TorchCVRunner(model_fn=make_net, cv_splitter=StratifiedGroupKFold(n_splits=3, random_state=0))
results = runner.run(
    dataset=(torch.tensor(X), torch.tensor(y)),
    epochs=5,
    optimizer_fn=lambda m: torch.optim.Adam(m.parameters(), lr=1e-3),
    loss_fn=nn.CrossEntropyLoss(),
    groups=patient_ids,
    inner_validation=0.2,  # val_loss for early stopping comes from this inner split
    callbacks=[EarlyStopping(monitor="val_loss", patience=2, verbose=False)],
)
print(f"{results.mean_score['val_acc']:.3f}")
print(results.metadata["model_selection"][0])  # inner split, epochs trained, restored or not
```

The final score keys start with `val_` (`val_loss`, `val_acc`), but they are computed on the held-out CV fold.

Since 1.2.2, `TorchCVRunner` and `MONAICVRunner` pass labels to the splitter, so stratified splitters such as `StratifiedGroupKFold` work. Labels are taken from an `(X, y)` tuple, a dataset's `targets` or `labels` attribute, or scalar MONAI `label` values, without loading any item. In 1.2.1 these runners call the splitter without labels, and stratified splitters fail with a "missing y" error; use `GroupKFoldMedical` there.

## TensorFlow and Keras

`KerasCVRunner` takes a model factory and `compile_kwargs`. With `inner_validation=0.1` it holds out 10% of each training fold for early stopping. The inner split keeps patients together when `groups=` is given, and is stratified when that is feasible.

```python
from tensorflow import keras
from trustcv import StratifiedGroupKFold
from trustcv.frameworks.tensorflow import KerasCVRunner


def make_model():
    return keras.Sequential([
        keras.Input(shape=(8,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


runner = KerasCVRunner(
    model_fn=make_model,
    cv_splitter=StratifiedGroupKFold(n_splits=3, random_state=0),
    compile_kwargs={
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", keras.metrics.AUC(name="auc")],
    },
)
results = runner.run(
    X, y,
    epochs=2,
    groups=patient_ids,
    inner_validation=0.2,  # val_loss for early stopping comes from this inner split
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)],
    verbose=0,
)
print(f"{results.mean_score['val_auc']:.3f}")
print(results.metadata["model_selection"][0])  # inner split used in fold 1
```

- With `inner_validation` and no `EarlyStopping` in `callbacks`, the runner adds one on the inner split.
- Each fold gets its own copy of every callback, so state such as `EarlyStopping.best` does not carry from one fold to the next.
- For chronologically ordered data, `inner_validation_strategy="last"` uses the last part of each training fold, so inner validation never comes before inner training. With `groups=` it also keeps a patient on one side.
- Keras's own `validation_split=` still works but warns: it takes the last rows without regard to groups and can split a patient. It cannot be combined with `inner_validation`, and passing `validation_data=` is rejected.

For your own training loop, `trustcv.core.inner_split.inner_validation_split` returns the same inner split:

```python
from trustcv import StratifiedGroupKFold
from trustcv.core.inner_split import inner_validation_split

train_idx, test_idx = next(StratifiedGroupKFold(n_splits=3, random_state=0).split(X, y, patient_ids))
split, reason = inner_validation_split(train_idx, y, patient_ids, fraction=0.2, random_state=0)
inner_train, inner_val = split
print(reason)  # group
assert not set(patient_ids[inner_train]) & set(patient_ids[inner_val])
```

Select epochs or checkpoints on `inner_val`, then score once on `test_idx`.

## MONAI

`MONAICVRunner` takes MONAI-style data dictionaries. The patient IDs go in `groups=`, so all images of a patient stay in one fold. This sketch shows the call; it needs real image files to run.

<!-- notest: needs MONAI and NIfTI files on disk -->
```python
import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ScaleIntensityd
from trustcv import GroupKFoldMedical
from trustcv.core.callbacks import EarlyStopping
from trustcv.frameworks.monai import MONAIAdapter, MONAICVRunner

data_dicts = [
    {"image": f"patient_{i:03d}_t1.nii.gz", "label": f"patient_{i:03d}_seg.nii.gz"}
    for i in range(100)
]
patient_ids = np.array([f"patient_{i:03d}" for i in range(100)])
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
])

runner = MONAICVRunner(
    model_fn=lambda: UNet(spatial_dims=3, in_channels=1, out_channels=2,
                          channels=(16, 32, 64, 128), strides=(2, 2, 2)),
    cv_splitter=GroupKFoldMedical(n_splits=5),
    adapter=MONAIAdapter(batch_size=2, roi_size=(96, 96, 96)),
)
results = runner.run(
    data_dicts=data_dicts,
    epochs=50,
    inner_validation=0.1,  # early stopping and checkpoints use 10% of each training fold
    train_transforms=transforms,
    val_transforms=transforms,
    optimizer_fn=lambda m: torch.optim.AdamW(m.parameters(), lr=1e-4),
    loss_fn=DiceLoss(to_onehot_y=True, softmax=True),
    metrics=[DiceMetric(include_background=False, reduction="mean")],
    groups=patient_ids,
    callbacks=[EarlyStopping(monitor="val_dice", mode="max", patience=3)],
)
print(results.mean_score["val_dice"])  # Dice on the held-out fold, scored once
```

The inner split is evaluated, and callbacks run, every `eval_every=5` epochs and on the last epoch. The labels in this example are mask paths, so there is nothing to stratify on; use `GroupKFoldMedical`. For classification with scalar `label` values, stratified splitters use them.

## JAX and Flax

`JAXCVRunner` takes a Flax module factory, `X`, `y` and an Optax optimizer. It takes the same `inner_validation=` option; restoring the best epoch restores the Flax `TrainState` (non-parameter collections such as BatchNorm statistics are not handled).

```python
import flax.linen as fnn
import optax
from trustcv import StratifiedGroupKFold
from trustcv.frameworks.jax import JAXAdapter, JAXCVRunner


class MLP(fnn.Module):
    @fnn.compact
    def __call__(self, x, training: bool = True):
        return fnn.Dense(2)(fnn.relu(fnn.Dense(16)(x)))


runner = JAXCVRunner(
    model_fn=MLP,
    cv_splitter=StratifiedGroupKFold(n_splits=3, random_state=0),
    adapter=JAXAdapter(batch_size=32, seed=0),
)
results = runner.run(X, y, epochs=2, groups=patient_ids, optimizer=optax.adam(1e-3))
print(f"{results.mean_score['val_acc']:.3f}")
```

## Callbacks

Callbacks from `trustcv.core.callbacks` work with the Torch, MONAI, JAX and Universal runners. Each epoch the runner calls every callback before acting on a stop request. A custom callback can read `logs["snapshot"]()` (a copy of the current weights) and append paths to `logs["checkpoints"]`.

| Callback | Use |
| --- | --- |
| `ProgressLogger` | Logs per-fold and per-epoch metrics, optionally to a JSON file |
| `ClassDistributionLogger` | Prints the class balance of each fold |
| `EarlyStopping`, `ModelCheckpoint` | Stop, restore the best weights, or checkpoint on a `val_*` metric from the inner split (`inner_validation=`) |
| `RegulatoryComplianceLogger` | Writes a JSON log of fold sizes and per-fold results to `output_dir` |

`RegulatoryComplianceLogger` can support documentation of a validation; it does not make a model or device compliant with any regulation. See [documenting validation](../regulatory/documentation-guide.md).

## Next steps

- [Choosing a split](choosing-cv.md) to pick the splitter for your data
- [Confidence intervals](confidence-intervals.md) for uncertainty around fold scores
- [Data leakage](data-leakage.md) for checks you can run on the folds
