"""MONAICVRunner selects on an inner split, restores the best weights and writes checkpoints."""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from monai.losses import DiceCELoss
from sklearn.model_selection import GroupKFold, KFold

from trustcv.core.callbacks import CVCallback, EarlyStopping, ModelCheckpoint
from trustcv.frameworks.monai import MONAIAdapter, MONAICVRunner


def _data(n=24):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        image = rng.normal(size=(1, 8, 8)).astype(np.float32)
        out.append({"id": i, "image": torch.tensor(image), "label": torch.tensor((image > 0).astype(np.int64))})
    return out


def _model():
    torch.manual_seed(0)
    return torch.nn.Conv2d(1, 2, 1)


class RecordingAdapter(MONAIAdapter):
    def __init__(self):
        super().__init__(batch_size=4, device="cpu", cache_rate=0.0, num_workers=0)
        self.trained, self.evaluated = [], []

    def train_epoch(self, model, train_data, *args, **kwargs):
        self.trained.append({d["id"] for d in train_data.dataset.data})
        return super().train_epoch(model, train_data, *args, **kwargs)

    def evaluate(self, model, val_data, *args, **kwargs):
        self.evaluated.append({d["id"] for d in val_data.dataset.data})
        return super().evaluate(model, val_data, *args, **kwargs)


def _run(callbacks=(), cv=None, groups=None, epochs=2, **kw):
    adapter = RecordingAdapter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = MONAICVRunner(_model, cv or KFold(2), adapter=adapter, store_models=True).run(
            _data(), epochs=epochs, train_transforms=lambda d: d, val_transforms=lambda d: d,
            loss_fn=DiceCELoss(to_onehot_y=True, softmax=True), metrics=[],
            callbacks=list(callbacks), groups=groups, eval_every=1, **kw)
    return result, adapter


def test_heldout_rows_never_reach_training_or_selection():
    groups = np.repeat(np.arange(8), 3)
    result, adapter = _run(cv=GroupKFold(2), groups=groups, inner_validation=0.25)
    for fold, (train_idx, test_idx) in enumerate(result.indices):
        fit = adapter.trained[2 * fold]
        select, heldout = adapter.evaluated[3 * fold], adapter.evaluated[3 * fold + 2]
        assert heldout == set(test_idx)
        assert fit | select == set(train_idx) and not fit & select
        assert not set(groups[list(fit)]) & set(groups[list(select)])


class FakeValLoss(CVCallback):
    def __init__(self, losses):
        self.losses, self.weights = losses, []

    def on_epoch_end(self, epoch, fold_idx, logs=None):
        logs["val_loss"] = self.losses[epoch]
        self.weights.append(logs["snapshot"]())


def test_best_weights_are_restored(tmp_path):
    fake = FakeValLoss([1.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    ckpt = ModelCheckpoint(str(tmp_path / "f{fold}" / "best.pt"), verbose=False)
    result, _ = _run([fake, EarlyStopping(patience=3, verbose=False), ckpt], epochs=6)
    final = result.models[-1].state_dict()
    best = fake.weights[-4]
    assert all(torch.equal(final[k], best[k]) for k in final)
    assert result.metadata["model_selection"][-1]["epochs_trained"] == 5
    saved = torch.load(tmp_path / "f1" / "best.pt")["model_state_dict"]
    assert all(torch.equal(saved[k], best[k]) for k in saved)  # the checkpoint is the selected epoch


def test_periodic_checkpoints_follow_the_evaluation_cadence(tmp_path):
    adapter = RecordingAdapter()
    ckpt = ModelCheckpoint(str(tmp_path / "f{fold}" / "e{epoch}.pt"), save_best_only=False, save_freq=5, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MONAICVRunner(_model, KFold(2), adapter=adapter).run(
            _data(), epochs=10, train_transforms=lambda d: d, val_transforms=lambda d: d,
            loss_fn=DiceCELoss(to_onehot_y=True, softmax=True), metrics=[], callbacks=[ckpt])
    assert sorted(p.name for p in (tmp_path / "f0").iterdir()) == ["e4.pt", "e9.pt"]  # epochs 5 and 10
