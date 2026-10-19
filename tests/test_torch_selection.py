"""TorchCVRunner selects models on an inner split of the training fold, never on the held-out fold."""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.model_selection import GroupKFold, KFold

from trustcv.core.callbacks import CVCallback, EarlyStopping, ModelCheckpoint
from trustcv.frameworks.pytorch import PyTorchAdapter, TorchCVRunner


def _data(n=60):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


class RecordingAdapter(PyTorchAdapter):
    def __init__(self):
        super().__init__(batch_size=16, device="cpu")
        self.trained, self.evaluated = [], []

    def train_epoch(self, model, train_data, *args, **kwargs):
        self.trained.append(set(train_data.dataset.indices))
        return super().train_epoch(model, train_data, *args, **kwargs)

    def evaluate(self, model, val_data, *args, **kwargs):
        self.evaluated.append(set(val_data.dataset.indices))
        return super().evaluate(model, val_data, *args, **kwargs)


class Logs(CVCallback):
    def __init__(self):
        self.keys = []

    def on_epoch_end(self, epoch, fold_idx, logs=None):
        self.keys.append(set(logs))


def _run(callbacks=(), cv=None, groups=None, epochs=3, **kw):
    adapter = RecordingAdapter()
    runner = TorchCVRunner(_model, cv or KFold(3), adapter=adapter, store_models=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = runner.run(_data(), epochs=epochs, loss_fn=torch.nn.CrossEntropyLoss(),
                            callbacks=list(callbacks), groups=groups, **kw)
    return result, adapter


def test_heldout_rows_never_reach_training_or_selection():
    groups = np.repeat(np.arange(12), 5)
    result, adapter = _run(cv=GroupKFold(3), groups=groups, inner_validation=0.25, epochs=2)
    # per fold: 2 x (train, select), then one held-out evaluation
    for fold, (train_idx, test_idx) in enumerate(result.indices):
        trained = adapter.trained[2 * fold:2 * fold + 2]
        evaluated = adapter.evaluated[3 * fold:3 * fold + 3]
        fit, select, heldout = trained[0], evaluated[0], evaluated[2]
        assert heldout == set(test_idx)
        assert fit | select == set(train_idx) and not fit & select
        assert not set(groups[list(fit)]) & set(groups[list(select)])
        assert all(t == fit for t in trained) and evaluated[1] == select
    assert result.metadata["model_selection"][0]["strategy"].startswith("inner_validation")


def test_no_val_metrics_without_inner_validation():
    logs = Logs()
    _, adapter = _run([logs])
    assert not any(k.startswith(("val_", "heldout")) for keys in logs.keys for k in keys)
    assert len(adapter.evaluated) == 3  # only the final held-out score per fold
    with pytest.warns(UserWarning, match="inner_validation"):
        TorchCVRunner(_model, KFold(2), adapter=PyTorchAdapter(device="cpu")).run(
            _data(), epochs=2, loss_fn=torch.nn.CrossEntropyLoss(),
            callbacks=[EarlyStopping(verbose=False)])


class FakeValLoss(CVCallback):
    """Puts a scripted val_loss in the logs and records the weights after each epoch."""

    def __init__(self, losses):
        self.losses, self.weights = losses, []

    def on_epoch_end(self, epoch, fold_idx, logs=None):
        logs["val_loss"] = self.losses[epoch]
        self.weights.append(logs["snapshot"]())


def test_best_weights_are_restored_before_scoring():
    fake = FakeValLoss([1.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    es = EarlyStopping(patience=3, verbose=False)
    result, _ = _run([fake, es], cv=KFold(2), epochs=6)
    final = result.models[-1].state_dict()
    best, last = fake.weights[-4], fake.weights[-1]  # fold 2: epoch 1 is best, stops at epoch 4
    assert all(torch.equal(final[k], best[k]) for k in final)
    assert not all(torch.equal(best[k], last[k]) for k in best)
    assert result.metadata["model_selection"][-1] == {"strategy": "none", "epochs_trained": 5,
                                                       "restored_best_weights": True}


def test_checkpoints_are_written_and_reload(tmp_path):
    pattern = str(tmp_path / "fold{fold}" / "epoch{epoch}.pt")
    _run([ModelCheckpoint(pattern, save_best_only=False, verbose=False)], cv=KFold(2), epochs=2)
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.pt"))
    assert files == ["fold0/epoch0.pt", "fold0/epoch1.pt", "fold1/epoch0.pt", "fold1/epoch1.pt"]
    assert isinstance(PyTorchAdapter(device="cpu").load_model(str(tmp_path / "fold1/epoch1.pt"), _model),
                      torch.nn.Linear)


def test_heldout_curves_are_metadata_only():
    logs = Logs()
    result, _ = _run([logs], report_heldout_per_epoch=True)
    curves = result.metadata["heldout_curves"]
    assert len(curves) == 3 and [c["epoch"] for c in curves[0]] == [0, 1, 2]
    assert not any(k.startswith(("val_", "heldout")) for keys in logs.keys for k in keys)


class WithExtraState(torch.nn.Linear):
    def get_extra_state(self):
        return {"calibration": [1.0, 2.0]}

    def set_extra_state(self, state):
        self.calibration = state["calibration"]


def test_snapshot_handles_non_tensor_extra_state():
    def model():
        torch.manual_seed(0)
        return WithExtraState(4, 2)

    runner = TorchCVRunner(model, KFold(2), adapter=PyTorchAdapter(device="cpu"), store_models=True)
    result = runner.run(_data(), epochs=3, loss_fn=torch.nn.CrossEntropyLoss(), inner_validation=0.25,
                        callbacks=[EarlyStopping(patience=1, verbose=False)])
    assert result.models[0].calibration == [1.0, 2.0]


@pytest.mark.parametrize("strategy", ["auto", "last"])
def test_documented_inner_strategies_run(strategy):
    result, _ = _run(epochs=1, inner_validation=0.25, inner_validation_strategy=strategy)
    assert result.metadata["model_selection"][0]["strategy"].startswith("inner_validation")
