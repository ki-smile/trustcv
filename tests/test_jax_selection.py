"""JAXCVRunner selects on an inner split, honours "stop" and restores the best state."""

import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nn = pytest.importorskip("flax.linen")
pytest.importorskip("optax")

from sklearn.model_selection import GroupKFold, KFold

from trustcv.core.callbacks import CVCallback, EarlyStopping, ModelCheckpoint
from trustcv.frameworks.jax import JAXAdapter, JAXCVRunner


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        return nn.Dense(2)(x)


def _data(n=60):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[:, 0] = np.arange(n)  # row id, so the test can see which rows each step used
    return X, (X[:, 1] > 0).astype(np.int32)


class RecordingAdapter(JAXAdapter):
    def __init__(self):
        super().__init__(batch_size=16, use_jit=False)
        self.trained, self.evaluated = [], []

    def train_epoch(self, model, train_data, *args, **kwargs):
        self.trained.append(set(train_data[0][:, 0].astype(int)))
        return super().train_epoch(model, train_data, *args, **kwargs)

    def evaluate(self, model, val_data, *args, **kwargs):
        self.evaluated.append(set(val_data[0][:, 0].astype(int)))
        return super().evaluate(model, val_data, *args, **kwargs)


def _run(callbacks=(), cv=None, groups=None, epochs=3, **kw):
    adapter = RecordingAdapter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = JAXCVRunner(MLP, cv or KFold(3), adapter=adapter, store_models=True).run(
            *_data(), epochs=epochs, callbacks=list(callbacks), groups=groups, **kw)
    return result, adapter


def test_heldout_rows_never_reach_training_or_selection():
    groups = np.repeat(np.arange(12), 5)
    result, adapter = _run(cv=GroupKFold(3), groups=groups, inner_validation=0.25, epochs=2)
    for fold, (train_idx, test_idx) in enumerate(result.indices):
        fit = adapter.trained[2 * fold]
        select, heldout = adapter.evaluated[3 * fold], adapter.evaluated[3 * fold + 2]
        assert heldout == set(test_idx)
        assert fit | select == set(train_idx) and not fit & select
        assert not set(groups[list(fit)]) & set(groups[list(select)])


class FakeValLoss(CVCallback):
    def __init__(self, losses):
        self.losses, self.states = losses, []

    def on_epoch_end(self, epoch, fold_idx, logs=None):
        logs["val_loss"] = self.losses[epoch]
        self.states.append(logs["snapshot"]())


def test_stops_and_restores_the_best_state():
    fake = FakeValLoss([1.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    result, _ = _run([fake, EarlyStopping(patience=3, verbose=False)], cv=KFold(2), epochs=6)
    assert result.metadata["model_selection"][-1]["epochs_trained"] == 5
    final, best = result.models[-1].params, fake.states[-4].params
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree_util.tree_leaves(final),
                                                     jax.tree_util.tree_leaves(best)))


def test_checkpoints_are_written(tmp_path):
    pattern = str(tmp_path / "fold{fold}" / "epoch{epoch}")
    _run([ModelCheckpoint(pattern, save_best_only=False, verbose=False)], cv=KFold(2), epochs=2)
    assert {p.name for p in tmp_path.iterdir()} == {"fold0", "fold1"}
    assert any((tmp_path / "fold1").iterdir())
