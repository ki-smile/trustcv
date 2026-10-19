"""UniversalCVRunner's epoch loop: inner selection, restore, and JAX state threading."""

import warnings

import numpy as np
import pytest
from sklearn.model_selection import KFold

from trustcv.core.callbacks import CVCallback, EarlyStopping
from trustcv.core.runner import UniversalCVRunner


def _data(n=60):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    return X, (X[:, 1] > 0).astype(np.int64)


class FakeValLoss(CVCallback):
    def __init__(self, losses):
        self.losses, self.snaps, self.keys = losses, [], []

    def on_epoch_end(self, epoch, fold_idx, logs=None):
        self.keys.append(set(logs))
        logs["val_loss"] = self.losses[epoch]
        self.snaps.append(logs["snapshot"]())


def _run(model, framework, callbacks, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return UniversalCVRunner(KFold(2), framework=framework, verbose=0).run(
            model, _data(), callbacks=callbacks, **kw)


def test_torch_restores_best_weights_and_hides_heldout_metrics():
    torch = pytest.importorskip("torch")

    def model():
        torch.manual_seed(0)
        return torch.nn.Linear(4, 2)

    fake = FakeValLoss([1.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    result = _run(model, "pytorch", [fake, EarlyStopping(patience=3, verbose=False)], epochs=6,
                  loss_fn=torch.nn.CrossEntropyLoss())
    assert not any(k.startswith("val_") for k in fake.keys[0])  # no inner_validation: nothing to select on
    final, best = result.models[-1].state_dict(), fake.snaps[-4]
    assert all(torch.equal(final[k].cpu(), best[k]) for k in final)
    assert result.metadata["model_selection"][-1]["epochs_trained"] == 5


def test_torch_inner_validation_feeds_val_metrics():
    torch = pytest.importorskip("torch")
    fake = FakeValLoss([1.0] * 3)
    result = _run(lambda: torch.nn.Linear(4, 2), "pytorch", [fake], epochs=3,
                  loss_fn=torch.nn.CrossEntropyLoss(), inner_validation=0.25)
    assert "val_loss" in fake.keys[0]
    sel = result.metadata["model_selection"][0]
    assert sel["n_inner_val"] > 0 and sel["n_inner_train"] + sel["n_inner_val"] == 30


def test_jax_state_is_threaded_through_training_and_scoring():
    pytest.importorskip("jax")
    nn = pytest.importorskip("flax.linen")
    optax = pytest.importorskip("optax")

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x, training=False):
            return nn.Dense(2)(x)

    fake = FakeValLoss([1.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    result = _run(MLP, "jax", [fake, EarlyStopping(patience=3, verbose=False)], epochs=6,
                  optimizer=optax.adam(1e-2))
    assert fake.snaps[0] is not None and fake.snaps[1].step > fake.snaps[0].step  # training continued from the state
    assert result.models[-1] is fake.snaps[-4]  # the restored best state is what was scored
    assert 0.0 <= result.scores[-1]["val_acc"] <= 1.0
