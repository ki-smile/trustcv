import warnings

import numpy as np
import pytest

from trustcv.core.callbacks import EarlyStopping, ModelCheckpoint
from trustcv.core.inner_split import inner_validation_split


def test_split_is_inside_the_training_fold_and_disjoint():
    train = np.arange(10, 110)
    (tr, va), reason = inner_validation_split(train, random_state=0)
    assert reason == "random"
    assert set(tr) | set(va) == set(train) and not set(tr) & set(va)


def test_groups_never_cross_the_inner_split():
    y = np.tile([0, 1], 60)
    groups = np.repeat(np.arange(30), 4)
    (tr, va), reason = inner_validation_split(np.arange(120), y, groups, fraction=0.2, random_state=0)
    assert reason == "group"
    assert not set(groups[tr]) & set(groups[va])


def test_stratified_when_feasible_and_random_otherwise():
    y = np.r_[np.zeros(80, int), np.ones(20, int)]
    (tr, va), reason = inner_validation_split(np.arange(100), y, random_state=0)
    assert reason == "stratified" and set(y[va]) == {0, 1}
    three = np.repeat([0, 1, 2], [7, 7, 6])  # 2 validation rows cannot hold 3 classes
    split, reason = inner_validation_split(np.arange(20), three, random_state=0)
    assert reason.startswith("random") and split is not None
    assert set(three[split[0]]) == {0, 1, 2}


def test_bootstrap_repeats_stay_on_one_side_and_keep_multiplicity():
    train = np.random.RandomState(0).randint(0, 100, 100)
    (tr, va), _ = inner_validation_split(train, np.tile([0, 1], 50), random_state=0)
    assert len(tr) + len(va) == len(train)
    assert not set(tr) & set(va)


def test_last_strategy_takes_the_tail_and_keeps_groups_whole():
    train = np.arange(100)
    (tr, va), reason = inner_validation_split(train, strategy="last", fraction=0.1)
    assert reason == "last" and list(va) == list(range(90, 100))
    groups = np.repeat(np.arange(25), 4)
    (tr, va), _ = inner_validation_split(np.arange(98), groups=groups, strategy="last", fraction=0.1)
    assert not set(groups[tr]) & set(groups[va])


@pytest.mark.parametrize(
    "train, kwargs",
    [
        (np.arange(3), {}),  # too few rows
        (np.arange(40), {"groups": np.zeros(40)}),  # a single group
        (np.arange(40), {"y": np.zeros(40, int)}),  # a single class
    ],
)
def test_infeasible_splits_return_none_with_reason(train, kwargs):
    split, reason = inner_validation_split(train, random_state=0, **kwargs)
    assert split is None and reason


@pytest.mark.parametrize("kwargs", [{"fraction": 0.0}, {"fraction": 1.0}, {"strategy": "time"}])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        inner_validation_split(np.arange(50), **kwargs)


def test_seeded_splits_are_reproducible():
    a, _ = inner_validation_split(np.arange(100), random_state=3)
    b, _ = inner_validation_split(np.arange(100), random_state=3)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def _epochs(cb, logs, n=3):
    cb.on_cv_start(2)
    cb.on_fold_start(0, None, None)
    for epoch in range(n):
        cb.on_epoch_end(epoch, 0, dict(logs))


@pytest.mark.parametrize("make", [lambda: EarlyStopping(monitor="heldout_loss"),
                                  lambda: ModelCheckpoint("m.pt", monitor="heldout_auc")])
def test_selecting_on_heldout_metrics_is_refused(make):
    with pytest.raises(ValueError, match="held-out CV fold"):
        make()


def test_missing_monitor_warns_once_per_run():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _epochs(EarlyStopping(monitor="val_loss", verbose=False), {"train_loss": 1.0}, n=5)
    assert sum("inner_validation" in str(w.message) for w in caught) == 1


def test_early_stopping_keeps_a_copy_of_the_best_weights():
    weights = {"w": [1.0]}
    es = EarlyStopping(monitor="val_loss", patience=10, verbose=False)
    es.on_cv_start(1)
    es.on_fold_start(0, None, None)
    snapshot = lambda: {"w": list(weights["w"])}
    es.on_epoch_end(0, 0, {"val_loss": 1.0, "snapshot": snapshot})
    weights["w"][0] = 2.0  # training continues and changes the live weights
    es.on_epoch_end(1, 0, {"val_loss": 5.0, "snapshot": snapshot})  # worse: no new snapshot
    assert es.best_weights == {"w": [1.0]}


def test_run_epoch_callbacks_runs_all_before_stopping():
    from trustcv.core.selection import run_epoch_callbacks

    seen = []

    class Stopper:
        def on_epoch_end(self, epoch, fold, logs):
            return "stop"

    class Recorder:
        def on_epoch_end(self, epoch, fold, logs):
            seen.append(epoch)
            logs["checkpoints"].append("a.pt")

    logs = {}
    assert run_epoch_callbacks([Stopper(), Recorder(), Recorder()], 3, 0, logs) is True
    assert seen == [3, 3] and logs["checkpoints"] == ["a.pt", "a.pt"]


def test_fold_selection_split_without_inner_validation_uses_the_whole_fold():
    from trustcv.core.selection import fold_selection_split

    fit, select, info = fold_selection_split(np.arange(10, 30), inner_validation=0.0)
    assert select is None and list(fit) == list(range(10, 30)) and info == {"strategy": "none"}
    fit, select, info = fold_selection_split(np.arange(100), np.tile([0, 1], 50), inner_validation=0.2)
    assert not set(fit) & set(select) and info["strategy"].startswith("inner_validation")
