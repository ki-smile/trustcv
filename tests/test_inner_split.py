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


def _epochs(cb, logs):
    cb.on_cv_start(2)
    cb.on_fold_start(0, None, None)
    for epoch in range(3):
        cb.on_epoch_end(epoch, 0, dict(logs))


def test_selection_on_heldout_metrics_warns_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _epochs(EarlyStopping(monitor="val_loss", verbose=False), {"val_loss": 1.0, "model": None})
    assert sum("held-out CV fold" in str(w.message) for w in caught) == 1


@pytest.mark.parametrize(
    "callback, logs",
    [
        (EarlyStopping(monitor="train_loss", verbose=False), {"train_loss": 1.0}),
        (ModelCheckpoint("m_{fold}.pt", monitor="val_loss", save_best_only=False, verbose=False), {"val_loss": 1.0, "model": None}),
    ],
)
def test_no_warning_when_nothing_is_selected_on_heldout_metrics(callback, logs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _epochs(callback, logs)
    assert not any("held-out CV fold" in str(w.message) for w in caught)


# --- regressions from the Codex implementation review ------------------------


def test_last_with_groups_purges_instead_of_validating_on_the_past():
    groups = np.arange(100)
    groups[95] = 0  # patient 0 appears at rows 0 and 95
    (tr, va), reason = inner_validation_split(np.arange(100), groups=groups, strategy="last", fraction=0.1)
    assert va.min() > tr.max()  # validation strictly after training
    assert 0 not in tr and 0 not in va and "purged" in reason
    assert not set(groups[tr]) & set(groups[va])


def test_continuous_regression_targets_are_not_treated_as_classes():
    y = np.linspace(0, 1, 20).repeat(1)  # 20 distinct continuous values
    split, reason = inner_validation_split(np.arange(20), y, fraction=0.2, random_state=0)
    assert split is not None and reason == "random"


def test_infeasible_group_fraction_falls_back_instead_of_raising():
    split, reason = inner_validation_split(np.arange(40), groups=np.repeat([0, 1], 20), fraction=0.6, random_state=0)
    assert split is None and "infeasible" in reason
