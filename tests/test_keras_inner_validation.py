"""KerasCVRunner never shows the held-out fold to training or model selection."""

import warnings

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
keras = tf.keras

from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402

from trustcv.frameworks.tensorflow import KerasCVRunner  # noqa: E402

N = 120


def _data():
    rng = np.random.RandomState(0)
    ids = np.arange(N, dtype="float32")
    X = np.column_stack([ids, rng.normal(size=(N, 3))]).astype("float32")  # column 0 = row id
    y = np.tile([0, 1], N // 2)
    groups = np.repeat(np.arange(N // 4), 4)
    return X, y, groups


def _runner(cv, record):
    def model_fn():
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(1, activation="sigmoid")])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        original = model.fit

        def fit(x, y=None, **kwargs):
            val = kwargs.get("validation_data")
            record.append(
                {
                    "fit_rows": set(np.asarray(x)[:, 0].astype(int)),
                    "val_rows": None if val is None else set(np.asarray(val[0])[:, 0].astype(int)),
                    "callbacks": list(kwargs.get("callbacks", [])),
                    "validation_split": kwargs.get("validation_split"),
                }
            )
            return original(x, y, **kwargs)

        model.fit = fit
        return model

    return KerasCVRunner(model_fn, cv)


def _heldout(result):
    return [set(te.tolist()) for _, te in result.indices]


def test_default_trains_without_validation_or_early_stopping():
    X, y, _ = _data()
    record = []
    result = _runner(StratifiedKFold(3, shuffle=True, random_state=0), record).run(X, y, epochs=3, verbose=0)
    for fold, held in zip(record, _heldout(result)):
        assert fold["val_rows"] is None and not fold["fit_rows"] & held
        assert not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in fold["callbacks"])
    assert all(s["strategy"] == "none" and s["epochs_trained"] == 3 for s in result.metadata["model_selection"])


def test_inner_validation_uses_training_patients_only():
    X, y, groups = _data()
    record = []
    result = _runner(GroupKFold(3), record).run(
        X, y, epochs=3, groups=groups, inner_validation=0.2, verbose=0
    )
    for fold, held in zip(record, _heldout(result)):
        val = fold["val_rows"]
        assert val and not val & held and not fold["fit_rows"] & held and not val & fold["fit_rows"]
        assert not set(groups[list(val)]) & set(groups[list(fold["fit_rows"])])
        assert any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in fold["callbacks"])
    assert all(s["strategy"] == "inner_validation:group" for s in result.metadata["model_selection"])


def test_user_callbacks_are_fresh_per_fold():
    X, y, _ = _data()
    record = []
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=1)
    _runner(StratifiedKFold(2, shuffle=True, random_state=0), record).run(
        X, y, epochs=2, callbacks=[stopper], inner_validation=0.2, verbose=0
    )
    used = [next(cb for cb in f["callbacks"] if isinstance(cb, keras.callbacks.EarlyStopping)) for f in record]
    assert stopper not in used and used[0] is not used[1]


def test_conflicting_or_manual_validation_is_rejected():
    X, y, _ = _data()
    runner = _runner(StratifiedKFold(2), [])
    with pytest.raises(ValueError):
        runner.run(X, y, epochs=1, inner_validation=0.1, validation_split=0.1)
    with pytest.raises(ValueError):
        runner.run(X, y, epochs=1, validation_data=(X, y))


def test_legacy_validation_split_warns_about_groups():
    X, y, groups = _data()
    record = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _runner(GroupKFold(2), record).run(X, y, epochs=1, groups=groups, validation_split=0.1, verbose=0)
    assert any("not group-aware" in str(w.message) for w in caught)
    assert all(f["validation_split"] == 0.1 and f["val_rows"] is None for f in record)


def test_small_regression_folds_still_get_inner_validation():
    from sklearn.model_selection import KFold

    rng = np.random.RandomState(0)
    X = np.column_stack([np.arange(40, dtype="float32"), rng.normal(size=(40, 3))]).astype("float32")
    y = np.linspace(0, 1, 40).astype("float32")

    def model_fn():
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        return model

    result = KerasCVRunner(model_fn, KFold(2)).run(X, y, epochs=2, inner_validation=0.2, verbose=0)
    assert all(s["strategy"].startswith("inner_validation") for s in result.metadata["model_selection"])


def test_infeasible_group_split_trains_without_selection_instead_of_failing():
    X, y, _ = _data()
    groups = np.repeat([0, 1, 2], N // 3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _runner(GroupKFold(3), []).run(X, y, epochs=1, groups=groups, inner_validation=0.6, verbose=0)
    assert all(s["strategy"] == "none" and "infeasible" in s["reason"] for s in result.metadata["model_selection"])
    assert any("No inner validation split" in str(w.message) for w in caught)
