import numpy as np
import pytest
from sklearn.datasets import load_diabetes, make_multilabel_classification
from sklearn.model_selection import KFold

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for Keras wrapper stress tests")

from trustcv import TrustCVValidator
from trustcv.core.runner import UniversalCVRunner
from trustcv.frameworks.tensorflow_sklearn import KerasRegressorWrap, KerasSkWrap


# ---- helper builders ----
def _build_multilabel(input_shape, n_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_regressor(input_shape, n_classes=None):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32, activation="relu")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mse"])
    return model


def _build_multi_input(input_shape, n_classes, lr=1e-3):
    in1 = tf.keras.layers.Input(shape=input_shape["x1"], name="x1")
    in2 = tf.keras.layers.Input(shape=input_shape["x2"], name="x2")
    a = tf.keras.layers.Dense(8, activation="relu")(in1)
    b = tf.keras.layers.Dense(4, activation="relu")(in2)
    x = tf.keras.layers.Concatenate()([a, b])
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model([in1, in2], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_multi_output(input_shape, n_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(12, activation="relu")(inputs)
    class_out = tf.keras.layers.Dense(1, activation="sigmoid", name="class")(x)
    reg_out = tf.keras.layers.Dense(1, name="reg")(x)
    model = tf.keras.Model(inputs=inputs, outputs={"class": class_out, "reg": reg_out})
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"class": "binary_crossentropy", "reg": "mse"},
        metrics={"class": ["accuracy"], "reg": ["mse"]},
    )
    return model


def _build_custom_loop_model(*args, **kwargs):
    class CustomModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

        def call(self, inputs):
            return self.dense(inputs)

        def train_step(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.compiled_metrics.update_state(y, y_pred)
            return {m.name: m.result() for m in self.metrics}

    model = CustomModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ---- tests ----
def test_multilabel_support_and_runner():
    X, y = make_multilabel_classification(
        n_samples=80,
        n_features=16,
        n_classes=3,
        n_labels=1,
        allow_unlabeled=False,
        random_state=0,
    )
    X = X.astype("float32")
    wrap = KerasSkWrap(build_fn=_build_multilabel, epochs=1, batch_size=16, verbose=0, task="multilabel")
    wrap.fit(X, y)

    proba = wrap.predict_proba(X[:5])
    pred = wrap.predict(X[:5])

    assert proba.shape == (5, 3)
    assert pred.shape == (5, 3)
    assert set(np.unique(pred)).issubset({0, 1})

    validator = TrustCVValidator(method="kfold", n_splits=2, shuffle=True, random_state=0)
    res = validator.validate(
        model=KerasSkWrap(build_fn=_build_multilabel, epochs=1, batch_size=16, verbose=0, task="multilabel"),
        X=X,
        y=y,
        scoring={"acc": "accuracy"},
    )
    assert "acc" in res.scores
    assert len(res.scores["acc"]) == 2


def test_regression_support_and_runner():
    X, y = load_diabetes(return_X_y=True)
    X = X.astype("float32")
    y = y.astype("float32")
    X_small, y_small = X[:80], y[:80]

    wrap = KerasRegressorWrap(build_fn=_build_regressor, epochs=1, batch_size=16, verbose=0)
    wrap.fit(X_small, y_small)
    pred = wrap.predict(X_small[:10])

    assert pred.shape == (10,)

    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    runner = UniversalCVRunner(cv_splitter=cv, framework="sklearn", verbose=0)
    res = runner.run(
        model=KerasRegressorWrap(build_fn=_build_regressor, epochs=1, batch_size=16, verbose=0),
        data=(X_small, y_small),
    )
    assert len(res.fold_details) == 2


def test_multi_input_dict_and_runner():
    rng = np.random.default_rng(42)
    n = 80
    X = {
        "x1": rng.standard_normal((n, 20), dtype=np.float32),
        "x2": rng.standard_normal((n, 10), dtype=np.float32),
    }
    y = (X["x1"][:, 0] + 0.4 * X["x2"][:, 0] > 0).astype("int32")

    wrap = KerasSkWrap(build_fn=_build_multi_input, epochs=1, batch_size=16, verbose=0, task="binary")
    wrap.fit(X, y)

    proba = wrap.predict_proba(X)
    assert proba.shape == (n, 2)

    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    runner = UniversalCVRunner(cv_splitter=cv, framework="sklearn", verbose=0)
    res = runner.run(
        model=KerasSkWrap(build_fn=_build_multi_input, epochs=1, batch_size=16, verbose=0, task="binary"),
        data=(X, y),
    )
    assert len(res.fold_details) == 2


def test_multi_output_selection():
    rng = np.random.default_rng(1)
    n = 60
    X = rng.standard_normal((n, 15), dtype=np.float32)
    y_class = (X[:, 0] + 0.25 * X[:, 1] > 0).astype("int32")
    y_reg = X[:, 2] * 0.5 + 0.1
    y = {"class": y_class, "reg": y_reg}

    wrap = KerasSkWrap(
        build_fn=_build_multi_output,
        epochs=1,
        batch_size=16,
        verbose=0,
        output_key="class",
        task="binary",
    )
    wrap.fit(X, y)

    proba = wrap.predict_proba(X[:8])
    pred = wrap.predict(X[:8])

    assert proba.shape == (8, 2)
    assert pred.shape == (8,)
    assert set(np.unique(pred)).issubset({0, 1})


def test_dataset_with_custom_train_step():
    rng = np.random.default_rng(3)
    n = 64
    X = rng.standard_normal((n, 6), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(16)

    wrap = KerasSkWrap(
        build_fn=_build_custom_loop_model,
        epochs=2,
        batch_size=16,
        verbose=0,
        task="binary",
        allow_dataset=True,
    )
    wrap.fit(ds, y=None)

    proba = wrap.predict_proba(X[:10])
    assert proba.shape == (10, 2)

