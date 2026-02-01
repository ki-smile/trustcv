import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for Keras multi-input tests")

from trustcv import TrustCVValidator
from trustcv.frameworks.tensorflow_sklearn import KerasSkWrap


def _build_multi_input(input_shape, n_classes, lr=1e-3):
    # input_shape is expected to be a dict with keys matching the inputs
    in1 = tf.keras.layers.Input(shape=input_shape["x1"], name="x1")
    in2 = tf.keras.layers.Input(shape=input_shape["x2"], name="x2")
    a = tf.keras.layers.Dense(8, activation="relu")(in1)
    b = tf.keras.layers.Dense(4, activation="relu")(in2)
    x = tf.keras.layers.Concatenate()([a, b])
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model([in1, in2], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def test_multi_input_dict_runs_without_cardinality_error():
    rng = np.random.default_rng(0)
    n = 120
    X1 = rng.standard_normal((n, 5), dtype=np.float32)
    X2 = rng.standard_normal((n, 3), dtype=np.float32)
    y = (X1[:, 0] + 0.2 * X2[:, 0] > 0).astype("int32")

    X = {"x1": X1, "x2": X2}

    model = KerasSkWrap(build_fn=_build_multi_input, epochs=1, batch_size=16, verbose=0, task="binary")
    v = TrustCVValidator(method="stratified_kfold", n_splits=2, shuffle=True, random_state=0)
    res = v.validate(model=model, X=X, y=y, scoring={"acc": "accuracy"})

    acc_scores = res.scores["acc"]
    assert not np.isnan(acc_scores).any()
    assert acc_scores.mean() > 0.4  # sanity: better than random
