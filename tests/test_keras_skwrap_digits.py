import numpy as np
import pytest
from sklearn.datasets import load_digits

from trustcv.validators import TrustCVValidator

# Skip if TensorFlow unavailable
tf = pytest.importorskip("tensorflow", reason="TensorFlow required for KerasSkWrap tests")

from trustcv.frameworks.tensorflow_sklearn import KerasSkWrap


def build_mlp(input_shape, n_classes, hidden=32):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(hidden, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def test_keras_skwrap_shapes_and_cv():
    X, y = load_digits(return_X_y=True)
    X = X.astype("float32") / 16.0
    X_small, y_small = X[:200], y[:200]  # keep test fast

    wrap = KerasSkWrap(build_fn=build_mlp, epochs=1, batch_size=64, verbose=0)
    wrap.fit(X_small, y_small)

    proba = wrap.predict_proba(X_small[:10])
    pred = wrap.predict(X_small[:10])

    assert proba.shape == (10, 10)
    assert pred.shape == (10,)

    validator = TrustCVValidator(method="stratified_kfold", n_splits=2, shuffle=True, random_state=0)
    res = validator.validate(model=KerasSkWrap(build_fn=build_mlp, epochs=1, batch_size=64, verbose=0),
                             X=X_small, y=y_small, metrics=["accuracy"])

    assert "accuracy" in res.mean_scores
    assert len(res.fold_details) == 2
