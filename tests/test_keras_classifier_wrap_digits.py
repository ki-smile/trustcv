import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold

from trustcv.frameworks.tensorflow_sklearn import KerasClassifierWrap
from trustcv.core.runner import UniversalCVRunner


def build_cnn(input_shape, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((8, 8, 1)),
        tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def test_keras_classifier_wrap_digits_predict_shapes_and_cv():
    X, y = load_digits(return_X_y=True)
    X = X.astype("float32") / 16.0
    X = X[:200]
    y = y[:200]

    wrap = KerasClassifierWrap(build_fn=build_cnn, epochs=1, batch_size=32, verbose=0)
    wrap.fit(X, y)
    preds = wrap.predict(X)
    proba = wrap.predict_proba(X)

    assert preds.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0], len(np.unique(y)))

    # quick CV run with UniversalCVRunner
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    runner = UniversalCVRunner(cv_splitter=cv, framework="sklearn", verbose=0)
    res = runner.run(model=KerasClassifierWrap(build_fn=build_cnn, epochs=1, batch_size=32, verbose=0), data=(X, y))

    # Expect 3 fold details and a mean_scores dict with accuracy
    assert hasattr(res, "fold_details")
    assert len(res.fold_details) == 3
    assert isinstance(res.mean_scores, dict)
    assert "accuracy" in res.mean_scores
