"""
Example: Using KerasSkWrap with UniversalCVRunner on the sklearn digits dataset.

TensorFlow must be installed to run this example.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_digits

from trustcv.core.runner import UniversalCVRunner
from trustcv.frameworks.tensorflow_sklearn import KerasSkWrap


def build_mlp(input_shape, n_classes, hidden=64):
    import tensorflow as tf  # local import to keep TF optional elsewhere

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


def main():
    X, y = load_digits(return_X_y=True)
    X = X.astype("float32") / 16.0

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    runner = UniversalCVRunner(cv_splitter=cv, framework="sklearn", verbose=1)

    model = KerasSkWrap(build_fn=build_mlp, epochs=3, batch_size=64, verbose=0)
    results = runner.run(model=model, data=(X, y))

    print("Per-fold metrics:")
    for i, m in enumerate(results.scores, 1):
        print(f"Fold {i}: {m}")

    print("\nSummary:", results.metrics)


if __name__ == "__main__":
    main()
