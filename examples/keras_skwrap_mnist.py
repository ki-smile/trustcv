"""Example: use `KerasSkWrap` with `UniversalCVRunner` and MNIST.

Run this example locally to quickly verify the sklearn-compatible wrapper for
Keras models. TensorFlow is optional; the example will print a friendly
message if TF isn't installed.
"""
from sklearn.model_selection import StratifiedKFold

try:
    from trustcv.frameworks.tensorflow_sklearn import KerasSkWrap
    from trustcv.core.runner import UniversalCVRunner
    from trustcv import ClassDistributionLogger
except Exception as exc:  # TF not available or import failed
    print("KerasSkWrap not available - ensure TensorFlow is installed.")
    raise

import numpy as np
import tensorflow as tf


def build_mnist_cnn(input_shape, n_classes):
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    N = 2000
    x = (x_train[:N].astype("float32") / 255.0)[..., None]
    y = y_train[:N].astype("int32")

    sk_wrap = KerasSkWrap(build_fn=build_mnist_cnn, epochs=1, batch_size=64, verbose=0)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    runner = UniversalCVRunner(cv_splitter=cv, framework="sklearn", verbose=1)

    logger = ClassDistributionLogger(labels=y, verbose=1)

    res = runner.run(model=sk_wrap, data=(x, y), callbacks=[logger])
    print(res.summary())


if __name__ == "__main__":
    main()
