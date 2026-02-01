"""Example: UniversalCVRunner with `framework='sklearn'` using KerasClassifierWrap.

Quick demo using a tiny CNN on MNIST (small subset) to illustrate passing a
Keras-based sklearn-compatible estimator to TrustCV's UniversalCVRunner.
"""
import numpy as np

from sklearn.model_selection import StratifiedKFold

try:
    from trustcv.frameworks.tensorflow_sklearn import KerasClassifierWrap
    from trustcv.core.runner import UniversalCVRunner
    from trustcv import ClassDistributionLogger
except Exception:
    print("KerasClassifierWrap not available - ensure TensorFlow is installed.")
    raise


def build_mnist_cnn(input_shape, n_classes):
    import tensorflow as tf

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
    import tensorflow as tf

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    N = 2000
    x = (x_train[:N].astype("float32") / 255.0)[..., None]
    y = y_train[:N].astype("int32")

    wrap = KerasClassifierWrap(build_fn=build_mnist_cnn, epochs=1, batch_size=64, verbose=0)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    runner = UniversalCVRunner(cv_splitter=cv, framework="sklearn", verbose=1)

    logger = ClassDistributionLogger(labels=y, verbose=1)

    res = runner.run(model=wrap, data=(x, y), callbacks=[logger])
    print(res.summary())


if __name__ == "__main__":
    main()
