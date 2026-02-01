"""
Mini-demo for the enhanced Keras sklearn-compatible wrapper.

Each example is intentionally tiny (few samples, 1 epoch) to show the API
for multilabel, regression, multi-input, multi-output, and dataset-based training.
"""

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - example is skipped if TF missing
    tf = None

from trustcv.frameworks.tensorflow_sklearn import KerasRegressorWrap, KerasSkWrap


def run_multilabel():
    from sklearn.datasets import make_multilabel_classification

    X, y = make_multilabel_classification(
        n_samples=40, n_features=12, n_classes=3, n_labels=1, random_state=0
    )
    X = X.astype("float32")

    def build(input_shape, n_classes):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(n_classes, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    wrap = KerasSkWrap(build_fn=build, epochs=1, batch_size=8, verbose=0, task="multilabel")
    wrap.fit(X, y)
    print("Multilabel proba:", wrap.predict_proba(X[:3]))
    print("Multilabel preds:", wrap.predict(X[:3]))


def run_regression():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    X = X[:60].astype("float32")
    y = y[:60].astype("float32")

    def build(input_shape, n_classes=None):
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Dense(32, activation="relu")(inputs)
        out = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs, out)
        model.compile(optimizer="adam", loss="mse")
        return model

    wrap = KerasRegressorWrap(build_fn=build, epochs=1, batch_size=8, verbose=0)
    wrap.fit(X, y)
    print("Regression preds:", wrap.predict(X[:5]))


def run_multi_input():
    rng = np.random.default_rng(2)
    n = 50
    X = {
        "x1": rng.standard_normal((n, 5), dtype=np.float32),
        "x2": rng.standard_normal((n, 3), dtype=np.float32),
    }
    y = (X["x1"][:, 0] + 0.3 * X["x2"][:, 0] > 0).astype("int32")

    def build(input_shape, n_classes):
        in1 = tf.keras.Input(shape=input_shape["x1"], name="x1")
        in2 = tf.keras.Input(shape=input_shape["x2"], name="x2")
        a = tf.keras.layers.Dense(8, activation="relu")(in1)
        b = tf.keras.layers.Dense(8, activation="relu")(in2)
        x = tf.keras.layers.Concatenate()([a, b])
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model([in1, in2], out)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    wrap = KerasSkWrap(build_fn=build, epochs=1, batch_size=8, verbose=0, task="binary")
    wrap.fit(X, y)
    print("Multi-input proba shape:", wrap.predict_proba(X).shape)


def run_multi_output():
    rng = np.random.default_rng(3)
    n = 40
    X = rng.standard_normal((n, 10), dtype=np.float32)
    y = {
        "class": (X[:, 0] > 0).astype("int32"),
        "reg": X[:, 1] * 0.2,
    }

    def build(input_shape, n_classes):
        inp = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Dense(12, activation="relu")(inp)
        class_out = tf.keras.layers.Dense(1, activation="sigmoid", name="class")(x)
        reg_out = tf.keras.layers.Dense(1, name="reg")(x)
        model = tf.keras.Model(inp, {"class": class_out, "reg": reg_out})
        model.compile(
            optimizer="adam",
            loss={"class": "binary_crossentropy", "reg": "mse"},
        )
        return model

    wrap = KerasSkWrap(
        build_fn=build,
        epochs=1,
        batch_size=8,
        verbose=0,
        output_key="class",
        task="binary",
    )
    wrap.fit(X, y)
    print("Multi-output class proba:", wrap.predict_proba(X[:3]))


def run_dataset_custom_loop():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((64, 6), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(16)

    def build(*args, **kwargs):
        class CustomModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

            def call(self, inputs):
                return self.dense(inputs)

            def train_step(self, data):
                x, y_true = data
                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
                grads = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                self.compiled_metrics.update_state(y_true, y_pred)
                return {m.name: m.result() for m in self.metrics}

        model = CustomModel()
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    wrap = KerasSkWrap(build_fn=build, epochs=2, batch_size=16, verbose=0, task="binary")
    wrap.fit(ds, y=None)
    print("Dataset-based proba shape:", wrap.predict_proba(X[:5]).shape)


if __name__ == "__main__":
    if tf is None:
        print("TensorFlow not installed; skip examples.")
    else:
        run_multilabel()
        run_regression()
        run_multi_input()
        run_multi_output()
        run_dataset_custom_loop()

