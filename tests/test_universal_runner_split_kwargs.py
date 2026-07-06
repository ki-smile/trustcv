"""Regression tests for UniversalCVRunner splitter/fit keyword routing."""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

from trustcv.core.base import CVResults
from trustcv.core.runner import UniversalCVRunner
from trustcv.splitters.spatial import (
    BufferedSpatialCV,
    EnvironmentalHealthCV,
    SpatialBlockCV,
    SpatiotemporalBlockCV,
)


class CoordinateRequiredSplitter:
    """Small splitter that proves splitter-only metadata is delivered."""

    def __init__(self):
        self.received_coordinates = None

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2

    def split(self, X, y=None, groups=None, *, coordinates):
        self.received_coordinates = coordinates
        indices = np.arange(len(X))
        midpoint = len(indices) // 2
        yield indices[:midpoint], indices[midpoint:]
        yield indices[midpoint:], indices[:midpoint]


class CoordinateRejectingClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-style estimator that fails if splitter metadata reaches fit."""

    fit_kwargs_seen = []

    def fit(self, X, y, **kwargs):
        if "coordinates" in kwargs:
            raise AssertionError("coordinates should not be passed to model.fit")
        type(self).fit_kwargs_seen.append(dict(kwargs))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        class_positions = np.arange(len(X)) % len(self.classes_)
        return self.classes_[class_positions]

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        class_positions = np.arange(len(X)) % len(self.classes_)
        proba[np.arange(len(X)), class_positions] = 1.0
        return proba

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


def _make_spatial_binary_data(n_per_axis=8):
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, n_per_axis),
        np.linspace(0.0, 1.0, n_per_axis),
    )
    coordinates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    X = np.column_stack([
        coordinates,
        np.sin(2 * np.pi * coordinates[:, 0]),
        np.cos(2 * np.pi * coordinates[:, 1]),
    ])
    y = (np.arange(len(X)) % 2).astype(int)
    return X, y, coordinates


def _assert_runner_completed(results):
    assert isinstance(results, CVResults)
    assert len(results.scores) > 0
    assert all(isinstance(score, dict) for score in results.scores)
    assert results.models is not None
    assert results.indices is not None


def test_standard_kfold_still_works_with_universal_runner():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    runner = UniversalCVRunner(
        cv_splitter=KFold(n_splits=3, shuffle=True, random_state=42),
        framework="sklearn",
        verbose=0,
    )
    results = runner.run(
        model=DecisionTreeClassifier(random_state=42),
        data=(X, y),
        metrics=["accuracy"],
    )

    _assert_runner_completed(results)
    assert len(results.scores) == 3
    assert any("accuracy" in fold for fold in results.scores)


def test_splitter_only_metadata_does_not_reach_model_fit():
    X, y, coordinates = _make_spatial_binary_data(n_per_axis=6)
    splitter = CoordinateRequiredSplitter()
    CoordinateRejectingClassifier.fit_kwargs_seen = []

    runner = UniversalCVRunner(
        cv_splitter=splitter,
        framework="sklearn",
        verbose=0,
    )
    results = runner.run(
        model=CoordinateRejectingClassifier(),
        data=(X, y),
        split_kwargs={"coordinates": coordinates},
        fit_kwargs={},
    )

    _assert_runner_completed(results)
    assert splitter.received_coordinates is coordinates
    assert CoordinateRejectingClassifier.fit_kwargs_seen
    assert all("coordinates" not in kwargs for kwargs in CoordinateRejectingClassifier.fit_kwargs_seen)


def test_spatial_block_cv_works_with_universal_runner_split_kwargs():
    X, y, coordinates = _make_spatial_binary_data()
    runner = UniversalCVRunner(
        cv_splitter=SpatialBlockCV(n_splits=4, random_state=42),
        framework="sklearn",
        verbose=0,
    )

    results = runner.run(
        model=DecisionTreeClassifier(random_state=42),
        data=(X, y),
        metrics=["accuracy", "roc_auc"],
        split_kwargs={"coordinates": coordinates},
    )

    _assert_runner_completed(results)


def test_buffered_spatial_cv_works_with_universal_runner_split_kwargs():
    X, y, coordinates = _make_spatial_binary_data()
    runner = UniversalCVRunner(
        cv_splitter=BufferedSpatialCV(n_splits=4, buffer_size=0.0, random_state=42),
        framework="sklearn",
        verbose=0,
    )

    results = runner.run(
        model=DecisionTreeClassifier(random_state=42),
        data=(X, y),
        metrics=["accuracy", "roc_auc"],
        split_kwargs={"coordinates": coordinates},
    )

    _assert_runner_completed(results)


def test_spatiotemporal_block_cv_works_with_universal_runner_split_kwargs():
    X, y, coordinates = _make_spatial_binary_data()
    timestamps = np.arange(len(X))
    runner = UniversalCVRunner(
        cv_splitter=SpatiotemporalBlockCV(
            n_spatial_blocks=2,
            n_temporal_blocks=2,
            random_state=42,
        ),
        framework="sklearn",
        verbose=0,
    )

    results = runner.run(
        model=DecisionTreeClassifier(random_state=42),
        data=(X, y),
        metrics=["accuracy", "roc_auc"],
        split_kwargs={"coordinates": coordinates, "timestamps": timestamps},
    )

    _assert_runner_completed(results)


def test_environmental_health_cv_works_with_universal_runner_split_kwargs():
    X, y, coordinates = _make_spatial_binary_data(n_per_axis=12)
    environmental_data = {"pm25": np.linspace(5.0, 25.0, len(X))}
    runner = UniversalCVRunner(
        cv_splitter=EnvironmentalHealthCV(
            spatial_blocks=2,
            temporal_strategy="seasonal",
            environmental_vars=["pm25"],
        ),
        framework="sklearn",
        verbose=0,
    )

    results = runner.run(
        model=DecisionTreeClassifier(random_state=42),
        data=(X, y),
        metrics=["accuracy", "roc_auc"],
        split_kwargs={
            "coordinates": coordinates,
            "environmental_data": environmental_data,
        },
    )

    _assert_runner_completed(results)