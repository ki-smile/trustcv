"""Nested CV lets callers control the inner grid search's parallelism."""

import numpy as np
import pytest
import sklearn.model_selection as skms
from sklearn.linear_model import LogisticRegression, Ridge

from trustcv.splitters import NestedCV, NestedGroupedCV, NestedTemporalCV


@pytest.fixture
def recorded(monkeypatch):
    seen = []
    real = skms.GridSearchCV

    def recording(*args, **kwargs):  # the library imports GridSearchCV at call time
        seen.append(kwargs.get("n_jobs"))
        return real(*args, **kwargs)

    monkeypatch.setattr(skms, "GridSearchCV", recording)
    return seen


def test_default_stays_all_cores():
    assert NestedCV().n_jobs == NestedGroupedCV().n_jobs == NestedTemporalCV().n_jobs == -1


def test_n_jobs_reaches_the_inner_grid_search(recorded):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.tile([0, 1], 30)
    NestedCV(n_jobs=1).fit_predict(LogisticRegression(), X, y, {"C": [0.1, 1.0]})
    NestedGroupedCV(n_jobs=1).fit_predict(LogisticRegression(), X, y, np.repeat(np.arange(15), 4), {"C": [0.1, 1.0]})
    assert recorded and set(recorded) == {1}
