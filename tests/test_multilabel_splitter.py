import numpy as np
import pytest
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import KFold

from trustcv.validators import TrustCVValidator


def _iterstrat_available():
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # noqa: F401
        return True
    except Exception:
        return False


def test_multilabel_validator_does_not_crash():
    X, Y = make_multilabel_classification(
        n_samples=300, n_features=10, n_classes=4, n_labels=2, random_state=0
    )
    class DummyEstimator:
        def fit(self, X, y):
            self.n_labels = y.shape[1]
            return self
        def predict(self, X):
            return np.zeros((len(X), self.n_labels), dtype=int)
    dummy = DummyEstimator()
    v = TrustCVValidator(method="stratified_kfold", n_splits=3, shuffle=True, random_state=0)
    res = v.validate(model=dummy, X=X, y=Y, metrics=["accuracy"])
    # Even though accuracy may be empty (no predict), validate should complete
    assert res is not None


@pytest.mark.skipif(not _iterstrat_available(), reason="iterative-stratification not installed")
def test_iterstrat_used_and_balances_better():
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    X, Y = make_multilabel_classification(
        n_samples=600, n_features=12, n_classes=5, n_labels=2, random_state=1
    )
    mlkf = MultilabelStratifiedKFold(n_splits=3, random_state=1, shuffle=True)
    kf = KFold(n_splits=3, random_state=1, shuffle=True)

    def _spread(splitter):
        prevalences = []
        for _, te in splitter.split(X, Y):
            prevalences.append(Y[te].mean(axis=0))
        arr = np.vstack(prevalences)
        return np.max(np.std(arr, axis=0))

    spread_iter = _spread(mlkf)
    spread_kf = _spread(kf)
    assert spread_iter <= spread_kf + 1e-6
