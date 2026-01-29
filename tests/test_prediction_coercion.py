import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from trustcv.validators import TrustCVValidator


class DummyProbEstimator:
    """Estimator that returns (n,1) probabilities to mimic SciKeras edge cases."""

    def fit(self, X, y, **kwargs):
        # store class prior as a simple probability
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        self.p_ = float(np.mean(y_arr))
        return self

    def predict(self, X):
        # shape (n, 1) probabilities instead of labels
        return np.full((len(X), 1), self.p_, dtype=float)

    def predict_proba(self, X):
        # shape (n, 1) probabilities (positive class only)
        return np.full((len(X), 1), self.p_, dtype=float)


def test_prediction_coercion_handles_single_column_probs():
    data = load_breast_cancer()
    X = data.data
    y = data.target

    validator = TrustCVValidator(method="stratified_kfold", n_splits=3)
    est = DummyProbEstimator()

    result = validator.validate(model=est, X=X, y=y)

    assert "accuracy" in result.mean_scores
    assert "roc_auc" in result.mean_scores
    assert np.isfinite(result.mean_scores["accuracy"])
    assert np.isfinite(result.mean_scores["roc_auc"])
    assert len(result.fold_details) == validator.n_splits


if __name__ == "__main__":
    pytest.main([__file__])
