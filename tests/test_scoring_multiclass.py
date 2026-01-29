import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from trustcv.validators import TrustCVValidator


def test_validate_accepts_sklearn_scorers_multiclass():
    data = load_digits()
    X, y = data.data, data.target

    validator = TrustCVValidator(method="stratified_kfold", n_splits=3)
    model = LogisticRegression(max_iter=500, multi_class="auto")

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    result = validator.validate(model=model, X=X, y=y, scoring=scoring)

    for key in scoring.keys():
        assert key in result.mean_scores
        assert np.isfinite(result.mean_scores[key])

    assert len(result.fold_details) == validator.n_splits
