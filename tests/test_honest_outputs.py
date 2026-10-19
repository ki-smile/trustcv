"""Outputs must not claim more than was established."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from trustcv import TrustCV
from trustcv.reporting.regulatory_report import RegulatoryReport


class _NoFolds:
    def get_n_splits(self, *a, **k):
        return 0

    def split(self, X, y=None, groups=None):
        return iter(())


def test_validate_rejects_a_splitter_that_produces_no_folds():
    X = np.random.RandomState(0).normal(size=(40, 3))
    y = np.tile([0, 1], 20)
    with pytest.raises(ValueError, match="no folds"):
        TrustCV(declare_independent_samples=True).validate(
            model=LogisticRegression(), X=X, y=y, groups=np.repeat(np.arange(10), 4), cv=_NoFolds()
        )


def test_regulatory_report_html_does_not_claim_compliance(tmp_path):
    report = RegulatoryReport(model_name="m", model_version="1", manufacturer="x", intended_use="research")
    path = report.generate_regulatory_report(str(tmp_path / "r.html"), format="html")
    html = open(path, encoding="utf-8").read().lower()
    assert "complies with" not in html
    assert "does not by itself establish compliance" in html
