"""Integrity semantics in regulatory and universal reports."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustcv import TrustCV
from trustcv.checks import CHECK_KEYS
from trustcv.core.runner import UniversalCVRunner
from trustcv.reporting import RegulatoryReport, UniversalRegulatoryReport
from trustcv.splitters import StratifiedKFoldMedical


def _report() -> RegulatoryReport:
    return RegulatoryReport(
        model_name="review-model",
        model_version="1.1.0",
        manufacturer="test",
        intended_use="test-only validation",
    )


def _assert_every_check_is_rendered(html: str, result) -> None:
    for name in CHECK_KEYS:
        label = name.replace("_", " ").title()
        assert label in html
        assert result.checks[name].status in html


def test_regulatory_html_does_not_pass_preselected_noise(tmp_path):
    rng = np.random.default_rng(123)
    X_raw = rng.normal(size=(120, 500))
    y = rng.integers(0, 2, size=120)
    X_selected = SelectKBest(f_classif, k=10).fit_transform(X_raw, y)

    validator = TrustCV(method="stratified_kfold", n_splits=3, random_state=7)
    result = validator.validate(
        model=LogisticRegression(max_iter=2000), X=X_selected, y=y
    )
    output = _report().generate_from_validator(
        validator, output_path=str(tmp_path / "preselected.html"), format="html"
    )
    html = (tmp_path / "preselected.html").read_text(encoding="utf-8")

    assert output == str(tmp_path / "preselected.html")
    assert result.overall_status == "NOT_FULLY_VERIFIED"
    assert "Overall Status:</strong> NOT_FULLY_VERIFIED" in html
    assert "Leakage Check:</strong> PASSED" not in html
    _assert_every_check_is_rendered(html, result)


def test_regulatory_html_passes_pipeline_with_declared_independence(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    validator = TrustCV(
        method="stratified_kfold",
        n_splits=3,
        random_state=7,
        declare_independent_samples=True,
    )
    result = validator.validate(
        model=make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=7)
        ),
        X=X,
        y=y,
    )
    _report().generate_from_validator(
        validator, output_path=str(tmp_path / "pipeline.html"), format="html"
    )
    html = (tmp_path / "pipeline.html").read_text(encoding="utf-8")

    assert result.overall_status == "PASSED"
    assert "Overall Status:</strong> PASSED" in html
    assert "Leakage Check:</strong> PASSED" in html
    _assert_every_check_is_rendered(html, result)


def test_universal_runner_and_report_disclose_integrity_not_run(tmp_path):
    X, y = load_breast_cancer(return_X_y=True)
    runner = UniversalCVRunner(
        cv_splitter=StratifiedKFoldMedical(n_splits=3, shuffle=True, random_state=7),
        framework="sklearn",
        verbose=0,
    )
    result = runner.run(
        model=make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=7)
        ),
        data=(X, y),
        metrics=["accuracy"],
    )

    assert result.overall_status == "NOT_FULLY_VERIFIED"
    assert all(check.status == "NOT_CHECKED" for check in result.checks.values())
    assert "Integrity Checks:" in result.summary()
    assert "NOT RUN: UniversalCVRunner does not run" in result.summary()
    assert "Overall Status: NOT_FULLY_VERIFIED" in result.summary()

    output = tmp_path / "universal.html"
    UniversalRegulatoryReport.from_runner(
        runner_results=result,
        model=LogisticRegression(),
        data=(X, y),
        output_path=str(output),
        clinical_metrics={"accuracy": 0.5},
    )
    html = output.read_text(encoding="utf-8")
    assert "UniversalCVRunner does not run the TrustCV integrity suite by default." in html
    assert "Overall Status:</strong> NOT_FULLY_VERIFIED" in html
    assert "Leakage Check:</strong> PASSED" not in html
    _assert_every_check_is_rendered(html, result)