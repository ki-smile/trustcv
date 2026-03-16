"""Tests for ValidationResult.dashboard() method."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from trustcv import TrustCV
from trustcv.validators import ValidationResult
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ── Fixture: run TrustCV once for all tests ──────────────────────────────
@pytest.fixture(scope="module")
def results():
    X, y = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    validator = TrustCV(method="stratified_kfold", n_splits=5, random_state=42,
                        check_leakage=True, check_balance=True)
    return validator.validate(model=model, X=X, y=y)


# ── Fixture: minimal synthetic ValidationResult ───────────────────────────
@pytest.fixture
def minimal_result():
    """A hand-crafted ValidationResult with 2 folds and 2 metrics."""
    return ValidationResult(
        scores={"accuracy": np.array([0.90, 0.92]),
                "roc_auc":  np.array([0.95, 0.96])},
        mean_scores={"accuracy": 0.91, "roc_auc": 0.955},
        std_scores={"accuracy": 0.01, "roc_auc": 0.005},
        confidence_intervals={"accuracy": (0.88, 0.94), "roc_auc": (0.93, 0.97)},
        fold_details=[
            {"fold": 1, "n_train": 80, "n_val": 20,
             "metrics": {"accuracy": 0.90, "roc_auc": 0.95}},
            {"fold": 2, "n_train": 80, "n_val": 20,
             "metrics": {"accuracy": 0.92, "roc_auc": 0.96}},
        ],
        leakage_check={"no_duplicate_samples": True,
                       "balanced_classes": True,
                       "has_leakage": True},
        recommendations=[],
        ci_method="bootstrap",
        ci_level=0.95,
    )


# ── 1. Method existence ───────────────────────────────────────────────────
class TestDashboardExists:
    def test_method_exists(self, results):
        assert hasattr(results, "dashboard"), \
            "ValidationResult must have a dashboard() method"

    def test_method_callable(self, results):
        assert callable(results.dashboard)

    def test_accepts_title_kwarg(self, results):
        """dashboard(title=...) must not raise TypeError."""
        with patch("plotly.graph_objects.Figure.show"):
            results.dashboard(title="Test Title")   # should not raise


# ── 2. Returns None (side-effect method) ──────────────────────────────────
class TestDashboardReturnValue:
    def test_returns_none(self, results):
        with patch("plotly.graph_objects.Figure.show"):
            ret = results.dashboard()
        assert ret is None, "dashboard() must return None"


# ── 3. Does not crash on real data ────────────────────────────────────────
class TestDashboardNoError:
    def test_no_exception_real_data(self, results):
        with patch("plotly.graph_objects.Figure.show"):
            results.dashboard()   # must not raise any exception

    def test_no_exception_minimal_data(self, minimal_result):
        with patch("plotly.graph_objects.Figure.show"):
            minimal_result.dashboard()

    def test_no_exception_with_recommendations(self):
        """dashboard() must handle non-empty recommendations list."""
        r = ValidationResult(
            scores={"accuracy": np.array([0.85])},
            mean_scores={"accuracy": 0.85},
            std_scores={"accuracy": 0.02},
            confidence_intervals={"accuracy": (0.82, 0.88)},
            fold_details=[{"fold": 1, "n_train": 80, "n_val": 20,
                           "metrics": {"accuracy": 0.85}}],
            leakage_check={"no_duplicate_samples": False},
            recommendations=["Remove duplicate samples before splitting."],
            ci_method="bootstrap",
            ci_level=0.95,
        )
        with patch("plotly.graph_objects.Figure.show"):
            r.dashboard()   # must not raise


# ── 4. Plotly figure structure ────────────────────────────────────────────
class TestDashboardFigureStructure:

    @pytest.fixture
    def captured_fig(self, results):
        """Capture the Figure object that dashboard() builds."""
        captured = {}
        original_show = MagicMock()

        import plotly.graph_objects as go
        original_update = go.Figure.update_layout

        def capturing_show(fig_self, *a, **kw):
            captured["fig"] = fig_self
            # do not actually render

        with patch.object(go.Figure, "show", capturing_show):
            results.dashboard()

        return captured.get("fig")

    def test_figure_created(self, captured_fig):
        assert captured_fig is not None, "dashboard() must create a plotly Figure"

    def test_has_six_subplots(self, captured_fig):
        assert len(captured_fig.data) >= 6, \
            "Dashboard must have at least 6 traces (one per subplot)"

    def test_contains_bar_trace(self, captured_fig):
        import plotly.graph_objects as go
        bar_traces = [t for t in captured_fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1, "Must contain at least one Bar trace"

    def test_contains_scatter_traces(self, captured_fig):
        import plotly.graph_objects as go
        scatter = [t for t in captured_fig.data if isinstance(t, go.Scatter)]
        assert len(scatter) >= 1, "Must contain Scatter traces for fold lines"

    def test_contains_heatmap(self, captured_fig):
        import plotly.graph_objects as go
        heatmaps = [t for t in captured_fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) >= 1, "Must contain a Heatmap trace"

    def test_contains_tables(self, captured_fig):
        import plotly.graph_objects as go
        tables = [t for t in captured_fig.data if isinstance(t, go.Table)]
        assert len(tables) >= 2, "Must contain at least 2 Table traces"

    def test_figure_height(self, captured_fig):
        assert captured_fig.layout.height >= 800, \
            "Dashboard height must be >= 800px"

    def test_bar_y_values_are_percentages(self, captured_fig):
        import plotly.graph_objects as go
        bar = next(t for t in captured_fig.data if isinstance(t, go.Bar))
        for v in bar.y:
            assert v >= 0 and v <= 101, \
                f"Bar y values must be percentages (0–100), got {v}"

    def test_heatmap_zrange(self, captured_fig):
        import plotly.graph_objects as go
        hm = next(t for t in captured_fig.data if isinstance(t, go.Heatmap))
        assert hm.zmin is not None and hm.zmin >= 0
        assert hm.zmax is not None and hm.zmax <= 101


# ── 5. Leakage status reflected correctly ─────────────────────────────────
class TestLeakageDisplay:

    def _get_integrity_table(self, result_obj):
        """Return the cells of the integrity check Table trace."""
        import plotly.graph_objects as go
        captured = {}
        def cap_show(self_fig, *a, **kw):
            captured["fig"] = self_fig
        with patch.object(go.Figure, "show", cap_show):
            result_obj.dashboard()
        fig = captured["fig"]
        tables = [t for t in fig.data if isinstance(t, go.Table)]
        assert tables, "No Table traces found"
        return tables[0]   # first table = integrity checks

    def test_passed_shown_when_no_leakage(self, results):
        tbl = self._get_integrity_table(results)
        # Flatten all cell values into a single string
        all_text = " ".join(str(v) for col in tbl.cells.values for v in col)
        assert "PASSED" in all_text, \
            "Table must show PASSED when leakage check passes"

    def test_failed_shown_when_leakage_detected(self):
        bad = ValidationResult(
            scores={"accuracy": np.array([0.85])},
            mean_scores={"accuracy": 0.85},
            std_scores={"accuracy": 0.0},
            confidence_intervals={"accuracy": (0.83, 0.87)},
            fold_details=[{"fold": 1, "n_train": 80, "n_val": 20,
                           "metrics": {"accuracy": 0.85}}],
            leakage_check={"no_duplicate_samples": False,
                           "balanced_classes": True,
                           "has_leakage": False},
            recommendations=["Remove duplicate samples."],
            ci_method="bootstrap",
            ci_level=0.95,
        )
        import plotly.graph_objects as go
        captured = {}
        def cap_show(sf, *a, **kw): captured["fig"] = sf
        with patch.object(go.Figure, "show", cap_show):
            bad.dashboard()
        fig = captured["fig"]
        tables = [t for t in fig.data if isinstance(t, go.Table)]
        all_text = " ".join(str(v) for col in tables[0].cells.values for v in col)
        assert "FAILED" in all_text, \
            "Table must show FAILED when no_duplicate_samples is False"


# ── 6. summary() still works after adding dashboard() ────────────────────
class TestSummaryUnchanged:
    def test_summary_still_returns_string(self, results):
        s = results.summary()
        assert isinstance(s, str)

    def test_summary_contains_accuracy(self, results):
        assert "accuracy" in results.summary()

    def test_summary_contains_leakage_check(self, results):
        assert "Leakage Check" in results.summary()

    def test_summary_and_dashboard_coexist(self, results):
        """Calling both in sequence must not raise."""
        results.summary()
        with patch("plotly.graph_objects.Figure.show"):
            results.dashboard()


# ── 7. Edge cases ─────────────────────────────────────────────────────────
class TestEdgeCases:

    def test_empty_recommendations(self, minimal_result):
        with patch("plotly.graph_objects.Figure.show"):
            minimal_result.dashboard()   # no crash

    def test_single_metric(self):
        r = ValidationResult(
            scores={"accuracy": np.array([0.90])},
            mean_scores={"accuracy": 0.90},
            std_scores={"accuracy": 0.01},
            confidence_intervals={"accuracy": (0.88, 0.92)},
            fold_details=[{"fold": 1, "n_train": 80, "n_val": 20,
                           "metrics": {"accuracy": 0.90}}],
            leakage_check={},
            recommendations=[],
            ci_method="",
            ci_level=0.95,
        )
        with patch("plotly.graph_objects.Figure.show"):
            r.dashboard()   # must not crash with single metric

    def test_custom_title_appears(self):
        """The custom title string must appear in the figure layout."""
        import plotly.graph_objects as go
        captured = {}
        def cap_show(sf, *a, **kw): captured["fig"] = sf
        r = ValidationResult(
            scores={"accuracy": np.array([0.90])},
            mean_scores={"accuracy": 0.90},
            std_scores={"accuracy": 0.0},
            confidence_intervals={"accuracy": (0.88, 0.92)},
            fold_details=[{"fold": 1, "n_train": 80, "n_val": 20,
                           "metrics": {"accuracy": 0.90}}],
            leakage_check={},
            recommendations=[],
            ci_method="bootstrap",
            ci_level=0.95,
        )
        with patch.object(go.Figure, "show", cap_show):
            r.dashboard(title="MY CUSTOM TITLE")
        title_text = captured["fig"].layout.title.text or ""
        assert "MY CUSTOM TITLE" in title_text, \
            "Custom title must appear in figure layout title"
