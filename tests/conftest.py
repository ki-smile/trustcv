"""Shared pytest configuration for trustcv's test suite."""

import os

import pytest

import trustcv

_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
_IMPORTED_FROM = os.path.realpath(os.path.dirname(trustcv.__file__))
if os.path.dirname(_IMPORTED_FROM) != _REPO_ROOT:
    # Another trustcv (e.g. an older editable install) would be tested instead of this code.
    raise pytest.UsageError(
        f"tests are running against trustcv at {_IMPORTED_FROM}, not this checkout "
        f"({_REPO_ROOT}). Install it with `pip install -e .` or run from the repo root."
    )


_PROTECTED_SLOW_TESTS = {
    "test_permutation_check_is_opt_in",
    "test_permutation_check_detects_leakage_inside_the_cv_loop",
    "test_permutation_check_passes_real_signal",
    "test_new_ci_methods_reach_nominal_coverage_on_noise",
}


def pytest_collection_modifyitems(items):
    """Mark expensive acceptance cases without editing the supplied test file."""
    slow = pytest.mark.slow
    for item in items:
        original_name = getattr(item, "originalname", None) or item.name.split("[")[0]
        if item.path.name == "test_trust_claims.py" and original_name in _PROTECTED_SLOW_TESTS:
            item.add_marker(slow)