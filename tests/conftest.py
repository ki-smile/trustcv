"""Shared pytest configuration for trustcv's test suite."""

import pytest


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