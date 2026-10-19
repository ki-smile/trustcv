"""The documented API covers every public name, and status values match the code."""

import re
from pathlib import Path

import numpy as np

import trustcv
from trustcv.checks import CheckResult, OVERALL_STATUSES, default_checks, overall_status

API_PAGE = Path(__file__).resolve().parent.parent / "docs" / "reference" / "api.md"


def test_every_public_name_is_in_the_api_reference():
    page = API_PAGE.read_text(encoding="utf-8")
    documented = {p.rsplit(".", 1)[1] for p in re.findall(r"^::: (\S+)$", page, re.M)}
    aliases = set(re.findall(r"^\| `(\w+)` \|", page, re.M))
    missing = sorted(set(trustcv.__all__) - documented - aliases)
    assert not missing, f"add to docs/reference/api.md: {missing}"


def test_overall_status_values_are_the_documented_set():
    seen = set()
    for status in ["PASSED", "NOT_APPLICABLE", "WARNING", "NOT_CHECKED", "FAILED"]:
        checks = default_checks()
        for name in checks:
            checks[name] = CheckResult(name, status, "")
        seen.add(overall_status(checks))
    assert seen == set(OVERALL_STATUSES)


def test_home_page_states_the_real_splitter_count():
    import inspect

    import trustcv.splitters as splitters

    count = len({id(getattr(splitters, n)) for n in splitters.__all__ if inspect.isclass(getattr(splitters, n))})
    home = (API_PAGE.parent.parent / "index.md").read_text(encoding="utf-8")
    assert f"{count} methods (the 29 in the paper plus later additions)" in home
