"""Build the MkDocs site once and check what it publishes.

Skipped unless the ``docs`` extra (MkDocs) is installed; CI runs it in the
dedicated docs job.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs")

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    proc = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(out)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return out


def test_strict_build_publishes_home(site):
    assert (site / "index.html").is_file()
    assert (site / "sitemap.xml").is_file()


def test_schema_matches_the_code(site):
    import inspect
    import json

    import trustcv
    from trustcv.checks import OVERALL_STATUSES

    schema = json.loads((site / "api-schema.json").read_text())
    assert schema["version"] == trustcv.__version__
    assert schema["overall_status_values"] == list(OVERALL_STATUSES)
    api = {item["path"]: item for item in schema["api"]}
    validator = api["trustcv.validators.TrustCVValidator"]
    validate = next(m for m in validator["methods"] if m["name"] == "validate")
    expected = [p for p in inspect.signature(trustcv.TrustCVValidator.validate).parameters if p != "self"]
    assert [p["name"] for p in validate["parameters"]] == expected
    keras = next(m for m in api["trustcv.frameworks.tensorflow.KerasCVRunner"]["methods"] if m["name"] == "run")
    assert "inner_validation" in [p["name"] for p in keras["parameters"]]


def test_machine_readable_files_are_published(site):
    llms = (site / "llms.txt").read_text()
    assert llms.startswith("# trustcv") and "NEEDS_REVIEW" in llms
    assert "inner_validation" in (site / "llms-full.txt").read_text()
