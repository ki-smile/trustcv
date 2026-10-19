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


OLD_URLS = {
    "getting-started.html": "get-started/quickstart.html",
    "tutorials.html": "learn/tutorials.html",
    "notebooks.html": "learn/notebooks.html",
    "demonstrations.html": "learn/visualisations.html",
    "quiz.html": "learn/quiz.html",
    "frameworks.html": "guides/deep-learning.html",
    "ai-agents.html": "reference/ai-agents.html",
    "data-leakage.html": "guides/data-leakage.html",
    "best-practices.html": "guides/best-practices.html",
    "cv-selection-guide.html": "guides/choosing-cv.html",
    "methods-comparison.html": "guides/methods-comparison.html",
    "regulatory-cv-tutorial.html": "regulatory/documentation-guide.html",
    "regulatory-report.html": "regulatory/report-draft.html",
    "docs/API_REFERENCE.html": "reference/api.html",
    "docs/LICENSE.html": "reference/license.html",
}


def test_old_urls_redirect_to_their_new_pages(site):
    import posixpath
    import re

    for old, new in OLD_URLS.items():
        page = (site / old).read_text(encoding="utf-8")
        target = re.search(r'http-equiv="refresh" content="0; url=([^"]+)"', page).group(1)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(old), target))
        assert resolved == new, (old, target)
        assert (site / new).is_file(), new
    home = (site / "index.html").read_text(encoding="utf-8")
    assert 'http-equiv="refresh"' not in home  # Home is never a redirect


def test_every_documented_constructor_matches_its_runtime_signature(site):
    import importlib
    import inspect
    import json

    schema = json.loads((site / "api-schema.json").read_text())
    checked = 0
    for item in schema["api"]:
        if item["kind"] != "class":
            continue
        module, _, name = item["path"].rpartition(".")
        try:
            cls = getattr(importlib.import_module(module), name)
        except ImportError:  # optional framework not installed
            continue
        if cls.__init__ is object.__init__:  # no constructor of its own
            runtime = []
        else:
            runtime = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
        assert [p["name"] for p in item["parameters"]] == runtime, item["path"]
        checked += 1
    assert checked >= 40


def test_generated_text_makes_no_compliance_claims(site):
    import re

    negations = ("not ", "no ", "never", "cannot", "neither")
    for path in ("reference/api.html", "llms-full.txt", "llms.txt"):
        text = re.sub(r"<[^>]+>", " ", (site / path).read_text(encoding="utf-8"))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            # the word, not identifiers such as RegulatoryComplianceLogger or compliance_standard
            # (also skips the docstring heading of a parameter named "compliance")
            if re.search(r"(?<![A-Za-z_])complian(?!ce_)(?!ce\s*:)|FDA[- ]approved|meets .{0,40}requirements", sentence, re.I):
                assert any(n in sentence.lower() for n in negations), (path, sentence.strip()[:200])
