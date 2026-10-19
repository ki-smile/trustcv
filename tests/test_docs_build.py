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
