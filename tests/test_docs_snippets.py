"""Execute the Python code blocks of every documentation page.

Each page runs in its own subprocess (temp directory, timeout, fixed seeds);
its blocks share one namespace, in page order. A block is skipped when the
line before its fence is ``<!-- notest: reason -->``, or when it imports an
optional framework that is not installed.
"""

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
SECTIONS = ("get-started", "guides", "learn", "regulatory", "reference")
FRAMEWORKS = ("torch", "tensorflow", "keras", "monai", "jax", "flax")
FENCE = re.compile(r"^(?P<indent>[ \t]*)```python[^\n]*\n(?P<body>.*?)^(?P=indent)```[ \t]*$", re.M | re.S)
NOTEST = re.compile(r"<!--\s*notest:\s*\S.*?-->\s*$")


def doc_pages():
    pages = [DOCS / "index.md"] + [p for s in SECTIONS for p in sorted((DOCS / s).rglob("*.md"))]
    return [p for p in pages if p.exists()]


def runnable_blocks(text):
    """(code, skip_reason) for each python block, in order."""
    blocks = []
    for m in FENCE.finditer(text):
        before = text[: m.start()].rstrip("\n").splitlines()
        code = textwrap.dedent(m.group("body"))
        reason = None
        if before and NOTEST.search(before[-1]):
            reason = "marked notest"
        else:
            missing = [
                f for f in FRAMEWORKS
                if re.search(rf"^\s*(import|from)\s+{f}\b", code, re.M) and importlib.util.find_spec(f) is None
            ]
            if missing:
                reason = f"needs {', '.join(missing)}"
        blocks.append((code, reason))
    return blocks


def page_script(blocks):
    # the child must run this checkout's trustcv, not another installed copy
    parts = [
        "import os, sys, random, numpy as np, trustcv",
        f"assert os.path.realpath(trustcv.__file__).startswith({str(REPO) + os.sep!r}), trustcv.__file__",
        "random.seed(0); np.random.seed(0)",
    ]
    for i, (code, reason) in enumerate(blocks, 1):
        if reason is None:
            parts.append(f"print('--- block {i}', file=sys.stderr, flush=True)")
            parts.append(code)
    return "\n".join(parts) + "\n"


@pytest.mark.parametrize("page", doc_pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_page_snippets_run(page, tmp_path):
    blocks = runnable_blocks(page.read_text(encoding="utf-8"))
    if not any(reason is None for _, reason in blocks):
        pytest.skip("no runnable python blocks")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0", OMP_NUM_THREADS="1",
               TF_CPP_MIN_LOG_LEVEL="3", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run(
        [sys.executable, "-c", page_script(blocks)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        marks = [l for l in proc.stderr.splitlines() if l.startswith("--- block")]
        where = marks[-1] if marks else "setup"
        pytest.fail(f"{page.name}: {where} failed\n{proc.stderr[-2000:]}")


def test_notest_marker_and_framework_skip():
    page = textwrap.dedent(
        """
        ```python
        x = 1
        ```

        <!-- notest: illustrative only -->
        ```python
        raise SystemExit("must not run")
        ```

        !!! example
            ```python
            assert x == 1
            ```

        ```python
        import definitely_missing_framework_xyz
        ```
        """
    )
    blocks = runnable_blocks(page)
    assert [reason for _, reason in blocks] == [None, "marked notest", None, None]
    assert blocks[2][0].strip() == "assert x == 1"  # indented fences are dedented
    fake = "```python\nimport torch\n```\n"
    if importlib.util.find_spec("torch") is None:
        assert runnable_blocks(fake)[0][1] == "needs torch"
