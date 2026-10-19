"""The deep-learning examples run end to end (fast mode) where their framework is installed."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "pytorch_medical_cv.py": ["torch"],
    "tensorflow_medical_cv.py": ["tensorflow"],
    "jax_flax_cv.py": ["jax", "flax", "optax"],
    "monai_3d_imaging_cv.py": ["torch", "monai"],
}


@pytest.mark.parametrize("script", sorted(EXAMPLES))
def test_example_runs(script, tmp_path):
    missing = [m for m in EXAMPLES[script] if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    env = {**os.environ, "TRUSTCV_EXAMPLE_FAST": "1", "PYTHONPATH": str(ROOT), "TF_CPP_MIN_LOG_LEVEL": "3",
           "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
