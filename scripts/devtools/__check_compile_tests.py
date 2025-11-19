import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tests_path = ROOT / 'tests' / 'test_cv_methods.py'
py_compile.compile(str(tests_path), doraise=True)
print('tests/test_cv_methods.py syntax OK')
