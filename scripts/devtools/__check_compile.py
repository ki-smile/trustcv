import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
validator_path = ROOT / 'trustcv' / 'validators.py'
try:
    py_compile.compile(str(validator_path), doraise=True)
    print('validators.py compiled OK')
except Exception as e:
    print('validators.py compile error:', e)
    sys.exit(1)
