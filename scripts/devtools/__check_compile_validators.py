import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
validator_path = ROOT / 'trustcv' / 'validators.py'
py_compile.compile(str(validator_path), doraise=True)
print('validators.py compiled OK')
