import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trustcv as t
# reload to ensure current module state
importlib.reload(t)

assert t.KFold is t.KFoldMedical
assert t.StratifiedKFold is t.StratifiedKFoldMedical
assert t.GroupKFold is t.GroupKFoldMedical
assert t.LeaveOneOut is t.LOOCV
assert t.LeavePOut is t.LPOCV

assert t.PurgedKFold is t.PurgedKFoldCV
assert t.CombinatorialPurgedKFold is t.CombinatorialPurgedCV

assert t.LeakageChecker is t.DataLeakageChecker
assert t.ClassBalanceChecker is t.BalanceChecker

print('Aliases OK')
