import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustcv import splitters as s
importlib.reload(s)
# Verify canonical names exist in submodule
names = ['KFold','StratifiedKFold','GroupKFold','LeaveOneOut','LeavePOut',
         'BlockedTimeSeriesSplit','RollingWindowSplit','ExpandingWindowSplit',
         'PurgedKFold','CombinatorialPurgedKFold','SpatialBlockSplit','BufferedSpatialSplit',
         'SpatiotemporalBlockSplit','EnvironmentalHealthSplit']
missing = [n for n in names if not hasattr(s, n)]
print('Missing in trustcv.splitters:', missing)
# Identity checks
print('KFold is KFoldMedical:', s.KFold is s.KFoldMedical)
print('GroupKFold is GroupKFoldMedical:', s.GroupKFold is s.GroupKFoldMedical)
print('PurgedKFold is PurgedKFoldCV:', s.PurgedKFold is s.PurgedKFoldCV)
