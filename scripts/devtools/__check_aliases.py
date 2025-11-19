import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import trustcv
print('trustcv version:', trustcv.__version__)
trustcv = importlib.reload(trustcv)
aliases = ['KFold','StratifiedKFold','LeaveOneOut','LeavePOut','GroupKFold',
           'BlockedTimeSeriesSplit','RollingWindowSplit','ExpandingWindowSplit',
           'PurgedKFold','CombinatorialPurgedKFold','SpatialBlockSplit','BufferedSpatialSplit',
           'SpatiotemporalBlockSplit','EnvironmentalHealthSplit','LeakageChecker','ClassBalanceChecker']
print('Aliases present:', [name for name in aliases if hasattr(trustcv, name)])
print('KFold is KFoldMedical:', getattr(trustcv,'KFold',None) is getattr(trustcv,'KFoldMedical',None))
print('GroupKFold is GroupKFoldMedical:', getattr(trustcv,'GroupKFold',None) is getattr(trustcv,'GroupKFoldMedical',None))
print('Has KFoldMedical:', hasattr(trustcv,'KFoldMedical'))
