"""
Advanced cross-validation splitters

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
"""

# I.I.D. methods
from .iid import (
    HoldOut, KFoldMedical, StratifiedKFoldMedical,
    RepeatedKFold, LOOCV, LPOCV, BootstrapValidation,
    MonteCarloCV, NestedCV
)

# Grouped methods
from .grouped import (
    GroupKFoldMedical, StratifiedGroupKFold,
    LeaveOneGroupOut, RepeatedGroupKFold,
    NestedGroupedCV, HierarchicalGroupKFold
)

# Temporal methods
from .temporal import (
    TimeSeriesSplit, BlockedTimeSeries,
    RollingWindowCV, ExpandingWindowCV,
    PurgedKFoldCV, CombinatorialPurgedCV,
    PurgedGroupTimeSeriesSplit, NestedTemporalCV
)

# Spatial methods
from .spatial import (
    SpatialBlockCV, BufferedSpatialCV,
    SpatiotemporalBlockCV, EnvironmentalHealthCV
)

__all__ = [
    # I.I.D.
    'HoldOut', 'KFoldMedical', 'StratifiedKFoldMedical',
    'RepeatedKFold', 'LOOCV', 'LPOCV', 'BootstrapValidation',
    'MonteCarloCV', 'NestedCV',
    # Grouped
    'GroupKFoldMedical', 'StratifiedGroupKFold',
    'LeaveOneGroupOut', 'RepeatedGroupKFold',
    'NestedGroupedCV', 'HierarchicalGroupKFold',
    # Temporal
    'TimeSeriesSplit', 'BlockedTimeSeries',
    'RollingWindowCV', 'ExpandingWindowCV',
    'PurgedKFoldCV', 'CombinatorialPurgedCV',
    'PurgedGroupTimeSeriesSplit', 'NestedTemporalCV',
    # Spatial
    'SpatialBlockCV', 'BufferedSpatialCV',
    'SpatiotemporalBlockCV', 'EnvironmentalHealthCV'
]

# Canonical alias names (sklearn-style) for convenience
# IID / Grouped
KFold = KFoldMedical
StratifiedKFold = StratifiedKFoldMedical
LeaveOneOut = LOOCV
LeavePOut = LPOCV
GroupKFold = GroupKFoldMedical

# Temporal
BlockedTimeSeriesSplit = BlockedTimeSeries
RollingWindowSplit = RollingWindowCV
ExpandingWindowSplit = ExpandingWindowCV
PurgedKFold = PurgedKFoldCV
CombinatorialPurgedKFold = CombinatorialPurgedCV

# Spatial
SpatialBlockSplit = SpatialBlockCV
BufferedSpatialSplit = BufferedSpatialCV
SpatiotemporalBlockSplit = SpatiotemporalBlockCV
EnvironmentalHealthSplit = EnvironmentalHealthCV

# Export aliases as well
__all__ += [
    # IID / Grouped
    'KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'GroupKFold',
    # Temporal
    'BlockedTimeSeriesSplit', 'RollingWindowSplit', 'ExpandingWindowSplit',
    'PurgedKFold', 'CombinatorialPurgedKFold',
    # Spatial
    'SpatialBlockSplit', 'BufferedSpatialSplit', 'SpatiotemporalBlockSplit', 'EnvironmentalHealthSplit',
]
