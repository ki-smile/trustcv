"""
Advanced cross-validation splitters

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
"""

# Grouped methods
from .grouped import (
    GroupKFoldMedical,
    HierarchicalGroupKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    MultilevelCV,
    NestedGroupedCV,
    RepeatedGroupKFold,
    StratifiedGroupKFold,
)

# I.I.D. methods
from .iid import (
    LOOCV,
    LPOCV,
    BootstrapValidation,
    HoldOut,
    KFoldMedical,
    MonteCarloCV,
    NestedCV,
    RepeatedKFold,
    StratifiedKFoldMedical,
)

# Spatial methods
from .spatial import BufferedSpatialCV, EnvironmentalHealthCV, SpatialBlockCV, SpatiotemporalBlockCV

# Temporal methods
from .temporal import (
    BlockedTimeSeries,
    CombinatorialPurgedCV,
    ExpandingWindowCV,
    NestedTemporalCV,
    PurgedGroupTimeSeriesSplit,
    PurgedKFoldCV,
    RollingWindowCV,
    TimeSeriesSplit,
)

__all__ = [
    # I.I.D.
    "HoldOut",
    "KFoldMedical",
    "StratifiedKFoldMedical",
    "RepeatedKFold",
    "LOOCV",
    "LPOCV",
    "BootstrapValidation",
    "MonteCarloCV",
    "NestedCV",
    # Grouped
    "GroupKFoldMedical",
    "StratifiedGroupKFold",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
    "RepeatedGroupKFold",
    "HierarchicalGroupKFold",
    "MultilevelCV",
    "NestedGroupedCV",
    # Temporal
    "TimeSeriesSplit",
    "BlockedTimeSeries",
    "RollingWindowCV",
    "ExpandingWindowCV",
    "PurgedKFoldCV",
    "CombinatorialPurgedCV",
    "PurgedGroupTimeSeriesSplit",
    "NestedTemporalCV",
    # Spatial
    "SpatialBlockCV",
    "BufferedSpatialCV",
    "SpatiotemporalBlockCV",
    "EnvironmentalHealthCV",
]

# Canonical alias names (sklearn-style) for convenience
# IID / Grouped
KFold = KFoldMedical
StratifiedKFold = StratifiedKFoldMedical
LeaveOneOut = LOOCV
LeavePOut = LPOCV
GroupKFold = GroupKFoldMedical
PatientGroupKFold = GroupKFoldMedical  # Legacy alias for backward compatibility

# Temporal
TemporalClinical = TimeSeriesSplit  # Legacy alias from docstring examples
BlockedTimeSeriesSplit = BlockedTimeSeries
BlockedTimeSeriesCV = BlockedTimeSeries  # Alternate alias
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
    "KFold",
    "StratifiedKFold",
    "LeaveOneOut",
    "LeavePOut",
    "GroupKFold",
    "PatientGroupKFold",
    # Temporal
    "TemporalClinical",
    "BlockedTimeSeriesSplit",
    "BlockedTimeSeriesCV",
    "RollingWindowSplit",
    "ExpandingWindowSplit",
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    # Spatial
    "SpatialBlockSplit",
    "BufferedSpatialSplit",
    "SpatiotemporalBlockSplit",
    "EnvironmentalHealthSplit",
]
