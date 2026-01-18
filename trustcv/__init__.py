"""
trustcv - Trustworthy Cross-Validation Toolkit

Quick Start:
    from trustcv import TrustCV  # or TrustCVValidator
    from sklearn.ensemble import RandomForestClassifier

    validator = TrustCV(method='stratified_kfold', n_splits=5)
    results = validator.validate(model=RandomForestClassifier(), X=X, y=y)
    print(results.summary())

Main Classes:
    - TrustCV / TrustCVValidator: Main cross-validation validator
    - DataLeakageChecker: Detect data leakage issues
    - ClinicalMetrics: Medical/clinical performance metrics

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se

Main Features:
- Framework-agnostic: Works with scikit-learn, PyTorch, TensorFlow, MONAI, JAX
- 29 cross-validation methods (IID, Grouped, Temporal, Spatial)
- Automatic data leakage detection
- Patient/group-level splitting to prevent data leakage
- Clinical metrics with confidence intervals
- Reporting utilities that support regulatory documentation

For more information: https://github.com/ki-smile/trustcv
"""

__version__ = "1.0.5"
__author__ = "SMAILE Team, Karolinska Institutet"
__institution__ = "SMAILE - Stockholm Medical AI and Learning Environments, Karolinska Institutet"
__website__ = "https://smile.ki.se"

from .checkers import BalanceChecker, DataLeakageChecker

# Import new framework-agnostic components
from .core import (
    ClassDistributionLogger,
    CVCallback,
    CVResults,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    UniversalCVRunner,
)
from .metrics import ClinicalMetrics

# Import all splitters
from .splitters import (  # I.I.D. methods; Grouped methods; Temporal methods; Spatial methods
    LOOCV,
    LPOCV,
    BlockedTimeSeries,
    BootstrapValidation,
    BufferedSpatialCV,
    CombinatorialPurgedCV,
    EnvironmentalHealthCV,
    ExpandingWindowCV,
    GroupKFoldMedical,
    HierarchicalGroupKFold,
    HoldOut,
    KFoldMedical,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    MonteCarloCV,
    NestedCV,
    NestedGroupedCV,
    NestedTemporalCV,
    PurgedGroupTimeSeriesSplit,
    PurgedKFoldCV,
    RepeatedGroupKFold,
    RepeatedKFold,
    RollingWindowCV,
    SpatialBlockCV,
    SpatiotemporalBlockCV,
    StratifiedGroupKFold,
    StratifiedKFoldMedical,
    TimeSeriesSplit,
)

# Import dataset loaders for convenience
from .datasets import (
    load_heart_disease,
    load_diabetic_readmission,
    load_cancer_imaging,
    generate_synthetic_ehr,
    generate_temporal_patient_data,
)
from .validators import MedicalValidator, NestedTemporalCV, TrustCVValidator

# Convenience alias - TrustCV is the same as TrustCVValidator
TrustCV = TrustCVValidator

# Conditionally import framework-specific runners if available
try:
    from .frameworks.pytorch import TorchCVRunner

    _pytorch_available = True
except ImportError:
    _pytorch_available = False

try:
    from .frameworks.tensorflow import KerasCVRunner

    _tensorflow_available = True
except ImportError:
    _tensorflow_available = False

try:
    from .frameworks.monai import MONAICVRunner

    _monai_available = True
except ImportError:
    _monai_available = False

__all__ = [
    # Core validators and checkers
    "TrustCV",  # Convenience alias (same as TrustCVValidator)
    "TrustCVValidator",
    "MedicalValidator",
    "DataLeakageChecker",
    "BalanceChecker",
    "ClinicalMetrics",
    "NestedTemporalCV",
    "NestedGroupedCV",
    # Framework-agnostic components
    "UniversalCVRunner",
    "CVResults",
    "CVCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "ClassDistributionLogger",
    # I.I.D. methods
    "HoldOut",
    "KFoldMedical",
    "StratifiedKFoldMedical",
    "RepeatedKFold",
    "LOOCV",
    "LPOCV",
    "BootstrapValidation",
    "MonteCarloCV",
    "NestedCV",
    # Grouped methods
    "GroupKFoldMedical",
    "StratifiedGroupKFold",
    "LeaveOneGroupOut",
    "LeavePGroupsOut",
    "RepeatedGroupKFold",
    "NestedGroupedCV",
    "HierarchicalGroupKFold",
    # Temporal methods
    "TimeSeriesSplit",
    "BlockedTimeSeries",
    "RollingWindowCV",
    "ExpandingWindowCV",
    "PurgedKFoldCV",
    "CombinatorialPurgedCV",
    "PurgedGroupTimeSeriesSplit",
    "NestedTemporalCV",
    # Spatial methods
    "SpatialBlockCV",
    "BufferedSpatialCV",
    "SpatiotemporalBlockCV",
    "EnvironmentalHealthCV",
    # Dataset loaders
    "load_heart_disease",
    "load_diabetic_readmission",
    "load_cancer_imaging",
    "generate_synthetic_ehr",
    "generate_temporal_patient_data",
]

# Add framework-specific runners if available
if _pytorch_available:
    __all__.append("TorchCVRunner")
if _tensorflow_available:
    __all__.append("KerasCVRunner")
if _monai_available:
    __all__.append("MONAICVRunner")

# === Canonical name aliases (consistency with sklearn/literature) ===
import warnings as _trustcv_warnings


def _trustcv_deprecated(old, new):
    _trustcv_warnings.warn(
        f"'{old}' is deprecated; use '{new}' instead.", category=DeprecationWarning, stacklevel=2
    )


# ---- Canonical aliases (bind canonical names to existing implementations) ----
# IID / Grouped
try:
    KFold = KFoldMedical
except NameError:
    pass
try:
    StratifiedKFold = StratifiedKFoldMedical
except NameError:
    pass
try:
    LeaveOneOut = LOOCV
except NameError:
    pass
try:
    LeavePOut = LPOCV
except NameError:
    pass
try:
    GroupKFold = GroupKFoldMedical
except NameError:
    pass

# Temporal
try:
    BlockedTimeSeriesSplit = BlockedTimeSeries
except NameError:
    pass
try:
    RollingWindowSplit = RollingWindowCV
except NameError:
    pass
try:
    ExpandingWindowSplit = ExpandingWindowCV
except NameError:
    pass
try:
    PurgedKFold = PurgedKFoldCV
except NameError:
    pass
try:
    CombinatorialPurgedKFold = CombinatorialPurgedCV
except NameError:
    pass

# Spatial
try:
    SpatialBlockSplit = SpatialBlockCV
except NameError:
    pass
try:
    BufferedSpatialSplit = BufferedSpatialCV
except NameError:
    pass
try:
    SpatiotemporalBlockSplit = SpatiotemporalBlockCV
except NameError:
    pass
try:
    EnvironmentalHealthSplit = EnvironmentalHealthCV
except NameError:
    pass

# Checkers
try:
    LeakageChecker = DataLeakageChecker
except NameError:
    pass
try:
    ClassBalanceChecker = BalanceChecker
except NameError:
    pass

# ---- Extend __all__ with canonical names (non-breaking) ----
__all__.extend(
    [
        # IID / Grouped
        "KFold",
        "StratifiedKFold",
        "LeaveOneOut",
        "LeavePOut",
        "GroupKFold",
        # Temporal
        "BlockedTimeSeriesSplit",
        "RollingWindowSplit",
        "ExpandingWindowSplit",
        "PurgedKFold",
        "CombinatorialPurgedKFold",
        # Spatial
        "SpatialBlockSplit",
        "BufferedSpatialSplit",
        "SpatiotemporalBlockSplit",
        "EnvironmentalHealthSplit",
        # Checkers
        "LeakageChecker",
        "ClassBalanceChecker",
    ]
)


# ---- Optional: gentle deprecation notices (future-proofing) ----
def __getattr__(name):
    _map = {
        "KFoldMedical": "KFold",
        "StratifiedKFoldMedical": "StratifiedKFold",
        "LOOCV": "LeaveOneOut",
        "LPOCV": "LeavePOut",
        "GroupKFoldMedical": "GroupKFold",
        "BlockedTimeSeries": "BlockedTimeSeriesSplit",
        "RollingWindowCV": "RollingWindowSplit",
        "ExpandingWindowCV": "ExpandingWindowSplit",
        "PurgedKFoldCV": "PurgedKFold",
        "CombinatorialPurgedCV": "CombinatorialPurgedKFold",
        "SpatialBlockCV": "SpatialBlockSplit",
        "BufferedSpatialCV": "BufferedSpatialSplit",
        "SpatiotemporalBlockCV": "SpatiotemporalBlockSplit",
        "EnvironmentalHealthCV": "EnvironmentalHealthSplit",
        "DataLeakageChecker": "LeakageChecker",
        "BalanceChecker": "ClassBalanceChecker",
    }
    if name in _map:
        _trustcv_deprecated(name, _map[name])
        return globals().get(_map[name])
    raise AttributeError(f"module 'trustcv' has no attribute '{name}'")
