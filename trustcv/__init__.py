"""
trustcv - Trustworthy Cross-Validation Toolkit

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility - https://smile.ki.se
Contributors: Farhad Abtahi, Abdelamir Karbalaie, SMAILE Team
=============================================

A framework-agnostic Python library for trustworthy cross-validation that provides a unified 
interface across scikit-learn, PyTorch, TensorFlow/Keras, MONAI, JAX, and other 
ML frameworks. It promotes best practices in model evaluation, providing both 
familiar interfaces and customizable validators tailored for reliable validation.

Developed at SMAILE (Stockholm Medical Artificial Intelligence and Learning Environments)
Karolinska Institutet Core Facility
Website: https://smile.ki.se

Main Features:
- Framework-agnostic: Works with any ML/DL framework
- Advanced cross-validation strategies (hierarchical, temporal, spatial)
- Automatic data leakage detection
- Patient/group-level and temporal splitting
- Regulatory compliance reporting (FDA, CE MDR)
- Interactive visualizations
- Support for PyTorch, TensorFlow, MONAI, and more

Contributors:
- Farhad Abtahi
- Abdelamir Karbalaie
- SMAILE Team

For more information about AI research and collaboration opportunities,
visit SMAILE at https://smile.ki.se
"""

__version__ = "1.0.1"
__author__ = "SMAILE Team, Karolinska Institutet"
__institution__ = "SMAILE - Stockholm Medical AI and Learning Environments, Karolinska Institutet"
__website__ = "https://smile.ki.se"

from .validators import TrustCVValidator, MedicalValidator, NestedTemporalCV
from .checkers import DataLeakageChecker, BalanceChecker
from .metrics import ClinicalMetrics

# Import all splitters
from .splitters import (
    # I.I.D. methods
    HoldOut, KFoldMedical, StratifiedKFoldMedical,
    RepeatedKFold, LOOCV, LPOCV, BootstrapValidation,
    MonteCarloCV, NestedCV,
    # Grouped methods
    GroupKFoldMedical, StratifiedGroupKFold,
    LeaveOneGroupOut, RepeatedGroupKFold,
    NestedGroupedCV, HierarchicalGroupKFold,
    # Temporal methods
    TimeSeriesSplit, BlockedTimeSeries,
    RollingWindowCV, ExpandingWindowCV,
    PurgedKFoldCV, CombinatorialPurgedCV,
    PurgedGroupTimeSeriesSplit, NestedTemporalCV,
    # Spatial methods
    SpatialBlockCV, BufferedSpatialCV,
    SpatiotemporalBlockCV, EnvironmentalHealthCV
)

# Import new framework-agnostic components
from .core import (
    UniversalCVRunner,
    CVResults,
    CVCallback,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
    ClassDistributionLogger,
)

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
    'TrustCVValidator', 'MedicalValidator',
    'DataLeakageChecker',
    'BalanceChecker',
    'ClinicalMetrics', 'NestedTemporalCV', 'NestedGroupedCV',
    # Framework-agnostic components
    'UniversalCVRunner',
    'CVResults',
    'CVCallback',
    'EarlyStopping',
    'ModelCheckpoint',
    'ProgressLogger',
    'ClassDistributionLogger',
    # I.I.D. methods
    'HoldOut', 'KFoldMedical', 'StratifiedKFoldMedical',
    'RepeatedKFold', 'LOOCV', 'LPOCV', 'BootstrapValidation',
    'MonteCarloCV', 'NestedCV',
    # Grouped methods
    'GroupKFoldMedical', 'StratifiedGroupKFold',
    'LeaveOneGroupOut', 'RepeatedGroupKFold',
    'NestedGroupedCV', 'HierarchicalGroupKFold',
    # Temporal methods
    'TimeSeriesSplit', 'BlockedTimeSeries',
    'RollingWindowCV', 'ExpandingWindowCV',
    'PurgedKFoldCV', 'CombinatorialPurgedCV',
    'PurgedGroupTimeSeriesSplit', 'NestedTemporalCV',
    # Spatial methods
    'SpatialBlockCV', 'BufferedSpatialCV',
    'SpatiotemporalBlockCV', 'EnvironmentalHealthCV'
]

# Add framework-specific runners if available
if _pytorch_available:
    __all__.append('TorchCVRunner')
if _tensorflow_available:
    __all__.append('KerasCVRunner')
if _monai_available:
    __all__.append('MONAICVRunner')

# === Canonical name aliases (consistency with sklearn/literature) ===
import warnings as _trustcv_warnings

def _trustcv_deprecated(old, new):
    _trustcv_warnings.warn(
        f"'{old}' is deprecated; use '{new}' instead.",
        category=DeprecationWarning,
        stacklevel=2
    )

# ---- Canonical aliases (bind canonical names to existing implementations) ----
# IID / Grouped
try: KFold = KFoldMedical
except NameError: pass
try: StratifiedKFold = StratifiedKFoldMedical
except NameError: pass
try: LeaveOneOut = LOOCV
except NameError: pass
try: LeavePOut = LPOCV
except NameError: pass
try: GroupKFold = GroupKFoldMedical
except NameError: pass

# Temporal
try: BlockedTimeSeriesSplit = BlockedTimeSeries
except NameError: pass
try: RollingWindowSplit = RollingWindowCV
except NameError: pass
try: ExpandingWindowSplit = ExpandingWindowCV
except NameError: pass
try: PurgedKFold = PurgedKFoldCV
except NameError: pass
try: CombinatorialPurgedKFold = CombinatorialPurgedCV
except NameError: pass

# Spatial
try: SpatialBlockSplit = SpatialBlockCV
except NameError: pass
try: BufferedSpatialSplit = BufferedSpatialCV
except NameError: pass
try: SpatiotemporalBlockSplit = SpatiotemporalBlockCV
except NameError: pass
try: EnvironmentalHealthSplit = EnvironmentalHealthCV
except NameError: pass

# Checkers
try: LeakageChecker = DataLeakageChecker
except NameError: pass
try: ClassBalanceChecker = BalanceChecker
except NameError: pass

# ---- Extend __all__ with canonical names (non-breaking) ----
__all__.extend([
    # IID / Grouped
    'KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'GroupKFold',
    # Temporal
    'BlockedTimeSeriesSplit', 'RollingWindowSplit', 'ExpandingWindowSplit',
    'PurgedKFold', 'CombinatorialPurgedKFold',
    # Spatial
    'SpatialBlockSplit', 'BufferedSpatialSplit', 'SpatiotemporalBlockSplit', 'EnvironmentalHealthSplit',
    # Checkers
    'LeakageChecker', 'ClassBalanceChecker',
])

# ---- Optional: gentle deprecation notices (future-proofing) ----
def __getattr__(name):
    _map = {
        'KFoldMedical': 'KFold',
        'StratifiedKFoldMedical': 'StratifiedKFold',
        'LOOCV': 'LeaveOneOut',
        'LPOCV': 'LeavePOut',
        'GroupKFoldMedical': 'GroupKFold',
        'BlockedTimeSeries': 'BlockedTimeSeriesSplit',
        'RollingWindowCV': 'RollingWindowSplit',
        'ExpandingWindowCV': 'ExpandingWindowSplit',
        'PurgedKFoldCV': 'PurgedKFold',
        'CombinatorialPurgedCV': 'CombinatorialPurgedKFold',
        'SpatialBlockCV': 'SpatialBlockSplit',
        'BufferedSpatialCV': 'BufferedSpatialSplit',
        'SpatiotemporalBlockCV': 'SpatiotemporalBlockSplit',
        'EnvironmentalHealthCV': 'EnvironmentalHealthSplit',
        'DataLeakageChecker': 'LeakageChecker',
        'BalanceChecker': 'ClassBalanceChecker',
    }
    if name in _map:
        _trustcv_deprecated(name, _map[name])
        return globals().get(_map[name])
    raise AttributeError(f"module 'trustcv' has no attribute '{name}'")
