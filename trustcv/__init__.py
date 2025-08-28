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

__version__ = "1.0.0"
__author__ = "SMAILE Team, Karolinska Institutet"
__institution__ = "SMAILE - Stockholm Medical AI and Learning Environments, Karolinska Institutet"
__website__ = "https://smile.ki.se"

from .validators import MedicalValidator
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
    ProgressLogger
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
    'MedicalValidator',
    'DataLeakageChecker',
    'BalanceChecker',
    'ClinicalMetrics',
    # Framework-agnostic components
    'UniversalCVRunner',
    'CVResults',
    'CVCallback',
    'EarlyStopping',
    'ModelCheckpoint',
    'ProgressLogger',
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