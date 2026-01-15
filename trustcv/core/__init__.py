"""
Core framework-agnostic components for trustcv

This module provides the base abstractions that allow trustcv to work
across different machine learning frameworks while maintaining backward
compatibility with scikit-learn.
"""

from .base import CVResults, CVSplitter, FrameworkAdapter
from .callbacks import (
    ClassDistributionLogger,
    CVCallback,
    EarlyStopping,
    ModelCheckpoint,
    ProgressLogger,
)
from .runner import UniversalCVRunner

__all__ = [
    "CVSplitter",
    "FrameworkAdapter",
    "CVResults",
    "CVCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressLogger",
    "ClassDistributionLogger",
    "UniversalCVRunner",
]
