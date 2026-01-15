"""
Framework-specific adapters for trustcv

This module provides adapters for different ML/DL frameworks to work
seamlessly with trustcv's cross-validation methods.
"""

# Import adapters conditionally based on available frameworks
adapters = {}

try:
    from .pytorch import PyTorchAdapter, TorchCVRunner

    adapters["pytorch"] = PyTorchAdapter
    __all__ = ["PyTorchAdapter", "TorchCVRunner"]
except ImportError:
    pass

try:
    from .tensorflow import KerasCVRunner, TensorFlowAdapter

    adapters["tensorflow"] = TensorFlowAdapter
    adapters["keras"] = TensorFlowAdapter
    __all__ = (
        __all__ + ["TensorFlowAdapter", "KerasCVRunner"]
        if "__all__" in locals()
        else ["TensorFlowAdapter", "KerasCVRunner"]
    )
except ImportError:
    pass

try:
    from .jax import JAXAdapter

    adapters["jax"] = JAXAdapter
    __all__ = __all__ + ["JAXAdapter"] if "__all__" in locals() else ["JAXAdapter"]
except ImportError:
    pass

try:
    from .xgboost import XGBoostAdapter

    adapters["xgboost"] = XGBoostAdapter
    __all__ = __all__ + ["XGBoostAdapter"] if "__all__" in locals() else ["XGBoostAdapter"]
except ImportError:
    pass

try:
    from .lightgbm import LightGBMAdapter

    adapters["lightgbm"] = LightGBMAdapter
    __all__ = __all__ + ["LightGBMAdapter"] if "__all__" in locals() else ["LightGBMAdapter"]
except ImportError:
    pass


def get_adapter(framework_name: str):
    """
    Get adapter for specified framework

    Parameters:
        framework_name: Name of the framework ('pytorch', 'tensorflow', etc.)

    Returns:
        Adapter class for the framework

    Raises:
        ValueError: If framework is not supported or not installed
    """
    if framework_name not in adapters:
        raise ValueError(
            f"Framework '{framework_name}' not supported or not installed. "
            f"Available frameworks: {list(adapters.keys())}"
        )
    return adapters[framework_name]
