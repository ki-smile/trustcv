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
    from .monai import MONAIAdapter, MONAICVRunner

    adapters["monai"] = MONAIAdapter
    __all__ = (
        __all__ + ["MONAIAdapter", "MONAICVRunner"]
        if "__all__" in locals()
        else ["MONAIAdapter", "MONAICVRunner"]
    )
except ImportError:
    pass

# XGBoost, LightGBM, CatBoost work via sklearn-compatible API (no adapter needed)

try:
    from .jax import JAXAdapter, JAXCVRunner

    adapters["jax"] = JAXAdapter
    __all__ = (
        __all__ + ["JAXAdapter", "JAXCVRunner"]
        if "__all__" in locals()
        else ["JAXAdapter", "JAXCVRunner"]
    )
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
