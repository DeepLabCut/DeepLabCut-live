
"""Factory to build runners for DeepLabCut-Live inference"""
from pathlib import Path

from dlclive.core.runner import BaseRunner


def build_runner(
    model_type: str,
    model_path: str | Path,
    **kwargs,
) -> BaseRunner:
    """

    Parameters
    ----------
    model_type
    model_path
    kwargs

    Returns
    -------

    """
    if model_type.lower() == "pytorch":
        from dlclive.pose_estimation_pytorch.runner import PyTorchRunner
        return PyTorchRunner(model_path, **kwargs)

    elif model_type.lower() in ("tensorflow", "base", "tensorrt", "lite"):
        from dlclive.pose_estimation_tensorflow.runner import TensorFlowRunner
        return TensorFlowRunner(model_path, model_type, **kwargs)

    raise ValueError(f"Unknown model type: {model_type}")
