"""Factory to build runners for DeepLabCut-Live inference"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from dlclive.core.runner import BaseRunner


def build_runner(
    model_type: Literal["pytorch", "tensorflow", "base", "tensorrt", "lite"],
    model_path: str | Path,
    **kwargs,
) -> BaseRunner:
    """

    Parameters
    ----------
    model_type: str, optional
        Which model to use. For the PyTorch engine, options are [`pytorch`]. For the
        TensorFlow engine, options are [`base`, `tensorrt`, `lite`].
    model_path: str, Path
        Full path to exported model (created when `deeplabcut.export_model(...)` was
        called). For PyTorch models, this is a single model file. For TensorFlow models,
        this is a directory containing the model snapshots.

    kwargs: dict, optional
        PyTorch Engine Kwargs:

        TensorFlow Engine Kwargs:

    Returns
    -------

    """
    if model_type.lower() == "pytorch":
        from dlclive.pose_estimation_pytorch.runner import PyTorchRunner

        valid = {"device", "precision", "single_animal", "dynamic", "top_down_config"}
        return PyTorchRunner(model_path, **filter_keys(valid, kwargs))

    elif model_type.lower() in ("tensorflow", "base", "tensorrt", "lite"):
        from dlclive.pose_estimation_tensorflow.runner import TensorFlowRunner

        if model_type.lower() == "tensorflow":
            model_type = "base"

        valid = {"tf_config", "precision"}
        return TensorFlowRunner(model_path, model_type, **filter_keys(valid, kwargs))

    raise ValueError(f"Unknown model type: {model_type}")


def filter_keys(valid: set[str], kwargs: dict) -> dict:
    """Filters the keys in kwargs, only keeping those in valid."""
    return {k: v for k, v in kwargs.items() if k in valid}
