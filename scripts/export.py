"""Exports DeepLabCut models for DeepLabCut-Live"""
import warnings
from pathlib import Path

import torch
from ruamel.yaml import YAML


def read_config_as_dict(config_path: str | Path) -> dict:
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes
    """
    with open(config_path, "r") as f:
        cfg = YAML(typ='safe', pure=True).load(f)

    return cfg


def export_dlc3_model(
    export_path: Path,
    model_config_path: Path,
    pose_snapshot: Path,
    detector_snapshot: Path | None = None,
) -> None:
    """Exports a DLC3 model

    Args:
        export_path:
        model_config_path:
        pose_snapshot:
        detector_snapshot:
    """
    model_cfg = read_config_as_dict(model_config_path)

    load_kwargs = dict(map_location="cpu", weights_only=True)
    pose_weights = torch.load(pose_snapshot, **load_kwargs)["model"]
    detector_weights = None
    if detector_snapshot is None:
        if model_cfg["method"].lower() == "td":
            warnings.warn(
                "The model is a top-down model but no detector snapshot was given."
                "The configuration will be changed to run the model in bottom-up mode."
            )
            model_cfg["method"] = "bu"

    else:
        if model_cfg["method"].lower() == "bu":
            raise ValueError(f"Cannot use a detector with a bottom-up model!")
        detector_weights = torch.load(detector_snapshot, **load_kwargs)["model"]

    torch.save(
        dict(config=model_cfg, detector=detector_weights, pose=pose_weights),
        export_path,
    )


if __name__ == "__main__":
    root = Path("/Users/john/Documents")
    project_dir = root / "2024-10-14-my-model"

    # Exporting a top-down model
    model_dir = project_dir / "top-down-resnet-50" / "model"
    export_dlc3_model(
        export_path=model_dir / "dlclive-export-fasterrcnnMobilenet-resnet50.pt",
        model_config_path=model_dir / "pytorch_config.yaml",
        pose_snapshot=model_dir / "snapshot-50.pt",
        detector_snapshot=model_dir / "snapshot-detector-100.pt",
    )

    # Exporting a bottom-up model
    model_dir = project_dir / "resnet-50" / "model"
    export_dlc3_model(
        export_path=model_dir / "dlclive-export-bu-resnet50.pt",
        model_config_path=model_dir / "pytorch_config.yaml",
        pose_snapshot=model_dir / "snapshot-50.pt",
        detector_snapshot=None,
    )
