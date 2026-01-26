"""
Utils for the DLC-Live Model Zoo
"""
# NOTE JR 2026-23-01: This file contains duplicated code from the DeepLabCut main repository.
# This should be removed once a solution is found to address duplicate code.

import copy
import logging
from pathlib import Path

from dlclibrary.dlcmodelzoo.modelzoo_download import download_huggingface_model
from ruamel.yaml import YAML

from dlclive.modelzoo.resolve_config import update_config

_MODELZOO_PATH = Path(__file__).parent


def get_super_animal_model_config_path(model_name: str) -> Path:
    """Get the path to the model configuration file for a model and validate choice of model"""
    cfg_path = _MODELZOO_PATH / "model_configs" / f"{model_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Modelzoo model configuration file not found: {cfg_path} Available models: {list_available_models()}"
        )
    return cfg_path


def get_super_animal_project_config_path(super_animal: str) -> Path:
    """Get the path to the project configuration file for a project and validate choice of project"""
    cfg_path = _MODELZOO_PATH / "project_configs" / f"{super_animal}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Modelzoo project configuration file not found: {cfg_path}Available projects: {list_available_projects()}"
        )
    return cfg_path


def get_snapshot_folder_path() -> Path:
    return _MODELZOO_PATH / "snapshots"


def list_available_models() -> list[str]:
    return [p.stem for p in _MODELZOO_PATH.glob("model_configs/*.yaml")]


def list_available_projects() -> list[str]:
    return [p.stem for p in _MODELZOO_PATH.glob("project_configs/*.yaml")]


def list_available_combinations() -> list[str]:
    models = list_available_models()
    projects = list_available_projects()
    combinations = ["_".join([p, m]) for p in projects for m in models]
    return combinations


def read_config_as_dict(config_path: str | Path) -> dict:
    """
    Args:
        config_path: the path to the configuration file to load

    Returns:
        The configuration file with pure Python classes
    """
    with open(config_path) as f:
        cfg = YAML(typ="safe", pure=True).load(f)

    return cfg


# NOTE - DUPLICATED @deruyter92 2026-23-01: Copied from the original DeepLabCut codebase
# from deeplabcut/pose_estimation_pytorch/config/make_pose_config.py
def add_metadata(
    project_config: dict,
    config: dict,
) -> dict:
    """Adds metadata to a pytorch pose configuration

    Args:
        project_config: the project configuration
        config: the pytorch pose configuration
        pose_config_path: the path where the pytorch pose configuration will be saved

    Returns:
        the configuration with a `meta` key added
    """
    config = copy.deepcopy(config)
    config["metadata"] = {
        "project_path": project_config["project_path"],
        "pose_config_path": "",
        "bodyparts": project_config.get("multianimalbodyparts") or project_config["bodyparts"],
        "unique_bodyparts": project_config.get("uniquebodyparts", []),
        "individuals": project_config.get("individuals", ["animal"]),
        "with_identity": project_config.get("identity", False),
    }
    return config


# NOTE - DUPLICATED @deruyter92 2026-23-01: Copied from the original DeepLabCut codebase
# from deeplabcut/pose_estimation_pytorch/modelzoo/utils.py
def load_super_animal_config(
    super_animal: str,
    model_name: str,
    detector_name: str | None = None,
    max_individuals: int = 30,
    device: str | None = None,
) -> dict:
    """Loads the model configuration file for a model, detector and SuperAnimal

    Args:
        super_animal: The name of the SuperAnimal for which to create the model config.
        model_name: The name of the model for which to create the model config.
        detector_name: The name of the detector for which to create the model config.
        max_individuals: The maximum number of detections to make in an image
        device: The device to use to train/run inference on the model

    Returns:
        The model configuration for a SuperAnimal-pretrained model.
    """
    project_cfg_path = get_super_animal_project_config_path(super_animal=super_animal)
    project_config = read_config_as_dict(project_cfg_path)

    model_cfg_path = get_super_animal_model_config_path(model_name=model_name)
    model_config = read_config_as_dict(model_cfg_path)
    model_config = add_metadata(project_config, model_config)
    model_config = update_config(model_config, max_individuals, device)

    if detector_name is None and super_animal != "superanimal_humanbody":
        model_config["method"] = "BU"
    else:
        model_config["method"] = "TD"
        if super_animal != "superanimal_humanbody":
            detector_cfg_path = get_super_animal_model_config_path(model_name=detector_name)
            detector_cfg = read_config_as_dict(detector_cfg_path)
            model_config["detector"] = detector_cfg
    return model_config


def download_super_animal_snapshot(dataset: str, model_name: str) -> Path:
    """Downloads a SuperAnimal snapshot

    Args:
        dataset: The name of the SuperAnimal dataset for which to download a snapshot.
        model_name: The name of the model for which to download a snapshot.

    Returns:
        The path to the downloaded snapshot.

    Raises:
        RuntimeError if the model fails to download.
    """
    snapshot_dir = get_snapshot_folder_path()
    model_name = f"{dataset}_{model_name}"
    model_filename = f"{model_name}.pt"
    model_path = snapshot_dir / model_filename

    if model_path.exists():
        logging.info(f"Snapshot {model_path} already exists, skipping download")
        return model_path

    try:
        download_huggingface_model(model_name, target_dir=str(snapshot_dir), rename_mapping=model_filename)

        if not model_path.exists():
            raise RuntimeError(f"Failed to download {model_name} to {model_path}")

    except Exception as e:
        logging.error(f"Failed to download superanimal snapshot {model_name} to {model_path}: {e}")
        raise e

    return model_path
