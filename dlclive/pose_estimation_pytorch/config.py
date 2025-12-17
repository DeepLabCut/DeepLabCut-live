import logging
from dataclasses import dataclass, fields, asdict

from collections import OrderedDict
from pathlib import Path

import torch

def _parse_dataclass_from_dict(cls: type[dataclass], cfg: dict) -> dataclass:
    """Parses a dictionary into a dataclass.

    Args:
        cls: The dataclass to parse into.
        cfg: The dictionary to parse from.

    Returns:
        The dataclass parsed from the dictionary.
    """
    # If the config is already a dataclass, return it (it was already parsed before)
    if isinstance(cfg, cls): 
        return cfg

    # Otherwise, parse the dictionary into the dataclass
    field_names = {f.name for f in fields(cls)}
    known = {k: v for k, v in cfg.items() if k in field_names}
    extras = {k: v for k, v in cfg.items() if k not in field_names}
    obj = cls(**known)
    obj.additional_kwargs = extras
    return obj


@dataclass
class SkipFrames:
    """Configuration for skip frames.

    Skip-frames can be used for top-down models running with a detector. If skip > 0,
    then the detector will only be run every `skip` frames. Between frames where the
    detector is run, bounding boxes will be computed from the pose estimated in the
    previous frame.

    Every `N` frames, the detector will be run to detect bounding boxes for individuals.
    In the "skipped" frames between the frames where the object detector is run, the
    bounding boxes will be computed from the poses estimated in the previous frame (with
    some margin added around the poses).

    Attributes:
        skip: The number of frames to skip between each run of the detector.
        margin: The margin (in pixels) to use when generating bboxes
    """

    skip: int
    margin: int
    _age: int = 0
    _detections: dict[str, torch.Tensor] | None = None

    def get_detections(self) -> dict[str, torch.Tensor] | None:
        return self._detections

    def update(self, pose: torch.Tensor, w: int, h: int) -> None:
        """Generates bounding boxes from a pose.

        Args:
            pose: The pose from which to generate bounding boxes.
            w: The width of the image.
            h: The height of the image.

        Returns:
            A dictionary containing the bounding boxes and scores for each detection.
        """
        if self._age >= self.skip:
            self._age = 0
            self._detections = None
            return

        num_det, num_kpts = pose.shape[:2]
        size = max(w, h)

        bboxes = torch.zeros((num_det, 4))
        bboxes[:, :2] = (
            torch.min(torch.nan_to_num(pose, size)[..., :2], dim=1)[0] - self.margin
        )
        bboxes[:, 2:4] = (
            torch.max(torch.nan_to_num(pose, 0)[..., :2], dim=1)[0] + self.margin
        )
        bboxes = torch.clip(bboxes, min=torch.zeros(4), max=torch.tensor([w, h, w, h]))
        self._detections = dict(boxes=bboxes, scores=torch.ones(num_det))
        self._age += 1


@dataclass
class TopDownConfig:
    """Configuration for top-down models.

    Attributes:
        bbox_cutoff: The minimum score required for a bounding box to be considered.
        max_detections: The maximum number of detections to keep in a frame. If None,
            the `max_detections` will be set to the number of individuals in the model
            configuration file when `read_config` is called.
        skip_frames: If defined, the detector will only be run every
            `skip_frames.skip` frames.
    """

    bbox_cutoff: float = 0.6
    max_detections: int | None = 30
    crop_size: tuple[int, int] = (256, 256)
    skip_frames: SkipFrames | None = None

    def read_config(self, model_cfg: dict) -> None:
        crop = model_cfg.get("data", {}).get("inference", {}).get("top_down_crop")
        if crop is not None:
            self.crop_size = (crop["width"], crop["height"])

        if self.max_detections is None:
            individuals = model_cfg.get("metadata", {}).get("individuals", [])
            self.max_detections = len(individuals)


@dataclass
class DataConfig:
    inference: dict 
    bbox_margin: int | None = None
    colormode: str | None = None
    train: dict | None = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "DataConfig":
        return _parse_dataclass_from_dict(cls, cfg)

@dataclass
class DetectorConfig:
    data: DataConfig | dict
    model: dict
    runner: str | None = None
    train_settings: dict | None = None

    @classmethod
    def from_dict(cls, cfg: dict) -> "DetectorConfig":
        return _parse_dataclass_from_dict(cls, cfg)

@dataclass
class BaseConfig:
    """Pytorch model configuration (DeepLabCut format)."""
    model: dict 
    net_type: str
    metadata: dict 
    data: DataConfig
    method: str
    detector: DetectorConfig | None = None
    train_settings: dict | None = None
    inference: dict | None = None

    def __post_init__(self) -> None:
        self.data = DataConfig.from_dict(self.data)
        if self.detector is not None:
            self.detector = DetectorConfig.from_dict(self.detector)

    @classmethod
    def from_dict(cls, cfg: dict) -> "BaseConfig":
        return _parse_dataclass_from_dict(cls, cfg)

    def to_dict(self) -> dict:
        return asdict(self)

StateDict=OrderedDict[str, torch.Tensor]

def load_exported_model(
    path: str | Path,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> tuple[BaseConfig, StateDict, StateDict | None]:
    """
    Loads a DeepLabCut exported model from a file.
    
    The exported model is a dictionary containing the following keys:
    - config: The base configuration of the model.
    - pose: The state dict of the model.
    - detector: The state dict of the detector.

    Args:
        path: The path to the exported model.
        map_location: The device to map the model to.
        weights_only: Whether to load only the weights of the model.

    Returns:
        A tuple containing the base configuration and the state dicts of the 
        pose and detector models. (The detector state dict is optional.)

    Raises:
        ValueError: If the exported model file does not contain a 'config' and 'pose' key.
        FileNotFoundError: If the exported model file does not exist.
    """
    raw_data = torch.load(path, map_location=map_location, weights_only=weights_only)
    if "config" not in raw_data or "pose" not in raw_data:
        raise ValueError(
            f"Invalid exported model file: {path}. The exported model must contain "
            "a 'config' and 'pose' key. For more information on how to export a model, "
            "visit https://deeplabcut.github.io/ and search for `export_model`."
        )

    base_config = BaseConfig.from_dict(raw_data["config"])
    return base_config, raw_data["pose"], raw_data["detector"]