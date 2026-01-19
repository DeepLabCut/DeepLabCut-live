#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""PyTorch and ONNX runners for DeepLabCut-Live"""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torchvision.transforms import v2

import dlclive.pose_estimation_pytorch.data as data
import dlclive.pose_estimation_pytorch.models as models
import dlclive.pose_estimation_pytorch.dynamic_cropping as dynamic_cropping
from dlclive.core.runner import BaseRunner
from dlclive.pose_estimation_pytorch.data.image import AutoPadToDivisor


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


class PyTorchRunner(BaseRunner):
    """PyTorch runner for live pose estimation using DeepLabCut-Live.

    Args:
        path: The path to the model to run inference with.
        device: The device on which to run inference, e.g. "cpu", "cuda", "cuda:0"
        precision: The precision of the model. One of "FP16" or "FP32".
        single_animal: This option is only available for single-animal pose estimation
            models. It makes the code behave in exactly the same way as DeepLabCut-Live
            with version < 3.0.0. This ensures backwards compatibility with any
            Processors that were implemented.
        dynamic: Whether to use dynamic cropping.
        top_down_config: Only for top-down models running with a detector.
    """

    def __init__(
        self,
        path: str | Path,
        device: str = "auto",
        precision: Literal["FP16", "FP32"] = "FP32",
        single_animal: bool = True,
        dynamic: dict | dynamic_cropping.DynamicCropper | None = None,
        top_down_config: dict | TopDownConfig | None = None,
    ) -> None:
        super().__init__(path)
        self.device = _parse_device(device)
        self.precision = precision
        self.single_animal = single_animal

        self.cfg = None
        self.detector = None
        self.model = None
        self.detector_transform = None
        self.pose_transform = None

        # Parse Dynamic Cropping parameters
        if isinstance(dynamic, dict):
            dynamic_type = dynamic.get("type", "DynamicCropper")
            if dynamic_type == "DynamicCropper":
                cropper_cls = dynamic_cropping.DynamicCropper
            else:
                cropper_cls = dynamic_cropping.TopDownDynamicCropper
            dynamic_params = dynamic.copy()
            dynamic_params.pop("type")
            dynamic = cropper_cls(**dynamic_params)

        # Parse Top-Down config
        if isinstance(top_down_config, dict):
            skip_frame_cfg = top_down_config.get("skip_frames")
            if skip_frame_cfg is not None:
                top_down_config["skip_frames"] = SkipFrames(**skip_frame_cfg)
            top_down_config = TopDownConfig(**top_down_config)

        self.dynamic = dynamic
        self.top_down_config = top_down_config

    def close(self) -> None:
        """Clears any resources used by the runner."""
        pass

    @torch.inference_mode()
    def get_pose(self, frame: np.ndarray) -> np.ndarray:
        c, h, w = frame.shape
        tensor = torch.from_numpy(frame).permute(2, 0, 1) # CHW, still on CPU

        offsets_and_scales = None
        if self.detector is not None:
            detections = None
            if self.top_down_config.skip_frames is not None:
                detections = self.top_down_config.skip_frames.get_detections()

            if detections is None:
                # Apply detector transform before inference
                detector_input = self.detector_transform(tensor).unsqueeze(0).to(self.device)
                if self.precision == "FP16":
                    detector_input = detector_input.half()
                detections = self.detector(detector_input)[0]

            frame_batch, offsets_and_scales = self._prepare_top_down(tensor, detections)
            if len(frame_batch) == 0:
                offsets_and_scales = [(0, 0), 1]            
            tensor = frame_batch  # still CHW, batched

        if self.dynamic is not None:
            tensor = self.dynamic.crop(tensor)

        # Apply pose transform
        model_input = self.pose_transform(tensor)
        # Ensure 4D input: (N, C, H, W)
        if model_input.dim() == 3:
            model_input = model_input.unsqueeze(0)
        # Send to device
        model_input = model_input.to(self.device)
        if self.precision == "FP16":
            model_input = model_input.half()

        outputs = self.model(model_input)
        batch_pose = self.model.get_predictions(outputs)["bodypart"]["poses"]

        if self.dynamic is not None:
            batch_pose = self.dynamic.update(batch_pose)

        if self.detector is None:
            pose = batch_pose[0]
        else:
            pose = self._postprocess_top_down(batch_pose, offsets_and_scales)
            if self.top_down_config.skip_frames is not None:
                self.top_down_config.skip_frames.update(pose, w, h)

        if self.single_animal:
            if len(pose) == 0:
                bodyparts, coords = pose.shape[-2:]
                return np.zeros((bodyparts, coords))

            pose = pose[0]

        return pose.cpu().numpy()

    def init_inference(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Initializes inference process on the provided frame.

        This method serves as an abstract base method, meant to be implemented by
        subclasses. It takes an input image frame and optional additional parameters
        to set up and perform inference. The method must return a processed result
        as a numpy array.

        Parameters
        ----------
        frame : np.ndarray
            The input image frame for which inference needs to be set up.
        kwargs : dict, optional
            Additional parameters that may be required for specific implementation
            of the inference initialization.

        Returns
        -------
        np.ndarray
            The result of the inference after being initialized and processed.
        """
        self.load_model()
        return self.get_pose(frame)

    def load_model(self) -> None:
        """Loads the model from the provided path."""
        raw_data = torch.load(self.path, map_location="cpu", weights_only=True)

        self.cfg = raw_data["config"]
        self.model = models.PoseModel.build(self.cfg["model"])
        self.model.load_state_dict(raw_data["pose"])
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.precision == "FP16":
            self.model = self.model.half()

        self.detector = None
        if self.dynamic is None and raw_data.get("detector") is not None:
            self.detector = models.DETECTORS.build(self.cfg["detector"]["model"])
            self.detector.to(self.device)
            self.detector.load_state_dict(raw_data["detector"])
            self.detector.eval()
            if self.precision == "FP16":
                self.detector = self.detector.half()

            if self.top_down_config is None:
                self.top_down_config = TopDownConfig()
            self.top_down_config.read_config(self.cfg)

            detector_transforms = [v2.ToDtype(torch.float32, scale=True)]
            if self.cfg["detector"]["data"]["inference"].get("normalize_images", False):
                detector_transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            self.detector_transform = v2.Compose(detector_transforms)

        if isinstance(self.dynamic, dynamic_cropping.TopDownDynamicCropper):
            crop = self.cfg["data"]["inference"].get("top_down_crop", {})
            w, h = crop.get("width", 256), crop.get("height", 256)
            self.dynamic.top_down_crop_size = w, h

        if (
            self.cfg["method"] == "td"
            and self.detector is None
            and self.dynamic is None
        ):
            raise ValueError(
                "Top-down models must either use a detector or a TopDownDynamicCropper."
            )

        pose_transforms = [v2.ToDtype(torch.float32, scale=True)]
        auto_padding_cfg = self.cfg["data"]["inference"].get("auto_padding", None)
        if auto_padding_cfg:
            pose_transforms.append(
                AutoPadToDivisor(
                    pad_height_divisor=auto_padding_cfg.get("pad_height_divisor", 1),
                    pad_width_divisor=auto_padding_cfg.get("pad_width_divisor", 1),
                )
            )
        if self.cfg["data"]["inference"].get("normalize_images", False):
            pose_transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.pose_transform = v2.Compose(pose_transforms)

    def read_config(self) -> dict:
        """Reads the configuration file"""
        if self.cfg is not None:
            return copy.deepcopy(self.cfg)

        raw_data = torch.load(self.path, map_location="cpu", weights_only=True)
        return raw_data["config"]

    def _prepare_top_down(
        self, frame: torch.Tensor, detections: dict[str, torch.Tensor]
    ):
        """Prepares a frame for top-down pose estimation."""
        # Accept unbatched frame (C, H, W) or batched frame (1, C, H, W)
        if frame.dim() == 4:
            if frame.size(0) != 1:
                raise ValueError(f"Expected batch size 1, got {frame.size(0)}")
            frame = frame[0]  # (C, H, W)
        elif frame.dim() != 3:
            raise ValueError(f"Expected frame of shape (C, H, W) or (1, C, H, W), got {frame.shape}")

        bboxes, scores = detections["boxes"], detections["scores"]
        bboxes = bboxes[scores >= self.top_down_config.bbox_cutoff]

        if len(bboxes) > 0 and self.top_down_config.max_detections is not None:
            bboxes = bboxes[: self.top_down_config.max_detections]

        crops = []
        offsets_and_scales = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.tolist()
            cropped_frame, offset, scale = data.top_down_crop_torch(
                frame,
                (x1, y1, x2 - x1, y2 - y1),
                output_size=self.top_down_config.crop_size,
                margin=0,
            )
            crops.append(cropped_frame)
            offsets_and_scales.append((offset, scale))

        if len(crops) > 0:
            frame_batch = torch.stack(crops, dim=0)
        else:
            crop_w, crop_h = self.top_down_config.crop_size
            frame_batch = torch.zeros((1, 3, crop_h, crop_w), device=frame.device)
            offsets_and_scales = [(0, 0), 1]

        return frame_batch, offsets_and_scales

    def _postprocess_top_down(
        self,
        batch_pose: torch.Tensor,
        offsets_and_scales: list[tuple[tuple[int, int], tuple[float, float]]],
    ) -> torch.Tensor:
        """Post-processes pose for top-down models."""
        if len(batch_pose) == 0:
            bodyparts, coords = batch_pose.shape[-2:]
            return torch.zeros((0, bodyparts, coords))

        poses = []
        for pose, (offset, scale) in zip(batch_pose, offsets_and_scales):
            poses.append(
                torch.cat(
                    [
                        pose[..., :2] * torch.tensor(scale) + torch.tensor(offset),
                        pose[..., 2:3],
                    ],
                    dim=-1,
                )
            )

        return torch.cat(poses)


def _parse_device(device: str | None) -> str:
    if device is None:
        device = "auto"

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return device
