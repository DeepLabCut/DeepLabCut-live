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
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from dlclive.pose_estimation_pytorch.models.predictors import BasePredictor
from dlclive.pose_estimation_pytorch.models.registry import Registry, build_from_cfg

HEADS = Registry("heads", build_func=build_from_cfg)


class BaseHead(ABC, nn.Module):
    """A head for pose estimation models

    Attributes:
        stride: The stride for the head (or neck + head pair), where positive values
            indicate an increase in resolution while negative values a decrease.
            Assuming that H and W are divisible by `stride`, this is the value such
            that if a backbone outputs an encoding of shape (C, H, W), the head will
            output heatmaps of shape:
                (C, H * stride, W * stride)       if stride > 0
                (C, -H/stride, -W/stride)         if stride < 0
        predictor: an object to generate predictions from the head outputs
    """

    def __init__(self, stride: int | float, predictor: BasePredictor) -> None:
        super().__init__()
        if stride == 0:
            raise ValueError(f"Stride must not be 0. Found {stride}.")

        self.stride = stride
        self.predictor = predictor

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Given the feature maps for an image ()

        Args:
            x: the feature maps, of shape (b, c, h, w)

        Returns:
            the head outputs (e.g. "heatmap", "locref")
        """
        pass
