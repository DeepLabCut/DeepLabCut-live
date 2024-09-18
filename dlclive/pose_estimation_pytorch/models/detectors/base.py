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

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from dlclive.pose_estimation_pytorch.models.registry import Registry, build_from_cfg

DETECTORS = Registry("detectors", build_func=build_from_cfg)


class BaseDetector(ABC, nn.Module):
    """
    Definition of the class BaseDetector object.
    This is an abstract class defining the common structure and inference for detectors.
    """

    def __init__(
        self,
        freeze_bn_stats: bool = False,
        freeze_bn_weights: bool = False,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.freeze_bn_stats = freeze_bn_stats
        self.freeze_bn_weights = freeze_bn_weights
        self._pretrained = pretrained

    @abstractmethod
    def forward(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        """
        Forward pass of the detector

        Args:
            x: images to be processed
            targets: ground-truth boxes present in each images

        Returns:
            losses: {'loss_name': loss_value}
            detections: for each of the b images, {"boxes": bounding_boxes}
        """
        pass
