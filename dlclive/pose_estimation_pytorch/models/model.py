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

import copy

import torch
import torch.nn as nn

from dlclive.pose_estimation_pytorch.models.backbones import BACKBONES, BaseBackbone
from dlclive.pose_estimation_pytorch.models.heads import HEADS, BaseHead
from dlclive.pose_estimation_pytorch.models.necks import NECKS, BaseNeck
from dlclive.pose_estimation_pytorch.models.predictors import PREDICTORS


class PoseModel(nn.Module):
    """A pose estimation model

    A pose estimation model is composed of a backbone, optionally a neck, and an
    arbitrary number of heads. Outputs are computed as follows:
    """

    def __init__(
        self,
        cfg: dict,
        backbone: BaseBackbone,
        heads: dict[str, BaseHead],
        neck: BaseNeck | None = None,
    ) -> None:
        """
        Args:
            cfg: configuration dictionary for the model.
            backbone: backbone network architecture.
            heads: the heads for the model
            neck: neck network architecture (default is None). Defaults to None.
        """
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self.neck = neck

        self._strides = {
            name: _model_stride(self.backbone.stride, head.stride)
            for name, head in heads.items()
        }

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass of the PoseModel.

        Args:
            x: input images

        Returns:
            Outputs of head groups
        """
        if x.dim() == 3:
            x = x[None, :]
        features = self.backbone(x)
        if self.neck:
            features = self.neck(features)

        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)
        return outputs

    def get_predictions(self, outputs: dict[str, dict[str, torch.Tensor]]) -> dict:
        """Abstract method for the forward pass of the Predictor.

        Args:
            outputs: outputs of the model heads

        Returns:
            A dictionary containing the predictions of each head group
        """
        return {
            name: head.predictor(self._strides[name], outputs[name])
            for name, head in self.heads.items()
        }

    @staticmethod
    def build(cfg: dict) -> "PoseModel":
        """
        Args:
            cfg: The configuration of the model to build.

        Returns:
            the built pose model
        """
        cfg["backbone"]["pretrained"] = False
        backbone = BACKBONES.build(dict(cfg["backbone"]))

        neck = None
        if cfg.get("neck"):
            neck = NECKS.build(dict(cfg["neck"]))

        heads = {}
        for name, head_cfg in cfg["heads"].items():
            head_cfg = copy.deepcopy(head_cfg)

            # Remove keys not needed for DLCLive inference
            for k in ("target_generator", "criterion", "aggregator", "weight_init"):
                if k in head_cfg:
                    head_cfg.pop(k)

            head_cfg["predictor"] = PREDICTORS.build(head_cfg["predictor"])
            heads[name] = HEADS.build(head_cfg)

        return PoseModel(cfg=cfg, backbone=backbone, neck=neck, heads=heads)


def _model_stride(backbone_stride: int | float, head_stride: int | float) -> float:
    """Computes the model stride from a backbone and a head"""
    if head_stride > 0:
        return backbone_stride / head_stride

    return backbone_stride * -head_stride
