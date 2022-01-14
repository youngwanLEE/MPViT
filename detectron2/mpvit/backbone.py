# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# CoaT: https://github.com/mlpc-ucsd/CoaT
# --------------------------------------------------------------------------------


import torch

from detectron2.layers import (
    ShapeSpec,
)
from detectron2.modeling import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelP6P7, LastLevelMaxPool

from .mpvit import mpvit_tiny, mpvit_xsmall, mpvit_small, mpvit_base

__all__ = [
    "build_mpvit_fpn_backbone",
]


class MPViT_Backbone(Backbone):
    """
    Implement MPViT backbone.
    """

    def __init__(self, name, out_features, norm, drop_path, model_kwargs):
        super().__init__()
        self._out_features = out_features
        self._out_feature_strides = {"stage2": 4, "stage3": 8, "stage4": 16, "stage5": 32}

        if name == 'mpvit_tiny':
            model_func = mpvit_tiny
            self._out_feature_channels = {"stage2": 96, "stage3": 176, "stage4": 216, "stage5": 216}
        elif name == 'mpvit_xsmall':
            model_func = mpvit_xsmall
            self._out_feature_channels = {"stage2": 128, "stage3": 192, "stage4": 256, "stage5": 256}
        elif name == 'mpvit_small':
            model_func = mpvit_small
            self._out_feature_channels = {"stage2": 128, "stage3": 216, "stage4": 288, "stage5": 288}
        elif name == 'mpvit_base':
            model_func = mpvit_base
            self._out_feature_channels = {"stage2": 224, "stage3": 368, "stage4": 480, "stage5": 480}
        else:
            raise ValueError()

        self.backbone = model_func(out_features=out_features, norm=norm,
                                   drop_path_rate=drop_path, **model_kwargs)


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"MPViT takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        return self.backbone.forward_features(x)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def build_mpvit_backbone(cfg):
    """
    Create a MPViT instance from config.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        A MPViT backbone instance.
    """
    # fmt: off
    name = cfg.MODEL.MPVIT.NAME
    out_features = cfg.MODEL.MPVIT.OUT_FEATURES
    norm = cfg.MODEL.MPVIT.NORM
    drop_path = cfg.MODEL.MPVIT.DROP_PATH

    model_kwargs = eval(str(cfg.MODEL.MPVIT.MODEL_KWARGS).replace("`", ""))

    return MPViT_Backbone(name, out_features, norm, drop_path, model_kwargs)


@BACKBONE_REGISTRY.register()
def build_mpvit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a MPViT w/ FPN backbone.

    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mpvit_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_mpvit_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_mpvit_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()[cfg.MODEL.MPVIT.OUT_FEATURES[-1]].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, cfg.MODEL.FPN.IN_FEATURES[-1]),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
