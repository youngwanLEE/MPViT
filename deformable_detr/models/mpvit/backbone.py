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
# Deformable-DETR : https://github.com/fundamentalvision/Deformable-DETR
# --------------------------------------------------------------------------------


import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List

from util.misc import NestedTensor

from ..position_encoding import build_position_encoding
from .mpvit import mpvit_tiny, mpvit_xsmall, mpvit_small, mpvit_base

__all__ = [
    "build_mpvit_backbone",
]


def load_pretrained_weights(model, pretrained_weights_path=''):
    """ Load MPViT model weights from pretrained checkpoint or https url. """
    print("MPViT model: Loading weights from {} ...".format(pretrained_weights_path))
    if pretrained_weights_path.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained_weights_path, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
    remove_list = [
        "cls_head.cls.weight", "cls_head.cls.bias"
    ]
    model_params = {k: v for k, v in checkpoint["model"].items() if k not in remove_list}
    model.load_state_dict(model_params)


class MPViT_Backbone(nn.Module):
    """
    Implement MPViT backbone.
    """

    def __init__(self, name, out_features, pretrained_weights_path, model_kwargs):
        super().__init__()

        # Options: FrozenBN, GN, "SyncBN", "BN"
        norm = 'SyncBN'
        drop_path = 0.

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

        # Extract strides and number of channels for output features.
        self.strides = [self._out_feature_strides[x] for x in self._out_features]
        self.num_channels = [self._out_feature_channels[x] for x in self._out_features]
        self.backbone = model_func(out_features=out_features, norm=norm,
                                   drop_path_rate=drop_path, **model_kwargs)
        if pretrained_weights_path:
            load_pretrained_weights(self.backbone, pretrained_weights_path)

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone.forward_features(tensor_list.tensors)  # NOTE tensor_list.tensors is merged tensors (padded).
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out  # Returns a dict of NestedTensors, containing the features and corresponding (interpolated) masks.


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_mpvit_backbone(args):

    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    # Always train the backbone with 4 feature levels (including 3 intermediate outputs from backbone).
    assert train_backbone and return_interm_layers and args.num_feature_levels == 4

    backbone = MPViT_Backbone(
        name=args.backbone,
        out_features=["stage3", "stage4", "stage5"],
        pretrained_weights_path=args.backbone_weights,
        model_kwargs=eval(str(args.backbone_kwargs).replace("`", "")),
    )
    model = Joiner(backbone, position_embedding)
    return model