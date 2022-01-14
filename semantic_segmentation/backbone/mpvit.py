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


import numpy as np
import math

import torch

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_

from einops import rearrange
from functools import partial
from torch import nn, einsum
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.runner import load_checkpoint
from mmcv.cnn import build_norm_layer

from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES

__all__ = [
    "mpvit_tiny",
    "mpvit_xsmall",
    "mpvit_small",
    "mpvit_base",
]

def _cfg_mpvit(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


class Mlp(nn.Module):
    """Feed-forward network (FFN, a.k.a. MLP) class."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        self.conv = torch.nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        )
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x


class DWConv2d_BN(nn.Module):
    """
    Depthwise Separable Conv
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = build_norm_layer(norm_cfg, out_ch)[1]
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):
    """
    Depthwise Convolutional Patch Embedding layer
    Image to Patch Embedding
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        patch_size=16,
        stride=1,
        pad=0,
        act_layer=nn.Hardswish,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        # TODO : confirm whether act_layer is effective or not
        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    def __init__(self, embed_dim, num_path=4, isPool=False, norm_cfg=dict(type="BN")):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList(
            [
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if isPool and idx == 0 else 1,
                    pad=1,
                    norm_cfg=norm_cfg,
                )
                for idx in range(num_path)
            ]
        )

        # scale

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
                )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = (
            x.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MHCABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3,
        drop_path=0.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        shared_cpe=None,
        shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, size):
        # x.shape = [B, N, C]

        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.drop_path(self.factoratt_crpe(cur, size))

        cur = self.norm2(x)
        x = x + self.drop_path(self.mlp(cur))
        return x


class MHCAEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        drop_path_list=[],
        qk_scale=None,
        crpe_window={3: 2, 5: 3, 7: 3},
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        self.MHCA_layers = nn.ModuleList(
            [
                MHCABlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_list[idx],
                    qk_scale=qk_scale,
                    shared_cpe=self.cpe,
                    shared_crpe=self.crpe,
                )
                for idx in range(self.num_layers)
            ]
        )

    def forward(self, x, size):
        H, W = size
        B = x.shape[0]
        # x' shape : [B, N, C]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Hardswish,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(
            in_features, hidden_features, act_layer=act_layer, norm_cfg=norm_cfg
        )
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        # self.norm = norm_layer(hidden_features)
        self.norm = build_norm_layer(norm_cfg, hidden_features)[1]
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCA_stage(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        norm_cfg=dict(type="BN"),
        drop_path_list=[],
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList(
            [
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                )
                for _ in range(num_path)
            ]
        )

        self.InvRes = ResBlock(
            in_features=embed_dim, out_features=embed_dim, norm_cfg=norm_cfg
        )
        self.aggregate = Conv2d_BN(
            embed_dim * (num_path + 1),
            out_embed_dim,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, inputs):
        att_outputs = [self.InvRes(inputs[0])]
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            att_outputs.append(encoder(x, size=(H, W)))

        out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(out_concat)

        return out


def dpr_generator(drop_path_rate, num_layers, num_stages):
    """
    Generate drop path rate list following linear decay rule
    """
    dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur : cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr


@BACKBONES.register_module()
class MPViT(nn.Module):
    """Multi-Path ViT class."""

    def __init__(
        self,
        num_classes=80,
        in_chans=3,
        num_stages=4,
        num_layers=[1, 1, 1, 1],
        mlp_ratios=[8, 8, 4, 4],
        num_path=[4, 4, 4, 4],
        embed_dims=[64, 128, 256, 512],
        num_heads=[8, 8, 8, 8],
        drop_path_rate=0.0,
        norm_cfg=dict(type="BN"),
        norm_eval=True,
        pretrained=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages
        self.conv_norm_cfg = norm_cfg
        self.norm_eval = norm_eval

        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                in_chans,
                embed_dims[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
            Conv2d_BN(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
                norm_cfg=self.conv_norm_cfg,
            ),
        )

        # Patch embeddings.
        self.patch_embed_stages = nn.ModuleList(
            [
                Patch_Embed_stage(
                    embed_dims[idx],
                    num_path=num_path[idx],
                    isPool=False if idx == 0 else True,
                    norm_cfg=self.conv_norm_cfg,
                )
                for idx in range(self.num_stages)
            ]
        )

        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stages = nn.ModuleList(
            [
                MHCA_stage(
                    embed_dims[idx],
                    embed_dims[idx + 1]
                    if not (idx + 1) == self.num_stages
                    else embed_dims[idx],
                    num_layers[idx],
                    num_heads[idx],
                    mlp_ratios[idx],
                    num_path[idx],
                    norm_cfg=self.conv_norm_cfg,
                    drop_path_list=dpr[idx],
                )
                for idx in range(self.num_stages)
            ]
        )

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward_features(self, x):

        # x's shape : [B, C, H, W]
        outs = []
        x = self.stem(x)  # Shape : [B, C, H/4, W/4]

        for idx in range(self.num_stages):
            att_inputs = self.patch_embed_stages[idx](x)
            x = self.mhca_stages[idx](att_inputs)

            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(MPViT, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def mpvit_tiny(**kwargs):
    """mpvit_tiny :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 96, 176, 216]
    - MLP_ratio : 2
    Number of params: 5843736
    FLOPs : 1654163812
    Activations : 16641952
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 96, 176, 216],
        mlp_ratios=[2, 2, 2, 2],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_xsmall(**kwargs):
    """mpvit_xsmall :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 2, 4, 1]
    - #channels : [64, 128, 192, 256]
    - MLP_ratio : 4
    Number of params : 10573448
    FLOPs : 2971396560
    Activations : 21983464
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 2, 4, 1],
        embed_dims=[64, 128, 192, 256],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_small(**kwargs):
    """mpvit_small :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 6, 3]
    - #channels : [64, 128, 216, 288]
    - MLP_ratio : 4
    Number of params : 22892400
    FLOPs : 4799650824
    Activations : 30601880
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model


def mpvit_base(**kwargs):
    """mpvit_base :

    - #paths : [2, 3, 3, 3]
    - #layers : [1, 3, 8, 3]
    - #channels : [128, 224, 368, 480]
    - MLP_ratio : 4
    Number of params: 74845976
    FLOPs : 16445326240
    Activations : 60204392
    """

    model = MPViT(
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[128, 224, 368, 480],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
        **kwargs,
    )
    model.default_cfg = _cfg_mpvit()
    return model
