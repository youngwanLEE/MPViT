# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------
# Modified from mmsegmentation (https://github.com/open-mmlab/mmsegmentation)
# Copyright (c) OpenMMLab. All rights reserved.
# --------------------------------------------------------------------------------

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MPViT',
        num_path=[4, 4, 4, 4],
        num_layers=[1, 1, 1, 1],
        embed_dims=[64, 128, 256, 512],
        num_heads=[8, 8, 8, 8],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
