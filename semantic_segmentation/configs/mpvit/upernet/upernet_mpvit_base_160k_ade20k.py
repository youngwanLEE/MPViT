# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------

_base_ = [
    '../../_base_/models/upernet_mpvit.py', '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]
model = dict(
    pretrained='https://dl.dropbox.com/s/6g44hu0ax71s74h/mpvit_base_mm.pth',
    backbone=dict(
        type='MPViT',
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 8, 3],
        embed_dims=[128, 224, 368, 480],
        drop_path_rate=0.4,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False
    ),
    decode_head=dict(
        in_channels=[224, 368, 480, 480],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=480,
        num_classes=150
    ))

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
