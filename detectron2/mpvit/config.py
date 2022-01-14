# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------------------


from detectron2.config import CfgNode as CN


def add_mpvit_config(cfg):
    """
    Add config for MPViT.
    """
    _C = cfg

    _C.MODEL.MPVIT = CN()

    # CoaT model name.
    _C.MODEL.MPVIT.NAME = ""

    # Output features from CoaT backbone.
    _C.MODEL.MPVIT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]


    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.MPVIT.NORM = "SyncBN"

    _C.MODEL.MPVIT.DROP_PATH = 0.

    _C.MODEL.MPVIT.MODEL_KWARGS = "{}"

    _C.SOLVER.OPTIMIZER = "ADAMW"

    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0


    _C.AUG = CN()

    _C.AUG.DETR = False