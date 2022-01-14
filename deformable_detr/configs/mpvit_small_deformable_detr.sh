#!/usr/bin/env bash

set -x

EXP_DIR=exps/mpvit_small_deformable_detr
PY_ARGS=${@:1}

mkdir -p ${EXP_DIR}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --backbone "mpvit_small" \
    --backbone_weights "https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth" \
    ${PY_ARGS} | tee -a ${EXP_DIR}/history.txt