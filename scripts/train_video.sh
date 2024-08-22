#!/usr/bin/bash
# Train on cat video
python train_video.py \
TRAIN_CONFIGS.out_dir='scone_cat' \
TRAIN_CONFIGS.lr=1e-4 \
model_config='scone'