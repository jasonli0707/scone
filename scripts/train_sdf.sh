#!/usr/bin/bash
# Train on SDF dataset
python train_sdf.py \
DATASET_CONFIGS.xyz_file='data/stanford3d/gt_thai.xyz' \
TRAIN_CONFIGS.out_dir='scone_thai' \
TRAIN_CONFIGS.lr=0.0005 \
model_config='scone'