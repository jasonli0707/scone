#!/usr/bin/bash
# Train on Kodak dataset
# for i in {01..24};
for i in 19;
do
    python train_image.py \
    TRAIN_CONFIGS.out_dir='siren_kodim'$i'' \
    TRAIN_CONFIGS.lr=1e-4 \
    DATASET_CONFIGS.img_file='data/kodak/kodim'$i'.png' \
    model_config='siren'
done