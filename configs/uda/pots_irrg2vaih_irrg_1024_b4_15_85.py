# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/segformer_b4_256.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../_base_/datasets/pots_irrg2vaih_irrg_1024.py',
    # DAFormer Self-Training
    '../_base_/uda/dacs_a999_fdthings.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 1  # seed with median performance

model = dict(
    decode_head=dict(
        num_classes=6,
    ),
)
data = dict(
    train=dict(

        rare_class_sampling=dict(
            class_temp=0.7, lower_quantile=15, upper_quantile=85
        ),
    ),
    # Use one separate thread/worker for data loading.
    workers_per_gpu=1,
    # Batch size
    samples_per_gpu=2,
)
# MIC Parameters
uda = dict(
    mix_train=False,
    masked_mix_train=True,
    original_mix=False,

    imnet_feature_dist_classes=None,
    alpha=0.999,
    pseudo_threshold=0.98,
    imnet_feature_dist_lambda=0,
    # Apply masking to color-augmented target images
    mask_mode='separatetrgaug',
    # Use the same teacher alpha for MIC as for DAFormer
    # self-training (0.999)
    mask_alpha='same',
    # Use the same pseudo label confidence threshold for
    # MIC as for DAFormer self-training (0.968)
    mask_pseudo_threshold='same',
    # Equal weighting of MIC loss
    mask_lambda=1,
    # Use random patch masking with a patch size of 64x64
    # and a mask ratio of 0.7
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True))
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01)
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=20)
evaluation = dict(interval=2000, metric=['mIoU','mFscore'])
# Meta Information for Result Analysis
name = 'pots_irrg2vaih_irrg_1024_b4_15_85'
exp = 'basic'
name_dataset = 'pots_irrg2vaih_irrg'

# For the other configurations used in the paper, please refer to experiment.py
