# ---------------------------------------------------------------
# Copyright (c) 2022-2023 TUM, Fahong Zhang. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

source_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(576, 576), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsPseudoLabelsV2',
         pseudo_labels_dir=None,
         reduce_zero_label=False,
         load_feats=False,
         pseudo_ratio=0.0),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='StrongAugmentation'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'ISPRSDataset'
data_root_pots = '../../../data/potsdam_1024_irrg'
data_root_vaih = '../../../data/vaihingen_1024_irrg'
gt_seg_map_loader_cfg=dict(reduce_zero_label=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            data_root=data_root_pots,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
            pipeline=source_pipeline),
        target=dict(
            type=dataset_type,
            data_root=data_root_vaih,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
            pipeline=target_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root_vaih,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root_vaih,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
        pipeline=test_pipeline),
)
