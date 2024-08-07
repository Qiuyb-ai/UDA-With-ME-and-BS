log_config = dict(
    interval=50,
    img_interval=1000,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b4.pth',
    backbone=dict(type='mit_b4', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256, conv_kernel_size=1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(
        work_dir=
        'work_dirs/local-basic/240807_0804_pots_irrg2vaih_irrg_1024_b4_15_85_711a8',
        log_config=dict(
            interval=50,
            img_interval=1000,
            hooks=[dict(type='TextLoggerHook', by_epoch=False)])),
    test_cfg=dict(mode='whole'))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
source_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(576, 576), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
target_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotationsPseudoLabelsV2',
        pseudo_labels_dir=None,
        reduce_zero_label=False,
        load_feats=False,
        pseudo_ratio=0.0),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomRotate90', prob=1.0),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='StrongAugmentation'),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
dataset_type = 'ISPRSDataset'
data_root_pots = '../../../data/potsdam_1024_irrg'
data_root_vaih = '../../../data/vaihingen_1024_irrg'
gt_seg_map_loader_cfg = dict(reduce_zero_label=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='UDADataset',
        source=dict(
            type='ISPRSDataset',
            data_root='../../../data/potsdam_1024_irrg',
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            gt_seg_map_loader_cfg=dict(reduce_zero_label=True),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=True),
                dict(
                    type='Resize',
                    img_scale=(576, 576),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomRotate90', prob=1.0),
                dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
                dict(
                    type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        target=dict(
            type='ISPRSDataset',
            data_root='../../../data/vaihingen_1024_irrg',
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            gt_seg_map_loader_cfg=dict(reduce_zero_label=True),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='LoadAnnotationsPseudoLabelsV2',
                    pseudo_labels_dir=None,
                    reduce_zero_label=False,
                    load_feats=False,
                    pseudo_ratio=0.0),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomRotate90', prob=1.0),
                dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
                dict(
                    type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
                dict(type='StrongAugmentation'),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        rare_class_sampling=dict(
            min_pixels=3000,
            class_temp=0.7,
            min_crop_ratio=1,
            lower_quantile=15,
            upper_quantile=85)),
    val=dict(
        type='ISPRSDataset',
        data_root='../../../data/vaihingen_1024_irrg',
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        gt_seg_map_loader_cfg=dict(reduce_zero_label=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ISPRSDataset',
        data_root='../../../data/vaihingen_1024_irrg',
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        gt_seg_map_loader_cfg=dict(reduce_zero_label=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
uda = dict(
    type='DACS',
    source_only=False,
    alpha=0.999,
    pseudo_threshold=0.98,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=0.75,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(type='block', mask_ratio=0.7, mask_block_size=64),
    debug_img_interval=1000,
    print_grad_magnitude=False,
    mix_train=False,
    masked_mix_train=True,
    original_mix=False)
use_ddp_wrapper = True
optimizer = dict(type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = None
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
seed = 1
n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=20)
evaluation = dict(interval=2000, metric=['mIoU', 'mFscore'])
name = '240807_0804_pots_irrg2vaih_irrg_1024_b4_15_85_711a8'
exp = 'basic'
name_dataset = 'pots_irrg2vaih_irrg'
work_dir = 'work_dirs/local-basic/240807_0804_pots_irrg2vaih_irrg_1024_b4_15_85_711a8'
git_rev = ''
gpu_ids = range(0, 1)
