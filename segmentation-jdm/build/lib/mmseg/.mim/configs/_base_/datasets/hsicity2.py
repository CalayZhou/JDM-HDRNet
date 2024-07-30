dataset_type = 'HSICity2Dataset'
data_root = '/home/ybwang/data0/datasets/Honor/person_v4/data/'
pad_size = (720, 960)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadHSIFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ResizeHSI', ratio=0.25), # wyb1108 0.5->0.25
    dict(type='PhotoMetricDistortion'),
    dict(type='PhotoMetricDistortionHSI'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=255),
    dict(type='PadHSI', size=pad_size, pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='DefaultFormatBundleHSI'),
    dict(
        type='Collect',
        keys=['img', 'hsi', 'gt_semantic_seg'],
        meta_keys=[
            'filename',
            'ori_filename',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadHSIFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 1057), # wyb1108 (1422, 1889)->(960, 1057)
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='ResizeHSI', ratio=0.25), # wyb1108 0.5->0.25
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=255),
            dict(type='PadHSI', size=pad_size, pad_val=0),
            dict(type='ImageToTensor', keys=['img', 'hsi']),
            dict(type='Collect', keys=['img', 'hsi']),
        ])
]
data = dict(
    workers_per_gpu=2,
    samples_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='test',
        pipeline=test_pipeline),
)