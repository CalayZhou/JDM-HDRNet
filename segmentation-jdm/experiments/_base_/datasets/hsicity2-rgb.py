dataset_type = 'HSICity2Dataset'
data_root = '/home/calay/DATASET/Mobile-Spec_jdm/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadReference'),
    dict(type='LoadShadingImage'),#calay 0906
    dict(type='LoadAnnotations'),
    dict(type='ResizeHSI', ratio=1), # wyb1108 0.5->0.25->1.0
    # dict(type='PhotoMetricDistortion'),
    dict(type='MaxNormalize', **img_norm_cfg),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'img_2', 'gt_semantic_seg','gt_shading'],#calay 0906
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
    dict(type='LoadReference'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 1057), # wyb1108 (1422, 1889)->(960, 1057)
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='ResizeHSI', ratio=1), # wyb1108 0.5->0.25
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='MaxNormalize', **img_norm_cfg),
            # dict(type='Pad', size=pad_size, pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img', 'img_2']),
            dict(type='Collect', keys=['img', 'img_2']),
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

