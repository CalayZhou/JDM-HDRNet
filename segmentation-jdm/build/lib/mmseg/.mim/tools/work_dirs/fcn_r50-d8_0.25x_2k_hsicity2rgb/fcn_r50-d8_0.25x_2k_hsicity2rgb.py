norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 2, 4),
        strides=(2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_2=dict(
        type='ResNetV1c',
        in_channels=10,
        depth=50,
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 2, 4),
        strides=(2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=2,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=1,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'HSICity2Dataset'
data_root = '/home/calayzhou/dataset/PersonOutdoorDatasetv3_seg/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadReference'),
    dict(type='LoadAnnotations'),
    dict(type='ResizeHSI', ratio=1),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'img_2', 'gt_semantic_seg'],
        meta_keys=[
            'filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadReference'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 1057),
        flip=False,
        transforms=[
            dict(type='ResizeHSI', ratio=1),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img', 'img_2']),
            dict(type='Collect', keys=['img', 'img_2'])
        ])
]
data = dict(
    workers_per_gpu=2,
    samples_per_gpu=2,
    train=dict(
        type='HSICity2Dataset',
        data_root='/home/calayzhou/dataset/PersonOutdoorDatasetv3_seg/',
        img_dir='train',
        ann_dir='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadReference'),
            dict(type='LoadAnnotations'),
            dict(type='ResizeHSI', ratio=1),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'img_2', 'gt_semantic_seg'],
                meta_keys=[
                    'filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor'
                ])
        ]),
    val=dict(
        type='HSICity2Dataset',
        data_root='/home/calayzhou/dataset/PersonOutdoorDatasetv3_seg/',
        img_dir='test',
        ann_dir='test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadReference'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(960, 1057),
                flip=False,
                transforms=[
                    dict(type='ResizeHSI', ratio=1),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img', 'img_2']),
                    dict(type='Collect', keys=['img', 'img_2'])
                ])
        ]),
    test=dict(
        type='HSICity2Dataset',
        data_root='/home/calayzhou/dataset/PersonOutdoorDatasetv3_seg/',
        img_dir='test',
        ann_dir='test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadReference'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(960, 1057),
                flip=False,
                transforms=[
                    dict(type='ResizeHSI', ratio=1),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img', 'img_2']),
                    dict(type='Collect', keys=['img', 'img_2'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/fcn_r50-d8_0.25x_2k_hsicity2rgb'
gpu_ids = [0]
auto_resume = False
