# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        # num_stages=4,
        # out_indices=(0, 1, 2, 3),
        # dilations=(1, 1, 2, 4),
        # strides=(1, 2, 1, 1),
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 2, 4),
        strides=(2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
# 第二路输入的编码层
#     backbone_2=dict(
#         type='DummyBackbone'),

    backbone_2=dict(
        type='ResNetV1c',
        in_channels=10,
        depth=50,
        num_stages=3,
        out_indices=(0, 1, 2),
        dilations=(1, 2, 4),
        strides=(2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=2,#3
        channels=512,#512,#512
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=2,#2
        channels=512,#256
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # type='L1Loss',  loss_weight=1.0)),#calay 0906
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
