_base_ = './fcn_hr18_0.5x_40k_hsicity2rgb.py'
model = dict(
    pretrained=None,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='work_dirs/fcn_hr48_512x1024_160k_cityscapes/latest.pth', prefix='backbone.'),
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))
