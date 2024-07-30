_base_ = ['./lawin_mit-b0_1024x1024_160k_cityscapes.py']

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]))
