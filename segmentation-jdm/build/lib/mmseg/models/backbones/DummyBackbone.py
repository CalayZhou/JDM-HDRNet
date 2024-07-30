# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..builder import BACKBONES
@BACKBONES.register_module()
# class DummyBackbone(nn.Module):
#     def __init__(self, out_channels, init_cfg=None):
#         super(DummyBackbone, self).__init__()
#         self.out_channels = out_channels
#         if init_cfg is not None:
#             self.init_weights(init_cfg)
#
#     def forward(self, x):
#         return x
#
#     def init_weights(self, init_cfg):
#         pass
class DummyBackbone(nn.Module):
    def __init__(self):
        super(DummyBackbone,self).__init__()
    def foward(self,x):
        return x