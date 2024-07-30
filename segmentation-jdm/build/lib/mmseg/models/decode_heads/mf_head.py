import torch
from torch import nn
import torch.nn.functional as F

from .decode_head import BaseDecodeHead
from ..builder import HEADS


class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


@HEADS.register_module()
class MFNetHead(BaseDecodeHead):

    def __init__(self,
                 rgb_ch=(16, 48, 48, 96, 96),
                 inf_ch=(16, 16, 16, 36, 36),
                 in_channels=[128, 128, 128, 128],
                 channels=2,
                 num_classes=19,
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            input_transform='multiple_select',
            **kwargs)
        self.decode4 = ConvBnLeakyRelu2d(rgb_ch[3] + inf_ch[3],
                                         rgb_ch[2] + inf_ch[2])
        self.decode3 = ConvBnLeakyRelu2d(rgb_ch[2] + inf_ch[2],
                                         rgb_ch[1] + inf_ch[1])
        self.decode2 = ConvBnLeakyRelu2d(rgb_ch[1] + inf_ch[1],
                                         rgb_ch[0] + inf_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(rgb_ch[0] + inf_ch[0], num_classes)

    def forward(self, inputs):
        x, d1, d2, d3 = self._transform_inputs(inputs)
        # decode
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # unpool4
        x = self.decode4(x + d1)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # unpool3
        x = self.decode3(x + d2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # unpool2
        x = self.decode2(x + d3)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # unpool1
        x = self.decode1(x)

        return x
