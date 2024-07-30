import torch
from torch import nn
import torch.nn.functional as F

from ..builder import BACKBONES
from mmcv.runner import BaseModule


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


class MiniInception(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left = ConvBnLeakyRelu2d(in_channels, out_channels // 2)
        self.conv1_right = ConvBnLeakyRelu2d(
            in_channels, out_channels // 2, padding=2, dilation=2)
        self.conv2_left = ConvBnLeakyRelu2d(out_channels, out_channels // 2)
        self.conv2_right = ConvBnLeakyRelu2d(
            out_channels, out_channels // 2, padding=2, dilation=2)
        self.conv3_left = ConvBnLeakyRelu2d(out_channels, out_channels // 2)
        self.conv3_right = ConvBnLeakyRelu2d(
            out_channels, out_channels // 2, padding=2, dilation=2)

    def forward(self, x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x


@BACKBONES.register_module()
class MFNet(BaseModule):

    def __init__(self,
                 in_channels=(3, 1),
                 rgb_ch=(16, 48, 48, 96, 96),
                 inf_ch=(16, 16, 16, 36, 36),
                 init_cfg=None):
        super().__init__(init_cfg)
        super(MFNet, self).__init__()

        self.conv1_rgb = ConvBnLeakyRelu2d(in_channels[0], rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb = MiniInception(rgb_ch[3], rgb_ch[4])

        self.conv1_inf = ConvBnLeakyRelu2d(in_channels[1], inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf = MiniInception(inf_ch[3], inf_ch[4])

    def forward(self, img, hsi):
        # split data into RGB and INF
        x_rgb = img
        x_inf = hsi

        # encode
        x_rgb = self.conv1_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb, kernel_size=2, stride=2)  # pool1
        x_rgb = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2)  # pool2
        x_rgb = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2)  # pool3
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2)  # pool4
        x_rgb = self.conv5_rgb(x_rgb)

        x_inf = self.conv1_inf(x_inf)
        x_inf = F.max_pool2d(x_inf, kernel_size=2, stride=2)  # pool1
        x_inf = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2)  # pool2
        x_inf = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2)  # pool3
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2)  # pool4
        x_inf = self.conv5_inf(x_inf)

        x = torch.cat((x_rgb, x_inf), dim=1)  # fusion RGB and INF

        d1 = torch.cat((x_rgb_p4, x_inf_p4), dim=1)
        d2 = torch.cat((x_rgb_p3, x_inf_p3), dim=1)
        d3 = torch.cat((x_rgb_p2, x_inf_p2), dim=1)

        return [x, d1, d2, d3]
