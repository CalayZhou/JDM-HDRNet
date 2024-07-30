import torch
import torch.nn as nn
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(
                planes,
                planes,
                kernel_size=2,
                stride=stride,
                padding=0,
                bias=False)
        else:
            self.conv3 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


@HEADS.register_module()
class RTFNetHead(BaseDecodeHead):

    def __init__(self,
                 num_classes: int,
                 inplanes: int,
                 in_channels: int = 128,
                 channels=128,
                 **kwargs):
        super(RTFNetHead, self).__init__(
            in_channels, channels, in_index=0, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.inplanes = inplanes

        ########  DECODER  ########

        self.deconv1 = self._make_transpose_layer(
            TransBottleneck, self.inplanes // 2, 2,
            stride=2)  # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(
            TransBottleneck, self.inplanes // 2, 2,
            stride=2)  # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(
            TransBottleneck, self.inplanes // 2, 2,
            stride=2)  # using // for python 3.6
        self.deconv4 = self._make_transpose_layer(
            TransBottleneck, self.inplanes // 2, 2,
            stride=2)  # using // for python 3.6
        self.conv_seg = self._make_transpose_layer(
            TransBottleneck, self.num_classes, 2, stride=2)

    def forward(self, inputs):
        fuse = self._transform_inputs(inputs)
        verbose = False

        ######################################################################
        # decoder

        fuse = self.deconv1(fuse)
        if verbose: print("fuse after deconv1: ", fuse.size())  # (30, 40)
        fuse = self.deconv2(fuse)
        if verbose: print("fuse after deconv2: ", fuse.size())  # (60, 80)
        fuse = self.deconv3(fuse)
        if verbose: print("fuse after deconv3: ", fuse.size())  # (120, 160)
        fuse = self.deconv4(fuse)
        if verbose: print("fuse after deconv4: ", fuse.size())  # (240, 320)

        fuse = self.cls_seg(fuse)

        return fuse

    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes,
                    kernel_size=2,
                    stride=stride,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(planes),
            )

        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
