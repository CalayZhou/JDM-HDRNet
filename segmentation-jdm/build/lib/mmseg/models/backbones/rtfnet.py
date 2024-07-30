import torch
import torch.nn as nn
import torchvision.models as models

from ..builder import BACKBONES
from mmcv.runner import BaseModule


@BACKBONES.register_module()
class RTFNet(BaseModule):

    def __init__(self,
                 in_channels=(3, 1),
                 num_resnet: int = 152,
                 init_cfg=None):
        super(RTFNet, self).__init__(init_cfg)
        self.rgb_channels, self.thr_channels = in_channels

        self.num_resnet_layers = num_resnet

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(
            self.thr_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        # self.encoder_thermal_conv1.weight.data = torch.unsqueeze(
        #     torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

    def forward(self, img, hsi):
        rgb = img
        thermal = hsi

        verbose = False

        # encoder

        ######################################################################

        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if verbose:
            print("thermal.size() original: ", thermal.size())  # (480, 640)

        ######################################################################

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

        thermal = self.encoder_thermal_conv1(thermal)
        if verbose:
            print("thermal.size() after conv1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose:
            print("thermal.size() after bn1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if verbose:
            print("thermal.size() after relu: ", thermal.size())  # (240, 320)

        rgb = rgb + thermal

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose:
            print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)

        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose:
            print("thermal.size() after maxpool: ",
                  thermal.size())  # (120, 160)

        ######################################################################

        rgb = self.encoder_rgb_layer1(rgb)
        if verbose:
            print("rgb.size() after layer1: ", rgb.size())  # (120, 160)
        thermal = self.encoder_thermal_layer1(thermal)
        if verbose:
            print("thermal.size() after layer1: ",
                  thermal.size())  # (120, 160)

        rgb = rgb + thermal

        ######################################################################

        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        thermal = self.encoder_thermal_layer2(thermal)
        if verbose:
            print("thermal.size() after layer2: ", thermal.size())  # (60, 80)

        rgb = rgb + thermal

        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        thermal = self.encoder_thermal_layer3(thermal)
        if verbose:
            print("thermal.size() after layer3: ", thermal.size())  # (30, 40)

        rgb = rgb + thermal

        ######################################################################

        rgb = self.encoder_rgb_layer4(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        thermal = self.encoder_thermal_layer4(thermal)
        if verbose:
            print("thermal.size() after layer4: ", thermal.size())  # (15, 20)

        fuse = rgb + thermal

        return [fuse]
