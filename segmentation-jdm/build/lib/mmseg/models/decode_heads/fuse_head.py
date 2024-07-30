import torch.nn.functional as F
from torch import nn
from .decode_head import BaseDecodeHead
from ..builder import HEADS


@HEADS.register_module()
class FuseNetHead(BaseDecodeHead):

    def __init__(self, num_classes=19, **kwargs):
        super().__init__(
            in_channels=[1, 2, 3, 4, 5, 6, 1],
            in_index=[0, 1, 2, 3, 4, 5, 6],
            channels=3,
            num_classes=num_classes,
            input_transform='multiple_select',
            **kwargs)
        bn_moment = 0.1
        # RGB DECODER
        self.CBR5_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_moment),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR4_Dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_moment),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR3_Dec = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=bn_moment),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.CBR2_Dec = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_moment),
            nn.ReLU(),
        )

        self.CBR1_Dec = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=bn_moment),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, inputs):
        y, id5, id4, id3, id2, id1, y_size = self._transform_inputs(inputs)
        # DECODER
        # Stage 5 dec
        y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
        y = self.CBR5_Dec(y)

        # Stage 4 dec
        y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
        y = self.CBR4_Dec(y)

        # Stage 3 dec
        y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
        y = self.CBR3_Dec(y)

        # Stage 2 dec
        y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
        y = self.CBR2_Dec(y)

        # Stage 1 dec
        y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
        y = self.CBR1_Dec(y)

        return y