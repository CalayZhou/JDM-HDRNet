from mmcv.runner import BaseModule

from .mit import MixVisionTransformer
from .hrnet import HRNet
from ..builder import BACKBONES


@BACKBONES.register_module()
class DuoHSI(BaseModule):
    """
    DuoHSI Backbone.
    HSI image will be fed into MiT branch, while RGB image will be fed into HRNet branch.
    """

    def __init__(self,
                 rgb_configs=dict(),
                 hsi_configs=dict(),
                 init_cfg=None,
                 rgb_pretrained=None,
                 hsi_pretrained=None,
                 **kwargs):
        super(DuoHSI, self).__init__(init_cfg=init_cfg)
        self.rgb_branch = HRNet(pretrained=rgb_pretrained, **rgb_configs)
        self.hsi_branch = MixVisionTransformer(
            pretrained=hsi_pretrained, **hsi_configs)

    def forward(self, img, hsi):
        rgb_out = self.rgb_branch(img)
        hsi_out = self.hsi_branch(hsi)
        return [rgb_out, hsi_out]
