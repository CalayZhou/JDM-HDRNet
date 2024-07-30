from .decode_head import BaseDecodeHead
from ..builder import HEADS


@HEADS.register_module()
class DuoHSIHead(BaseDecodeHead):

    def __init__(self, in_channels, channels, **kwargs):
        super().__init__(in_channels, channels, **kwargs)

    def forward(self, inputs):
        img_features, hsi_features = inputs
        [c1, c2, c3, c4] = self._transform_inputs(hsi_features)
        x = self._transform_inputs(x)

        return super().forward(inputs)
