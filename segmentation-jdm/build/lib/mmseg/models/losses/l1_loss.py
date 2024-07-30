
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

def l1_loss(pred, target,reduction='none'):
    return F.l1_loss(pred, target, reduction=reduction)

from ..builder import LOSSES
@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean',loss_name='L1Loss'):
        super(L1Loss, self).__init__()
        # if reduction not in ['none', 'mean', 'sum']:
        #     raise ValueError(f'Unsupported reduction mode: {reduction}. '
        #                      f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        psnr = 10 * torch.log10(1 / F.mse_loss(pred, target))
        print("psnr:",psnr)
        return self.loss_weight * l1_loss(
            pred, target, reduction=self.reduction)

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name