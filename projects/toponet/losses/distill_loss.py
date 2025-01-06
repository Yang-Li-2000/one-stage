# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from mmdet.models.builder import LOSSES
from mmdet.models import weighted_loss
import mmcv
import torch.nn.functional as F
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import functools


@weighted_loss
def l2_loss(pred, target):
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)**2
    return loss


@LOSSES.register_module()
class L2Loss(nn.Module):
    def __init__(self,
                 neg_pos_ub: int = -1,
                 pos_margin: float = -1,
                 neg_margin: float = -1,
                 hard_mining: bool = False,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0):
        super(L2Loss, self).__init__()
        self.neg_pos_ub = neg_pos_ub
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.hard_mining = hard_mining
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l2_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox