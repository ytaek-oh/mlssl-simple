# code in this file is adpated from
# https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/models/losses/resample_loss.py, derived from
# https://github.com/open-mmlab/mmdetection/tree/master/mmdet/models/losses, licensed under Apache License 2.0 License

import torch
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # reduction in class dimension
    if weight is not None:
        valid_inds = torch.any(weight > 0, dim=1)
        loss = loss[valid_inds]
        weight = weight[valid_inds]

        loss = loss * weight
        loss = loss.sum(dim=1) / weight.sum(dim=1)
    else:
        loss = loss.mean(dim=1)

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def partial_cross_entropy(pred, label, loss_weight=1.0, valid_mask=None):
    if valid_mask is not None:
        valid_mask = valid_mask.float()
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    return loss_weight * weight_reduce_loss(loss, weight=valid_mask)


def mse_loss(pred, target, loss_weight=1.0, weight=None):
    """Warpper of mse loss."""
    loss = F.mse_loss(pred, target, reduction='none')
    return loss_weight * weight_reduce_loss(loss, weight=weight)
