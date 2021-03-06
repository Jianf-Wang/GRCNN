import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def cross_entropy_loss(input, target, size_average=True, smooth=0.1):
    input = F.log_softmax(input, dim=1)
    target_1 = (target > 0).int().long()
    target_2 = (1. - target_1)
  
    target_1 = ((target_1 * smooth)/target_1.sum(dim=1, keepdim=True)).float()
    target_2 = ((target_2 * smooth)/target_2.sum(dim=1, keepdim=True)).float()

    target = target - target_1 + target_2

    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLossSmooth(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)
