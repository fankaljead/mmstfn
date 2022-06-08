import torch
from torch import nn
from ssim import msssim
import torch.nn.functional as F


class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.8, normalize=True):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.normalize = normalize
        self.lamda = 0

    def forward(self, prediction, target):
        loss_l1 = F.l1_loss(prediction, target)
        loss_l2 = F.mse_loss(prediction, target)
        loss_content = self.lamda * loss_l2 + (1 - self.lamda) * loss_l1
        loss_ssim = self.alpha * (1.0 - msssim(prediction, target, normalize=self.normalize))
        return (loss_content + loss_ssim)