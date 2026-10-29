"""Class-weighted focal loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'): 
        super(DynamicFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Class weights and the loss function are constructed from optimization labels in each fold.
