import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------- #
# Focal loss definition
# taken from: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
# default values: alpha = 0.25, gamma = 2 ->  https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
# ------------------------------------------------------- #
# converted as a class to define it as criterion 
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = "mean"): # changed default reduction to mean instead of none
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # if self.reduction == "none":
            # pass
            # loss = loss.mean() # Reduce the loss to a scalar before passing it to backward
            # to reduce loss to a scalar in case of batch of images -> loss.mean() or loss.sum()
            # -> change deafualt reduction to mean
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss