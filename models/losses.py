"""
Loss Functions
  - FocalLoss: down-weights easy Normal examples, focuses on minority events
  - LabelSmoothingFocalLoss: adds label smoothing regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: Per-class weighting tensor (n_classes,) or scalar.
               If None, no per-class weighting is applied.
        gamma: Focusing parameter. gamma=0 reduces to standard cross-entropy.
               gamma=2 is the standard recommendation.
        reduction: "mean", "sum", or "none".
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = 4,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw unnormalized scores.
            targets: (B,) integer class labels.

        Returns:
            Scalar loss.
        """
        B, C = logits.shape

        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        probs = torch.exp(log_probs)               # (B, C)

        # Gather probabilities of the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)           # (B,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]  # (B,)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss with label smoothing.
    Label smoothing prevents overconfidence by softening the one-hot targets.

    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing).
        alpha, gamma: Same as FocalLoss.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        reduction: str = "mean",
        num_classes: int = 4,
    ):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = num_classes

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape

        # Smoothed targets: (B, C)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (C - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight based on true class probability
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma

        # KL divergence between smooth targets and predictions
        loss = -(smooth_targets * log_probs).sum(dim=-1)  # (B,)

        # Apply focal weighting
        loss = focal_weight * loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    smoothing: float = 0.1,
    num_classes: int = 4,
) -> nn.Module:
    """
    Factory function for building the loss criterion.

    Args:
        loss_type: "focal", "focal_smooth", or "cross_entropy"
        class_weights: Per-class weight tensor (n_classes,).
        gamma: Focal loss focusing parameter.
        smoothing: Label smoothing factor.
        num_classes: Number of output classes.
    """
    if loss_type == "focal":
        return FocalLoss(
            alpha=class_weights, gamma=gamma, num_classes=num_classes
        )
    elif loss_type == "focal_smooth":
        return LabelSmoothingFocalLoss(
            alpha=class_weights, gamma=gamma, smoothing=smoothing, num_classes=num_classes
        )
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Choose from: focal, focal_smooth, cross_entropy")
