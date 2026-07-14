"""Composable multi-class segmentation losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _targets(target: torch.Tensor, num_classes: int, ignore_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    valid = target != ignore_index
    safe = target.masked_fill(~valid, 0).clamp(0, num_classes - 1)
    one_hot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).float()
    return one_hot * valid.unsqueeze(1), valid.unsqueeze(1)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int, include_background: bool = False) -> torch.Tensor:
    """Soft multi-class Dice loss."""
    probabilities = logits.softmax(dim=1)
    one_hot, valid = _targets(target, logits.shape[1], ignore_index)
    probabilities = probabilities * valid
    if not include_background and logits.shape[1] > 1:
        probabilities, one_hot = probabilities[:, 1:], one_hot[:, 1:]
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int, gamma: float) -> torch.Tensor:
    """Multi-class focal loss based on per-pixel cross entropy."""
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    ce = ce[valid]
    return (((1.0 - torch.exp(-ce)) ** gamma) * ce).mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int, alpha: float, beta: float) -> torch.Tensor:
    """Tversky loss with controllable false-positive/false-negative costs."""
    probabilities = logits.softmax(dim=1)
    one_hot, valid = _targets(target, logits.shape[1], ignore_index)
    probabilities = probabilities * valid
    probabilities, one_hot = probabilities[:, 1:], one_hot[:, 1:]
    dims = (0, 2, 3)
    tp = (probabilities * one_hot).sum(dims)
    fp = (probabilities * (1.0 - one_hot)).sum(dims)
    fn = ((1.0 - probabilities) * one_hot).sum(dims)
    return 1.0 - ((tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)).mean()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Dice loss between differentiable predicted and target boundary maps."""
    probabilities = logits.softmax(dim=1)
    one_hot, valid = _targets(target, logits.shape[1], ignore_index)
    probabilities, one_hot = probabilities[:, 1:] * valid, one_hot[:, 1:]
    pred_boundary = F.max_pool2d(probabilities, 3, 1, 1) - (-F.max_pool2d(-probabilities, 3, 1, 1))
    true_boundary = F.max_pool2d(one_hot, 3, 1, 1) - (-F.max_pool2d(-one_hot, 3, 1, 1))
    intersection = (pred_boundary * true_boundary).sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (pred_boundary.sum() + true_boundary.sum() + 1e-6)


class CompositeSegmentationLoss(nn.Module):
    """Weighted CE/BCE/Dice/Focal/Tversky/boundary objective."""

    def __init__(self, config: dict[str, Any], num_classes: int, ignore_index: int) -> None:
        super().__init__()
        self.weights = {key: float(value) for key, value in config["weights"].items()}
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = float(config.get("focal_gamma", 2.0))
        self.alpha = float(config.get("tversky_alpha", 0.3))
        self.beta = float(config.get("tversky_beta", 0.7))
        class_weights = config.get("class_weights")
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32) if class_weights else None)
        if not any(value > 0 for value in self.weights.values()):
            raise ValueError("하나 이상의 loss weight가 0보다 커야 합니다.")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, torch.Tensor] = {}
        if self.weights.get("cross_entropy", 0.0):
            parts["cross_entropy"] = F.cross_entropy(logits, target, self.class_weights, ignore_index=self.ignore_index)
        if self.weights.get("binary_cross_entropy", 0.0):
            one_hot, valid = _targets(target, self.num_classes, self.ignore_index)
            value = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
            parts["binary_cross_entropy"] = (value * valid).sum() / (valid.sum().clamp_min(1) * self.num_classes)
        if self.weights.get("dice", 0.0):
            parts["dice"] = soft_dice_loss(logits, target, self.ignore_index)
        if self.weights.get("focal", 0.0):
            parts["focal"] = focal_loss(logits, target, self.ignore_index, self.gamma)
        if self.weights.get("tversky", 0.0):
            parts["tversky"] = tversky_loss(logits, target, self.ignore_index, self.alpha, self.beta)
        if self.weights.get("boundary", 0.0):
            parts["boundary"] = boundary_loss(logits, target, self.ignore_index)
        total = sum(self.weights[name] * value for name, value in parts.items())
        return total, {name: float(value.detach()) for name, value in parts.items()}
