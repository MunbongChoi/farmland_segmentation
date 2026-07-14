"""Shared validation and test evaluation loop."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import CompositeSegmentationLoss
from .metrics.segmentation_metrics import SegmentationMetrics


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    criterion: CompositeSegmentationLoss,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    mixed_precision: bool = False,
) -> dict[str, Any]:
    """Evaluate a model and return loss plus background-aware metrics."""
    model.eval()
    meter = SegmentationMetrics(num_classes, ignore_index)
    loss_sum = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
            logits = model(images)
            loss, _ = criterion(logits, masks)
        meter.update(logits.argmax(1), masks)
        loss_sum += torch.tensor([float(loss.detach()) * images.shape[0], images.shape[0]], device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(loss_sum)
    meter.synchronize()
    metrics = meter.compute(include_background=True)
    foreground = meter.compute(include_background=False)
    metrics["loss"] = float(loss_sum[0] / loss_sum[1].clamp_min(1))
    metrics["mean_iou_no_background"] = foreground["mean_iou"]
    metrics["mean_dice_no_background"] = foreground["mean_dice"]
    return metrics

