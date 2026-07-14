"""Confusion-matrix based semantic segmentation metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulate a confusion matrix and derive stable class metrics."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate integer class predictions and targets."""
        prediction = prediction.detach().view(-1).cpu().long()
        target = target.detach().view(-1).cpu().long()
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        encoded = self.num_classes * target[valid] + prediction[valid].clamp(0, self.num_classes - 1)
        self.confusion += torch.bincount(encoded, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def synchronize(self) -> None:
        """Sum the confusion matrix across initialized DDP processes."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            value = self.confusion.to(device)
            torch.distributed.all_reduce(value)
            self.confusion = value.cpu()

    def compute(self, include_background: bool = True) -> dict[str, Any]:
        """Return per-class and aggregate accuracy/IoU/Dice metrics."""
        matrix = self.confusion.double()
        tp = matrix.diag()
        support = matrix.sum(1)
        predicted = matrix.sum(0)
        union = support + predicted - tp
        precision = tp / predicted.clamp_min(1)
        recall = tp / support.clamp_min(1)
        iou = tp / union.clamp_min(1)
        dice = 2 * tp / (support + predicted).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        selected = slice(None) if include_background else slice(1, None)
        total = matrix.sum().clamp_min(1)
        return {
            "confusion_matrix": matrix.long().tolist(),
            "pixel_accuracy": float(tp.sum() / total),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "dice_per_class": dice.tolist(),
            "iou_per_class": iou.tolist(),
            "mean_iou": float(iou[selected].mean()),
            "mean_dice": float(dice[selected].mean()),
            "frequency_weighted_iou": float(((support / total) * iou).sum()),
        }


def save_class_metrics(metrics: dict[str, Any], class_names: list[str], path: str | Path) -> str:
    """Write per-class metrics and return the lowest-IoU class name."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, name in enumerate(class_names):
        rows.append({
            "class_id": index,
            "class_name": name,
            "precision": metrics["precision_per_class"][index],
            "recall": metrics["recall_per_class"][index],
            "f1": metrics["f1_per_class"][index],
            "dice": metrics["dice_per_class"][index],
            "iou": metrics["iou_per_class"][index],
        })
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    foreground = rows[1:] if len(rows) > 1 else rows
    return min(foreground, key=lambda row: row["iou"])["class_name"]

