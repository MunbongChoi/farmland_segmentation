"""Shared CLI implementation for validation and test commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets.dataset import build_datasets
from .evaluate import evaluate_model
from .losses import CompositeSegmentationLoss
from .metrics.segmentation_metrics import save_class_metrics
from .model import build_model
from .utils.checkpoint import load_checkpoint
from .utils.config import apply_overrides, load_config
from .utils.logger import setup_logger


def run_cli(split: str) -> None:
    """Evaluate the requested split and write JSON/CSV reports."""
    parser = argparse.ArgumentParser(description=f"{split} 평가")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    output = Path(config["project"]["output_dir"]) / "metrics" / split
    logger = setup_logger(split, output / f"{split}.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, validation, test = build_datasets(config)
    dataset = validation if split == "validation" else test
    loader = DataLoader(dataset, batch_size=int(config["training"]["validation_batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]))
    # 체크포인트가 전 가중치를 덮어쓰므로 사전학습(HF Hub) 로드는 생략한다.
    config["model"]["pretrained"] = False
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, current_config=config, map_location=device)
    criterion = CompositeSegmentationLoss(config["loss"], int(config["dataset"]["num_classes"]), int(config["dataset"]["ignore_index"])).to(device)
    metrics = evaluate_model(model, loader, criterion, device, int(config["dataset"]["num_classes"]), int(config["dataset"]["ignore_index"]), bool(config["training"]["mixed_precision"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lowest = save_class_metrics(metrics, list(config["dataset"]["class_names"]), output / "class_metrics.csv")
    logger.info("loss=%.5f mIoU(fg)=%.5f 최저 IoU 클래스=%s", metrics["loss"], metrics["mean_iou_no_background"], lowest)

