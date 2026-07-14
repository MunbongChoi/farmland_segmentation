"""U-Net training entry point with CPU, single-GPU, and DDP support."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .datasets.dataset import build_datasets
from .evaluate import evaluate_model
from .losses import CompositeSegmentationLoss
from .model import build_model, parameter_counts
from .utils.checkpoint import load_checkpoint, save_checkpoint
from .utils.config import apply_overrides, load_config, save_config
from .utils.logger import setup_logger
from .utils.seed import seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="논/밭 U-Net 학습")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    """Initialize torchrun environment and return rank/world/local-rank."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank


def select_device(local_rank: int) -> torch.device:
    """Select a CUDA process-local device or CPU fallback."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def build_loader(dataset: Any, config: dict[str, Any], training: bool, world_size: int, rank: int) -> tuple[DataLoader[Any], Any]:
    """Create a DataLoader and optional distributed sampler."""
    settings = config["training"]
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=training) if world_size > 1 else None
    workers = int(settings["num_workers"])
    batch_size = int(settings["batch_size"] if training else settings["validation_batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(settings.get("pin_memory", True)),
        persistent_workers=bool(settings.get("persistent_workers", True)) and workers > 0,
        worker_init_fn=seed_worker,
        drop_last=training,
    )
    return loader, sampler


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: CompositeSegmentationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    settings: dict[str, Any],
    show_progress: bool = False,
) -> float:
    """Run one gradient-accumulated training epoch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(settings["gradient_accumulation_steps"])
    mixed = bool(settings["mixed_precision"]) and device.type == "cuda"
    total = torch.zeros(2, dtype=torch.float64, device=device)
    progress = tqdm(loader, desc="train", leave=False, disable=not show_progress)
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        should_step = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        synchronization = model.no_sync() if hasattr(model, "no_sync") and not should_step else nullcontext()
        with synchronization:
            with torch.autocast(device_type=device.type, enabled=mixed):
                logits = model(images)
                raw_loss, _ = criterion(logits, masks)
                loss = raw_loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf loss가 발생했습니다: step={step}")
            scaler.scale(loss).backward()
        if should_step:
            scaler.unscale_(optimizer)
            clip = float(settings.get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += torch.tensor([float(raw_loss.detach()) * images.shape[0], images.shape[0]], device=device)
        if show_progress:
            progress.set_postfix(loss=f"{float(raw_loss.detach()):.4f}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return float(total[0] / total[1].clamp_min(1))


def append_history(path: Path, row: dict[str, Any]) -> None:
    """Append one flat epoch record to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_scheduler(optimizer: torch.optim.Optimizer, settings: dict[str, Any]) -> Any:
    """Build the configured epoch scheduler."""
    name = str(settings.get("scheduler", "cosine")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(settings["epochs"]), eta_min=float(settings["min_learning_rate"]))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(settings.get("scheduler_step_size", 20)), gamma=float(settings.get("scheduler_gamma", 0.5)))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=int(settings.get("scheduler_patience", 5)), factor=float(settings.get("scheduler_gamma", 0.5)))
    raise ValueError(f"지원하지 않는 scheduler입니다: {name}")


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    if args.resume:
        config["training"]["resume"] = args.resume
    rank, world_size, local_rank = setup_distributed()
    device = select_device(local_rank)
    seed_everything(int(config["project"]["seed"]) + rank)
    output = Path(config["project"]["output_dir"])
    logger = setup_logger("train", output / "logs" / "train.log", rank)
    if rank == 0:
        save_config(config, output / "resolved_config.yaml")
        (output / "logs").mkdir(parents=True, exist_ok=True)
        (output / "logs" / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    train_set, validation_set, _ = build_datasets(config)
    train_loader, train_sampler = build_loader(train_set, config, True, world_size, rank)
    validation_loader, _ = build_loader(validation_set, config, False, world_size, rank)
    model = build_model(config).to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    logger.info("device=%s world_size=%d train=%d val=%d params=%d trainable=%d", device, world_size, len(train_set), len(validation_set), total_parameters, trainable_parameters)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
    criterion = CompositeSegmentationLoss(config["loss"], int(config["dataset"]["num_classes"]), int(config["dataset"]["ignore_index"])).to(device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    scheduler = build_scheduler(optimizer, settings)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(settings["mixed_precision"]) and device.type == "cuda")
    start_epoch, best_metric = 0, float("-inf")
    if settings.get("resume"):
        checkpoint = load_checkpoint(settings["resume"], model, optimizer, scheduler, scaler, config, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        logger.info("학습 재개: epoch=%d best=%.6f", start_epoch, best_metric)
    writer = None
    wandb_run = None
    if rank == 0 and config.get("logging", {}).get("tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(output / "logs" / "tensorboard")
        except ImportError:
            logger.warning("tensorboard가 설치되지 않아 비활성화합니다.")
    if rank == 0 and config.get("logging", {}).get("wandb", False):
        try:
            import wandb

            wandb_run = wandb.init(project=config["logging"]["wandb_project"], name=config["project"]["name"], config=config)
        except ImportError:
            logger.warning("wandb가 설치되지 않아 비활성화합니다.")
    no_improvement = 0
    training_started = time.perf_counter()
    for epoch in range(start_epoch, int(settings["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, settings, show_progress=rank == 0)
        validation = evaluate_model(model, validation_loader, criterion, device, int(config["dataset"]["num_classes"]), int(config["dataset"]["ignore_index"]), bool(settings["mixed_precision"]))
        metric = float(validation[str(settings["monitor"])])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        else:
            scheduler.step()
        elapsed = time.perf_counter() - started
        gpu_memory = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation["loss"], "mean_iou": validation["mean_iou"], "mean_iou_no_background": validation["mean_iou_no_background"], "mean_dice": validation["mean_dice_no_background"], "learning_rate": optimizer.param_groups[0]["lr"], "gpu_memory_mb": gpu_memory, "epoch_seconds": elapsed}
        for class_index, class_iou in enumerate(validation["iou_per_class"]):
            row[f"iou_class_{class_index}"] = class_iou
        if rank == 0:
            logger.info("epoch=%d train=%.5f val=%.5f mIoU(fg)=%.5f time=%.1fs gpu=%.0fMB", epoch, train_loss, validation["loss"], validation["mean_iou_no_background"], elapsed, gpu_memory)
            append_history(output / "logs" / "history.csv", row)
            (output / "logs" / "latest_metrics.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
            if writer:
                for key, value in row.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch)
            if wandb_run:
                wandb_run.log(row, step=epoch)
            save_checkpoint(output / "checkpoints" / "last.pt", model, optimizer, scheduler, scaler, epoch, max(best_metric, metric), config)
            if metric > best_metric:
                best_metric = metric
                no_improvement = 0
                save_checkpoint(output / "checkpoints" / "best.pt", model, optimizer, scheduler, scaler, epoch, best_metric, config)
            else:
                no_improvement += 1
        stop = torch.tensor(int(no_improvement >= int(settings["early_stopping_patience"])), device=device)
        if world_size > 1:
            torch.distributed.broadcast(stop, src=0)
        if stop.item():
            logger.info("조기 종료: %d epoch 동안 개선 없음", no_improvement)
            break
    if rank == 0:
        logger.info("학습 완료: %.1f초, best=%.6f", time.perf_counter() - training_started, best_metric)
        if writer:
            writer.close()
        if wandb_run:
            wandb_run.finish()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
