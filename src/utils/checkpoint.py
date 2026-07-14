"""Training checkpoint serialization and compatibility checks."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn


def git_commit_hash() -> str:
    """Return the current commit or ``unknown`` outside a committed repository."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the wrapped module for DDP/DataParallel models."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    """Atomically save all state required to resume training."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "model_config": config["model"],
        "dataset_config": config["dataset"],
        "class_names": config["dataset"]["class_names"],
        "git_commit": git_commit_hash(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    current_config: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and warn when model/data contracts changed."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {source}")
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    if current_config:
        for section in ("model", "dataset"):
            saved = checkpoint.get(f"{section}_config")
            if saved is not None and saved != current_config.get(section):
                logging.getLogger(__name__).warning("체크포인트의 %s 설정이 현재 설정과 다릅니다.", section)
    try:
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise RuntimeError(f"체크포인트와 모델 구조가 호환되지 않습니다: {error}") from error
    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint

