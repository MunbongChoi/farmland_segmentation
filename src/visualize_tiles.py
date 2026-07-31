"""Render tile samples as an RGB / ground-truth / optional-prediction grid PNG."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import rasterio
import torch

from .datasets.dataset import build_datasets
from .model import build_model
from .utils.checkpoint import load_checkpoint
from .utils.config import apply_overrides, load_config
from .utils.visualization import _title_tile, _write_rgb, blend_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="타일 데이터셋 샘플/예측 시각화")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="지정하면 모델 예측 열을 추가한다")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    datasets = dict(zip(("train", "val", "test"), build_datasets(config)))
    dataset = datasets[args.split]
    indices = random.Random(int(config["project"]["seed"])).sample(range(len(dataset)), min(args.count, len(dataset)))

    model = None
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config).to(device).eval()
        load_checkpoint(args.checkpoint, model, current_config=config, map_location=device)

    channels = tuple(int(value) for value in config["dataset"]["channel_indices"][:3])
    rows = []
    for index in indices:
        sample = dataset[index]
        with rasterio.open(sample["path"]) as source:
            rgb = np.moveaxis(source.read(channels), 0, -1)
        mask = sample["mask"].numpy()
        tiles = [_title_tile(rgb, Path(sample["path"]).stem[-12:]), _title_tile(blend_mask(rgb, mask), "Ground truth")]
        if model is not None:
            with torch.inference_mode():
                logits = model(sample["image"][None].to(next(model.parameters()).device))
            prediction = logits.argmax(dim=1)[0].cpu().numpy()
            tiles.append(_title_tile(blend_mask(rgb, prediction), "Prediction"))
        rows.append(np.hstack(tiles))

    output = Path(args.output) if args.output else Path(config["project"]["output_dir"]) / "visualizations" / f"{args.split}_samples.png"
    _write_rgb(output, np.vstack(rows))
    print(output)


if __name__ == "__main__":
    main()
