"""Render tile samples as an RGB / ground-truth / optional-prediction grid PNG."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import rasterio
import torch

from .datasets.dataset import build_datasets
from .infer import close_parcel_boundaries
from .model import build_model
from .utils.checkpoint import load_checkpoint
from .utils.config import apply_overrides, load_config
from .utils.visualization import _title_tile, _write_rgb, blend_mask


def _read_context_window(root: Path, name: str, channels: tuple[int, ...], tile_size: int, margin: int) -> np.ndarray:
    """Read a tile plus ``margin`` pixels of real neighbor-tile context around it."""
    scene, row, col = name.rsplit("_", 2)
    # Missing neighbors (scene edges) stay zero, matching the no-data black border.
    mosaic = np.zeros((len(channels), tile_size * 3, tile_size * 3), dtype=np.uint8)
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            path = root / "images" / f"{scene}_{int(row) + delta_row:03d}_{int(col) + delta_col:03d}.tif"
            if not path.is_file():
                continue
            with rasterio.open(path) as source:
                mosaic[:, (delta_row + 1) * tile_size : (delta_row + 2) * tile_size, (delta_col + 1) * tile_size : (delta_col + 2) * tile_size] = source.read(channels)
    low, high = tile_size - margin, 2 * tile_size + margin
    return mosaic[:, low:high, low:high]


def main() -> None:
    parser = argparse.ArgumentParser(description="타일 데이터셋 샘플/예측 시각화")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="지정하면 모델 예측 열을 추가한다")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--context-margin", type=int, default=256, help="이웃 타일에서 가져올 추론 문맥 픽셀. 0이면 타일 단독 추론")
    parser.add_argument("--watershed", action=argparse.BooleanOptionalAction, default=True, help="경계 확률 watershed로 끊긴 필지 경계를 닫는다")
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
            device = next(model.parameters()).device
            margin = max(0, args.context_margin)
            if margin:
                window = _read_context_window(dataset.root, Path(sample["path"]).stem, dataset.channels, int(config["dataset"]["tile_size"]), margin)
                values = (window.astype(np.float32) / 255.0 - dataset.mean) / dataset.std
                inputs = torch.from_numpy(np.ascontiguousarray(values))[None].float()
            else:
                inputs = sample["image"][None]
            with torch.inference_mode():
                logits = model(inputs.to(device))
            if margin:
                logits = logits[..., margin:-margin, margin:-margin]
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
            prediction = close_parcel_boundaries(probabilities) if args.watershed else probabilities.argmax(axis=0)
            tiles.append(_title_tile(blend_mask(rgb, prediction), "Prediction"))
        rows.append(np.hstack(tiles))

    output = Path(args.output) if args.output else Path(config["project"]["output_dir"]) / "visualizations" / f"{args.split}_samples.png"
    _write_rgb(output, np.vstack(rows))
    print(output)


if __name__ == "__main__":
    main()
