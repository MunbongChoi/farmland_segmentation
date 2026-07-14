"""Compute channel statistics and optionally materialize georeferenced tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window, transform

from .datasets.dataset import RasterPair, discover_pairs, load_label_mask
from .utils.config import apply_overrides, load_config
from .utils.logger import setup_logger


def compute_channel_stats(pairs: list[RasterPair], channels: tuple[int, ...]) -> tuple[list[float], list[float]]:
    """Compute exact per-channel mean/std in [0, 1] using streaming sums."""
    sums = np.zeros(len(channels), dtype=np.float64)
    squares = np.zeros(len(channels), dtype=np.float64)
    pixels = 0
    for pair in pairs:
        with rasterio.open(pair.image) as source:
            values = source.read(channels).astype(np.float64) / 255.0
        sums += values.sum(axis=(1, 2))
        squares += np.square(values).sum(axis=(1, 2))
        pixels += values.shape[1] * values.shape[2]
    mean = sums / max(1, pixels)
    variance = np.maximum(0.0, squares / max(1, pixels) - np.square(mean))
    return mean.tolist(), np.sqrt(variance).tolist()


def _origins(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    values = list(range(0, size - tile_size + 1, stride))
    if values[-1] != size - tile_size:
        values.append(size - tile_size)
    return values


def tile_pair(
    pair: RasterPair,
    output: Path,
    channels: tuple[int, ...],
    tile_size: int,
    overlap: int,
    max_background_fraction: float,
    dataset_config: dict[str, Any],
) -> int:
    """Write tiles using CRS/transform reconstructed from the existing Meta JSON."""
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap은 0 이상 tile_size 미만이어야 합니다.")
    written = 0
    label_mask, metadata, _ = load_label_mask(pair, dataset_config)
    with rasterio.open(pair.image) as image:
        if image.shape != (metadata.height, metadata.width):
            raise ValueError(f"크기 불일치: {pair.image} / {pair.meta_json}")
        for row in _origins(image.height, tile_size, tile_size - overlap):
            for col in _origins(image.width, tile_size, tile_size - overlap):
                window = Window(col, row, tile_size, tile_size)
                mask_tile = label_mask[row : row + tile_size, col : col + tile_size]
                if mask_tile.shape != (tile_size, tile_size):
                    padded = np.zeros((tile_size, tile_size), dtype=mask_tile.dtype)
                    padded[: mask_tile.shape[0], : mask_tile.shape[1]] = mask_tile
                    mask_tile = padded
                foreground_fraction = (mask_tile > 0).mean()
                if 1.0 - foreground_fraction > max_background_fraction:
                    continue
                image_tile = image.read(channels, window=window, boundless=True, fill_value=0)
                stem = f"{pair.image.stem}_r{row:05d}_c{col:05d}"
                image_profile = image.profile.copy()
                image_profile.update(driver="GTiff", width=tile_size, height=tile_size, count=len(channels), crs=metadata.crs, transform=transform(window, metadata.transform))
                mask_profile = {"driver": "GTiff", "dtype": "uint8", "width": tile_size, "height": tile_size, "count": 1, "crs": metadata.crs, "transform": transform(window, metadata.transform), "compress": "deflate"}
                (output / "images").mkdir(parents=True, exist_ok=True)
                (output / "masks").mkdir(parents=True, exist_ok=True)
                with rasterio.open(output / "images" / f"{stem}.tif", "w", **image_profile) as destination:
                    destination.write(image_tile)
                with rasterio.open(output / "masks" / f"{stem}.tif", "w", **mask_profile) as destination:
                    destination.write(mask_tile, 1)
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="통계 계산 및 GeoTIFF 타일 생성")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--stats-output", default="outputs/data_stats.json")
    parser.add_argument("--tile-output")
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-background-fraction", type=float, default=1.0)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    dataset = config["dataset"]
    logger = setup_logger("prepare_data")
    pairs = discover_pairs(dataset["root_dir"], dataset["train"], dataset["raw_class_map"], dataset.get("target_property", "ANN_CD"))
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    channels = tuple(int(value) for value in dataset["channel_indices"])
    mean, std = compute_channel_stats(pairs, channels)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({"samples": len(pairs), "mean": mean, "std": std}, indent=2), encoding="utf-8")
    logger.info("mean=%s std=%s", mean, std)
    if args.tile_output:
        count = sum(tile_pair(pair, Path(args.tile_output), channels, int(dataset["tile_size"]), args.overlap, args.max_background_fraction, dataset) for pair in pairs)
        logger.info("타일 %d개 저장: %s", count, args.tile_output)


if __name__ == "__main__":
    main()
