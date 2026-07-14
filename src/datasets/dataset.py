"""Paired GeoTIFF dataset with explicit raw-label remapping."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from .transforms import SegmentationTransform


@dataclass(frozen=True)
class RasterPair:
    """A source image and its class-mask raster."""

    image: Path
    mask: Path


def _collect_tiffs(root: Path, relative_dirs: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for relative in relative_dirs:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"데이터 디렉터리를 찾을 수 없습니다: {directory}")
        for path in sorted(directory.glob("*.tif")):
            if path.name in paths:
                raise ValueError(f"중복 파일명이 있습니다: {path.name}")
            paths[path.name] = path
    return paths


def discover_pairs(root: str | Path, split_config: dict[str, Any]) -> list[RasterPair]:
    """Pair images and masks by exact filename and reject mismatches."""
    root_path = Path(root)
    images = _collect_tiffs(root_path, list(split_config["image_dirs"]))
    masks = _collect_tiffs(root_path, list(split_config["mask_dirs"]))
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        detail = f"mask 누락={missing_masks[:5]}, image 누락={missing_images[:5]}"
        raise ValueError(f"영상과 마스크 파일명이 대응되지 않습니다: {detail}")
    if not images:
        raise ValueError("발견된 GeoTIFF 영상이 없습니다.")
    return [RasterPair(images[name], masks[name]) for name in sorted(images)]


def deterministic_partition(items: list[RasterPair], test_fraction: float, seed: int) -> tuple[list[RasterPair], list[RasterPair]]:
    """Split records without moving files."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("validation_test_fraction은 [0, 1) 범위여야 합니다.")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    count = round(len(shuffled) * test_fraction)
    return shuffled[count:], shuffled[:count]


class GeoTiffPairDataset(Dataset[dict[str, Any]]):
    """Read 512-sized windows, pad borders, remap labels, and normalize RGB."""

    def __init__(
        self,
        pairs: list[RasterPair],
        dataset_config: dict[str, Any],
        augmentation_config: dict[str, Any] | None = None,
        training: bool = False,
    ) -> None:
        self.pairs = pairs
        self.config = dataset_config
        self.training = training
        self.tile_size = int(dataset_config["tile_size"])
        self.channels = tuple(int(value) for value in dataset_config["channel_indices"])
        self.raw_class_map = {int(k): int(v) for k, v in dataset_config["raw_class_map"].items()}
        self.unmapped_value = int(dataset_config.get("unmapped_value", 0))
        self.mean = np.asarray(dataset_config["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(dataset_config["std"], dtype=np.float32).reshape(-1, 1, 1)
        if len(self.channels) != int(dataset_config["input_channels"]):
            raise ValueError("channel_indices 개수와 input_channels가 다릅니다.")
        self.transform = SegmentationTransform(augmentation_config or {}, training)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        with rasterio.open(pair.image) as image_source, rasterio.open(pair.mask) as mask_source:
            if image_source.shape != mask_source.shape:
                raise ValueError(f"영상/마스크 크기가 다릅니다: {pair.image} / {pair.mask}")
            if max(self.channels) > image_source.count:
                raise ValueError(f"입력 밴드가 부족합니다: {pair.image} (bands={image_source.count})")
            row, col = self._window_origin(image_source.height, image_source.width)
            window = Window(col, row, self.tile_size, self.tile_size)
            image = image_source.read(
                self.channels,
                window=window,
                boundless=True,
                fill_value=0,
                out_shape=(len(self.channels), self.tile_size, self.tile_size),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            raw_mask = mask_source.read(
                1,
                window=window,
                boundless=True,
                fill_value=self.unmapped_value,
                out_shape=(self.tile_size, self.tile_size),
                resampling=rasterio.enums.Resampling.nearest,
            )
        mask = np.full(raw_mask.shape, self.unmapped_value, dtype=np.int64)
        for raw_value, class_value in self.raw_class_map.items():
            mask[raw_mask == raw_value] = class_value
        image_hwc = np.moveaxis(image, 0, -1)
        image_hwc, mask = self.transform(image_hwc, mask)
        image_chw = np.moveaxis(image_hwc.astype(np.float32) / 255.0, -1, 0)
        image_chw = (image_chw - self.mean) / self.std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_chw)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "path": str(pair.image),
        }

    def _window_origin(self, height: int, width: int) -> tuple[int, int]:
        if not self.training:
            return max(0, (height - self.tile_size) // 2), max(0, (width - self.tile_size) // 2)
        max_row, max_col = max(0, height - self.tile_size), max(0, width - self.tile_size)
        return random.randint(0, max_row), random.randint(0, max_col)


def build_datasets(config: dict[str, Any]) -> tuple[GeoTiffPairDataset, GeoTiffPairDataset, GeoTiffPairDataset]:
    """Build train/validation/test datasets from the configured existing split."""
    dataset_config = config["dataset"]
    root = dataset_config["root_dir"]
    train_pairs = discover_pairs(root, dataset_config["train"])
    validation_pairs = discover_pairs(root, dataset_config["validation"])
    validation_pairs, test_pairs = deterministic_partition(
        validation_pairs,
        float(dataset_config.get("validation_test_fraction", 0.0)),
        int(config["project"]["seed"]),
    )
    train = GeoTiffPairDataset(train_pairs, dataset_config, config.get("augmentation"), training=True)
    validation = GeoTiffPairDataset(validation_pairs, dataset_config, training=False)
    test = GeoTiffPairDataset(test_pairs or validation_pairs, dataset_config, training=False)
    return train, validation, test

