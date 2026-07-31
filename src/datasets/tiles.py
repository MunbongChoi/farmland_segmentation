"""Pre-tiled image/label GeoTIFF pairs with manifest.csv train/val/test splits."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .transforms import SegmentationTransform


def read_manifest_splits(root: Path) -> dict[str, list[str]]:
    """Group tile IDs by the manifest ``split`` column."""
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.csv를 찾을 수 없습니다: {manifest}")
    splits: dict[str, list[str]] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            splits.setdefault(row["split"], []).append(row["tile"])
    return splits


class TileDataset(Dataset[dict[str, Any]]):
    """Read a pre-tiled multiband image TIF and its single-band mask TIF."""

    def __init__(
        self,
        root: Path,
        tiles: list[str],
        dataset_config: dict[str, Any],
        augmentation_config: dict[str, Any] | None = None,
        training: bool = False,
    ) -> None:
        self.root = root
        self.tiles = tiles
        self.channels = tuple(int(value) for value in dataset_config["channel_indices"])
        if len(self.channels) != int(dataset_config["input_channels"]):
            raise ValueError("channel_indices 개수와 input_channels가 다릅니다.")
        self.mean = np.asarray(dataset_config["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(dataset_config["std"], dtype=np.float32).reshape(-1, 1, 1)
        self.transform = SegmentationTransform(augmentation_config or {}, training)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.tiles[index]
        image_path = self.root / "images" / f"{name}.tif"
        label_path = self.root / "labels" / f"{name}.tif"
        with rasterio.open(image_path) as source:
            if max(self.channels) > source.count:
                raise ValueError(f"입력 밴드가 부족합니다: {image_path} (bands={source.count})")
            image = source.read(self.channels)
        with rasterio.open(label_path) as source:
            mask = source.read(1).astype(np.int64)
        if mask.shape != image.shape[1:]:
            raise ValueError(f"영상과 라벨 크기가 다릅니다: {image_path} / {label_path}")
        image_hwc, mask = self.transform(np.moveaxis(image, 0, -1), mask)
        image_chw = np.moveaxis(image_hwc.astype(np.float32) / 255.0, -1, 0)
        image_chw = (image_chw - self.mean) / self.std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_chw)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "path": str(image_path),
            "label_path": str(label_path),
        }


def build_tile_datasets(config: dict[str, Any]) -> tuple[TileDataset, TileDataset, TileDataset]:
    """Build train/val/test datasets from a pre-tiled dataset directory."""
    dataset_config = config["dataset"]
    root = Path(dataset_config["root_dir"]).expanduser()
    splits = read_manifest_splits(root)
    empty = [name for name in ("train", "val", "test") if not splits.get(name)]
    if empty:
        raise ValueError(f"manifest.csv에 비어 있는 split이 있습니다: {empty}")
    train = TileDataset(root, splits["train"], dataset_config, config.get("augmentation"), training=True)
    validation = TileDataset(root, splits["val"], dataset_config, training=False)
    test = TileDataset(root, splits["test"], dataset_config, training=False)
    return train, validation, test
