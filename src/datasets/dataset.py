"""RGB GeoTIFF images paired directly with existing polygon JSON labels."""

from __future__ import annotations

import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.validation import make_valid
from torch.utils.data import Dataset

from .transforms import SegmentationTransform


@dataclass(frozen=True)
class SpatialMetadata:
    """Pixel-grid definition read from an existing ``_META.json`` file."""

    width: int
    height: int
    crs: CRS
    transform: Affine
    resolution: float


@dataclass(frozen=True)
class RasterPair:
    """A source image, polygon-label JSON, and spatial metadata JSON."""

    image: Path
    label_json: Path
    meta_json: Path


def _collect(root: Path, relative_dirs: list[str], pattern: str, key: Any) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for relative in relative_dirs:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"데이터 디렉터리를 찾을 수 없습니다: {directory}")
        for path in sorted(directory.glob(pattern)):
            item_key = str(key(path))
            if item_key in paths:
                raise ValueError(f"중복 데이터 ID가 있습니다: {item_key}")
            paths[item_key] = path
    return paths


def _has_target_feature(path: Path, target_codes: set[int], property_name: str) -> bool:
    """Quickly reject JSON files without paddy/field features before indexing."""
    escaped = re.escape(property_name.encode("ascii"))
    codes = b"|".join(str(code).encode("ascii") for code in sorted(target_codes))
    pattern = re.compile(rb'"' + escaped + rb'"\s*:\s*(?:' + codes + rb')\b')
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            payload = tail + chunk
            if pattern.search(payload) is not None:
                return True
            tail = payload[-128:]
    return False


def _index_signature(root: Path, split_config: dict[str, Any], images: dict[str, Path], labels: dict[str, Path], metadata: dict[str, Path], target_codes: set[int], property_name: str) -> dict[str, Any]:
    """Describe the immutable inputs used to decide cached target IDs."""
    directories = [root / item for key in ("image_dirs", "label_dirs", "meta_dirs") for item in split_config[key]]
    return {
        "root": str(root.resolve()),
        "directories": {str(path.resolve()): path.stat().st_mtime_ns for path in directories},
        "counts": {"images": len(images), "labels": len(labels), "metadata": len(metadata)},
        "target_codes": sorted(target_codes),
        "target_property": property_name,
    }


def _target_ids(labels: dict[str, Path], target_codes: set[int], property_name: str, cache_path: str | Path | None, signature: dict[str, Any]) -> list[str]:
    """Load or concurrently create the list of JSON files containing targets."""
    cache = Path(cache_path) if cache_path else None
    if cache and cache.is_file():
        try:
            cached = read_json(cache)
            if cached.get("signature") == signature:
                return [item_id for item_id in cached["included_ids"] if item_id in labels]
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            logging.getLogger(__name__).warning("JSON target cache가 손상되어 다시 생성합니다: %s", cache)
    workers = min(16, max(1, (os.cpu_count() or 4)))
    items = list(sorted(labels.items()))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        flags = executor.map(lambda item: _has_target_feature(item[1], target_codes, property_name), items)
        included = [item_id for (item_id, _), has_target in zip(items, flags) if has_target]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(cache.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"signature": signature, "included_ids": included}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache)
    return included


def discover_pairs(
    root: str | Path,
    split_config: dict[str, Any],
    raw_class_map: dict[int, int] | None = None,
    target_property: str = "ANN_CD",
    cache_path: str | Path | None = None,
) -> list[RasterPair]:
    """Match existing image/label/meta files and omit JSONs without target classes."""
    root_path = Path(root)
    images = _collect(root_path, list(split_config["image_dirs"]), "*.tif", lambda path: path.stem)
    labels = _collect(root_path, list(split_config["label_dirs"]), "*.json", lambda path: path.stem)
    metadata = _collect(root_path, list(split_config["meta_dirs"]), "*_META.json", lambda path: path.stem.removesuffix("_META"))
    missing_labels = sorted(images.keys() - labels.keys())
    missing_metadata = sorted(images.keys() - metadata.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_metadata or missing_images:
        detail = f"label 누락={missing_labels[:5]}, meta 누락={missing_metadata[:5]}, image 누락={missing_images[:5]}"
        raise ValueError(f"영상/JSON/Meta 파일명이 대응되지 않습니다: {detail}")
    if not images:
        raise ValueError("발견된 GeoTIFF 영상이 없습니다.")
    target_codes = set(int(code) for code in (raw_class_map or {50: 1, 60: 2}))
    signature = _index_signature(root_path, split_config, images, labels, metadata, target_codes, target_property)
    included = _target_ids(labels, target_codes, target_property, cache_path or split_config.get("index_cache"), signature)
    logging.getLogger(__name__).info("JSON target 필터: 전체=%d 사용=%d 제외=%d", len(images), len(included), len(images) - len(included))
    if not included:
        raise ValueError("논/밭 feature가 포함된 JSON 라벨이 없습니다.")
    return [RasterPair(images[item_id], labels[item_id], metadata[item_id]) for item_id in included]


def read_json(path: str | Path) -> Any:
    """Read dataset JSON, accepting its UTF-8 or CP949 encodings."""
    payload = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return json.loads(payload.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8/cp949", payload, 0, len(payload), f"JSON 인코딩을 읽을 수 없습니다: {path}")


def load_spatial_metadata(path: str | Path) -> SpatialMetadata:
    """Build an upper-left pixel-corner transform from center coordinates in Meta JSON."""
    data = read_json(path)
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError(f"Meta JSON 구조가 올바르지 않습니다: {path}")
    item = data[0]
    width, height = int(item["img_width"]), int(item["img_height"])
    resolution = float(item["img_resolution"])
    if resolution <= 0:
        raise ValueError(f"해상도는 0보다 커야 합니다: {path}")
    coordinates = [float(value.strip()) for value in str(item["coordinates"]).split(",")]
    if len(coordinates) != 2:
        raise ValueError(f"coordinates는 'x, y' 형식이어야 합니다: {path}")
    crs = CRS.from_string(str(item["img_coordinate"]))
    # Meta coordinates denote the upper-left pixel center; raster transforms use its corner.
    transform_value = from_origin(coordinates[0] - resolution / 2, coordinates[1] + resolution / 2, resolution, resolution)
    return SpatialMetadata(width, height, crs, transform_value, resolution)


def load_label_mask(pair: RasterPair, config: dict[str, Any]) -> tuple[np.ndarray, SpatialMetadata, dict[int, int]]:
    """Rasterize only existing ANN_CD 50/60 polygons to background/paddy/field classes."""
    metadata = load_spatial_metadata(pair.meta_json)
    collection = read_json(pair.label_json)
    property_name = str(config.get("target_property", "ANN_CD"))
    class_map = {int(key): int(value) for key, value in config["raw_class_map"].items()}
    geometries: list[tuple[dict[str, Any], int]] = []
    feature_counts = {code: 0 for code in class_map}
    json_crs_name = collection.get("crs", {}).get("properties", {}).get("name")
    if json_crs_name and CRS.from_string(str(json_crs_name)) != metadata.crs:
        raise ValueError(f"Label JSON과 Meta JSON의 CRS가 다릅니다: {pair.label_json}")
    for feature in collection.get("features", []):
        properties = feature.get("properties") or {}
        raw_code = int(properties.get(property_name, -1))
        if raw_code not in class_map or not feature.get("geometry"):
            continue
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            continue
        geometries.append((mapping(geometry), class_map[raw_code]))
        feature_counts[raw_code] += 1
    if not geometries:
        raise ValueError(f"논/밭 polygon이 없는 JSON은 사용할 수 없습니다: {pair.label_json}")
    mask = rasterize(
        geometries,
        out_shape=(metadata.height, metadata.width),
        transform=metadata.transform,
        fill=int(config.get("unmapped_value", 0)),
        all_touched=bool(config.get("geometry_all_touched", False)),
        dtype="uint8",
    )
    return mask, metadata, feature_counts


def deterministic_partition(items: list[RasterPair], test_fraction: float, seed: int) -> tuple[list[RasterPair], list[RasterPair]]:
    """Split records without moving files."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("validation_test_fraction은 [0, 1) 범위여야 합니다.")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    count = round(len(shuffled) * test_fraction)
    return shuffled[count:], shuffled[:count]


class GeoTiffPairDataset(Dataset[dict[str, Any]]):
    """Read an RGB window and rasterize its existing polygon JSON label."""

    def __init__(self, pairs: list[RasterPair], dataset_config: dict[str, Any], augmentation_config: dict[str, Any] | None = None, training: bool = False) -> None:
        self.pairs = pairs
        self.config = dataset_config
        self.training = training
        self.tile_size = int(dataset_config["tile_size"])
        self.channels = tuple(int(value) for value in dataset_config["channel_indices"])
        self.mean = np.asarray(dataset_config["mean"], dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(dataset_config["std"], dtype=np.float32).reshape(-1, 1, 1)
        if len(self.channels) != int(dataset_config["input_channels"]):
            raise ValueError("channel_indices 개수와 input_channels가 다릅니다.")
        self.transform = SegmentationTransform(augmentation_config or {}, training)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        mask, metadata, _ = load_label_mask(pair, self.config)
        with rasterio.open(pair.image) as image_source:
            if image_source.shape != (metadata.height, metadata.width):
                raise ValueError(f"영상과 Meta JSON 크기가 다릅니다: {pair.image} / {pair.meta_json}")
            if max(self.channels) > image_source.count:
                raise ValueError(f"입력 밴드가 부족합니다: {pair.image} (bands={image_source.count})")
            row, col = self._window_origin(image_source.height, image_source.width)
            window = Window(col, row, self.tile_size, self.tile_size)
            image = image_source.read(self.channels, window=window, boundless=True, fill_value=0)
        mask_tile = np.full((self.tile_size, self.tile_size), int(self.config.get("unmapped_value", 0)), dtype=np.int64)
        crop = mask[row : row + self.tile_size, col : col + self.tile_size]
        mask_tile[: crop.shape[0], : crop.shape[1]] = crop
        image_hwc = np.moveaxis(image, 0, -1)
        image_hwc, mask_tile = self.transform(image_hwc, mask_tile)
        image_chw = np.moveaxis(image_hwc.astype(np.float32) / 255.0, -1, 0)
        image_chw = (image_chw - self.mean) / self.std
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_chw)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_tile)).long(),
            "path": str(pair.image),
            "label_path": str(pair.label_json),
        }

    def _window_origin(self, height: int, width: int) -> tuple[int, int]:
        if not self.training:
            return max(0, (height - self.tile_size) // 2), max(0, (width - self.tile_size) // 2)
        return random.randint(0, max(0, height - self.tile_size)), random.randint(0, max(0, width - self.tile_size))


def build_datasets(config: dict[str, Any]) -> tuple[GeoTiffPairDataset, GeoTiffPairDataset, GeoTiffPairDataset]:
    """Build filtered train/validation/test datasets from existing JSON labels."""
    dataset_config = config["dataset"]
    root = dataset_config["root_dir"]
    class_map = {int(key): int(value) for key, value in dataset_config["raw_class_map"].items()}
    property_name = str(dataset_config.get("target_property", "ANN_CD"))
    train_pairs = discover_pairs(root, dataset_config["train"], class_map, property_name, dataset_config["train"].get("index_cache"))
    validation_pairs = discover_pairs(root, dataset_config["validation"], class_map, property_name, dataset_config["validation"].get("index_cache"))
    validation_pairs, test_pairs = deterministic_partition(validation_pairs, float(dataset_config.get("validation_test_fraction", 0.0)), int(config["project"]["seed"]))
    train = GeoTiffPairDataset(train_pairs, dataset_config, config.get("augmentation"), training=True)
    validation = GeoTiffPairDataset(validation_pairs, dataset_config, training=False)
    test = GeoTiffPairDataset(test_pairs or validation_pairs, dataset_config, training=False)
    return train, validation, test
