"""Sliding-window GeoTIFF inference, postprocessing, and vector export."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage

from .datasets.dataset import load_spatial_metadata
from .model import build_model
from .utils.checkpoint import load_checkpoint
from .utils.config import apply_overrides, load_config
from .utils.logger import setup_logger


def window_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return origins that cover every pixel exactly to the last border."""
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size > 0이고 0 <= overlap < tile_size여야 합니다.")
    if length <= tile_size:
        return [0]
    origins = list(range(0, length - tile_size + 1, tile_size - overlap))
    if origins[-1] != length - tile_size:
        origins.append(length - tile_size)
    return origins


def blending_weight(tile_size: int, method: str) -> np.ndarray:
    """Build uniform or edge-tapered overlap weights."""
    if method == "average":
        return np.ones((tile_size, tile_size), dtype=np.float32)
    if method != "hann":
        raise ValueError(f"지원하지 않는 병합 방식입니다: {method}")
    one_dimensional = np.hanning(tile_size).astype(np.float32)
    return np.maximum(np.outer(one_dimensional, one_dimensional), 1e-3)


def normalize_tile(tile: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize a channel-first uint8 tile."""
    return (tile.astype(np.float32) / 255.0 - mean) / std


def _infer_tiles(
    model: torch.nn.Module,
    tiles: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    auto_reduce_batch: bool,
) -> list[np.ndarray]:
    """Infer a tile list, reducing batch size after CUDA OOM."""
    outputs: list[np.ndarray] = []
    index = 0
    current_batch = max(1, batch_size)
    while index < len(tiles):
        chunk = tiles[index : index + current_batch]
        try:
            inputs = torch.from_numpy(np.stack(chunk)).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probabilities = model(inputs).softmax(1).float().cpu().numpy()
            outputs.extend(probabilities)
            index += len(chunk)
        except torch.OutOfMemoryError as error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not auto_reduce_batch or current_batch == 1:
                raise RuntimeError("CUDA OOM: inference.batch_size 또는 tile_size를 줄이세요.") from error
            current_batch = max(1, current_batch // 2)
            logging.getLogger(__name__).warning("CUDA OOM으로 추론 batch_size를 %d로 줄입니다.", current_batch)
    return outputs


def sliding_window_predict(
    read_tile: Callable[[int, int], np.ndarray],
    height: int,
    width: int,
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    tile_size: int,
    overlap: int,
    batch_size: int,
    mean: list[float],
    std: list[float],
    merge: str = "hann",
    auto_reduce_batch: bool = True,
) -> np.ndarray:
    """Predict a virtual raster and average all overlapping probabilities."""
    rows, columns = window_origins(height, tile_size, overlap), window_origins(width, tile_size, overlap)
    coordinates = [(row, col) for row in rows for col in columns]
    total = np.zeros((num_classes, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    weight = blending_weight(tile_size, merge)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    for start in range(0, len(coordinates), batch_size):
        chunk_coordinates = coordinates[start : start + batch_size]
        tiles = [normalize_tile(read_tile(row, col), mean_array, std_array) for row, col in chunk_coordinates]
        predictions = _infer_tiles(model, tiles, device, batch_size, auto_reduce_batch)
        for (row, col), prediction in zip(chunk_coordinates, predictions):
            valid_height = min(tile_size, height - row)
            valid_width = min(tile_size, width - col)
            local_weight = weight[:valid_height, :valid_width]
            total[:, row : row + valid_height, col : col + valid_width] += prediction[:, :valid_height, :valid_width] * local_weight
            weight_sum[row : row + valid_height, col : col + valid_width] += local_weight
    return total / np.maximum(weight_sum, 1e-6)[None]


def sliding_window_predict_array(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    tile_size: int,
    overlap: int,
    batch_size: int,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Array wrapper used by tests and non-raster callers (expects C,H,W)."""
    if image.ndim != 3:
        raise ValueError("image shape은 (channels, height, width)여야 합니다.")

    def reader(row: int, col: int) -> np.ndarray:
        tile = np.zeros((image.shape[0], tile_size, tile_size), dtype=image.dtype)
        crop = image[:, row : row + tile_size, col : col + tile_size]
        tile[:, : crop.shape[1], : crop.shape[2]] = crop
        return tile

    return sliding_window_predict(reader, image.shape[1], image.shape[2], model, device, num_classes, tile_size, overlap, batch_size, mean, std)


def _minimum_pixels(value: float, unit: str, transform_value: Affine, crs: CRS | None) -> int:
    if unit == "pixels":
        return max(0, round(value))
    if unit != "m2":
        raise ValueError("min_area_unit은 pixels 또는 m2여야 합니다.")
    if crs is None or not crs.is_projected:
        raise ValueError("m2 면적 필터에는 투영 CRS가 필요합니다. 참조 래스터를 지정하세요.")
    pixel_area = abs(transform_value.a * transform_value.e - transform_value.b * transform_value.d)
    if pixel_area <= 0:
        raise ValueError("유효하지 않은 GeoTIFF Transform입니다.")
    return max(0, round(value / pixel_area))


def postprocess_mask(mask: np.ndarray, config: dict[str, Any], transform_value: Affine, crs: CRS | None) -> tuple[np.ndarray, np.ndarray]:
    """Clean each class and derive per-object connected-component IDs."""
    processed = np.zeros(mask.shape, dtype=np.uint8)
    instances = np.zeros(mask.shape, dtype=np.uint32)
    next_instance = 1
    minimums = {int(key): float(value) for key, value in config.get("min_area", {}).items()}
    structure = ndimage.generate_binary_structure(2, 2)
    for class_id in sorted(int(value) for value in np.unique(mask) if value != 0):
        binary = mask == class_id
        if config.get("fill_holes", False):
            binary = ndimage.binary_fill_holes(binary)
        opening = int(config.get("opening_iterations", 0))
        closing = int(config.get("closing_iterations", 0))
        smoothing = int(config.get("boundary_smoothing_iterations", 0))
        erosion = int(config.get("boundary_erosion_pixels", 0))
        if opening:
            binary = ndimage.binary_opening(binary, structure, iterations=opening)
        if closing:
            binary = ndimage.binary_closing(binary, structure, iterations=closing)
        if smoothing:
            binary = ndimage.binary_closing(ndimage.binary_opening(binary, structure, iterations=smoothing), structure, iterations=smoothing)
        if erosion:
            binary = ndimage.binary_erosion(binary, structure, iterations=erosion)
        labels, count = ndimage.label(binary, structure)
        minimum = _minimum_pixels(minimums.get(class_id, 0.0), str(config.get("min_area_unit", "pixels")), transform_value, crs)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= minimum
        keep[0] = False
        binary = keep[labels]
        processed[binary] = class_id
        labels, count = ndimage.label(binary, structure)
        if count:
            nonzero = labels > 0
            instances[nonzero] = labels[nonzero].astype(np.uint32) + next_instance - 1
            next_instance += count
    return processed, instances


def write_raster(path: str | Path, array: np.ndarray, crs: CRS, transform_value: Affine, dtype: str, nodata: int | float | None = None) -> None:
    """Write a 2D or channel-first GeoTIFF preserving spatial metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = array[None] if array.ndim == 2 else array
    profile = {"driver": "GTiff", "height": values.shape[1], "width": values.shape[2], "count": values.shape[0], "dtype": dtype, "crs": crs, "transform": transform_value, "compress": "deflate", "nodata": nodata}
    with rasterio.open(destination, "w", **profile) as output:
        output.write(values.astype(dtype, copy=False))


def run_inference(
    config: dict[str, Any],
    checkpoint: str,
    input_path: str,
    output_mask: str,
    reference_meta: str | None = None,
    output_vector: str | None = None,
) -> None:
    """Run streaming-tile inference using existing Meta JSON georeferencing."""
    logger = logging.getLogger("infer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, current_config=config, map_location=device)
    model.eval()
    dataset, inference = config["dataset"], config["inference"]
    channels = tuple(int(value) for value in dataset["channel_indices"])
    with rasterio.open(input_path) as source:
        if max(channels) > source.count:
            raise ValueError(f"입력 채널 부족: required={channels}, actual={source.count}")
        if reference_meta:
            metadata = load_spatial_metadata(reference_meta)
            if (metadata.height, metadata.width) != source.shape:
                raise ValueError("reference Meta JSON과 입력 영상 크기가 다릅니다.")
            crs, transform_value = metadata.crs, metadata.transform
        else:
            crs, transform_value = source.crs, source.transform
        if bool(inference.get("require_georeference", True)) and (crs is None or transform_value.is_identity):
            raise ValueError("입력 GeoTIFF에 CRS/Transform이 없습니다. 기존 대응 _META.json을 --reference-meta로 지정하세요.")

        def reader(row: int, col: int) -> np.ndarray:
            return source.read(channels, window=Window(col, row, int(inference["tile_size"]), int(inference["tile_size"])), boundless=True, fill_value=0)

        probabilities = sliding_window_predict(reader, source.height, source.width, model, device, int(dataset["num_classes"]), int(inference["tile_size"]), int(inference["overlap"]), int(inference["batch_size"]), list(dataset["mean"]), list(dataset["std"]), str(inference.get("merge", "hann")), bool(inference.get("auto_reduce_batch", True)))
    if crs is None:
        raise ValueError("출력 공간정보가 없습니다.")
    raw_mask = probabilities.argmax(0).astype(np.uint8)
    confidence_threshold = float(inference.get("confidence_threshold", 0.0))
    foreground = raw_mask > 0
    raw_mask[foreground & (probabilities.max(0) < confidence_threshold)] = 0
    output_path = Path(output_mask)
    raw_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}")
    write_raster(raw_path, raw_mask, crs, transform_value, "uint8", 0)
    post_config = config.get("postprocess", {})
    if post_config.get("enabled", True):
        processed, instances = postprocess_mask(raw_mask, post_config, transform_value, crs)
    else:
        processed, instances = raw_mask, np.zeros(raw_mask.shape, dtype=np.uint32)
    write_raster(output_path, processed, crs, transform_value, "uint8", 0)
    if config.get("output", {}).get("save_probability_map", True):
        write_raster(output_path.with_name(f"{output_path.stem}_probability.tif"), probabilities, crs, transform_value, "float32")
    instance_path = output_path.with_name(f"{output_path.stem}_instances.tif")
    if post_config.get("save_instances", True) or output_vector:
        write_raster(instance_path, instances, crs, transform_value, "uint32", 0)
    if output_vector:
        try:
            subprocess.run(
                [sys.executable, "-m", "src.vectorize", "--instances", str(instance_path), "--classes", str(output_path), "--output", str(output_vector)],
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError("벡터 변환에 실패했습니다. GeoPandas/pyogrio 설치와 출력 확장자를 확인하세요.") from error
    logger.info("추론 완료: raw=%s postprocessed=%s", raw_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="논/밭 GeoTIFF 추론")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-mask", required=True)
    parser.add_argument("--output-vector")
    parser.add_argument("--reference-meta", help="입력에 공간정보가 없을 때 대응하는 기존 _META.json")
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    for key, value in (("tile_size", args.tile_size), ("overlap", args.overlap), ("batch_size", args.batch_size)):
        if value is not None:
            config["inference"][key] = value
    setup_logger("infer", Path(config["project"]["output_dir"]) / "logs" / "infer.log")
    run_inference(config, args.checkpoint, args.input, args.output_mask, args.reference_meta, args.output_vector)


if __name__ == "__main__":
    main()
