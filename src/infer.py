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
from .resolution import open_resampled_vrt
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
    processed = 0
    for start in range(0, len(coordinates), batch_size):
        chunk_coordinates = coordinates[start : start + batch_size]
        # 장면 밖 전부-검정 창은 추론 없이 배경(가중치 0)으로 남긴다.
        kept = [(coordinate, tile) for coordinate in chunk_coordinates if (tile := read_tile(*coordinate)).any()]
        processed += len(chunk_coordinates)
        if not kept:
            continue
        tiles = [normalize_tile(tile, mean_array, std_array) for _, tile in kept]
        predictions = _infer_tiles(model, tiles, device, batch_size, auto_reduce_batch)
        for ((row, col), _), prediction in zip(kept, predictions):
            valid_height = min(tile_size, height - row)
            valid_width = min(tile_size, width - col)
            local_weight = weight[:valid_height, :valid_width]
            total[:, row : row + valid_height, col : col + valid_width] += prediction[:, :valid_height, :valid_width] * local_weight
            weight_sum[row : row + valid_height, col : col + valid_width] += local_weight
        if processed % (batch_size * 20) < batch_size:
            print(f"  추론 {processed}/{len(coordinates)} 창", flush=True)
    return total / np.maximum(weight_sum, 1e-6)[None]


def sliding_window_predict_compact(
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
    boundary_class: int,
    merge: str = "hann",
    auto_reduce_batch: bool = True,
    band_rows: int = 3072,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Streaming Hann-blended prediction keeping only postprocess-relevant maps.

    Returns ``(argmax uint8, max-probability uint8(x255), interior-sum float16,
    boundary float16)``. Identical blending to :func:`sliding_window_predict`;
    bands only bound RAM (~C x band x W instead of C x H x W), and windows
    spanning band edges are recomputed so every core pixel still receives all
    of its overlapping-window contributions — no seams.
    """
    row_origins = window_origins(height, tile_size, overlap)
    column_origins = window_origins(width, tile_size, overlap)
    argmax_map = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.uint8)
    interior_map = np.zeros((height, width), dtype=np.float16)
    boundary_map = np.zeros((height, width), dtype=np.float16)
    interior_indices = [index for index in range(1, num_classes) if index != boundary_class]
    weight = blending_weight(tile_size, merge)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    band_count = (height + band_rows - 1) // band_rows
    for band_index, band_start in enumerate(range(0, height, band_rows), start=1):
        band_end = min(height, band_start + band_rows)
        band_row_origins = [row for row in row_origins if row + tile_size > band_start and row < band_end]
        buffer_top = min(band_row_origins)
        buffer_height = max(band_row_origins) + tile_size - buffer_top
        total = np.zeros((num_classes, buffer_height, width), dtype=np.float32)
        weight_sum = np.zeros((buffer_height, width), dtype=np.float32)
        coordinates = [(row, col) for row in band_row_origins for col in column_origins]
        skipped = 0
        for start in range(0, len(coordinates), batch_size):
            chunk_coordinates = coordinates[start : start + batch_size]
            # 장면 밖 전부-검정 창은 추론 없이 배경(가중치 0)으로 남긴다.
            kept = [(coordinate, tile) for coordinate in chunk_coordinates if (tile := read_tile(*coordinate)).any()]
            skipped += len(chunk_coordinates) - len(kept)
            if not kept:
                continue
            tiles = [normalize_tile(tile, mean_array, std_array) for _, tile in kept]
            predictions = _infer_tiles(model, tiles, device, batch_size, auto_reduce_batch)
            for ((row, col), _), prediction in zip(kept, predictions):
                valid_height = min(tile_size, height - row)
                valid_width = min(tile_size, width - col)
                local_weight = weight[:valid_height, :valid_width]
                local_row = row - buffer_top
                total[:, local_row : local_row + valid_height, col : col + valid_width] += prediction[:, :valid_height, :valid_width] * local_weight
                weight_sum[local_row : local_row + valid_height, col : col + valid_width] += local_weight
        print(f"  밴드 {band_index}/{band_count} (행 {band_start}-{band_end}): 창 {len(coordinates)}개, 검정 스킵 {skipped}개", flush=True)
        # 밴드 core 행만 축약해 저장한다. 512행 단위로 나눠 임시 확률 복사본을 작게 유지한다.
        for reduce_start in range(band_start, band_end, 512):
            reduce_end = min(band_end, reduce_start + 512)
            local = slice(reduce_start - buffer_top, reduce_end - buffer_top)
            probabilities = total[:, local] / np.maximum(weight_sum[local], 1e-6)[None]
            target = slice(reduce_start, reduce_end)
            argmax_map[target] = probabilities.argmax(axis=0).astype(np.uint8)
            confidence_map[target] = np.round(probabilities.max(axis=0) * 255.0).astype(np.uint8)
            interior_map[target] = probabilities[interior_indices].sum(axis=0).astype(np.float16)
            boundary_map[target] = probabilities[boundary_class].astype(np.float16)
        total = weight_sum = None
    return argmax_map, confidence_map, interior_map, boundary_map


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


def close_parcel_boundaries(
    probabilities: np.ndarray,
    boundary_class: int = 2,
    seed_threshold: float = 0.5,
    seed_boundary_maximum: float = 0.15,
    line_iterations: int = 1,
    seed_erosion_iterations: int = 0,
) -> np.ndarray:
    """Bridge argmax boundary gaps with watershed lines over the boundary probability.

    Argmax drops faint boundary ridges, leaving open parcel outlines. Seeding
    confident interior regions and flooding the boundary-probability surface
    draws a closed dividing line wherever two parcels meet, even through gaps.
    Every non-background class except ``boundary_class`` counts as interior, so
    the same routine serves both the 3-class and crop-class label schemes.
    """
    interior_indices = [index for index in range(1, probabilities.shape[0]) if index != boundary_class]
    return close_parcel_boundaries_from_maps(
        probabilities.argmax(axis=0).astype(np.uint8),
        probabilities[interior_indices].sum(axis=0),
        probabilities[boundary_class],
        boundary_class,
        seed_threshold,
        seed_boundary_maximum,
        line_iterations,
        seed_erosion_iterations,
    )


def close_parcel_boundaries_from_maps(
    mask: np.ndarray,
    interior_probability: np.ndarray,
    boundary_probability: np.ndarray,
    boundary_class: int = 2,
    seed_threshold: float = 0.5,
    seed_boundary_maximum: float = 0.15,
    line_iterations: int = 1,
    seed_erosion_iterations: int = 0,
) -> np.ndarray:
    """Watershed closing from precomputed maps (streaming path needs no full C,H,W array)."""
    try:
        from skimage.segmentation import watershed
    except ImportError as error:
        raise RuntimeError("watershed 후처리에는 scikit-image가 필요합니다: pip install scikit-image") from error
    parcel = mask > 0
    interior = parcel & (mask != boundary_class)
    structure = ndimage.generate_binary_structure(2, 2)
    seeds = (interior_probability > seed_threshold) & (boundary_probability < seed_boundary_maximum) & parcel
    if seed_erosion_iterations:
        # 약한 두렁 위로 이어진 얇은 씨앗 다리를 끊어 인접 필지 병합을 억제한다.
        seeds = ndimage.binary_erosion(seeds, structure, iterations=seed_erosion_iterations)
    # Every argmax-interior component must own a seed, or it would flood as boundary.
    interior_labels, count = ndimage.label(interior, structure)
    seeded = np.zeros(count + 1, dtype=bool)
    seeded[interior_labels[seeds]] = True
    seeds |= (interior_labels > 0) & ~seeded[interior_labels]
    markers, marker_count = ndimage.label(seeds, structure)
    if not marker_count:
        return mask
    basins = watershed(boundary_probability.astype(np.float32, copy=False), markers, mask=parcel, watershed_line=True)
    lines = parcel & (basins == 0)
    if line_iterations:
        # 1px watershed lines would be re-bridged by the postprocess 3x3 closing.
        lines = ndimage.binary_dilation(lines, structure, iterations=line_iterations) & parcel
    closed = mask.copy()
    # 두껍게 예측된 argmax 경계 띠를 각 basin의 다수 내부 클래스로 되돌린다.
    # 경계는 아래에서 다시 그리는 분수령 선만 남아 폭이 균일해지고 필지 면적이 보존된다.
    band = parcel & (mask == boundary_class) & (basins > 0)
    if band.any():
        num_classes = int(mask.max()) + 1
        encoded = basins[interior].astype(np.int64) * num_classes + mask[interior]
        counts = np.bincount(encoded, minlength=(int(basins.max()) + 1) * num_classes).reshape(-1, num_classes)
        basin_class = counts.argmax(axis=1).astype(np.uint8)
        basin_class[counts.max(axis=1) == 0] = boundary_class
        closed[band] = basin_class[basins[band]]
    closed[lines] = boundary_class
    return closed


def postprocess_mask(mask: np.ndarray, config: dict[str, Any], transform_value: Affine, crs: CRS | None, boundary_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Clean each class and derive per-object connected-component IDs."""
    processed = np.zeros(mask.shape, dtype=np.uint8)
    instances = np.zeros(mask.shape, dtype=np.uint32)
    next_instance = 1
    minimums = {int(key): float(value) for key, value in config.get("min_area", {}).items()}
    structure = ndimage.generate_binary_structure(2, 2)
    for class_id in sorted(int(value) for value in np.unique(mask) if value != 0):
        binary = mask == class_id
        # 경계 클래스는 닫힌 고리라서 fill_holes가 필지 내부 전체를 경계로 메워버린다.
        if config.get("fill_holes", False) and class_id != boundary_class:
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


# 클래스 표시색 — visualization.DEFAULT_PALETTE와 동일하게 유지한다.
CLASS_COLORMAP = {
    0: (0, 0, 0, 0),
    1: (60, 180, 75, 255),    # 논 (tiles: 필지 내부)
    2: (255, 165, 0, 255),    # 밭 (tiles: 필지 경계)
    3: (150, 80, 200, 255),   # 과수
    4: (70, 140, 230, 255),   # 시설
    5: (235, 110, 180, 255),  # 인삼
    6: (150, 150, 150, 255),  # 비경지
    7: (230, 50, 50, 255),    # 필지 경계
}


def write_raster(path: str | Path, array: np.ndarray, crs: CRS, transform_value: Affine, dtype: str, nodata: int | float | None = None, colormap: dict[int, tuple[int, int, int, int]] | None = None) -> None:
    """Write a 2D or channel-first GeoTIFF preserving spatial metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = array[None] if array.ndim == 2 else array
    profile = {"driver": "GTiff", "height": values.shape[1], "width": values.shape[2], "count": values.shape[0], "dtype": dtype, "crs": crs, "transform": transform_value, "compress": "deflate", "nodata": nodata, "bigtiff": "if_safer"}
    with rasterio.open(destination, "w", **profile) as output:
        output.write(values.astype(dtype, copy=False))
        if colormap and values.shape[0] == 1:
            output.write_colormap(1, colormap)


def load_inference_model(
    config: dict[str, Any],
    checkpoint: str,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """Load one checkpoint for reuse across one or many inference rasters."""
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(selected_device)
    load_checkpoint(checkpoint, model, current_config=config, map_location=selected_device)
    model.eval()
    return model, selected_device


def run_inference(
    config: dict[str, Any],
    checkpoint: str,
    input_path: str,
    output_mask: str,
    reference_meta: str | None = None,
    output_vector: str | None = None,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> None:
    """Run streaming-tile inference using existing Meta JSON georeferencing."""
    logger = logging.getLogger("infer")
    if model is None:
        model, device = load_inference_model(config, checkpoint, device)
    elif device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
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

        if crs is None:
            raise ValueError("25cm 해상도 변환에는 입력 CRS가 필요합니다.")
        target_resolution_m = inference.get("target_resolution_m")
        if target_resolution_m is not None:
            inference_source, grid = open_resampled_vrt(
                source,
                crs,
                transform_value,
                float(target_resolution_m),
                inference.get("target_crs"),
                str(inference.get("resampling", "bilinear")),
            )
            crs, transform_value = grid.crs, grid.transform
            logger.info(
                "추론 격자 변환: source=%dx%d target=%dx%d resolution=%.3fm crs=%s",
                source.width,
                source.height,
                grid.width,
                grid.height,
                float(target_resolution_m),
                grid.crs,
            )
        else:
            inference_source = source

        try:
            max_pixels = int(inference.get("max_resampled_pixels", 0))
            output_pixels = inference_source.width * inference_source.height
            if max_pixels > 0 and output_pixels > max_pixels:
                raise ValueError(
                    f"25cm 변환 결과가 너무 큽니다: {inference_source.width}x{inference_source.height}="
                    f"{output_pixels:,} pixels > max_resampled_pixels={max_pixels:,}. "
                    "입력을 공간 타일로 나누거나 inference.max_resampled_pixels를 조정하세요."
                )

            def reader(row: int, col: int) -> np.ndarray:
                tile_size = int(inference["tile_size"])
                window_width = min(tile_size, inference_source.width - col)
                window_height = min(tile_size, inference_source.height - row)
                crop = inference_source.read(channels, window=Window(col, row, window_width, window_height))
                if (window_height, window_width) == (tile_size, tile_size):
                    return crop
                tile = np.zeros((len(channels), tile_size, tile_size), dtype=crop.dtype)
                tile[:, :window_height, :window_width] = crop
                return tile

            save_probability = bool(config.get("output", {}).get("save_probability_map", True))
            boundary_class = int(inference.get("boundary_class", 2))
            common = (inference_source.height, inference_source.width, model, device, int(dataset["num_classes"]), int(inference["tile_size"]), int(inference["overlap"]), int(inference["batch_size"]), list(dataset["mean"]), list(dataset["std"]))
            if save_probability:
                # 확률맵 저장이 필요할 때만 전체 C,H,W float32를 유지한다 (~4B*C*H*W RAM).
                probabilities = sliding_window_predict(reader, *common, str(inference.get("merge", "hann")), bool(inference.get("auto_reduce_batch", True)))
                argmax_map = probabilities.argmax(0).astype(np.uint8)
                confidence_ok = probabilities.max(0) >= float(inference.get("confidence_threshold", 0.0))
                interior_map = boundary_map = None
            else:
                probabilities = None
                argmax_map, confidence_map, interior_map, boundary_map = sliding_window_predict_compact(reader, *common, boundary_class, str(inference.get("merge", "hann")), bool(inference.get("auto_reduce_batch", True)))
                confidence_ok = confidence_map >= round(float(inference.get("confidence_threshold", 0.0)) * 255.0)
        finally:
            if inference_source is not source:
                inference_source.close()
    if crs is None:
        raise ValueError("출력 공간정보가 없습니다.")
    raw_mask = argmax_map.copy()
    raw_mask[(raw_mask > 0) & ~confidence_ok] = 0
    output_path = Path(output_mask)
    raw_path = output_path.with_name(f"{output_path.stem}_raw{output_path.suffix}")
    write_raster(raw_path, raw_mask, crs, transform_value, "uint8", 0, CLASS_COLORMAP)
    post_input = raw_mask
    if bool(inference.get("close_boundaries", False)):
        # Deliberately rescues sub-threshold ridge pixels; min_area pruning still applies.
        watershed_options = {
            "seed_threshold": float(inference.get("watershed_seed_threshold", 0.5)),
            "seed_boundary_maximum": float(inference.get("watershed_seed_boundary_maximum", 0.15)),
            "line_iterations": int(inference.get("watershed_line_iterations", 1)),
            "seed_erosion_iterations": int(inference.get("watershed_seed_erosion_iterations", 0)),
        }
        if probabilities is not None:
            post_input = close_parcel_boundaries(probabilities, boundary_class, **watershed_options)
        else:
            post_input = close_parcel_boundaries_from_maps(argmax_map, interior_map, boundary_map, boundary_class, **watershed_options)
    post_config = config.get("postprocess", {})
    if post_config.get("enabled", True):
        processed, instances = postprocess_mask(post_input, post_config, transform_value, crs, boundary_class)
    else:
        processed, instances = post_input, np.zeros(raw_mask.shape, dtype=np.uint32)
    if bool(inference.get("erase_boundary", True)):
        # 경계 클래스는 인스턴스 분리 신호일 뿐 산출물이 아니다 (RGBvsRGBN infer_building의
        # grow_labels와 동일 원리): 경계 픽셀을 이웃 필지로 흡수시켜 필지끼리 맞닿게 하고,
        # 흡수되지 못한 잔여 경계는 배경 처리해 마스크/벡터에서 클래스 7을 제거한다.
        # 모폴로지(closing 등)가 경계를 원래 선 밖으로 넓힐 수 있으므로 양쪽 모두 흡수 대상이다.
        line = (post_input == boundary_class) | (processed == boundary_class)
        instances[line] = 0
        instance_class = np.zeros(int(instances.max()) + 1, dtype=np.uint8)
        occupied = instances > 0
        instance_class[instances[occupied]] = processed[occupied]
        for _ in range(8):
            grown = ndimage.maximum_filter(instances, 3)
            update = line & (instances == 0) & (grown > 0)
            if not update.any():
                break
            instances[update] = grown[update]
            processed[update] = instance_class[instances[update]]
        processed[line & (instances == 0)] = 0
    write_raster(output_path, processed, crs, transform_value, "uint8", 0, CLASS_COLORMAP)
    if config.get("output", {}).get("save_probability_map", True):
        write_raster(output_path.with_name(f"{output_path.stem}_probability.tif"), probabilities, crs, transform_value, "float32")
    instance_path = output_path.with_name(f"{output_path.stem}_instances.tif")
    if post_config.get("save_instances", True) or output_vector:
        write_raster(instance_path, instances, crs, transform_value, "uint32", 0)
    if output_vector:
        try:
            command = [sys.executable, "-m", "src.vectorize", "--instances", str(instance_path), "--classes", str(output_path), "--output", str(output_vector)]
            class_names = dataset.get("class_names")
            if class_names:
                command += ["--class-names", ",".join(str(name) for name in class_names)]
            simplify = float(config.get("output", {}).get("vector_simplify_m", 0.0))
            if simplify > 0:
                command += ["--simplify", str(simplify)]
            subprocess.run(command, check=True)
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
    parser.add_argument("--target-resolution-m", type=float, help="추론 격자의 지상 해상도(m/pixel), 예: 0.25")
    parser.add_argument("--target-crs", help="목표 투영 CRS, 예: EPSG:5179. 경위도 입력에는 필수")
    parser.add_argument("--resampling", choices=("nearest", "bilinear", "cubic", "lanczos"))
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    for key, value in (("tile_size", args.tile_size), ("overlap", args.overlap), ("batch_size", args.batch_size), ("target_resolution_m", args.target_resolution_m), ("target_crs", args.target_crs), ("resampling", args.resampling)):
        if value is not None:
            config["inference"][key] = value
    setup_logger("infer", Path(config["project"]["output_dir"]) / "logs" / "infer.log")
    run_inference(config, args.checkpoint, args.input, args.output_mask, args.reference_meta, args.output_vector)


if __name__ == "__main__":
    main()
