"""Dependency-light segmentation visualization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_PALETTE = np.asarray(
    [
        [0, 0, 0],       # 0: background
        [60, 180, 75],   # 1: paddy
        [255, 165, 0],   # 2: field
    ],
    dtype=np.uint8,
)


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Convert CHW/HWC RGB imagery to uint8 with robust per-band stretching."""
    if image.ndim != 3:
        raise ValueError("image는 CHW 또는 HWC 3차원 배열이어야 합니다.")
    values = np.moveaxis(image[:3], 0, -1) if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4) else image[..., :3]
    if values.shape[2] != 3:
        raise ValueError("시각화에는 RGB 3개 밴드가 필요합니다.")
    if values.dtype == np.uint8:
        return np.ascontiguousarray(values)
    result = np.zeros(values.shape, dtype=np.uint8)
    for channel in range(3):
        band = values[..., channel].astype(np.float32)
        finite = np.isfinite(band)
        if not finite.any():
            continue
        low, high = np.percentile(band[finite], (2.0, 98.0))
        if high <= low:
            high = low + 1.0
        stretched = np.clip((band - low) / (high - low), 0.0, 1.0)
        result[..., channel] = np.where(finite, stretched * 255.0, 0.0).astype(np.uint8)
    return result


def colorize_mask(mask: np.ndarray, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    """Map integer semantic classes to RGB colors."""
    if mask.ndim != 2:
        raise ValueError("mask는 HW 2차원 배열이어야 합니다.")
    if mask.size and (mask.min() < 0 or mask.max() >= len(palette)):
        raise ValueError(f"palette가 mask class id를 포함하지 않습니다: min={mask.min()} max={mask.max()}")
    return palette[mask.astype(np.int64)]


def blend_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend paddy/field colors over RGB while leaving background unchanged."""
    rgb = to_display_rgb(image)
    if rgb.shape[:2] != mask.shape:
        raise ValueError("image와 mask의 공간 크기가 같아야 합니다.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha는 0~1 범위여야 합니다.")
    colored = colorize_mask(mask)
    blended = cv2.addWeighted(rgb, 1.0 - alpha, colored, alpha, 0)
    overlay = rgb.copy()
    overlay[mask > 0] = blended[mask > 0]
    return overlay


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"시각화 파일을 저장할 수 없습니다: {path}")


def _heatmap(values: np.ndarray) -> np.ndarray:
    scaled = np.clip(values.astype(np.float32), 0.0, 1.0)
    colored = cv2.applyColorMap(np.round(scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _title_tile(image: np.ndarray, title: str) -> np.ndarray:
    tile = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    cv2.putText(tile, title, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return tile


def save_inference_visualizations(
    image: np.ndarray,
    mask: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str | Path,
    stem: str,
    alpha: float = 0.45,
) -> dict[str, Path]:
    """Save RGB, semantic mask, overlay, confidence, class maps, and a panel."""
    rgb = to_display_rgb(image)
    if rgb.shape[:2] != mask.shape:
        raise ValueError("원본 영상과 추론 mask의 크기가 다릅니다.")
    if probabilities.ndim != 3 or probabilities.shape[1:] != mask.shape:
        raise ValueError("probabilities는 C,H,W이고 mask와 공간 크기가 같아야 합니다.")
    if probabilities.shape[0] < len(DEFAULT_PALETTE):
        raise ValueError("배경/논/밭 확률 3개 밴드가 필요합니다.")

    destination = Path(output_dir)
    mask_rgb = colorize_mask(mask)
    overlay = blend_mask(rgb, mask, alpha)
    confidence = _heatmap(probabilities.max(axis=0))
    paths = {
        "rgb": destination / f"{stem}_rgb.png",
        "mask": destination / f"{stem}_mask_color.png",
        "overlay": destination / f"{stem}_overlay.png",
        "confidence": destination / f"{stem}_confidence.png",
        "panel": destination / f"{stem}_panel.png",
    }
    for key, values in (("rgb", rgb), ("mask", mask_rgb), ("overlay", overlay), ("confidence", confidence)):
        _write_rgb(paths[key], values)
    for class_id, class_name in ((1, "paddy"), (2, "field")):
        path = destination / f"{stem}_prob_{class_id}_{class_name}.png"
        _write_rgb(path, _heatmap(probabilities[class_id]))
        paths[f"probability_{class_name}"] = path

    panel = np.vstack(
        [
            np.hstack([_title_tile(rgb, "RGB"), _title_tile(mask_rgb, "Mask (green / orange)")]),
            np.hstack([_title_tile(overlay, "Overlay"), _title_tile(confidence, "Max class confidence")]),
        ]
    )
    _write_rgb(paths["panel"], panel)
    return paths


def save_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.45) -> None:
    """Save an RGB image blended with background/paddy/field colors."""
    destination = Path(path)
    _write_rgb(destination, blend_mask(image, mask, alpha))
