"""Small, dependency-light segmentation visualization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.45) -> None:
    """Save an RGB image blended with background/paddy/field colors."""
    if image.ndim != 3 or image.shape[2] < 3 or image.shape[:2] != mask.shape:
        raise ValueError("image는 HWC RGB이고 mask와 공간 크기가 같아야 합니다.")
    palette = np.asarray([[0, 0, 0], [60, 180, 75], [255, 165, 0]], dtype=np.uint8)
    colored = palette[np.clip(mask, 0, len(palette) - 1)]
    overlay = cv2.addWeighted(image[..., :3].astype(np.uint8), 1.0 - alpha, colored, alpha, 0)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"시각화 파일을 저장할 수 없습니다: {destination}")

