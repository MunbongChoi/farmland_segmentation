"""Synchronized image/mask augmentation for segmentation."""

from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np


class SegmentationTransform:
    """Apply configurable geometry and color transforms to a pair."""

    def __init__(self, config: dict[str, Any], training: bool) -> None:
        self.config = config
        self.training = training

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.training:
            return np.ascontiguousarray(image), np.ascontiguousarray(mask)
        if random.random() < float(self.config.get("horizontal_flip", 0.0)):
            image, mask = np.flip(image, 1), np.flip(mask, 1)
        if random.random() < float(self.config.get("vertical_flip", 0.0)):
            image, mask = np.flip(image, 0), np.flip(mask, 0)
        if random.random() < float(self.config.get("rotate_90", 0.0)):
            k = random.randint(1, 3)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        if random.random() < float(self.config.get("random_scale", 0.0)):
            image, mask = self._random_scale(image, mask)
        image = self._photometric(image)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def _random_scale(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = mask.shape
        mask_dtype = mask.dtype
        limit = float(self.config.get("scale_limit", 0.15))
        scale = random.uniform(1.0 - limit, 1.0 + limit)
        resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        scaled_image = cv2.resize(image, resized_size, interpolation=cv2.INTER_LINEAR)
        # OpenCV does not support int64 images; use int32 without changing class IDs.
        scaled_mask = cv2.resize(mask.astype(np.int32), resized_size, interpolation=cv2.INTER_NEAREST)
        if scale >= 1.0:
            y = random.randint(0, scaled_mask.shape[0] - height)
            x = random.randint(0, scaled_mask.shape[1] - width)
            return scaled_image[y : y + height, x : x + width], scaled_mask[y : y + height, x : x + width].astype(mask_dtype)
        pad_y, pad_x = height - scaled_mask.shape[0], width - scaled_mask.shape[1]
        top, left = random.randint(0, pad_y), random.randint(0, pad_x)
        image_out = cv2.copyMakeBorder(scaled_image, top, pad_y - top, left, pad_x - left, cv2.BORDER_REFLECT_101)
        mask_out = cv2.copyMakeBorder(scaled_mask, top, pad_y - top, left, pad_x - left, cv2.BORDER_CONSTANT, value=0)
        return image_out, mask_out.astype(mask_dtype)

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        result = image.astype(np.float32)
        if random.random() < float(self.config.get("brightness_contrast", 0.0)):
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-20.0, 20.0)
            result = result * alpha + beta
        if random.random() < float(self.config.get("color_jitter", 0.0)):
            gains = np.random.uniform(0.9, 1.1, size=(1, 1, result.shape[2]))
            result *= gains
        if random.random() < float(self.config.get("gaussian_noise", 0.0)):
            result += np.random.normal(0.0, 5.0, size=result.shape)
        result = np.clip(result, 0.0, 255.0).astype(np.uint8)
        if random.random() < float(self.config.get("blur", 0.0)):
            result = cv2.GaussianBlur(result, (3, 3), 0)
        return result
