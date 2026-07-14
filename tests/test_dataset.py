from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from src.datasets.dataset import GeoTiffPairDataset, RasterPair
from src.datasets.transforms import SegmentationTransform


class DatasetTests(unittest.TestCase):
    def test_dataset_returns_aligned_tensors_and_remaps_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, mask_path = root / "image.tif", root / "mask.tif"
            image = np.zeros((3, 32, 32), dtype=np.uint8)
            image[:, 8:24, 8:24] = 255
            mask = np.full((32, 32), 10, dtype=np.uint8)
            mask[8:16, 8:24] = 50
            mask[16:24, 8:24] = 60
            profile = {"driver": "GTiff", "height": 32, "width": 32, "crs": "EPSG:5186", "transform": from_origin(200000, 600000, 0.25, 0.25)}
            with rasterio.open(image_path, "w", count=3, dtype="uint8", **profile) as output:
                output.write(image)
            with rasterio.open(mask_path, "w", count=1, dtype="uint8", **profile) as output:
                output.write(mask, 1)
            config = {"tile_size": 32, "channel_indices": [1, 2, 3], "input_channels": 3, "raw_class_map": {50: 1, 60: 2}, "unmapped_value": 0, "mean": [0, 0, 0], "std": [1, 1, 1]}
            sample = GeoTiffPairDataset([RasterPair(image_path, mask_path)], config)[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["mask"].shape), (32, 32))
            self.assertEqual(set(torch.unique(sample["mask"]).tolist()), {0, 1, 2})

    def test_geometry_augmentation_is_synchronized(self) -> None:
        random.seed(1)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.int64)
        image[0, 0] = 255
        mask[0, 0] = 2
        transform = SegmentationTransform({"horizontal_flip": 1.0}, training=True)
        transformed_image, transformed_mask = transform(image, mask)
        self.assertEqual(int(transformed_mask[0, 3]), 2)
        self.assertEqual(int(transformed_image[0, 3, 0]), 255)

    def test_random_scale_accepts_int64_masks(self) -> None:
        random.seed(4)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.int64)
        image[4:12, 4:12] = 200
        mask[4:12, 4:12] = 2
        transform = SegmentationTransform({"random_scale": 1.0, "scale_limit": 0.1}, training=True)
        transformed_image, transformed_mask = transform(image, mask)
        self.assertEqual(transformed_image.shape, image.shape)
        self.assertEqual(transformed_mask.shape, mask.shape)
        self.assertEqual(transformed_mask.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
