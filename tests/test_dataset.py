from __future__ import annotations

import random
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from src.datasets.dataset import GeoTiffPairDataset, RasterPair, discover_pairs
from src.datasets.transforms import SegmentationTransform


class DatasetTests(unittest.TestCase):
    def test_dataset_returns_aligned_tensors_and_remaps_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, label_path, meta_path = root / "image.tif", root / "image.json", root / "image_META.json"
            image = np.zeros((3, 32, 32), dtype=np.uint8)
            image[:, 8:24, 8:24] = 255
            profile = {"driver": "GTiff", "height": 32, "width": 32, "crs": "EPSG:5186", "transform": from_origin(200000, 600000, 1, 1)}
            with rasterio.open(image_path, "w", count=3, dtype="uint8", **profile) as output:
                output.write(image)
            label = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:5186"}}, "features": [
                {"type": "Feature", "properties": {"ANN_CD": 50}, "geometry": {"type": "Polygon", "coordinates": [[[200008, 599992], [200024, 599992], [200024, 599984], [200008, 599984], [200008, 599992]]]}},
                {"type": "Feature", "properties": {"ANN_CD": 60}, "geometry": {"type": "Polygon", "coordinates": [[[200008, 599984], [200024, 599984], [200024, 599976], [200008, 599976], [200008, 599984]]]}}
            ]}
            meta = [{"img_width": 32, "img_height": 32, "img_coordinate": "EPSG:5186", "coordinates": "200000.5, 599999.5", "img_resolution": 1}]
            label_path.write_text(json.dumps(label), encoding="utf-8")
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            config = {"tile_size": 32, "channel_indices": [1, 2, 3], "input_channels": 3, "raw_class_map": {50: 1, 60: 2}, "target_property": "ANN_CD", "unmapped_value": 0, "mean": [0, 0, 0], "std": [1, 1, 1]}
            sample = GeoTiffPairDataset([RasterPair(image_path, label_path, meta_path)], config)[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["mask"].shape), (32, 32))
            self.assertEqual(set(torch.unique(sample["mask"]).tolist()), {0, 1, 2})

    def test_discovery_ignores_json_without_paddy_or_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("images", "labels", "meta"):
                (root / name).mkdir()
            profile = {"driver": "GTiff", "height": 4, "width": 4, "count": 3, "dtype": "uint8"}
            meta = [{"img_width": 4, "img_height": 4, "img_coordinate": "EPSG:5186", "coordinates": "0.5, 3.5", "img_resolution": 1}]
            for stem, code in (("target", 50), ("empty", 10)):
                with rasterio.open(root / "images" / f"{stem}.tif", "w", **profile) as output:
                    output.write(np.zeros((3, 4, 4), dtype=np.uint8))
                feature = {"type": "Feature", "properties": {"ANN_CD": code}, "geometry": {"type": "Polygon", "coordinates": [[[0, 4], [1, 4], [1, 3], [0, 3], [0, 4]]]}}
                (root / "labels" / f"{stem}.json").write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
                (root / "meta" / f"{stem}_META.json").write_text(json.dumps(meta), encoding="utf-8")
            pairs = discover_pairs(root, {"image_dirs": ["images"], "label_dirs": ["labels"], "meta_dirs": ["meta"]}, {50: 1, 60: 2})
            self.assertEqual([pair.image.stem for pair in pairs], ["target"])

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
