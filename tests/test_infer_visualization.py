from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

from src.infer import write_raster
from src.datasets.dataset import RasterPair
from src.infer_visualize import load_visualization_arrays, select_sample_pairs
from src.utils.visualization import blend_mask, save_inference_visualizations


class InferenceVisualizationTests(unittest.TestCase):
    @patch("src.infer_visualize.discover_pairs")
    def test_batch_selection_is_reproducible_and_limited(self, discover) -> None:
        discover.return_value = [
            RasterPair(Path(f"image_{index}.tif"), Path(f"label_{index}.json"), Path(f"meta_{index}.json"))
            for index in range(20)
        ]
        config = {
            "project": {"seed": 42},
            "dataset": {
                "root_dir": "data",
                "validation": {"index_cache": "cache.json"},
                "validation_test_fraction": 0.0,
                "raw_class_map": {50: 1, 60: 2},
                "target_property": "ANN_CD",
            },
        }
        first = select_sample_pairs(config, "validation", 10)
        second = select_sample_pairs(config, "validation", 10)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len({pair.image for pair in first}), 10)

    def test_overlay_preserves_background_pixels(self) -> None:
        image = np.full((8, 9, 3), 100, dtype=np.uint8)
        mask = np.zeros((8, 9), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        overlay = blend_mask(image, mask, alpha=0.5)
        np.testing.assert_array_equal(overlay[0, 0], image[0, 0])
        self.assertFalse(np.array_equal(overlay[3, 4], image[3, 4]))

    def test_visualization_bundle_is_written(self) -> None:
        image = np.random.default_rng(3).integers(0, 255, (3, 12, 14), dtype=np.uint8)
        mask = np.zeros((12, 14), dtype=np.uint8)
        mask[:, 4:8] = 1
        mask[:, 8:] = 2
        probabilities = np.eye(3, dtype=np.float32)[mask].transpose(2, 0, 1)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_inference_visualizations(image, mask, probabilities, directory, "sample")
            self.assertEqual(
                set(paths),
                {"rgb", "mask", "overlay", "confidence", "panel", "probability_paddy", "probability_field"},
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths.values()))

    def test_loader_validates_meta_mask_probability_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "source.tif"
            mask_path = root / "mask.tif"
            probability_path = root / "mask_probability.tif"
            meta_path = root / "source_META.json"
            transform = from_origin(199999.875, 600000.125, 0.25, 0.25)
            crs = rasterio.crs.CRS.from_epsg(5186)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(image_path, "w", driver="GTiff", width=10, height=8, count=3, dtype="uint8") as destination:
                    destination.write(np.ones((3, 8, 10), dtype=np.uint8))
            write_raster(mask_path, np.zeros((8, 10), dtype=np.uint8), crs, transform, "uint8", 0)
            write_raster(probability_path, np.zeros((3, 8, 10), dtype=np.float32), crs, transform, "float32")
            meta_path.write_text(
                json.dumps(
                    [{"img_width": 10, "img_height": 8, "img_resolution": 0.25, "coordinates": "200000, 600000", "img_coordinate": "EPSG:5186"}]
                ),
                encoding="utf-8",
            )
            image, mask, probabilities = load_visualization_arrays(image_path, mask_path, probability_path, [1, 2, 3], meta_path)
            self.assertEqual(image.shape, (3, 8, 10))
            self.assertEqual(mask.shape, (8, 10))
            self.assertEqual(probabilities.shape, (3, 8, 10))


if __name__ == "__main__":
    unittest.main()
