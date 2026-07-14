from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from src.infer import sliding_window_predict_array, write_raster
from src.model import UNet
from src.utils.checkpoint import load_checkpoint, save_checkpoint


class TinyModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack((inputs[:, 0], -inputs[:, 0]), dim=1)


class CheckpointInferenceTests(unittest.TestCase):
    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            model = UNet(3, 3, base_channels=4)
            optimizer = torch.optim.Adam(model.parameters())
            config = {"model": {"name": "unet"}, "dataset": {"class_names": ["배경", "논", "밭"]}}
            save_checkpoint(path, model, optimizer, None, None, 3, 0.75, config)
            restored = UNet(3, 3, base_channels=4)
            payload = load_checkpoint(path, restored)
            self.assertEqual(payload["epoch"], 3)
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_sliding_window_preserves_original_shape(self) -> None:
        image = np.random.default_rng(2).integers(0, 255, (3, 73, 91), dtype=np.uint8)
        probabilities = sliding_window_predict_array(image, TinyModel(), torch.device("cpu"), 2, 32, 8, 4, [0, 0, 0], [1, 1, 1])
        self.assertEqual(probabilities.shape, (2, 73, 91))
        np.testing.assert_allclose(probabilities.sum(0), 1.0, atol=1e-5)

    def test_geotiff_output_preserves_crs_and_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.tif"
            transform = from_origin(200000, 600000, 0.25, 0.25)
            write_raster(path, np.ones((10, 12), dtype=np.uint8), rasterio.crs.CRS.from_epsg(5186), transform, "uint8", 0)
            with rasterio.open(path) as source:
                self.assertEqual(source.crs.to_epsg(), 5186)
                self.assertEqual(source.transform, transform)
                self.assertEqual(source.shape, (10, 12))


if __name__ == "__main__":
    unittest.main()

