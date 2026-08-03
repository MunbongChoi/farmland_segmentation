from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.datasets.tiles import build_tile_datasets
from src.visualize_tiles import _read_context_window


def _write_tiles(root: Path, names_by_split: dict[str, list[str]]) -> None:
    (root / "images").mkdir()
    (root / "labels").mkdir()
    rows = ["tile,scene,row,col,x,y,parcels,cover,boundary,split"]
    for split, names in names_by_split.items():
        for name in names:
            with rasterio.open(root / "images" / f"{name}.tif", "w", driver="GTiff", height=16, width=16, count=4, dtype="uint8") as output:
                output.write(np.full((4, 16, 16), 128, dtype=np.uint8))
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 1
            mask[4:12, 4] = 2
            with rasterio.open(root / "labels" / f"{name}.tif", "w", driver="GTiff", height=16, width=16, count=1, dtype="uint8") as output:
                output.write(mask, 1)
            rows.append(f"{name},scene,0,0,0.0,0.0,1,0.25,0.03,{split}")
    (root / "manifest.csv").write_text("\n".join(rows), encoding="utf-8")


class TileDatasetTests(unittest.TestCase):
    def test_build_splits_and_returns_aligned_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tiles(root, {"train": ["a", "b"], "val": ["c"], "test": ["d"]})
            config = {
                "dataset": {"type": "tiles", "root_dir": str(root), "input_channels": 3, "channel_indices": [1, 2, 3], "mean": [0, 0, 0], "std": [1, 1, 1]},
                "augmentation": {},
            }
            train, validation, test = build_tile_datasets(config)
            self.assertEqual((len(train), len(validation), len(test)), (2, 1, 1))
            sample = validation[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 16, 16))
            self.assertEqual(tuple(sample["mask"].shape), (16, 16))
            self.assertEqual(set(torch.unique(sample["mask"]).tolist()), {0, 1, 2})
            self.assertAlmostEqual(float(sample["image"].max()), 128 / 255, places=5)

    def test_context_window_uses_neighbors_and_zero_fills_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            for row in (0, 1):
                for col in (0, 1):
                    value = 10 * row + col + 1
                    with rasterio.open(root / "images" / f"s_{row:03d}_{col:03d}.tif", "w", driver="GTiff", height=8, width=8, count=1, dtype="uint8") as output:
                        output.write(np.full((1, 8, 8), value, dtype=np.uint8))
            window = _read_context_window(root, "s_001_001", (1,), tile_size=8, margin=4)
            self.assertEqual(window.shape, (1, 16, 16))
            self.assertTrue((window[0, 4:12, 4:12] == 12).all())  # center = tile itself
            self.assertTrue((window[0, :4, :4] == 1).all())       # top-left from neighbor (0,0)
            self.assertTrue((window[0, :4, 4:12] == 2).all())     # top from neighbor (0,1)
            self.assertTrue((window[0, 12:, :] == 0).all())       # missing bottom neighbors -> zeros

    def test_missing_split_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tiles(root, {"train": ["a"]})
            config = {"dataset": {"type": "tiles", "root_dir": str(root), "input_channels": 3, "channel_indices": [1, 2, 3], "mean": [0, 0, 0], "std": [1, 1, 1]}}
            with self.assertRaises(ValueError):
                build_tile_datasets(config)


if __name__ == "__main__":
    unittest.main()
