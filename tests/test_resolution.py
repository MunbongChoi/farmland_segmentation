from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.resolution import make_target_grid, open_resampled_vrt


class ResolutionTests(unittest.TestCase):
    def test_one_metre_raster_is_resampled_to_25cm(self) -> None:
        crs = CRS.from_epsg(5186)
        transform = from_origin(200000, 600000, 1.0, 1.0)
        grid = make_target_grid(crs, transform, 10, 8, 0.25)
        self.assertEqual((grid.width, grid.height), (40, 32))
        self.assertAlmostEqual(abs(grid.transform.a), 0.25)
        self.assertAlmostEqual(abs(grid.transform.e), 0.25)
        self.assertEqual(grid.crs, crs)

    def test_geographic_source_requires_projected_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "투영 CRS"):
            make_target_grid(CRS.from_epsg(4326), from_origin(127, 37, 0.001, 0.001), 10, 8, 0.25)

    def test_vrt_reads_target_grid_without_intermediate_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tif"
            crs = CRS.from_epsg(5186)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=10,
                height=8,
                count=3,
                dtype="uint8",
                crs=crs,
                transform=from_origin(200000, 600000, 1.0, 1.0),
            ) as destination:
                destination.write(np.full((3, 8, 10), 100, dtype=np.uint8))
            with rasterio.open(path) as source:
                vrt, grid = open_resampled_vrt(source, source.crs, source.transform, 0.25)
                try:
                    values = vrt.read()
                finally:
                    vrt.close()
            self.assertEqual(values.shape, (3, 32, 40))
            self.assertEqual((grid.height, grid.width), values.shape[1:])

    def test_vrt_uses_external_grid_for_unreferenced_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unreferenced.tif"
            with rasterio.open(path, "w", driver="GTiff", width=10, height=8, count=3, dtype="uint8") as destination:
                destination.write(np.full((3, 8, 10), 80, dtype=np.uint8))
            external_crs = CRS.from_epsg(5186)
            external_transform = from_origin(200000, 600000, 1.0, 1.0)
            with rasterio.open(path) as source:
                vrt, _ = open_resampled_vrt(source, external_crs, external_transform, 0.25)
                try:
                    values = vrt.read()
                finally:
                    vrt.close()
            self.assertEqual(values.shape, (3, 32, 40))
            self.assertGreater(values.mean(), 0)


if __name__ == "__main__":
    unittest.main()
