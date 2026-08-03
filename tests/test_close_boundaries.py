from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage

from src.infer import close_parcel_boundaries


class CloseParcelBoundariesTests(unittest.TestCase):
    def test_bridges_argmax_gap_along_probability_ridge(self) -> None:
        # Two parcels split by a boundary ridge at columns 9:11; rows 8:12 are a
        # "gap" where interior narrowly wins argmax but the ridge is still elevated.
        probabilities = np.zeros((3, 20, 20), dtype=np.float32)
        probabilities[1] = 0.9
        probabilities[2] = 0.05
        probabilities[1, :, 9:11] = 0.3
        probabilities[2, :, 9:11] = 0.6
        probabilities[1, 8:12, 9:11] = 0.5
        probabilities[2, 8:12, 9:11] = 0.45

        open_mask = probabilities.argmax(axis=0)
        _, open_count = ndimage.label(open_mask == 1, ndimage.generate_binary_structure(2, 2))
        self.assertEqual(open_count, 1)  # gap merges both parcels before closing

        closed = close_parcel_boundaries(probabilities)
        structure = ndimage.generate_binary_structure(2, 2)
        _, closed_count = ndimage.label(closed == 1, structure)
        self.assertEqual(closed_count, 2)
        self.assertTrue((closed[8:12, 9:11] == 2).any())  # gap now holds boundary pixels

        # The dividing line must survive the postprocess 3x3 binary closing.
        reclosed = ndimage.binary_closing(closed == 1, structure, iterations=1)
        _, reclosed_count = ndimage.label(reclosed, structure)
        self.assertEqual(reclosed_count, 2)

    def test_pure_interior_returns_argmax_unchanged(self) -> None:
        probabilities = np.zeros((3, 8, 8), dtype=np.float32)
        probabilities[1] = 0.9
        self.assertTrue((close_parcel_boundaries(probabilities) == 1).all())


if __name__ == "__main__":
    unittest.main()
