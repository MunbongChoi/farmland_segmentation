from __future__ import annotations

import unittest

import torch

from src.losses import CompositeSegmentationLoss
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.model import UNet


class ModelLossMetricTests(unittest.TestCase):
    def test_model_output_matches_input_size(self) -> None:
        model = UNet(3, 3, base_channels=8)
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 65, 67))
        self.assertEqual(tuple(output.shape), (2, 3, 65, 67))

    def test_composite_loss_is_finite(self) -> None:
        config = {"weights": {"cross_entropy": 1.0, "binary_cross_entropy": 0.0, "dice": 1.0, "focal": 0.0, "tversky": 0.0, "boundary": 0.1}}
        criterion = CompositeSegmentationLoss(config, 3, 255)
        loss, parts = criterion(torch.randn(2, 3, 16, 16), torch.randint(0, 3, (2, 16, 16)))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"cross_entropy", "dice", "boundary"})

    def test_metrics_for_perfect_prediction(self) -> None:
        meter = SegmentationMetrics(3)
        target = torch.tensor([[0, 1], [2, 2]])
        meter.update(target, target)
        metrics = meter.compute(False)
        self.assertAlmostEqual(metrics["pixel_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["foreground_pixel_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_iou"], 1.0)

    def test_foreground_accuracy_is_not_inflated_by_background(self) -> None:
        meter = SegmentationMetrics(3)
        target = torch.tensor([[0, 0], [1, 2]])
        prediction = torch.tensor([[0, 0], [0, 2]])
        meter.update(prediction, target)
        metrics = meter.compute()
        self.assertAlmostEqual(metrics["pixel_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["foreground_pixel_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
