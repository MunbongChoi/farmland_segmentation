from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from src.losses import CompositeSegmentationLoss
from src.metrics.segmentation_metrics import SegmentationMetrics
from src.model import SegFormerAdapter, UNet, build_model


class DummySegFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 3, kernel_size=1, stride=4)
        self.decode_head = SimpleNamespace(classifier=self.head)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.head(pixel_values))


class ModelLossMetricTests(unittest.TestCase):
    def test_model_output_matches_input_size(self) -> None:
        model = UNet(3, 3, base_channels=8)
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 65, 67))
        self.assertEqual(tuple(output.shape), (2, 3, 65, 67))

    def test_segformer_adapter_restores_mask_resolution(self) -> None:
        model = SegFormerAdapter(DummySegFormer()).eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 65, 67))
        self.assertEqual(tuple(output.shape), (2, 3, 65, 67))

    def test_segformer_classifier_gradient_matches_parameter_stride(self) -> None:
        backbone = DummySegFormer()
        model = SegFormerAdapter(backbone)
        model(torch.randn(2, 3, 32, 32)).mean().backward()
        self.assertEqual(backbone.head.weight.grad.stride(), backbone.head.weight.stride())

    @patch("src.model._load_segformer_model")
    def test_build_model_selects_segformer(self, loader: Mock) -> None:
        loader.return_value = DummySegFormer()
        config = {
            "model": {
                "name": "segformer",
                "input_channels": 3,
                "num_classes": 3,
                "checkpoint": "nvidia/segformer-b2-finetuned-ade-512-512",
                "revision": "safe-revision",
                "use_safetensors": True,
                "pretrained": True,
            },
            "dataset": {
                "input_channels": 3,
                "num_classes": 3,
                "class_names": ["배경", "논", "밭"],
                "ignore_index": 255,
            },
        }
        model = build_model(config)
        self.assertIsInstance(model, SegFormerAdapter)
        loader.assert_called_once_with(
            checkpoint="nvidia/segformer-b2-finetuned-ade-512-512",
            revision="safe-revision",
            input_channels=3,
            num_classes=3,
            class_names=["배경", "논", "밭"],
            ignore_index=255,
            pretrained=True,
            local_files_only=False,
            use_safetensors=True,
        )

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
