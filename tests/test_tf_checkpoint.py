from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from src.utils.tf_checkpoint import _canonical_parameter_name, detect_tf_segformer_depths, load_tf_segformer_h5


class _PatchEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 2, kernel_size=3)
        self.layer_norm = nn.LayerNorm(2)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = nn.ModuleList([_PatchEmbedding()])


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()


class _DecodeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(2, 3, kernel_size=1)


class _FakeSegFormer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segformer = _Backbone()
        self.decode_head = _DecodeHead()


class TensorFlowCheckpointTests(unittest.TestCase):
    def test_transformers_5_modular_names_map_to_tf_h5_names(self) -> None:
        mappings = {
            "segformer.stages.0.patch_embeddings.proj.weight": "segformer.encoder.patch_embeddings.0.proj.weight",
            "segformer.stages.1.blocks.7.layernorm_before.bias": "segformer.encoder.block.1.7.layer_norm_1.bias",
            "segformer.stages.2.blocks.26.attention.q_proj.weight": "segformer.encoder.block.2.26.attention.self.query.weight",
            "segformer.stages.2.blocks.0.attention.sequence_reduction.sequence_reduction.weight": "segformer.encoder.block.2.0.attention.self.sr.weight",
            "segformer.stages.0.blocks.1.attention.o_proj.bias": "segformer.encoder.block.0.1.attention.output.dense.bias",
            "segformer.stages.3.blocks.2.mlp.fc1.weight": "segformer.encoder.block.3.2.mlp.dense1.weight",
            "segformer.stages.3.layer_norm.weight": "segformer.encoder.layer_norm.3.weight",
            "decode_head.linear_projections.2.proj.weight": "decode_head.linear_c.2.proj.weight",
        }
        for current, legacy in mappings.items():
            with self.subTest(current=current):
                self.assertEqual(_canonical_parameter_name(current), legacy)

    def test_b4_depths_are_detected_from_encoder_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b4.h5"
            with h5py.File(path, "w") as handle:
                encoder = handle.create_group("segformer/wrapper/segformer/encoder")
                handle.create_group("decode_head/wrapper/decode_head")
                for stage, depth in enumerate((3, 8, 27, 3)):
                    for block in range(depth):
                        encoder.create_group(f"block.{stage}.{block}")
            self.assertEqual(detect_tf_segformer_depths(path), (3, 8, 27, 3))

    def test_h5_kernels_are_transposed_and_mismatched_head_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.h5"
            kernel = np.arange(3 * 3 * 3 * 2, dtype=np.float32).reshape(3, 3, 3, 2)
            with h5py.File(path, "w") as handle:
                handle.attrs["backend"] = "tensorflow"
                segformer = handle.create_group("segformer/wrapper/segformer/encoder/patch_embeddings.0")
                projection = segformer.create_group("proj")
                projection.create_dataset("kernel:0", data=kernel)
                projection.create_dataset("bias:0", data=np.array([1.0, 2.0], dtype=np.float32))
                normalization = segformer.create_group("layer_norm")
                normalization.create_dataset("gamma:0", data=np.array([3.0, 4.0], dtype=np.float32))
                normalization.create_dataset("beta:0", data=np.array([5.0, 6.0], dtype=np.float32))
                classifier = handle.create_group("decode_head/wrapper/decode_head/classifier")
                classifier.create_dataset("kernel:0", data=np.zeros((1, 1, 2, 150), dtype=np.float32))
                classifier.create_dataset("bias:0", data=np.zeros(150, dtype=np.float32))

            model = _FakeSegFormer()
            classifier_before = model.decode_head.classifier.weight.detach().clone()
            report = load_tf_segformer_h5(model, path)
            expected = torch.from_numpy(kernel.transpose(3, 2, 0, 1).copy())
            torch.testing.assert_close(model.segformer.encoder.patch_embeddings[0].proj.weight, expected)
            torch.testing.assert_close(model.segformer.encoder.patch_embeddings[0].layer_norm.weight, torch.tensor([3.0, 4.0]))
            torch.testing.assert_close(model.decode_head.classifier.weight, classifier_before)
            self.assertEqual(set(report.skipped_classifier), {"decode_head.classifier.weight", "decode_head.classifier.bias"})


if __name__ == "__main__":
    unittest.main()
