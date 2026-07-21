"""Load Hugging Face TensorFlow SegFormer HDF5 weights into PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class H5LoadReport:
    loaded: tuple[str, ...]
    skipped_classifier: tuple[str, ...]
    missing: tuple[str, ...]


def detect_tf_segformer_depths(path: str | Path) -> tuple[int, int, int, int]:
    """Return encoder block counts stored in a TensorFlow SegFormer H5 file."""
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(".h5 SegFormer 가중치를 읽으려면 `python -m pip install h5py`가 필요합니다.") from error
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TensorFlow H5 가중치를 찾을 수 없습니다: {source}")
    with h5py.File(source, "r") as handle:
        roots = _find_roots(handle)
        encoder = handle[f"{roots['segformer']}/encoder"]
        counts = [0, 0, 0, 0]
        for name in encoder.keys():
            match = re.fullmatch(r"block\.(\d+)\.(\d+)", name)
            if match:
                stage, block = int(match.group(1)), int(match.group(2))
                if stage >= len(counts):
                    raise ValueError(f"지원하지 않는 SegFormer stage입니다: {stage}")
                counts[stage] = max(counts[stage], block + 1)
    if any(count == 0 for count in counts):
        raise ValueError(f"H5에서 SegFormer encoder 깊이를 확인할 수 없습니다: {counts}")
    return tuple(counts)


def _weight_dataset_name(parameter_name: str) -> str:
    leaf = parameter_name.rsplit(".", 1)[-1]
    if leaf == "weight":
        return "gamma:0" if "layer_norm" in parameter_name or "batch_norm" in parameter_name else "kernel:0"
    if leaf == "bias":
        return "beta:0" if "layer_norm" in parameter_name or "batch_norm" in parameter_name else "bias:0"
    if leaf == "running_mean":
        return "moving_mean:0"
    if leaf == "running_var":
        return "moving_variance:0"
    raise KeyError(parameter_name)


def _h5_path(parameter_name: str, roots: dict[str, str]) -> str:
    if parameter_name.startswith("segformer."):
        relative = parameter_name[len("segformer.") :]
        base = roots["segformer"]
    elif parameter_name.startswith("decode_head."):
        relative = parameter_name[len("decode_head.") :]
        base = roots["decode_head"]
    else:
        raise KeyError(parameter_name)
    module_name = relative.rsplit(".", 1)[0]
    parts = module_name.split(".")
    groups: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "block" and index + 2 < len(parts) and parts[index + 1].isdigit() and parts[index + 2].isdigit():
            groups.append(f"block.{parts[index + 1]}.{parts[index + 2]}")
            index += 3
        elif part in {"patch_embeddings", "layer_norm", "linear_c"} and index + 1 < len(parts) and parts[index + 1].isdigit():
            groups.append(f"{part}.{parts[index + 1]}")
            index += 2
        else:
            groups.append(part)
            index += 1
    module_name = "/".join(groups)
    return f"{base}/{module_name}/{_weight_dataset_name(parameter_name)}"


def _find_roots(handle) -> dict[str, str]:
    roots: dict[str, str] = {}
    for component in ("segformer", "decode_head"):
        if component not in handle:
            raise ValueError(f"TensorFlow SegFormer H5 그룹이 없습니다: {component}")
        wrappers = list(handle[component].keys())
        if len(wrappers) != 1:
            raise ValueError(f"H5 {component} wrapper를 하나로 결정할 수 없습니다: {wrappers}")
        roots[component] = f"{component}/{wrappers[0]}/{component}"
        if roots[component] not in handle:
            raise ValueError(f"TensorFlow SegFormer H5 경로가 없습니다: {roots[component]}")
    return roots


def _convert_array(values: np.ndarray, target: torch.Tensor, name: str) -> torch.Tensor:
    target_shape = tuple(target.shape)
    if tuple(values.shape) == target_shape:
        converted = values
    elif values.ndim == 2 and tuple(values.T.shape) == target_shape:
        converted = values.T
    elif values.ndim == 4 and tuple(values.transpose(3, 2, 0, 1).shape) == target_shape:
        converted = values.transpose(3, 2, 0, 1)
    else:
        raise ValueError(f"H5 가중치 shape 불일치: {name} h5={values.shape} torch={target_shape}")
    return torch.from_numpy(np.ascontiguousarray(converted)).to(dtype=target.dtype)


def load_tf_segformer_h5(model: nn.Module, path: str | Path) -> H5LoadReport:
    """Load a TF/Keras SegFormer weights-only H5 file without TensorFlow.

    TensorFlow dense and convolution kernels are transposed to PyTorch layout.
    A classifier whose class count differs from the target model is deliberately
    skipped so the downstream segmentation head can be trained from scratch.
    """
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(".h5 SegFormer 가중치를 읽으려면 `python -m pip install h5py`가 필요합니다.") from error

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"TensorFlow H5 가중치를 찾을 수 없습니다: {source}")

    current = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    loaded: list[str] = []
    skipped_classifier: list[str] = []
    missing: list[str] = []
    with h5py.File(source, "r") as handle:
        backend = handle.attrs.get("backend")
        if backend is not None and str(backend) != "tensorflow":
            raise ValueError(f"TensorFlow H5 파일이 아닙니다: backend={backend}")
        roots = _find_roots(handle)
        for name, target in current.items():
            if name.endswith("num_batches_tracked"):
                continue
            try:
                dataset_path = _h5_path(name, roots)
            except KeyError:
                missing.append(name)
                continue
            if dataset_path not in handle:
                missing.append(name)
                continue
            values = np.asarray(handle[dataset_path])
            try:
                converted[name] = _convert_array(values, target, name)
                loaded.append(name)
            except ValueError:
                if name.startswith("decode_head.classifier."):
                    skipped_classifier.append(name)
                else:
                    raise

    required_missing = [name for name in missing if not name.startswith("decode_head.classifier.")]
    if required_missing:
        preview = ", ".join(required_missing[:8])
        raise ValueError(f"H5에서 필수 SegFormer 가중치를 찾지 못했습니다({len(required_missing)}개): {preview}")
    model.load_state_dict(converted, strict=False)
    return H5LoadReport(tuple(loaded), tuple(skipped_classifier), tuple(missing))
