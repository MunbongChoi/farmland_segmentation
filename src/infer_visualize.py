"""Run semantic inference and create pixel-aligned PNG visualizations."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from .datasets.dataset import load_spatial_metadata
from .infer import run_inference
from .utils.config import apply_overrides, load_config
from .utils.logger import setup_logger
from .utils.visualization import save_inference_visualizations


def _same_transform(left: Affine, right: Affine) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=1e-8))


def load_visualization_arrays(
    input_path: str | Path,
    mask_path: str | Path,
    probability_path: str | Path,
    channel_indices: list[int],
    reference_meta: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate source, semantic mask, probability, and spatial grids."""
    channels = tuple(int(value) for value in channel_indices[:3])
    if len(channels) != 3:
        raise ValueError("RGB 시각화를 위해 dataset.channel_indices에 3개 밴드가 필요합니다.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(input_path) as source:
            if max(channels) > source.count:
                raise ValueError(f"입력 영상 밴드 부족: required={channels}, actual={source.count}")
            image = source.read(channels)
            source_shape, source_crs, source_transform = source.shape, source.crs, source.transform

    with rasterio.open(mask_path) as mask_source:
        mask = mask_source.read(1)
        if mask_source.crs is None or mask_source.transform.is_identity:
            raise ValueError("추론 mask에 유효한 CRS/Transform이 없습니다.")
        mask_crs, mask_transform = mask_source.crs, mask_source.transform
        if mask_source.shape != source_shape:
            raise ValueError("원본 영상과 추론 mask 크기가 다릅니다.")

    if reference_meta:
        metadata = load_spatial_metadata(reference_meta)
        if (metadata.height, metadata.width) != source_shape:
            raise ValueError("reference Meta JSON과 원본 영상 크기가 다릅니다.")
        if metadata.crs != mask_crs or not _same_transform(metadata.transform, mask_transform):
            raise ValueError("reference Meta JSON과 추론 mask의 CRS/Transform이 다릅니다.")
    elif source_crs is not None and not source_transform.is_identity:
        if source_crs != mask_crs or not _same_transform(source_transform, mask_transform):
            raise ValueError("원본 GeoTIFF와 추론 mask의 CRS/Transform이 다릅니다.")
    else:
        raise ValueError("원본 영상에 공간정보가 없습니다. 대응하는 --reference-meta를 지정하세요.")

    with rasterio.open(probability_path) as probability_source:
        if probability_source.shape != source_shape:
            raise ValueError("확률 raster와 원본 영상 크기가 다릅니다.")
        if probability_source.crs != mask_crs or not _same_transform(probability_source.transform, mask_transform):
            raise ValueError("확률 raster와 추론 mask의 CRS/Transform이 다릅니다.")
        probabilities = probability_source.read()
    return image, mask, probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="논/밭 SegFormer 추론 및 PNG 시각화")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference-meta", help="공간정보가 없는 원본 TIF에 대응하는 기존 _META.json")
    parser.add_argument("--output-dir", help="기본값: project.output_dir/predictions/<입력 파일명>")
    parser.add_argument("--name", help="출력 파일 접두어; 기본값은 입력 파일 stem")
    parser.add_argument("--output-vector", action="store_true", help="인스턴스 polygon GPKG도 생성")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    for key, value in (("tile_size", args.tile_size), ("overlap", args.overlap), ("batch_size", args.batch_size)):
        if value is not None:
            config["inference"][key] = value
    config.setdefault("output", {})["save_probability_map"] = True

    stem = args.name or Path(args.input).stem
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["project"]["output_dir"]) / "predictions" / stem
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("infer_visualize", output_dir / "infer_visualize.log")
    mask_path = output_dir / f"{stem}.tif"
    vector_path = output_dir / f"{stem}.gpkg" if args.output_vector else None
    run_inference(config, args.checkpoint, args.input, str(mask_path), args.reference_meta, str(vector_path) if vector_path else None)

    probability_path = mask_path.with_name(f"{mask_path.stem}_probability.tif")
    image, mask, probabilities = load_visualization_arrays(
        args.input,
        mask_path,
        probability_path,
        list(config["dataset"]["channel_indices"]),
        args.reference_meta,
    )
    paths = save_inference_visualizations(image, mask, probabilities, output_dir, stem, args.alpha)
    logging.getLogger("infer_visualize").info("시각화 완료: %s", {key: str(path) for key, path in paths.items()})


if __name__ == "__main__":
    main()
