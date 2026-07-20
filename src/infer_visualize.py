"""Run semantic inference and create pixel-aligned PNG visualizations."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from .datasets.dataset import RasterPair, deterministic_partition, discover_pairs, load_spatial_metadata
from .infer import load_inference_model, run_inference
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


def select_sample_pairs(config: dict[str, Any], split: str, sample_count: int) -> list[RasterPair]:
    """Select reproducible target-containing pairs without loading image pixels."""
    if sample_count <= 0:
        raise ValueError("sample_count는 1 이상이어야 합니다.")
    dataset = config["dataset"]
    class_map = {int(key): int(value) for key, value in dataset["raw_class_map"].items()}
    property_name = str(dataset.get("target_property", "ANN_CD"))
    if split == "train":
        split_config = dataset["train"]
        pairs = discover_pairs(dataset["root_dir"], split_config, class_map, property_name, split_config.get("index_cache"))
    else:
        split_config = dataset["validation"]
        all_validation = discover_pairs(dataset["root_dir"], split_config, class_map, property_name, split_config.get("index_cache"))
        validation, test = deterministic_partition(
            all_validation,
            float(dataset.get("validation_test_fraction", 0.0)),
            int(config["project"]["seed"]),
        )
        pairs = validation if split == "validation" else (test or validation)
    if not pairs:
        raise ValueError(f"{split} split에 추론 가능한 논/밭 샘플이 없습니다.")
    count = min(sample_count, len(pairs))
    return random.Random(int(config["project"]["seed"])).sample(pairs, count)


def infer_and_visualize_one(
    config: dict[str, Any],
    checkpoint: str,
    input_path: str | Path,
    reference_meta: str | Path | None,
    output_dir: str | Path,
    stem: str,
    alpha: float,
    output_vector: bool,
    model,
    device,
) -> dict[str, Path]:
    """Create geospatial inference rasters and their PNG visualization bundle."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = destination / f"{stem}.tif"
    vector_path = destination / f"{stem}.gpkg" if output_vector else None
    run_inference(
        config,
        checkpoint,
        str(input_path),
        str(mask_path),
        str(reference_meta) if reference_meta else None,
        str(vector_path) if vector_path else None,
        model=model,
        device=device,
    )
    probability_path = mask_path.with_name(f"{mask_path.stem}_probability.tif")
    image, mask, probabilities = load_visualization_arrays(
        input_path,
        mask_path,
        probability_path,
        list(config["dataset"]["channel_indices"]),
        reference_meta,
    )
    return save_inference_visualizations(image, mask, probabilities, destination, stem, alpha)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="논/밭 SegFormer 추론 및 PNG 시각화")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="단일 원본 TIF")
    source.add_argument("--split", choices=("train", "validation", "test"), help="설정된 데이터 split에서 batch 샘플 선택")
    parser.add_argument("--sample-count", type=int, default=10, help="--split 사용 시 추론할 샘플 수")
    parser.add_argument("--reference-meta", help="공간정보가 없는 원본 TIF에 대응하는 기존 _META.json")
    parser.add_argument("--output-dir", help="단일/배치 결과 루트")
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
    if args.split and (args.reference_meta or args.name):
        raise ValueError("--split batch 모드에서는 pair의 Meta와 파일명을 자동 사용하므로 --reference-meta/--name을 지정하지 마세요.")

    default_name = Path(args.input).stem if args.input else f"{args.split}_samples"
    output_root = Path(args.output_dir) if args.output_dir else Path(config["project"]["output_dir"]) / "predictions" / default_name
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logger("infer_visualize", output_root / "infer_visualize.log")
    logger = logging.getLogger("infer_visualize")

    if args.input:
        model, device = load_inference_model(config, args.checkpoint)
        stem = args.name or Path(args.input).stem
        paths = infer_and_visualize_one(
            config, args.checkpoint, args.input, args.reference_meta, output_root, stem, args.alpha, args.output_vector, model, device
        )
        logger.info("시각화 완료: %s", {key: str(path) for key, path in paths.items()})
        return

    pairs = select_sample_pairs(config, args.split, args.sample_count)
    logger.info("batch 추론 시작: split=%s requested=%d selected=%d", args.split, args.sample_count, len(pairs))
    model, device = load_inference_model(config, args.checkpoint)
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        stem = pair.image.stem
        sample_dir = output_root / f"{index:02d}_{stem}"
        paths = infer_and_visualize_one(
            config, args.checkpoint, pair.image, pair.meta_json, sample_dir, stem, args.alpha, args.output_vector, model, device
        )
        rows.append(
            {
                "index": index,
                "image": str(pair.image),
                "label_json": str(pair.label_json),
                "meta_json": str(pair.meta_json),
                "panel": str(paths["panel"]),
            }
        )
        logger.info("batch 진행: %d/%d %s", index, len(pairs), pair.image.name)
    manifest = output_root / "samples.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("batch 추론 완료: samples=%d manifest=%s", len(rows), manifest)


if __name__ == "__main__":
    main()
