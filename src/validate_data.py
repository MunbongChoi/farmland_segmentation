"""Dataset integrity and geospatial-contract validation CLI."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from .datasets.dataset import RasterPair, discover_pairs, load_label_mask
from .utils.config import apply_overrides, load_config
from .utils.logger import setup_logger


def validate_pair(pair: RasterPair, config: dict[str, Any]) -> tuple[dict[str, Any], Counter[int]]:
    """Inspect one image/JSON/Meta triplet and return rasterized pixel counts."""
    result: dict[str, Any] = {"image": str(pair.image), "label_json": str(pair.label_json), "meta_json": str(pair.meta_json), "status": "ok", "issues": []}
    counts: Counter[int] = Counter()
    try:
        mask, metadata, feature_counts = load_label_mask(pair, config)
        with rasterio.open(pair.image) as image:
            result.update({
                "width": image.width,
                "height": image.height,
                "image_bands": image.count,
                "image_crs": str(image.crs) if image.crs else None,
                "label_crs": str(metadata.crs),
                "label_transform": tuple(metadata.transform),
                "resolution": metadata.resolution,
                "target_features": json.dumps(feature_counts, ensure_ascii=False),
            })
            if image.shape != (metadata.height, metadata.width):
                result["issues"].append("영상과 Meta JSON 크기 불일치")
            if image.count < int(config["input_channels"]):
                result["issues"].append("입력 채널 부족")
            if metadata.crs is None or metadata.transform.is_identity:
                result["issues"].append("JSON 공간정보 없음")
            values, frequencies = np.unique(mask, return_counts=True)
            counts.update({int(value): int(count) for value, count in zip(values, frequencies)})
            foreground = int((mask > 0).sum())
            result["foreground_fraction"] = foreground / max(1, metadata.width * metadata.height)
            if foreground == 0:
                result["issues"].append("논/밭 픽셀이 없는 마스크")
    except Exception as error:
        result["issues"].append(f"읽기 오류: {error}")
    if result["issues"]:
        result["status"] = "warning"
    result["issues"] = "; ".join(result["issues"])
    return result, counts


def write_reports(rows: list[dict[str, Any]], pixel_counts: Counter[int], output: Path) -> None:
    """Write per-file CSV and aggregate JSON reports."""
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / "files.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    total = sum(pixel_counts.values())
    summary = {
        "checked_pairs": len(rows),
        "warnings": sum(row["status"] != "ok" for row in rows),
        "rasterized_class_pixels": {str(key): value for key, value in sorted(pixel_counts.items())},
        "rasterized_class_ratios": {str(key): value / max(1, total) for key, value in sorted(pixel_counts.items())},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoTIFF 데이터 검증")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--max-samples", type=int, default=0, help="0이면 전체 검사")
    parser.add_argument("--output", default="outputs/data_validation")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    logger = setup_logger("validate_data", Path(args.output) / "validation.log")
    dataset = config["dataset"]
    split_config = dataset["train" if args.split == "train" else "validation"]
    pairs = discover_pairs(dataset["root_dir"], split_config, dataset["raw_class_map"], dataset.get("target_property", "ANN_CD"))
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    rows: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    for index, pair in enumerate(pairs, 1):
        row, pair_counts = validate_pair(pair, dataset)
        rows.append(row)
        counts.update(pair_counts)
        if row["status"] != "ok":
            logger.warning("%s: %s", pair.image, row["issues"])
        if index % 1000 == 0:
            logger.info("%d/%d 검사", index, len(pairs))
    write_reports(rows, counts, Path(args.output) / args.split)
    logger.info("검사 완료: pairs=%d warnings=%d", len(rows), sum(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    main()
