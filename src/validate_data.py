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

from .datasets.dataset import RasterPair, discover_pairs
from .utils.config import apply_overrides, load_config
from .utils.logger import setup_logger


def validate_pair(pair: RasterPair, config: dict[str, Any]) -> tuple[dict[str, Any], Counter[int]]:
    """Inspect one image/mask pair and return a report plus raw pixel counts."""
    result: dict[str, Any] = {"image": str(pair.image), "mask": str(pair.mask), "status": "ok", "issues": []}
    counts: Counter[int] = Counter()
    try:
        with rasterio.open(pair.image) as image, rasterio.open(pair.mask) as mask:
            result.update({
                "width": image.width,
                "height": image.height,
                "image_bands": image.count,
                "image_crs": str(image.crs) if image.crs else None,
                "mask_crs": str(mask.crs) if mask.crs else None,
                "mask_transform": tuple(mask.transform),
                "resolution": tuple(abs(value) for value in mask.res),
            })
            if image.shape != mask.shape:
                result["issues"].append("영상과 마스크 크기 불일치")
            if image.count < int(config["input_channels"]):
                result["issues"].append("입력 채널 부족")
            if mask.count != 1:
                result["issues"].append("마스크가 단일 밴드가 아님")
            if mask.crs is None or mask.transform.is_identity:
                result["issues"].append("마스크 공간정보 없음")
            values, frequencies = np.unique(mask.read(1), return_counts=True)
            counts.update({int(value): int(count) for value, count in zip(values, frequencies)})
            allowed = set(int(value) for value in config.get("allowed_raw_values", config["raw_class_map"]))
            invalid = sorted(int(value) for value in values if int(value) not in allowed)
            result["invalid_raw_values"] = invalid
            if invalid:
                result["issues"].append(f"허용되지 않은 클래스 값: {invalid}")
            foreground = sum(counts[value] for value in config["raw_class_map"])
            result["foreground_fraction"] = foreground / max(1, mask.width * mask.height)
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
        "raw_class_pixels": {str(key): value for key, value in sorted(pixel_counts.items())},
        "raw_class_ratios": {str(key): value / max(1, total) for key, value in sorted(pixel_counts.items())},
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
    pairs = discover_pairs(dataset["root_dir"], split_config)
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
