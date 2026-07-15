"""Diagnose JSON-to-pixel alignment against an optional reference label raster."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from src.datasets.dataset import discover_pairs, load_label_mask
from src.utils.config import load_config


def intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON polygon과 기준 라벨 격자 정렬 진단")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--root", default="01.데이터")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    dataset["root_dir"] = args.root
    pairs = discover_pairs(args.root, dataset["train"], dataset["raw_class_map"], dataset["target_property"])
    reference_dir = Path(args.reference_dir)
    rows: list[tuple[str, float, float, int]] = []
    for pair in pairs[: args.samples]:
        mask, metadata, _ = load_label_mask(pair, dataset)
        reference_path = reference_dir / f"{pair.image.stem}.tif"
        with rasterio.open(reference_path) as reference:
            raw = reference.read(1)
            reference_mask = np.zeros_like(raw, dtype=np.uint8)
            reference_mask[raw == 50] = 1
            reference_mask[raw == 60] = 2
            print(
                f"{pair.image.name}: meta_transform={tuple(metadata.transform)} "
                f"reference_transform={tuple(reference.transform)} "
                f"meta_crs={metadata.crs.to_epsg()} reference_crs={reference.crs.to_epsg()}"
            )
        paddy_iou = intersection_over_union(mask == 1, reference_mask == 1)
        field_iou = intersection_over_union(mask == 2, reference_mask == 2)
        difference = int((mask != reference_mask).sum())
        rows.append((pair.image.name, paddy_iou, field_iou, difference))
        print(f"  paddy_iou={paddy_iou:.6f} field_iou={field_iou:.6f} different_pixels={difference}")

    print(
        "summary:",
        {
            "samples": len(rows),
            "mean_paddy_iou": float(np.mean([row[1] for row in rows])),
            "mean_field_iou": float(np.mean([row[2] for row in rows])),
            "mean_different_pixels": float(np.mean([row[3] for row in rows])),
        },
    )


if __name__ == "__main__":
    main()
