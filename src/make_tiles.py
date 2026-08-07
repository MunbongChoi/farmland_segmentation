"""Cut a georeferenced mosaic into 512px training tiles with a block-split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform


def block_split(row: int, col: int, tiles_per_block: int, seed: int) -> str:
    """Assign a spatial block of tiles to a split (80/10/10, deterministic)."""
    key = f"{seed}:{row // tiles_per_block}:{col // tiles_per_block}"
    value = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
    return "train" if value < 8 else "val" if value == 8 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="모자이크 GeoTIFF를 512px 학습 타일로 분할")
    parser.add_argument("--image", required=True, help="georeferenced 다중밴드 모자이크")
    parser.add_argument("--output", required=True, help="타일 데이터셋 root (images/, manifest.csv 생성)")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--min-valid", type=float, default=0.5, help="nodata가 아닌 픽셀 비율 하한")
    parser.add_argument("--block-tiles", type=int, default=8, help="split 블록 한 변의 타일 수 (공간 누수 방지)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    size = args.tile_size
    rows: list[tuple[str, str]] = []
    with rasterio.open(args.image) as source:
        scene = Path(args.image).stem.rsplit("_", 1)[0]
        for row in range(0, source.height - size + 1, size):
            for col in range(0, source.width - size + 1, size):
                window = Window(col, row, size, size)
                data = source.read(window=window)
                if (data.max(axis=0) > 0).mean() < args.min_valid:
                    continue
                name = f"{scene}_r{row:05d}_c{col:05d}"
                profile = source.profile.copy()
                profile.update(width=size, height=size, transform=transform(window, source.transform), compress="deflate", tiled=False)
                with rasterio.open(output / "images" / f"{name}.tif", "w", **profile) as destination:
                    destination.write(data)
                rows.append((name, block_split(row // size, col // size, args.block_tiles, args.seed)))
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tile", "split"])
        writer.writerows(rows)
    counts = {split: sum(1 for _, s in rows if s == split) for split in ("train", "val", "test")}
    print(f"tiles={len(rows)} {counts} -> {output}")


if __name__ == "__main__":
    main()
