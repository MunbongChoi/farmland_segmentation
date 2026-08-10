"""Cut a georeferenced mosaic into 512px training tiles with a block-split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform


def block_split(block_row: int, block_col: int, seed: int) -> str:
    """Assign a spatial block to a split (80/10/10, deterministic)."""
    value = int(hashlib.md5(f"{seed}:{block_row}:{block_col}".encode()).hexdigest(), 16) % 10
    return "train" if value < 8 else "val" if value == 8 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="모자이크 GeoTIFF를 512px 학습 타일로 분할")
    parser.add_argument("--image", required=True, help="georeferenced 다중밴드 모자이크")
    parser.add_argument("--output", required=True, help="타일 데이터셋 root (images/, manifest.csv 생성)")
    parser.add_argument("--label-image", help="전장면 라벨 래스터. 지정하면 타일과 같은 창으로 잘라 label-dir에 저장")
    parser.add_argument("--label-dir", default="labels_crop")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, help="타일 간격(px). 기본=tile-size(겹침 없음). 작게 주면 겹침 타일 추가")
    parser.add_argument("--min-valid", type=float, default=0.5, help="nodata가 아닌 픽셀 비율 하한")
    parser.add_argument("--block-tiles", type=int, default=8, help="split 블록 한 변의 타일 수 (공간 누수 방지)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    size = args.tile_size
    stride = args.stride or size
    if not 0 < stride <= size:
        raise SystemExit("stride는 1 이상 tile-size 이하여야 합니다.")
    block_px = size * args.block_tiles
    rows: list[tuple[str, str]] = []
    label_source = None
    with rasterio.open(args.image) as source:
        if args.label_image:
            label_source = rasterio.open(args.label_image)
            if label_source.shape != source.shape or label_source.transform != source.transform:
                raise SystemExit("--label-image의 크기/Transform이 영상과 다릅니다.")
            (output / args.label_dir).mkdir(parents=True, exist_ok=True)
        scene = Path(args.image).stem.rsplit("_", 1)[0]
        for row in range(0, source.height - size + 1, stride):
            for col in range(0, source.width - size + 1, stride):
                on_grid = row % size == 0 and col % size == 0
                if on_grid:
                    split = block_split(row // block_px, col // block_px, args.seed)
                else:
                    # 겹침 타일이 val/test 블록에 걸치면 공간 누수가 생기므로 train 블록 내부에서만 뽑는다.
                    corner_splits = {
                        block_split(edge_row // block_px, edge_col // block_px, args.seed)
                        for edge_row in (row, row + size - 1)
                        for edge_col in (col, col + size - 1)
                    }
                    if corner_splits != {"train"}:
                        continue
                    split = "train"
                window = Window(col, row, size, size)
                data = source.read(window=window)
                if (data.max(axis=0) > 0).mean() < args.min_valid:
                    continue
                name = f"{scene}_r{row:05d}_c{col:05d}"
                profile = source.profile.copy()
                profile.update(width=size, height=size, transform=transform(window, source.transform), compress="deflate", tiled=False)
                with rasterio.open(output / "images" / f"{name}.tif", "w", **profile) as destination:
                    destination.write(data)
                if label_source is not None:
                    label_profile = label_source.profile.copy()
                    label_profile.update(width=size, height=size, transform=transform(window, source.transform), compress="deflate", tiled=False)
                    with rasterio.open(output / args.label_dir / f"{name}.tif", "w", **label_profile) as destination:
                        destination.write(label_source.read(window=window))
                rows.append((name, split))
    if label_source is not None:
        label_source.close()
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tile", "split"])
        writer.writerows(rows)
    counts = {split: sum(1 for _, s in rows if s == split) for split in ("train", "val", "test")}
    print(f"tiles={len(rows)} {counts} -> {output}")


if __name__ == "__main__":
    main()
