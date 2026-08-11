"""Drop label-less (pure background) tiles from an existing dataset manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨 전경이 없는 타일을 manifest에서 제외")
    parser.add_argument("--root", required=True, help="타일 데이터셋 root")
    parser.add_argument("--label-dir", default="labels_crop")
    parser.add_argument("--min-foreground", type=float, default=0.01, help="전경(1~7, 255 제외) 비율 하한")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = root / "manifest.csv"
    backup = root / "manifest_full.csv"
    if not backup.exists():  # 원본 보존 — 재실행해도 전체 목록에서 다시 거른다
        shutil.copy2(manifest, backup)
    with backup.open(newline="", encoding="utf-8-sig") as stream:
        entries = list(csv.DictReader(stream))
    kept: list[dict[str, str]] = []
    for index, entry in enumerate(entries, start=1):
        with rasterio.open(root / args.label_dir / f"{entry['tile']}.tif") as source:
            label = source.read(1)
        if (((label > 0) & (label != 255)).mean()) >= args.min_foreground:
            kept.append(entry)
        if index % 5000 == 0:
            print(f"  {index}/{len(entries)} 검사, 유지 {len(kept)}", flush=True)
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tile", "split"])
        writer.writeheader()
        writer.writerows(kept)
    counts = {split: sum(1 for entry in kept if entry["split"] == split) for split in ("train", "val", "test")}
    print(f"{len(entries)} -> {len(kept)}타일 (제외 {len(entries) - len(kept)}) {counts}")


if __name__ == "__main__":
    main()
