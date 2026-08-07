"""Merge tile dataset roots into one root via hardlinks and a combined manifest."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def link_tree(source: Path, destination: Path) -> int:
    """Hardlink every file from source into destination (copy if linking fails)."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.glob("*.tif")):
        target = destination / path.name
        if target.exists():
            raise SystemExit(f"타일 이름이 충돌합니다: {target}")
        try:
            os.link(path, target)
        except OSError:  # 다른 볼륨 등 하드링크 불가 시 복사로 대체
            shutil.copy2(path, target)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="타일 데이터셋 root 병합 (hardlink)")
    parser.add_argument("--roots", required=True, nargs="+", help="병합할 데이터셋 root 목록")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-dir", default="labels_crop")
    args = parser.parse_args()

    output = Path(args.output)
    rows: list[dict[str, str]] = []
    for root in (Path(item) for item in args.roots):
        images = link_tree(root / "images", output / "images")
        labels = link_tree(root / args.label_dir, output / args.label_dir)
        with (root / "manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
            manifest = list(csv.DictReader(stream))
        if images != len(manifest) or labels != len(manifest):
            raise SystemExit(f"{root}: manifest {len(manifest)}개와 타일 수(images={images}, labels={labels})가 다릅니다.")
        rows.extend(manifest)
        print(f"{root}: {images} tiles")
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tile", "split"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")}
    print(f"merged={len(rows)} {counts} -> {output}")


if __name__ == "__main__":
    main()
