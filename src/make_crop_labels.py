"""Rasterize farmmap crop-class labels with parcel boundaries for existing tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

# 팜맵 경지구분 코드 → 라벨 값. 7은 필지 경계 전용.
CLASS_BY_CODE = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6}
BOUNDARY_CLASS = 7


def main() -> None:
    parser = argparse.ArgumentParser(description="팜맵 경지구분 다중 클래스 타일 라벨 생성")
    parser.add_argument("--farmmap", required=True, nargs="+", help="팜맵 shapefile 경로 (시군별 여러 개 지정 가능)")
    parser.add_argument("--root", required=True, help="타일 데이터셋 root (images/ 포함)")
    parser.add_argument("--label-dir", default="labels_crop")
    parser.add_argument("--boundary-iterations", type=int, default=1, help="경계선 추가 팽창 횟수 (0=순수 에지 ~2px)")
    args = parser.parse_args()

    import pandas as pd

    frames = [gpd.read_file(path, columns=["CLSF_CD"]) for path in args.farmmap]
    if len({str(item.crs) for item in frames}) != 1:
        raise SystemExit(f"팜맵 CRS가 서로 다릅니다: {[str(item.crs) for item in frames]}")
    frame = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    frame["class_value"] = frame["CLSF_CD"].map(CLASS_BY_CODE)
    frame = frame.dropna(subset=["class_value"])
    class_of_id = np.zeros(len(frame) + 1, dtype=np.uint8)
    class_of_id[1:] = frame["class_value"].to_numpy(dtype=np.uint8)
    spatial_index = frame.sindex
    geometries = frame.geometry.to_numpy()

    root = Path(args.root)
    output_dir = root / args.label_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    structure = ndimage.generate_binary_structure(2, 2)
    images = sorted((root / "images").glob("*.tif"))
    for count, path in enumerate(images, start=1):
        with rasterio.open(path) as source:
            shape, transform, crs, bounds = source.shape, source.transform, source.crs, source.bounds
        candidates = list(spatial_index.intersection(bounds))
        if candidates:
            # 필지 ID로 래스터화해야 같은 클래스인 인접 필지 사이에도 경계가 생긴다.
            identifiers = rasterize(
                [(geometries[index], index + 1) for index in candidates],
                out_shape=shape, transform=transform, fill=0, dtype="int32",
            )
        else:
            identifiers = np.zeros(shape, dtype=np.int32)
        label = class_of_id[identifiers]
        parcel = identifiers > 0
        edges = parcel & (ndimage.maximum_filter(identifiers, 3) != ndimage.minimum_filter(identifiers, 3))
        if args.boundary_iterations:
            edges = ndimage.binary_dilation(edges, structure, iterations=args.boundary_iterations) & parcel
        label[edges] = BOUNDARY_CLASS
        profile = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1, "dtype": "uint8", "crs": crs, "transform": transform, "compress": "deflate", "nodata": None}
        with rasterio.open(output_dir / path.name, "w", **profile) as destination:
            destination.write(label, 1)
        if count % 1000 == 0:
            print(f"{count}/{len(images)}")
    print(output_dir)


if __name__ == "__main__":
    main()
