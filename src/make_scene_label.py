"""Rasterize farmmap crop classes over a whole scene mosaic (strip-based, RAM-safe)."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy import ndimage

from .make_crop_labels import BOUNDARY_CLASS, CLASS_BY_CODE

STRIP, HALO = 4096, 8
IGNORE = 255


def main() -> None:
    parser = argparse.ArgumentParser(description="장면 모자이크 전체의 팜맵 8클래스 GT 래스터 생성")
    parser.add_argument("--image", required=True, help="georeferenced 장면 모자이크")
    parser.add_argument("--farmmap", required=True, nargs="+", help="팜맵 shapefile (여러 시군 가능)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-iterations", type=int, default=1)
    parser.add_argument("--nodata-ignore", type=int, default=0, help="영상 전밴드가 이 값인 픽셀을 255로 마스킹. 음수면 끔")
    args = parser.parse_args()

    with rasterio.open(args.image) as source:
        height, width, transform, crs, bounds = source.height, source.width, source.transform, source.crs, source.bounds
    pixel = abs(transform.a)

    frames = []
    for pattern in args.farmmap:
        for shp in sorted(glob.glob(pattern)) or [pattern]:
            frame = gpd.read_file(shp, columns=["CLSF_CD"], bbox=tuple(bounds))
            if len(frame):
                frames.append(frame)
            print(Path(shp).stem, len(frame))
    if not frames:
        raise SystemExit("장면 범위와 겹치는 팜맵 필지가 없습니다.")
    farm = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    farm["class_value"] = farm["CLSF_CD"].map(CLASS_BY_CODE)
    farm = farm.dropna(subset=["class_value"]).reset_index(drop=True)
    class_of = np.zeros(len(farm) + 1, dtype=np.uint8)
    class_of[1:] = farm["class_value"].to_numpy(dtype=np.uint8)
    spatial_index = farm.sindex
    geometries = farm.geometry.to_numpy()

    profile = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": "uint8",
               "crs": crs, "transform": transform, "compress": "deflate", "tiled": True,
               "blockxsize": 512, "blockysize": 512, "bigtiff": "if_safer"}
    structure = ndimage.generate_binary_structure(2, 2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.image) as image, rasterio.open(output, "w", **profile) as destination:
        for row0 in range(0, height, STRIP):
            core = min(STRIP, height - row0)
            top = max(0, row0 - HALO)
            rows = min(height, row0 + core + HALO) - top
            y1 = bounds.top - top * pixel
            strip_transform = from_origin(bounds.left, y1, pixel, pixel)
            candidates = list(spatial_index.intersection((bounds.left, y1 - rows * pixel, bounds.right, y1)))
            if candidates:
                # 필지 ID로 래스터화해야 같은 클래스인 인접 필지 사이에도 경계가 생긴다.
                identifiers = rasterize(
                    [(geometries[index], index + 1) for index in candidates],
                    out_shape=(rows, width), transform=strip_transform, fill=0, dtype="int32")
            else:
                identifiers = np.zeros((rows, width), dtype=np.int32)
            label = class_of[identifiers]
            parcel = identifiers > 0
            edges = parcel & (ndimage.maximum_filter(identifiers, 3) != ndimage.minimum_filter(identifiers, 3))
            if args.boundary_iterations:
                edges = ndimage.binary_dilation(edges, structure, iterations=args.boundary_iterations) & parcel
            label[edges] = BOUNDARY_CLASS
            offset = row0 - top
            label_core = label[offset : offset + core]
            if args.nodata_ignore >= 0:
                invalid = image.read(window=Window(0, row0, width, core)).max(axis=0) == args.nodata_ignore
                label_core[invalid] = IGNORE
            destination.write(label_core, 1, window=Window(0, row0, width, core))
            print(f"strip {row0}-{row0 + core} ({len(candidates)} parcels)")
    print(output)


if __name__ == "__main__":
    main()
