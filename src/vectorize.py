"""Raster-instance polygonization in an isolated geospatial process."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def vectorize(instance_raster: str | Path, class_raster: str | Path, output: str | Path) -> None:
    """Convert connected-component IDs to class-attributed polygons."""
    with rasterio.open(instance_raster) as instance_source, rasterio.open(class_raster) as class_source:
        if instance_source.shape != class_source.shape or instance_source.transform != class_source.transform:
            raise ValueError("instance/class raster의 크기 또는 Transform이 다릅니다.")
        if instance_source.crs is None:
            raise ValueError("instance raster에 CRS가 없습니다.")
        # shapes()는 uint32를 받지 않는다. 인스턴스 수는 int32 범위를 넘지 않는다.
        instances = instance_source.read(1).astype("int32")
        classes = class_source.read(1)
        records = []
        for geometry, value in shapes(instances, mask=instances > 0, transform=instance_source.transform, connectivity=8):
            geometry_object = shape(geometry)
            point = geometry_object.representative_point()
            row, col = rasterio.transform.rowcol(instance_source.transform, point.x, point.y)
            row = min(max(row, 0), classes.shape[0] - 1)
            col = min(max(col, 0), classes.shape[1] - 1)
            records.append({"instance_id": int(value), "class_id": int(classes[row, col]), "geometry": geometry_object})
        crs = instance_source.crs
    if records:
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    else:
        frame = gpd.GeoDataFrame(
            {"instance_id": pd.Series(dtype="int64"), "class_id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs=crs),
        )
    if crs.is_projected:
        frame["area_m2"] = frame.geometry.area
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    driver = {".gpkg": "GPKG", ".geojson": "GeoJSON", ".json": "GeoJSON", ".shp": "ESRI Shapefile"}.get(destination.suffix.lower())
    if driver is None:
        raise ValueError("벡터 출력 확장자는 .gpkg, .geojson 또는 .shp여야 합니다.")
    options = {"driver": driver}
    if driver == "GPKG":
        options["layer"] = destination.stem
    frame.to_file(destination, **options)


def main() -> None:
    parser = argparse.ArgumentParser(description="인스턴스 래스터 벡터화")
    parser.add_argument("--instances", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    vectorize(args.instances, args.classes, args.output)


if __name__ == "__main__":
    main()
