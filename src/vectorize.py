"""Raster-instance polygonization in an isolated geospatial process."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon, shape


def _smooth_instance_polygon(instances: "np.ndarray", instance_id: int, window: tuple[slice, slice], transform_value, sigma: float):
    """Sub-pixel outline via Gaussian smoothing + marching-squares 0.5 iso-contour."""
    from scipy import ndimage
    from skimage import measure

    margin = int(3 * sigma) + 2
    rows, cols = window
    row_start = max(0, rows.start - margin)
    col_start = max(0, cols.start - margin)
    crop = instances[row_start : rows.stop + margin, col_start : cols.stop + margin] == instance_id
    smooth = ndimage.gaussian_filter(crop.astype("float32"), sigma)
    contours = [contour for contour in measure.find_contours(smooth, 0.5) if len(contour) >= 4]
    if not contours:
        return None
    # ponytail: 외곽 링만 사용 — 필지 내부 구멍은 드물어 그리지 않는다 (기존 동작과 동일).
    contour = max(contours, key=len)
    points = [transform_value * (col_start + column + 0.5, row_start + row + 0.5) for row, column in contour]
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return None if polygon.is_empty else polygon


def vectorize(instance_raster: str | Path, class_raster: str | Path, output: str | Path, class_names: list[str] | None = None, simplify: float = 0.0, smooth: float = 0.0, drop_classes: frozenset[int] = frozenset()) -> None:
    """Convert connected-component IDs to class-attributed polygons."""
    with rasterio.open(instance_raster) as instance_source, rasterio.open(class_raster) as class_source:
        if instance_source.shape != class_source.shape or instance_source.transform != class_source.transform:
            raise ValueError("instance/class raster의 크기 또는 Transform이 다릅니다.")
        if instance_source.crs is None:
            raise ValueError("instance raster에 CRS가 없습니다.")
        # shapes()는 uint32를 받지 않는다. 인스턴스 수는 int32 범위를 넘지 않는다.
        instances = instance_source.read(1).astype("int32")
        classes = class_source.read(1)
        windows = {}
        if smooth > 0:
            from scipy import ndimage

            windows = {index: window for index, window in enumerate(ndimage.find_objects(instances), start=1) if window is not None}
        records = []
        smoothed_ids: set[int] = set()
        for geometry, value in shapes(instances, mask=instances > 0, transform=instance_source.transform, connectivity=8):
            geometry_object = shape(geometry)
            # 클래스 조회는 단순화 전 원본 도형의 대표점으로 해야 안전하다.
            point = geometry_object.representative_point()
            row, col = rasterio.transform.rowcol(instance_source.transform, point.x, point.y)
            row = min(max(row, 0), classes.shape[0] - 1)
            col = min(max(col, 0), classes.shape[1] - 1)
            if int(classes[row, col]) in drop_classes:
                continue  # 경계 등 산출 제외 클래스 — erase_boundary 이전 래스터 재벡터화 안전망
            if smooth > 0 and int(value) in windows:
                if int(value) in smoothed_ids:
                    continue  # 한 인스턴스가 여러 조각으로 나와도 평활 폴리곤은 한 번만
                smoothed_ids.add(int(value))
                smoothed = _smooth_instance_polygon(instances, int(value), windows[int(value)], instance_source.transform, smooth)
                if smoothed is not None:
                    geometry_object = smoothed
            if simplify > 0:
                simplified = geometry_object.simplify(simplify, preserve_topology=True)
                if not simplified.is_empty:
                    geometry_object = simplified
            records.append({"instance_id": int(value), "class_id": int(classes[row, col]), "geometry": geometry_object})
        crs = instance_source.crs
    if records:
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    else:
        frame = gpd.GeoDataFrame(
            {"instance_id": pd.Series(dtype="int64"), "class_id": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs=crs),
        )
    if class_names:
        frame["class_name"] = frame["class_id"].map(lambda index: class_names[index] if 0 <= index < len(class_names) else str(index))
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
    parser.add_argument(
        "--class-names",
        default="배경,논,밭,과수,시설,인삼,비경지,필지 경계",
        help="쉼표로 구분한 클래스 이름 목록 (class_id 순서). 기본=8클래스 경지구분. 3클래스 결과는 직접 지정",
    )
    parser.add_argument("--simplify", type=float, default=0.0, help="Douglas-Peucker 허용 오차(m). 0이면 픽셀 계단 그대로")
    parser.add_argument("--smooth", type=float, default=0.0, help="서브픽셀 평활 가우시안 sigma(px). marching squares 등고선 폴리곤화, 1.5 권장. 0=끔")
    parser.add_argument("--drop-classes", default="7", help="폴리곤으로 내보내지 않을 class_id (쉼표 구분). 기본=7(필지 경계). 빈 문자열=전부 유지")
    args = parser.parse_args()
    drop = frozenset(int(value) for value in args.drop_classes.split(",") if value.strip())
    vectorize(args.instances, args.classes, args.output, args.class_names.split(",") if args.class_names else None, args.simplify, args.smooth, drop)


if __name__ == "__main__":
    main()
