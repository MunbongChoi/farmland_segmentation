"""Geospatial image-grid normalization for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform


@dataclass(frozen=True)
class RasterGrid:
    """A raster grid with an explicit CRS, affine transform, and dimensions."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def resolution(self) -> tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)


def parse_resampling(value: str) -> Resampling:
    """Resolve a safe continuous-image resampling algorithm."""
    methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
    }
    try:
        return methods[value.lower()]
    except KeyError as error:
        raise ValueError(f"지원하지 않는 리샘플링 방식입니다: {value}. 선택값={sorted(methods)}") from error


def _metres_per_crs_unit(crs: CRS) -> float:
    if not crs.is_projected:
        raise ValueError("25cm 해상도는 미터 단위 연산입니다. 투영 CRS를 --target-crs로 지정하세요.")
    _, factor = crs.linear_units_factor
    factor = float(factor)
    if factor <= 0:
        raise ValueError(f"CRS 선형 단위를 미터로 변환할 수 없습니다: {crs}")
    return factor


def make_target_grid(
    source_crs: CRS,
    source_transform: Affine,
    source_width: int,
    source_height: int,
    target_resolution_m: float,
    target_crs: str | CRS | None = None,
) -> RasterGrid:
    """Create a north-up target grid at an exact ground resolution in metres."""
    if target_resolution_m <= 0:
        raise ValueError("target_resolution_m은 0보다 커야 합니다.")
    destination_crs = CRS.from_user_input(target_crs) if target_crs else source_crs
    metres_per_unit = _metres_per_crs_unit(destination_crs)
    resolution_in_crs_units = target_resolution_m / metres_per_unit
    left, bottom, right, top = array_bounds(source_height, source_width, source_transform)
    transform, width, height = calculate_default_transform(
        source_crs,
        destination_crs,
        source_width,
        source_height,
        left,
        bottom,
        right,
        top,
        resolution=resolution_in_crs_units,
    )
    if width <= 0 or height <= 0:
        raise ValueError("목표 25cm 격자의 크기를 계산할 수 없습니다.")
    return RasterGrid(destination_crs, transform, width, height)


def open_resampled_vrt(
    source: rasterio.io.DatasetReader,
    source_crs: CRS,
    source_transform: Affine,
    target_resolution_m: float,
    target_crs: str | CRS | None = None,
    resampling: str = "bilinear",
) -> tuple[WarpedVRT, RasterGrid]:
    """Open a virtual raster aligned to the requested metre-based resolution."""
    grid = make_target_grid(source_crs, source_transform, source.width, source.height, target_resolution_m, target_crs)
    if (
        grid.crs == source_crs
        and grid.width == source.width
        and grid.height == source.height
        and grid.transform.almost_equals(source_transform, precision=1e-9)
    ):
        # 이미 목표 격자와 동일하다 — identity warp는 창 읽기마다 순수 오버헤드다.
        return source, RasterGrid(source_crs, source_transform, source.width, source.height)
    vrt_options: dict[str, Any] = {
        "src_crs": source_crs,
        "src_transform": source_transform,
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "resampling": parse_resampling(resampling),
    }
    if source.nodata is not None:
        vrt_options["src_nodata"] = source.nodata
        vrt_options["nodata"] = source.nodata
    return WarpedVRT(source, **vrt_options), grid
