"""Fetch VWorld WMTS aerial tiles into a georeferenced EPSG:5179 GeoTIFF mosaic."""

from __future__ import annotations

import argparse
import io
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

TILE = 256
WEB_MERCATOR_ORIGIN = 20037508.342789244


def mercator_resolution(zoom: int) -> float:
    return 2 * WEB_MERCATOR_ORIGIN / (TILE * 2**zoom)


def main() -> None:
    parser = argparse.ArgumentParser(description="VWorld 항공영상 WMTS → EPSG:5179 GeoTIFF")
    parser.add_argument("--key", required=True, help="VWorld API 키")
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="EPSG:5179 범위")
    parser.add_argument("--output", required=True)
    parser.add_argument("--zoom", type=int, default=18, help="18≈0.5m/px (위도 36도 기준)")
    parser.add_argument("--resolution", type=float, default=0.5, help="출력 해상도 m/pixel")
    parser.add_argument("--layer", default="Satellite")
    parser.add_argument("--sleep", type=float, default=0.05, help="타일 요청 간격(초) — 쿼터 보호")
    args = parser.parse_args()

    import requests

    # AOI를 Web Mercator로 투영해 타일 범위 계산
    to_mercator = Transformer.from_crs("EPSG:5179", "EPSG:3857", always_xy=True)
    corners = [to_mercator.transform(x, y) for x, y in
               ((args.bbox[0], args.bbox[1]), (args.bbox[0], args.bbox[3]), (args.bbox[2], args.bbox[1]), (args.bbox[2], args.bbox[3]))]
    xs, ys = zip(*corners)
    resolution = mercator_resolution(args.zoom)
    tile_span = TILE * resolution
    tx0 = int((min(xs) + WEB_MERCATOR_ORIGIN) // tile_span)
    tx1 = int((max(xs) + WEB_MERCATOR_ORIGIN) // tile_span)
    ty0 = int((WEB_MERCATOR_ORIGIN - max(ys)) // tile_span)
    ty1 = int((WEB_MERCATOR_ORIGIN - min(ys)) // tile_span)
    columns, rows = tx1 - tx0 + 1, ty1 - ty0 + 1
    print(f"zoom={args.zoom} ({resolution:.3f}m/px) 타일 {columns}x{rows} = {columns * rows}개", flush=True)

    mosaic = np.zeros((3, rows * TILE, columns * TILE), dtype=np.uint8)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "http://localhost/"})
    fetched = failed = consecutive = 0
    for row in range(rows):
        for column in range(columns):
            url = f"https://api.vworld.kr/req/wmts/1.0.0/{args.key}/{args.layer}/{args.zoom}/{ty0 + row}/{tx0 + column}.jpeg"
            tile = None
            for attempt in range(3):  # 일시 오류는 지수 백오프로 재시도
                try:
                    response = session.get(url, timeout=20)
                    response.raise_for_status()
                    tile = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
                    break
                except Exception as error:
                    if attempt == 2 and failed < 5:
                        print(f"타일 실패 ({ty0 + row},{tx0 + column}): {error}", flush=True)
                    time.sleep(2**attempt)
            if tile is None:
                failed += 1
                consecutive += 1
                if consecutive >= 20:
                    raise SystemExit("연속 20타일 실패 — VWorld 서버 장애(503)이거나 쿼터 초과입니다. 잠시 후 다시 실행하세요.")
            else:
                mosaic[:, row * TILE : (row + 1) * TILE, column * TILE : (column + 1) * TILE] = np.moveaxis(tile, -1, 0)
                fetched += 1
                consecutive = 0
            time.sleep(args.sleep)
        print(f"  행 {row + 1}/{rows} (수신 {fetched}, 실패 {failed})", flush=True)
    if fetched == 0:
        raise SystemExit("타일을 하나도 받지 못했습니다 — API 키/쿼터/레이어를 확인하세요.")

    # Web Mercator 모자이크 → EPSG:5179 재투영
    mercator_transform = from_origin(tx0 * tile_span - WEB_MERCATOR_ORIGIN, WEB_MERCATOR_ORIGIN - ty0 * tile_span, resolution, resolution)
    x0, y0, x1, y1 = args.bbox
    width = math.ceil((x1 - x0) / args.resolution)
    height = math.ceil((y1 - y0) / args.resolution)
    destination = np.zeros((3, height, width), dtype=np.uint8)
    output_transform = from_origin(x0, y1, args.resolution, args.resolution)
    reproject(
        mosaic, destination,
        src_transform=mercator_transform, src_crs="EPSG:3857",
        dst_transform=output_transform, dst_crs="EPSG:5179",
        resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": height, "width": width, "count": 3, "dtype": "uint8",
               "crs": "EPSG:5179", "transform": output_transform, "compress": "deflate",
               "tiled": True, "blockxsize": 512, "blockysize": 512, "bigtiff": "if_safer", "nodata": 0}
    with rasterio.open(output, "w", **profile) as dataset:
        dataset.write(destination)
    print(f"{output} ({width}x{height} @ {args.resolution}m, 수신 {fetched}/{columns * rows})")


if __name__ == "__main__":
    main()
