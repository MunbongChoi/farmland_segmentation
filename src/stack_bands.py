"""Stack single-band uint16 satellite bands into an 8-bit RGB(N) inference GeoTIFF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from rasterio.windows import Window


def band_percentiles(path: Path, low: float, high: float, sample_strips: int = 40, strip_rows: int = 256) -> tuple[float, float]:
    """Estimate stretch percentiles from sampled strips, ignoring zero nodata."""
    samples = []
    with rasterio.open(path) as source:
        step = max(strip_rows, source.height // sample_strips)
        for row in range(0, source.height, step):
            strip = source.read(1, window=Window(0, row, source.width, min(strip_rows, source.height - row)))
            nonzero = strip[strip > 0]
            if nonzero.size:
                samples.append(nonzero[:: max(1, nonzero.size // 200_000)])
    if not samples:
        raise ValueError(f"0이 아닌 픽셀이 없습니다: {path}")
    merged = np.concatenate(samples)
    return float(np.percentile(merged, low)), float(np.percentile(merged, high))


def main() -> None:
    parser = argparse.ArgumentParser(description="단일밴드 uint16 위성 밴드를 8비트 다중밴드 GeoTIFF로 통합")
    parser.add_argument("--red", required=True)
    parser.add_argument("--green", required=True)
    parser.add_argument("--blue", required=True)
    parser.add_argument("--nir", help="지정하면 4번째 밴드로 추가한다")
    parser.add_argument("--output", required=True)
    parser.add_argument("--low", type=float, default=2.0, help="스트레치 하한 퍼센타일")
    parser.add_argument("--high", type=float, default=98.0, help="스트레치 상한 퍼센타일")
    parser.add_argument("--resolution", type=float, default=0.5, help="지상 해상도 m/pixel")
    parser.add_argument("--crs", default="EPSG:5179")
    # 원본에 georeferencing이 없을 때 추론 파이프라인이 요구하는 좌표계를 채우는 자리표시자.
    # 실제 좌표를 알게 되면 --origin으로 좌상단 (x, y)를 지정한다.
    parser.add_argument("--origin", type=float, nargs=2, default=[1000000.0, 2000000.0], metavar=("X", "Y"))
    args = parser.parse_args()

    band_paths = [Path(args.red), Path(args.green), Path(args.blue)] + ([Path(args.nir)] if args.nir else [])
    sources = [rasterio.open(path) for path in band_paths]
    width, height = sources[0].width, sources[0].height
    if any((source.width, source.height) != (width, height) for source in sources):
        raise SystemExit("밴드 크기가 서로 다릅니다.")

    stretches = []
    for path in band_paths:
        low_value, high_value = band_percentiles(path, args.low, args.high)
        stretches.append((low_value, high_value))
        print(f"{path.name}: p{args.low:g}={low_value:.0f} p{args.high:g}={high_value:.0f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": len(sources), "dtype": "uint8",
        "crs": args.crs, "transform": from_origin(args.origin[0], args.origin[1], args.resolution, args.resolution),
        "compress": "deflate", "tiled": True, "blockxsize": 512, "blockysize": 512, "bigtiff": "if_safer", "nodata": 0,
    }
    strip = 512
    with rasterio.open(output, "w", **profile) as destination:
        interpretations = [ColorInterp.red, ColorInterp.green, ColorInterp.blue] + [ColorInterp.undefined] * (len(sources) - 3)
        destination.colorinterp = interpretations
        for row in range(0, height, strip):
            window = Window(0, row, width, min(strip, height - row))
            for index, (source, (low_value, high_value)) in enumerate(zip(sources, stretches), start=1):
                values = source.read(1, window=window).astype(np.float32)
                scaled = np.clip((values - low_value) / max(high_value - low_value, 1.0), 0.0, 1.0) * 254.0 + 1.0
                scaled[values <= 0] = 0.0  # keep zero as nodata
                destination.write(scaled.astype(np.uint8), index, window=window)
    for source in sources:
        source.close()
    print(output)


if __name__ == "__main__":
    main()
