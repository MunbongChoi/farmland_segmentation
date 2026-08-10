"""Merge per-scene tiles into single GeoTIFFs and optionally run full inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import rasterio
from rasterio.enums import ColorInterp
from rasterio.merge import merge

from .utils.config import apply_overrides, load_config
from .utils.logger import setup_logger


def merge_scene(paths: list[Path], output: Path) -> Path:
    """Mosaic georeferenced tiles of one scene into a single compressed GeoTIFF."""
    with rasterio.open(paths[0]) as first:
        profile = first.profile
    # Paths (not open datasets) so rasterio opens them one at a time.
    mosaic, transform = merge([str(path) for path in paths], nodata=0)
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, compress="deflate", tiled=True, blockxsize=512, blockysize=512, bigtiff="if_safer")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp name first so a crash never leaves a reusable corrupt mosaic.
    temporary = output.with_name(output.name + ".tmp")
    with rasterio.open(temporary, "w", driver="GTiff", **{key: value for key, value in profile.items() if key != "driver"}) as destination:
        destination.write(mosaic)
        if destination.count == 4:
            # Source tiles mislabel NIR as alpha, which viewers render as transparency.
            destination.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined]
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="타일을 장면별 단일 영상으로 통합하고 선택적으로 추론")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="지정하면 각 모자이크에 슬라이딩 윈도우 추론까지 실행한다")
    parser.add_argument("--scene", help="이 장면만 처리한다 (기본: 전체)")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    images = Path(config["dataset"]["root_dir"]).expanduser() / "images"
    output_dir = Path(config["project"]["output_dir"]) / "mosaics"
    logger = setup_logger("mosaic", output_dir / "mosaic.log")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(images.glob("*.tif")):
        groups[path.stem.rsplit("_", 2)[0]].append(path)
    if not groups:
        raise SystemExit(f"타일을 찾을 수 없습니다: {images}")
    if args.scene:
        if args.scene not in groups:
            raise SystemExit(f"장면을 찾을 수 없습니다: {args.scene} (가능: {sorted(groups)})")
        groups = {args.scene: groups[args.scene]}

    model = device = None
    if args.checkpoint:
        # Torch import stays out of the mosaic-only path.
        from .infer import load_inference_model

        model, device = load_inference_model(config, args.checkpoint)
    for scene, paths in groups.items():
        mosaic_path = output_dir / f"{scene}.tif"
        if mosaic_path.exists():
            logger.info("모자이크 재사용: %s", mosaic_path)
        else:
            logger.info("모자이크 생성: %s (tiles=%d)", scene, len(paths))
            merge_scene(paths, mosaic_path)
        if model is not None:
            mask_path = mosaic_path.with_name(f"{scene}_mask.tif")
            instance_path = mask_path.with_name(f"{mask_path.stem}_instances.tif")
            vector = mosaic_path.with_name(f"{scene}_parcels.gpkg") if config.get("output", {}).get("save_vector", True) else None
            # Instances are the last raster written, so their presence marks a finished inference.
            if mask_path.exists() and (vector is None or instance_path.exists()):
                logger.info("추론 결과 재사용: %s", mask_path)
                if vector is not None and not vector.exists():
                    command = [sys.executable, "-m", "src.vectorize", "--instances", str(instance_path), "--classes", str(mask_path), "--output", str(vector)]
                    class_names = config["dataset"].get("class_names")
                    if class_names:
                        command += ["--class-names", ",".join(str(name) for name in class_names)]
                    simplify = float(config.get("output", {}).get("vector_simplify_m", 0.0))
                    if simplify > 0:
                        command += ["--simplify", str(simplify)]
                    smooth = float(config.get("output", {}).get("vector_smooth_px", 0.0))
                    if smooth > 0:
                        command += ["--smooth", str(smooth)]
                    subprocess.run(command, check=True)
                continue
            from .infer import run_inference

            run_inference(config, args.checkpoint, str(mosaic_path), str(mask_path), output_vector=str(vector) if vector else None, model=model, device=device)


if __name__ == "__main__":
    main()
