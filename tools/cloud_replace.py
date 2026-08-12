"""Replace cloudy pixels in one georeferenced RGB raster from older CAS500 bands.

The donor GeoTIFFs in this workflow have lost their georeferencing tags, so this
tool estimates a donor-to-target image transform from visual features and then
warps/radiometrically matches the donor before cloud-only compositing.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling


def stretched_preview(paths: list[Path], max_size: int = 2200) -> tuple[np.ndarray, list[tuple[float, float]], tuple[int, int]]:
    with rasterio.open(paths[0]) as source:
        scale = min(1.0, max_size / max(source.width, source.height))
        shape = (max(1, round(source.height * scale)), max(1, round(source.width * scale)))
    bands = []
    stretches = []
    for path in paths:
        with rasterio.open(path) as source:
            values = source.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
        valid = values > 0
        low, high = np.percentile(values[valid], (2, 98))
        stretches.append((float(low), float(high)))
        bands.append(np.clip((values - low) / max(high - low, 1), 0, 1))
    return np.moveaxis(np.stack(bands), 0, -1), stretches, (source.height, source.width)


def estimate_homography(donor: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int, float]:
    detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.015)
    donor_gray = cv2.cvtColor((donor * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    key_d, desc_d = detector.detectAndCompute(donor_gray, None)
    key_t, desc_t = detector.detectAndCompute(target_gray, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_d, desc_t, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(good) < 20:
        raise RuntimeError(f"정합 특징점 부족: {len(good)}")
    src = np.float32([key_d[item.queryIdx].pt for item in good])
    dst = np.float32([key_t[item.trainIdx].pt for item in good])
    matrix, inliers = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, maxIters=10000, confidence=0.999)
    if matrix is None or inliers is None:
        raise RuntimeError("homography 추정 실패")
    count = int(inliers.sum())
    error = np.linalg.norm(cv2.perspectiveTransform(src[inliers.ravel() > 0, None], matrix)[:, 0] - dst[inliers.ravel() > 0], axis=1)
    return matrix, count, float(np.median(error))


def clean_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(count, np.uint8)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    return keep[labels] > 0


def color_luts(donor: np.ndarray, target: np.ndarray, valid: np.ndarray, excluded: np.ndarray) -> list[np.ndarray]:
    sample = valid & ~excluded & (target.max(2) > 15) & (donor.max(2) > 15)
    luts = []
    for channel in range(3):
        source_values = donor[..., channel][sample].astype(np.float32)
        target_values = target[..., channel][sample].astype(np.float32)
        quantiles = (0, 2, 10, 25, 50, 75, 90, 98, 100)
        source_q = np.percentile(source_values, quantiles)
        target_q = np.percentile(target_values, quantiles)
        source_q = np.maximum.accumulate(source_q + np.arange(len(source_q)) * 1e-3)
        luts.append(np.interp(np.arange(256), source_q, target_q).clip(0, 255).astype(np.uint8))
    return luts


def match_colors(donor: np.ndarray, luts: list[np.ndarray]) -> np.ndarray:
    result = np.empty_like(donor)
    for channel in range(3):
        result[..., channel] = luts[channel][donor[..., channel]]
    return result


def full_homography(matrix: np.ndarray, donor_shape: tuple[int, int], donor_preview_shape: tuple[int, int], target_shape: tuple[int, int], target_preview_shape: tuple[int, int]) -> np.ndarray:
    donor_scale_x = donor_preview_shape[1] / donor_shape[1]
    donor_scale_y = donor_preview_shape[0] / donor_shape[0]
    target_scale_x = target_preview_shape[1] / target_shape[1]
    target_scale_y = target_preview_shape[0] / target_shape[0]
    donor_to_preview = np.diag([donor_scale_x, donor_scale_y, 1.0])
    preview_to_target = np.diag([1.0 / target_scale_x, 1.0 / target_scale_y, 1.0])
    return preview_to_target @ matrix @ donor_to_preview


def write_full_composite(target_path: Path, output: Path, scenes: list[dict]) -> None:
    """Windowed full-resolution warp/composite with bounded RAM."""
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_path, output)
    with rasterio.open(output, "r+") as destination:
            target_h, target_w = destination.height, destination.width
            preview_h, preview_w = scenes[0]["mask"].shape
            xp_full = np.minimum(preview_w - 1, (np.arange(target_w) * preview_w / target_w).astype(np.int32))
            for row in range(0, target_h, 256):
                height = min(256, target_h - row)
                yp = np.minimum(preview_h - 1, (np.arange(row, row + height) * preview_h / target_h).astype(np.int32))
                combined = np.zeros((height, target_w), bool)
                for scene in scenes:
                    combined |= scene["mask"][yp[:, None], xp_full[None, :]]
                occupied_columns = np.flatnonzero(combined.any(axis=0))
                if not len(occupied_columns):
                    continue
                col_start = max(0, int(occupied_columns[0]) - 64)
                col_end = min(target_w, int(occupied_columns[-1]) + 65)
                width = col_end - col_start
                window = rasterio.windows.Window(col_start, row, width, height)
                target_rgb = np.moveaxis(destination.read((1, 2, 3), window=window), 0, -1)
                composite = target_rgb.copy()
                xx, yy = np.meshgrid(np.arange(col_start, col_end, dtype=np.float32), np.arange(row, row + height, dtype=np.float32))
                homogeneous = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
                for scene in scenes:
                    inverse = scene["inverse_full"]
                    mapped = homogeneous @ inverse.T
                    donor_x = mapped[..., 0] / mapped[..., 2]
                    donor_y = mapped[..., 1] / mapped[..., 2]
                    donor_h, donor_w = scene["donor_shape"]
                    valid_geometry = (donor_x >= 0) & (donor_x < donor_w - 1) & (donor_y >= 0) & (donor_y < donor_h - 1)
                    if not valid_geometry.any():
                        continue
                    xmin = max(0, int(np.floor(donor_x[valid_geometry].min())) - 2)
                    xmax = min(donor_w, int(np.ceil(donor_x[valid_geometry].max())) + 3)
                    ymin = max(0, int(np.floor(donor_y[valid_geometry].min())) - 2)
                    ymax = min(donor_h, int(np.ceil(donor_y[valid_geometry].max())) + 3)
                    local_x = (donor_x - xmin).astype(np.float32)
                    local_y = (donor_y - ymin).astype(np.float32)
                    warped_channels = []
                    donor_valid = np.zeros(valid_geometry.shape, bool)
                    for path, (low, high) in zip(scene["paths"], scene["stretches"]):
                        with rasterio.open(path) as donor_source:
                            raw = donor_source.read(1, window=rasterio.windows.Window(xmin, ymin, xmax - xmin, ymax - ymin))
                        warped_raw = cv2.remap(raw, local_x, local_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        donor_valid |= warped_raw > 0
                        scaled = np.clip((warped_raw.astype(np.float32) - low) * (255.0 / max(high - low, 1)), 0, 255).astype(np.uint8)
                        warped_channels.append(scaled)
                    donor_rgb = np.stack(warped_channels, axis=-1)
                    valid = valid_geometry & donor_valid
                    matched = match_colors(donor_rgb, scene["luts"])
                    preview_mask = scene["mask"]
                    xp = xp_full[col_start:col_end]
                    candidate = preview_mask[yp[:, None], xp[None, :]] & valid
                    target_float = target_rgb.astype(np.float32) / 255.0
                    donor_float = matched.astype(np.float32) / 255.0
                    brightness = target_float.mean(2)
                    donor_brightness = donor_float.mean(2)
                    chroma = target_float.max(2) - target_float.min(2)
                    core = (brightness > 0.56) & (chroma < 0.29) & (brightness - donor_brightness > 0.06)
                    shadow = (brightness < 0.45) & (donor_brightness - brightness > 0.09)
                    # With an explicit expansion, the user's cutline is the
                    # compositing contract: replace the whole expanded mask,
                    # including cloud fringes that no longer pass core tests.
                    refined = candidate if int(scene.get("mask_expand_px", 0)) > 0 else candidate & (core | shadow)
                    if bool(scene.get("replace_all_valid", False)):
                        alpha = scene["alpha_preview"][yp[:, None], xp[None, :]][..., None]
                        alpha *= refined[..., None]
                    else:
                        alpha = cv2.GaussianBlur(refined.astype(np.float32), (0, 0), 3.0)[..., None]
                    composite = np.round(composite * (1 - alpha) + matched * alpha).astype(np.uint8)
                new_nodata = (composite.max(2) == 0) & (target_rgb.max(2) > 0)
                composite[new_nodata] = target_rgb[new_nodata]
                destination.write(np.moveaxis(composite, -1, 0), window=window)
                print(f"full-resolution {row + height}/{target_h}", flush=True)


def cloud_mask(target: np.ndarray, donor: np.ndarray, valid: np.ndarray, include_thin_clouds: bool = False) -> np.ndarray:
    rgb = target.astype(np.float32) / 255.0
    replacement = donor.astype(np.float32) / 255.0
    brightness = rgb.mean(2)
    chroma = rgb.max(2) - rgb.min(2)
    donor_brightness = replacement.mean(2)
    # Bright, nearly neutral pixels that are materially brighter than the
    # cloud-free date. Requiring temporal contrast rejects most roofs/roads.
    core = valid & (
        ((brightness > 0.68) & (chroma < 0.28))
        | ((brightness > 0.55) & (chroma < 0.24) & (brightness - donor_brightness > 0.08))
    )
    if include_thin_clouds:
        # Thin cloud/haze is less bright than the opaque core but remains
        # broadly neutral and brighter than the registered cloud-free date.
        thin = valid & (brightness > 0.36) & (chroma < 0.34) & (brightness - donor_brightness > 0.045)
        thin = clean_components(thin, 100)
        thin = cv2.morphologyEx(thin.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2) > 0
        thin = clean_components(thin, 100)
        core |= thin
    core = clean_components(core, 60)
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2) > 0
    core = clean_components(core, 60)
    # Cloud shadow: substantially darker than donor and spatially close to a
    # confirmed cloud. The radius is in preview pixels (~7.5 m each here).
    vicinity = cv2.dilate(core.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1) > 0
    shadow = valid & vicinity & (donor_brightness - brightness > 0.14) & (brightness < 0.42)
    shadow = clean_components(shadow, 8)
    mask = core | shadow
    return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0


def snow_mask(donor_raw: np.ndarray, donor_matched: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Conservative snow/ice mask for winter donor scenes.

    Snow is bright and weakly saturated in the donor stretch. Histogram
    matching can additionally turn shaded snow cyan, so reject that signature
    as well and dilate slightly to remove mixed snow-edge pixels.
    """
    raw = donor_raw.astype(np.float32) / 255.0
    matched = donor_matched.astype(np.float32) / 255.0
    raw_brightness = raw.mean(2)
    raw_chroma = raw.max(2) - raw.min(2)
    cyan = ((matched[..., 1] + matched[..., 2]) * 0.5 - matched[..., 0] > 0.12) & (matched[..., 1] > 0.45) & (matched[..., 2] > 0.45)
    snow = valid & (((raw_brightness > 0.55) & (raw_chroma < 0.28)) | cyan)
    snow = clean_components(snow, 12)
    snow = cv2.dilate(snow.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1) > 0
    return snow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--donor-prefix", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--mask-expand-px", type=int, default=0, help="구름 마스크를 대상 raster pixel 단위로 추가 확장")
    parser.add_argument("--include-thin-clouds", action="store_true", help="donor 대비 밝은 얇은 구름과 연무까지 포함")
    parser.add_argument("--replace-all-valid", action="store_true", help="구름 임계값 없이 donor 유효영역 전체를 교체")
    parser.add_argument("--last-donor-fill-only", action="store_true", help="마지막 donor는 앞선 donor 마스크의 빈 구간만 채움")
    parser.add_argument("--last-donor-exclude-snow", action="store_true", help="마지막 겨울 donor의 눈/잔설과 혼합 경계 픽셀 제외")
    args = parser.parse_args()
    target_path = Path(args.target)
    with rasterio.open(target_path) as source:
        target_full_shape = (source.height, source.width)
        scale = min(1.0, 2200 / max(source.width, source.height))
        target = np.moveaxis(source.read((1, 2, 3), out_shape=(3, round(source.height * scale), round(source.width * scale)), resampling=Resampling.average), 0, -1).astype(np.float32) / 255
    diagnostics = []
    scenes = []
    preview = (target * 255).astype(np.uint8)
    composite = preview.copy()
    combined_mask = np.zeros(target.shape[:2], bool)
    for donor_index, prefix in enumerate(args.donor_prefix):
        paths = [Path(prefix + suffix) for suffix in ("_L2G_R_PS.tif", "_L2G_G_PS.tif", "_L2G_B_PS.tif")]
        donor, stretches, donor_shape = stretched_preview(paths)
        matrix, inliers, error = estimate_homography(donor, target)
        warped = cv2.warpPerspective((donor * 255).astype(np.uint8), matrix, (target.shape[1], target.shape[0]))
        valid = cv2.warpPerspective(np.full(donor.shape[:2], 255, np.uint8), matrix, (target.shape[1], target.shape[0])) > 0
        valid &= warped.max(2) > 0
        obvious_cloud = (preview.mean(2) > 158) & ((preview.max(2) - preview.min(2)) < 62)
        luts = color_luts(warped, preview, valid, obvious_cloud)
        matched = match_colors(warped, luts)
        excluded_snow = np.zeros(valid.shape, bool)
        if args.last_donor_exclude_snow and donor_index == len(args.donor_prefix) - 1:
            excluded_snow = snow_mask(warped, matched, valid)
            valid &= ~excluded_snow
            print(f"{Path(prefix).name}: excluded_snow={excluded_snow.mean():.3%}")
        mask = valid.copy() if args.replace_all_valid else cloud_mask(preview, matched, valid, args.include_thin_clouds)
        full_pixels_per_preview = max(target_full_shape[0] / preview.shape[0], target_full_shape[1] / preview.shape[1])
        if args.mask_expand_px > 0:
            radius = args.mask_expand_px / full_pixels_per_preview
            distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
            mask |= distance <= radius
            mask &= valid
        if args.last_donor_fill_only and donor_index == len(args.donor_prefix) - 1:
            mask &= ~combined_mask
        if args.replace_all_valid:
            edge_radius = 200.0 / full_pixels_per_preview
            alpha_preview = np.clip(cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) / max(edge_radius, 1e-3), 0, 1).astype(np.float32)
            alpha = alpha_preview[..., None]
        else:
            alpha_preview = np.ones(mask.shape, np.float32)
            alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)[..., None]
        composite = np.round(composite * (1 - alpha) + matched * alpha).astype(np.uint8)
        combined_mask |= mask
        overlay = preview.copy()
        overlay[valid] = np.round(0.5 * overlay[valid] + 0.5 * warped[valid]).astype(np.uint8)
        diagnostics.append((Path(prefix).name, inliers, error, float(valid.mean()), overlay, mask))
        matrix_full = full_homography(matrix, donor_shape, donor.shape[:2], target_full_shape, target.shape[:2])
        scenes.append({"paths": paths, "stretches": stretches, "donor_shape": donor_shape, "inverse_full": np.linalg.inv(matrix_full), "luts": luts, "mask": mask, "mask_expand_px": args.mask_expand_px, "replace_all_valid": args.replace_all_valid, "alpha_preview": alpha_preview})
    mask_rgb = preview.copy()
    mask_rgb[combined_mask] = (255, 0, 255)
    canvas = np.hstack([preview, mask_rgb, composite] + [item[4] for item in diagnostics])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output.with_suffix(".jpg")), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    for name, inliers, error, coverage, _, mask in diagnostics:
        print(f"{name}: inliers={inliers} median_error_px={error:.3f} preview_coverage={coverage:.3%}")
        print(f"{name}: replace_mask={mask.mean():.3%}")
    if not args.preview_only:
        write_full_composite(target_path, output, scenes)


if __name__ == "__main__":
    main()
