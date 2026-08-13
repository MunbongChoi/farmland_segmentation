"""Render tile samples as an RGB / ground-truth / optional-prediction grid PNG."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from scipy import ndimage

from .datasets.dataset import build_datasets
from .infer import close_parcel_boundaries
from .model import build_model
from .utils.checkpoint import load_checkpoint
from .utils.config import apply_overrides, load_config
from .utils.visualization import _title_tile, _write_rgb, blend_mask


def _read_context_window(root: Path, name: str, channels: tuple[int, ...], tile_size: int, margin: int) -> np.ndarray:
    """Read a tile plus ``margin`` pixels of real neighbor-tile context around it."""
    scene, row, col = name.rsplit("_", 2)
    # make_tiles 이름은 픽셀 오프셋(r08704_c19968), 구 형식은 타일 인덱스(001_002)다.
    prefixed = row.startswith("r") and col.startswith("c")
    step = tile_size if prefixed else 1
    row_value, col_value = int(row.lstrip("r")), int(col.lstrip("c"))
    # Missing neighbors (scene edges) stay zero, matching the no-data black border.
    mosaic = np.zeros((len(channels), tile_size * 3, tile_size * 3), dtype=np.uint8)
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            neighbor_row, neighbor_col = row_value + delta_row * step, col_value + delta_col * step
            stem = f"{scene}_r{neighbor_row:05d}_c{neighbor_col:05d}" if prefixed else f"{scene}_{neighbor_row:03d}_{neighbor_col:03d}"
            path = root / "images" / f"{stem}.tif"
            if neighbor_row < 0 or neighbor_col < 0 or not path.is_file():
                continue
            with rasterio.open(path) as source:
                mosaic[:, (delta_row + 1) * tile_size : (delta_row + 2) * tile_size, (delta_col + 1) * tile_size : (delta_col + 2) * tile_size] = source.read(channels)
    low, high = tile_size - margin, 2 * tile_size + margin
    return mosaic[:, low:high, low:high]


def _parcel_polygons(interior: np.ndarray, minimum_pixels: int) -> tuple[list[tuple[np.ndarray, int]], np.ndarray]:
    """Exterior pixel rings (with component id) of connected interior components."""
    structure = ndimage.generate_binary_structure(2, 2)
    labels, count = ndimage.label(interior, structure)
    if count:
        sizes = np.bincount(labels.ravel())
        small = sizes < minimum_pixels
        small[0] = True
        labels[small[labels]] = 0
    # ponytail: exterior rings only; holes inside a parcel are rare and not drawn.
    rings = [(np.asarray(geometry["coordinates"][0]), int(value)) for geometry, value in shapes(labels.astype(np.int32), mask=labels > 0, connectivity=8)]
    return rings, labels


def _component_majority_class(labels: np.ndarray, prediction: np.ndarray, num_classes: int) -> np.ndarray:
    """Majority prediction class per connected component id."""
    valid = labels > 0
    encoded = labels[valid].astype(np.int64) * num_classes + prediction[valid].clip(0, num_classes - 1)
    counts = np.bincount(encoded, minlength=(int(labels.max()) + 1) * num_classes).reshape(-1, num_classes)
    return counts.argmax(axis=1)


def _draw_outlines(image: np.ndarray, rings: list[np.ndarray], color: tuple[int, int, int]) -> None:
    for ring in rings:
        cv2.polylines(image, [np.round(ring).astype(np.int32)], True, color, 2, cv2.LINE_AA)


def _predict_tile(model: torch.nn.Module, dataset: Any, config: dict, sample: dict, margin: int, use_watershed: bool) -> np.ndarray:
    """Predict one tile with optional neighbor context and watershed closing."""
    device = next(model.parameters()).device
    if margin:
        window = _read_context_window(dataset.root, Path(sample["path"]).stem, dataset.channels, int(config["dataset"]["tile_size"]), margin)
        values = (window.astype(np.float32) / 255.0 - dataset.mean) / dataset.std
        inputs = torch.from_numpy(np.ascontiguousarray(values))[None].float()
    else:
        inputs = sample["image"][None]
    with torch.inference_mode():
        logits = model(inputs.to(device))
    if margin:
        logits = logits[..., margin:-margin, margin:-margin]
    probabilities = logits.softmax(dim=1)[0].cpu().numpy()
    boundary_class = int(config.get("inference", {}).get("boundary_class", 2))
    return close_parcel_boundaries(probabilities, boundary_class) if use_watershed else probabilities.argmax(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="타일 데이터셋 샘플/예측 시각화")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="지정하면 모델 예측 열을 추가한다")
    parser.add_argument("--compare-config", help="비교 모델의 config (–compare-checkpoint와 함께)")
    parser.add_argument("--compare-checkpoint", help="지정하면 두 번째 예측 열을 추가한다")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, help="샘플 추첨 시드. 미지정 시 실행마다 다른 타일이 뽑힌다")
    parser.add_argument("--min-foreground", type=float, default=0.02, help="GT 전경 비율이 이 값 미만인 빈 타일은 건너뛴다")
    parser.add_argument("--context-margin", type=int, default=256, help="이웃 타일에서 가져올 추론 문맥 픽셀. 0이면 타일 단독 추론")
    parser.add_argument("--watershed", action=argparse.BooleanOptionalAction, default=True, help="경계 확률 watershed로 끊긴 필지 경계를 닫는다")
    parser.add_argument("--min-parcel-pixels", type=int, default=50, help="이보다 작은 폴리곤은 잡음으로 버린다")
    parser.add_argument("--output")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    datasets = dict(zip(("train", "val", "test"), build_datasets(config)))
    dataset = datasets[args.split]
    # 빈 타일(GT 전경 없음)을 건너뛰며 무작위로 count개를 모은다. 라벨 파일만 읽어 빠르다.
    order = list(range(len(dataset)))
    random.Random(args.sample_seed).shuffle(order)
    indices: list[int] = []
    for index in order:
        with rasterio.open(dataset.root / dataset.label_dir / f"{dataset.tiles[index]}.tif") as label_source:
            label = label_source.read(1)
        if ((label > 0) & (label != 255)).mean() >= args.min_foreground:
            indices.append(index)
        if len(indices) >= args.count:
            break
    if len(indices) < args.count:
        print(f"전경 조건을 만족하는 타일이 {len(indices)}개뿐입니다 (요청 {args.count})")

    model = None
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 체크포인트가 전 가중치를 덮어쓰므로 사전학습(HF Hub) 로드는 생략한다.
        model = build_model({**config, "model": {**config["model"], "pretrained": False}}).to(device).eval()
        load_checkpoint(args.checkpoint, model, current_config=config, map_location=device)

    if bool(args.compare_config) != bool(args.compare_checkpoint):
        parser.error("--compare-config와 --compare-checkpoint는 함께 지정해야 합니다.")
    compare_model = compare_config = compare_dataset = None
    if args.compare_checkpoint:
        compare_config = apply_overrides(load_config(args.compare_config), args.set)
        compare_dataset = dict(zip(("train", "val", "test"), build_datasets(compare_config)))[args.split]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compare_model = build_model({**compare_config, "model": {**compare_config["model"], "pretrained": False}}).to(device).eval()
        load_checkpoint(args.compare_checkpoint, compare_model, current_config=compare_config, map_location=device)

    channels = tuple(int(value) for value in config["dataset"]["channel_indices"][:3])
    rows = []
    vector_records: list[dict] = []
    for index in indices:
        sample = dataset[index]
        with rasterio.open(sample["path"]) as source:
            rgb = np.moveaxis(source.read(channels), 0, -1)
        mask = sample["mask"].numpy()
        # ponytail: ignore(255)는 배경으로 표시 — 해당 픽셀은 영상도 nodata 검정이라 그대로 읽힌다.
        mask = np.where(mask == 255, 0, mask)
        tiles = [_title_tile(rgb, Path(sample["path"]).stem[-12:]), _title_tile(blend_mask(rgb, mask), "Ground truth")]
        margin = max(0, args.context_margin)
        prediction = compare_prediction = None
        if model is not None:
            prediction = _predict_tile(model, dataset, config, sample, margin, args.watershed)
            tiles.append(_title_tile(blend_mask(rgb, prediction), f"Pred A: {Path(args.config).stem}"))
        if compare_model is not None:
            compare_prediction = _predict_tile(compare_model, compare_dataset, compare_config, compare_dataset[index], margin, args.watershed)
            tiles.append(_title_tile(blend_mask(rgb, compare_prediction), f"Pred B: {Path(args.compare_config).stem}"))
        panel = rgb.copy()
        boundary = int(config.get("inference", {}).get("boundary_class", 2))
        gt_rings, _ = _parcel_polygons((mask > 0) & (mask != boundary), args.min_parcel_pixels)
        _draw_outlines(panel, [ring for ring, _ in gt_rings], (40, 110, 255))
        legend = ["GT blue"]
        if prediction is not None:
            predicted_rings, component_labels = _parcel_polygons((prediction > 0) & (prediction != boundary), args.min_parcel_pixels)
            _draw_outlines(panel, [ring for ring, _ in predicted_rings], (255, 40, 40))
            legend.append("A red")
            class_names = list(config["dataset"]["class_names"])
            majority = _component_majority_class(component_labels, prediction, len(class_names))
            with rasterio.open(sample["path"]) as source:
                if source.crs is not None and not source.transform.is_identity:
                    for ring, component in predicted_rings:
                        class_id = int(majority[component])
                        vector_records.append({
                            "tile": Path(sample["path"]).stem,
                            "class_id": class_id,
                            "class_name": class_names[class_id],
                            "crs": source.crs,
                            "ring": [tuple(source.transform * tuple(point)) for point in ring],
                        })
        if compare_prediction is not None:
            compare_boundary = int(compare_config.get("inference", {}).get("boundary_class", 2))
            compare_rings, _ = _parcel_polygons((compare_prediction > 0) & (compare_prediction != compare_boundary), args.min_parcel_pixels)
            _draw_outlines(panel, [ring for ring, _ in compare_rings], (255, 210, 0))
            legend.append("B yellow")
        tiles.append(_title_tile(panel, f"Parcels ({' / '.join(legend)})"))
        rows.append(np.hstack(tiles))

    output = Path(args.output) if args.output else Path(config["project"]["output_dir"]) / "visualizations" / f"{args.split}_samples.png"
    _write_rgb(output, np.vstack(rows))
    print(output)
    if vector_records:
        import geopandas as gpd
        from shapely.geometry import Polygon

        frame = gpd.GeoDataFrame(
            {
                "tile": [record["tile"] for record in vector_records],
                "class_id": [record["class_id"] for record in vector_records],
                "class_name": [record["class_name"] for record in vector_records],
            },
            geometry=[Polygon(record["ring"]) for record in vector_records],
            crs=vector_records[0]["crs"],
        )
        vector_path = output.with_suffix(".gpkg")
        frame.to_file(vector_path, driver="GPKG")
        print(vector_path)


if __name__ == "__main__":
    main()
