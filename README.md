# 팜맵 경지구분 세그멘테이션 (farmland_segmentation)

국토위성(CAS500) / VWorld 항공 정사영상에서 팜맵 경지구분 8클래스를 분할하는 PyTorch 파이프라인.
SegFormer-B5 기반이며, 데이터 구축 → 학습 → 평가 → 전체 장면 추론 → GIS 벡터 출력을 독립 CLI로 제공한다.

## 클래스 체계

| 값 | 클래스 | 출처 | 비고 |
|---|---|---|---|
| 0 | 배경 | 팜맵 필지 없음 | 폴리곤화 안 됨 |
| 1 | 논 | CLSF_CD 01 | |
| 2 | 밭 | CLSF_CD 02 | |
| 3 | 과수 | CLSF_CD 03 | |
| 4 | 시설 | CLSF_CD 04 | 비닐하우스 등 |
| 5 | 인삼 | CLSF_CD 05 | 익산 데이터엔 표본 없음 |
| 6 | 비경지 | CLSF_CD 06 | |
| 7 | 필지 경계 | 래스터화 시 인접 필지 에지 | 분리 신호 전용, 최종 산출물에서 제거 |
| 255 | ignore | nodata·구름 | 손실/평가 제외 |

## 요구사항

- Python 3.11, PyTorch(CUDA), rasterio, geopandas, scipy, scikit-image, transformers
- 팜맵 shapefile (시군별, EPSG:5179, `CLSF_CD` 속성)
- 영상: 아래 셋 중 하나
  - 국토위성 L2G/L3M 단일밴드 → `src.stack_bands`로 병합
  - VWorld 항공 → `src.fetch_vworld`로 다운로드 (API 키 필요)
  - 이미 georeferenced된 RGB GeoTIFF (EPSG:5179, 0.5m 권장)

## 1. 데이터 구축

### 1-1. 영상 준비

```bash
# (a) 국토위성 단일밴드 병합 — Aux.xml의 모서리 좌표로 원점 지정
python -m src.stack_bands --red R.tif --green G.tif --blue B.tif \
    --output scene_RGB.tif --resolution 0.5 --crs EPSG:5179 --origin <X> <Y>

# (b) VWorld 항공영상 다운로드 (z18 ≈ 0.5m)
python -m src.fetch_vworld --key <VWorld_API_키> \
    --bbox <XMIN> <YMIN> <XMAX> <YMAX> --output scene_RGB.tif
```

원점이 불확실한 위성 장면은 팜맵 경계×영상 에지 상관 매칭으로 복원한다
(과거 전주 L3M: 자리표시자 원점 → (954676.5, 1767453.0) 복원 사례 참고).

### 1-2. 팜맵 GT 래스터 (장면 전체)

```bash
python -m src.make_scene_label \
    --image scene_RGB.tif \
    --farmmap 팜맵_시군A.shp 팜맵_시군B.shp \
    --output scene_GT.tif --nodata-ignore 0
# 장면과 겹치는 모든 시군 shapefile을 나열할 것 (접경부 누락 방지)
# --nodata-ignore 0: 영상 nodata 픽셀을 255(ignore)로 마스킹
```

### 1-3. 타일 + manifest 생성

```bash
python -m src.make_tiles \
    --image scene_RGB.tif --label-image scene_GT.tif \
    --output ../data/<데이터셋이름> \
    --scene <장면접두어> --stride 256 --min-valid 0.2 --min-foreground 0.01
```

- `--stride 256`: 512px 타일을 절반 겹침으로 생성 (train만 증가, val/test는 원본 격자)
- `--min-foreground 0.01`: 라벨 전경 1% 미만(순수 배경) 타일 제외
- `--scene`: 타일 이름 접두어. **여러 장면을 병합할 계획이면 반드시 서로 다르게 지정**
- split은 EPSG:5179 절대좌표 4km 블록 기준 80/10/10 — 장면이 겹쳐도 같은 땅은 같은 split (누수 방지)

### 1-4. (선택) 여러 장면 병합 / 사후 필터

```bash
# hardlink 병합 (디스크 추가 사용 없음). 이름 충돌 시 에러로 멈춤
python -m src.merge_tile_datasets --roots rootA rootB --output ../data/merged

# 이미 만든 데이터셋에서 배경 타일 제외 (manifest만 수정, 원본은 manifest_full.csv 백업)
python -m src.filter_manifest --root ../data/merged --min-foreground 0.01
```

### 1-5. (선택) 구름 처리

- 학습: 구름 픽셀을 GT에서 255로 마킹 (`src.infer.detect_cloud_mask` 참고 — 전 밴드 밝은 큰 덩어리 탐지)
- 추론: config `inference.cloud_mask: true`로 구름 지역 예측 자동 제거
- 또는 다른 날짜 장면으로 구름을 대체한 합성 영상 사용 (현재 운용 방식)

## 2. 학습

config는 `_base_` 상속 구조다. 새 데이터셋 config는 기존 것을 상속해 root와 통계만 바꾼다:

```yaml
# configs/tiles_b5_crop_c1_composite.yaml (현재 활성 예시)
_base_: [tiles_b5_crop_rgb_v3.yaml]   # 클래스 가중·경계 손실 0.3·증강 강화 상속
project:
  name: farmland_tiles_b5_crop_c1_composite
  output_dir: outputs/tiles_segformer_b5_crop_c1_composite
dataset:
  root_dir: "../data/farmmap_c1_composite"
  mean: [0.193, 0.1802, 0.1653]   # train 표본 실측 (아래 통계 명령 참고)
  std: [0.154, 0.1412, 0.1364]
training:
  min_epochs: 0                    # 0=조기종료(patience 15)에 맡김
```

```bash
# 통계 계산 (config의 mean/std)
python - <<'EOF'
import csv, numpy as np, rasterio
from pathlib import Path
root = Path("../data/<데이터셋이름>")
with (root/"manifest.csv").open(newline="", encoding="utf-8-sig") as s:
    train = [r["tile"] for r in csv.DictReader(s) if r["split"] == "train"]
rng = np.random.default_rng(42); sums = np.zeros(3); squares = np.zeros(3); pixels = 0
for name in rng.choice(train, min(300, len(train)), replace=False):
    v = rasterio.open(root/"images"/f"{name}.tif").read([1,2,3]).astype(np.float64)/255.0
    sums += v.sum(axis=(1,2)); squares += (v**2).sum(axis=(1,2)); pixels += v.size//3
mean = sums/pixels; print("mean:", mean.round(4).tolist(), "std:", np.sqrt(squares/pixels-mean**2).round(4).tolist())
EOF

# 학습 (다중 GPU)
torchrun --nproc_per_node=4 -m src.train --config configs/tiles_b5_crop_c1_composite.yaml

# 중단 후 재개
torchrun --nproc_per_node=4 -m src.train --config <같은 config> \
    --resume outputs/<프로젝트>/checkpoints/last.pt
```

- AMP는 bf16 기본 (`training.amp_dtype`) — fp16 오버플로 NaN 방지. NaN 배치는 0-손실로 건너뛰고 카운트
- best.pt는 `mean_iou_no_background` 기준. 이 지표는 빈 클래스(인삼 등)를 0으로 포함하므로
  절대값이 낮게 보인다 — 클래스별 IoU는 `logs/history.csv` 참고

## 3. 평가·시각화

```bash
# test split 정량 평가 → metrics/test/class_metrics.csv (클래스별 P/R/F1/IoU)
python -m src.test --config <config> --checkpoint outputs/<프로젝트>/checkpoints/best.pt

# GT vs 예측 비교 패널 (실행마다 무작위 표본, 빈 타일 자동 제외)
python -m src.visualize_tiles --config <config> --checkpoint <best.pt> --split test --count 20
# --sample-seed N: 표본 고정 | --min-foreground 0.1: 더 알찬 타일만 | --no-watershed: 순수 모델 출력
```

## 4. 전체 장면 추론

```bash
python -m src.infer --config <config> --checkpoint <best.pt> \
    --input scene_RGB.tif \
    --output-mask outputs/infer/<이름>_mask.tif \
    --output-vector outputs/infer/<이름>_parcels.gpkg \
    --set output.save_probability_map=false
```

산출물: `_mask.tif`(8클래스, 경계 제거·필지 맞닿음), `_raw.tif`(원시 argmax, 진단용),
`_instances.tif`(필지 ID), `_parcels.gpkg`(폴리곤: class_id/class_name/area_m2, 평활+직선화).

동작 특성:
- 스트리밍 추론(RAM ~10GB), 전부-검정 창 스킵, Hann 겹침 병합(이음새 없음), 단계별 진행 로그
- watershed로 인접 필지 분리 후 경계 클래스는 필지로 흡수(`erase_boundary`)
- 벡터는 marching-squares 서브픽셀 평활(`vector_smooth_px`) + Douglas-Peucker(`vector_simplify_m`)

속도 참고 (0.5m, GPU 1장): 5×5km ≈ 5~9분, 전체 장면(16×14km) ≈ 30~60분.
병목은 GPU가 아니라 CPU 후처리(watershed)이며 `watershed_downscale: 2`(기본)로 3배 가속돼 있다.

### 주요 추론 튜닝 (configs/default.yaml)

| 키 | 기본 | 용도 |
|---|---|---|
| `inference.watershed_seed_erosion_iterations` | 1 | 인접 필지 병합 억제 (심하면 2) |
| `inference.watershed_seed_boundary_maximum` | 0.15 | 낮추면 분리 강해짐 (0.08) |
| `inference.watershed_surface_sigma` | 1.5 | 분할선 잔떨림 평활 |
| `inference.watershed_downscale` | 2 | 분수령 계산 축소 가속 (1=정밀) |
| `inference.erase_boundary` | true | 경계 클래스 산출물 제거 |
| `inference.cloud_mask` | false | 구름 지역 예측 비우기 |
| `output.vector_smooth_px` / `vector_simplify_m` | 1.5 / 1.5 | 폴리곤 계단 제거·직선화 |
| `postprocess.min_area` | {1:100, 2:100} | 클래스별 최소 면적(㎡) |

재벡터화만 다시 (추론 없이 폴리곤 옵션 변경):

```bash
python -m src.vectorize --instances <_instances.tif> --classes <_mask.tif> \
    --output parcels.gpkg --smooth 1.5 --simplify 1.5
# 클래스 이름은 8클래스 기본 내장, 경계(7)는 기본 제외(--drop-classes)
```

## 트러블슈팅

- **VWorld 503**: 서버 간헐 장애. 연속 20타일 실패 시 자동 중단되니 잠시 후 재실행
- **추론 중 Killed**: RAM 부족. `output.save_probability_map=false`(스트리밍 경로) 확인
- **학습 NaN**: bf16 기본으로 해소. `skipped` 카운터가 계속 늘면 데이터/LR 점검
- **gdal_translate 없음**: rasterio로 대체 (Window 읽기 → 새 GeoTIFF 쓰기)
- **타일 이름 충돌(병합 시)**: `make_tiles --scene`으로 장면별 접두어를 다르게
- **폴리곤에 경계 클래스가 보임**: 옛 추론 산출물. 최신 코드로 재추론
