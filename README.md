# 논·밭 U-Net GeoTIFF Segmentation

항공 RGB GeoTIFF에서 논과 밭을 분할하는 PyTorch 파이프라인이다. 데이터 검증, 통계/타일 전처리, 학습, 검증, 테스트, sliding-window 추론, 형태학적 후처리, 인스턴스 연결요소 생성 및 GIS 벡터 출력을 독립 CLI로 제공한다.

## 확인된 데이터 계약

- 학습 입력: 논/밭 구분 라벨이 있는 3밴드 RGB 512×512 항공사진
- 라벨 입력: 기존 `항공사진_FGT_512픽셀_Json/*.json` 폴리곤의 `ANN_CD 50=논`, `60=밭`
- 논/밭 feature가 하나도 없는 JSON은 데이터셋 인덱스에서 제외하며, 다른 토지피복 polygon은 배경으로 처리
- 모델 출력: `0=배경`, `1=논`, `2=밭`의 3개 logits
- 라벨 CRS: EPSG:5186, 512/1024 항공사진 해상도 0.25 m
- 원천 TIF에는 CRS/Transform이 없으므로 기존 `_META.json`의 EPSG, 좌상단 픽셀 중심 좌표, 해상도로 격자를 구성하고 기존 GeoJSON polygon을 직접 rasterize한다. 독립 추론에는 `--reference-meta`가 필요하다.
- 기존 JSON polygon은 학습 시 `배경/논/밭` semantic mask로 메모리에서 rasterize된다. 추론의 `instances.tif`는 클래스별 경계 침식 후 connected components로 만든 파생 인스턴스다.
- 별도 1024 자료는 `90=농경지`까지만 표시되어 논/밭을 구분하지 못한다. 이를 배경으로 잘못 학습시키지 않도록 기본 설정에서 제외했다. 학습된 모델의 sliding-window 추론은 1024 이상 임의 크기를 지원한다.

제곱미터 면적 필터는 투영 CRS가 확인된 경우에만 실행된다. EPSG:4326이나 EPSG:3857에서 권위 있는 면적을 계산하지 않는다.

## 구조

```text
configs/                 데이터, 모델, 실행 설정
src/datasets/            영상-기존 JSON/Meta 매칭, polygon rasterization, 동기 증강
src/metrics/             confusion matrix 기반 평가
src/utils/               설정, 로그, seed, checkpoint, 시각화
src/model.py             U-Net 정의
src/train.py             CPU/단일 GPU/DDP 학습
src/validate.py          validation 평가
src/test.py              고정 seed로 분리한 test 평가
src/infer.py             GeoTIFF sliding-window 추론 및 GIS 출력
src/vectorize.py         추론 프로세스와 격리된 래스터 polygonization
src/validate_data.py     데이터 품질/공간정보 보고서
src/prepare_data.py      통계 계산과 선택적 물리 타일 생성
src/split_data.py        이동 없는 분할 manifest 생성
tests/                   단위/스모크 테스트
outputs/                 checkpoint, log, metric, prediction
```

## 설치

Python 3.11과 CUDA 12.1 조합:

```bash
conda env create -f environment.yml
conda activate farmland-segmentation
```

pip 환경에서는 CUDA 12.1 PyTorch를 먼저 설치한다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CPU 검증 환경은 PyTorch CPU wheel을 설치한 뒤 동일한 `requirements.txt`를 사용한다.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## 데이터 배치

기본 설정은 현재 저장소의 다음 폴더를 직접 읽는다.

```text
../data/01.데이터/1.Training/원천데이터/TS_항공사진_FGT_512픽셀
../data/01.데이터/1.Training/라벨링데이터/항공사진_FGT_512픽셀_Json
../data/01.데이터/1.Training/라벨링데이터/항공사진_FGT_512픽셀_Meta
../data/01.데이터/2.Validation/원천데이터/VS_항공사진_FGT_512픽셀
../data/01.데이터/2.Validation/라벨링데이터/항공사진_FGT_512픽셀_Json
../data/01.데이터/2.Validation/라벨링데이터/항공사진_FGT_512픽셀_Meta
```

Training은 그대로 사용하고, 제공 Validation은 seed 42로 validation/test에 50:50 분리한다. 파일을 복사하거나 이동하지 않는다.

## 데이터 검증과 준비

빠른 표본 검사 후 전체 검사를 수행한다.

```bash
python -m src.validate_data --config configs/default.yaml --split train --max-samples 100
python -m src.validate_data --config configs/default.yaml --split train
python -m src.validate_data --config configs/default.yaml --split validation
python -m src.split_data --config configs/default.yaml
python -m src.prepare_data --config configs/default.yaml --max-samples 1000
```

실제 파일로 공간정보가 유지된 512 타일을 물리적으로 만들려면 다음을 사용한다. 배경 비율 단위는 0~1이다.

```bash
python -m src.prepare_data --config configs/default.yaml \
  --max-samples 100 --tile-output outputs/tiles \
  --overlap 128 --max-background-fraction 0.95
```

계산된 `outputs/data_stats.json`의 mean/std를 `configs/dataset.yaml`에 반영한다.

## 학습

CPU 또는 단일 GPU:

```bash
python -m src.train --config configs/default.yaml
```

두 GPU DDP:

```bash
torchrun --standalone --nproc_per_node=2 -m src.train --config configs/default.yaml
```

체크포인트 재개와 CLI 우선 덮어쓰기:

```bash
python -m src.train --config configs/default.yaml \
  --resume outputs/checkpoints/last.pt \
  --set training.batch_size=4 \
  --set training.epochs=150
```

`outputs/checkpoints/best.pt`와 `last.pt`에는 모델, optimizer, scheduler, AMP scaler, epoch, best metric, 모델/데이터 설정, 클래스 목록, Git hash 및 UTC 저장시각이 포함된다.

## 검증과 테스트

```bash
python -m src.validate --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
python -m src.test --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

JSON에는 pixel accuracy, precision, recall, F1, Dice, 클래스 IoU, mean IoU, frequency-weighted IoU, confusion matrix가 저장된다. CSV에는 클래스별 지표가 저장되고 최저 IoU 클래스가 로그에 표시된다.

## 추론

현재 원천 영상은 공간정보가 없으므로 대응하는 기존 `_META.json`을 지정한다.

```bash
python -m src.infer \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --input "../data/01.데이터/2.Validation/원천데이터/VS_항공사진_FGT_512픽셀/LC_GS_AP25_34801025_006_2019_FGT.tif" \
  --reference-meta "../data/01.데이터/2.Validation/라벨링데이터/항공사진_FGT_512픽셀_Meta/LC_GS_AP25_34801025_006_2019_FGT_META.json" \
  --output-mask outputs/predictions/sample.tif \
  --output-vector outputs/predictions/sample.gpkg \
  --tile-size 512 --overlap 128 --batch-size 8
```

출력:

- `sample_raw.tif`: 후처리 전 argmax 클래스
- `sample.tif`: 후처리 후 클래스
- `sample_probability.tif`: 3밴드 클래스 확률
- `sample_instances.tif`: 연결요소 인스턴스 ID
- `sample.gpkg`: `class_id`, `instance_id`, `area_m2` 폴리곤

GPU OOM 시 추론 batch size는 자동으로 절반씩 감소한다. 타일 overlap은 덮어쓰지 않고 Hann 가중 확률 평균으로 병합한다.
전경 예측 확률이 `inference.confidence_threshold`보다 낮으면 배경으로 되돌린다.

## Loss 선택

기본은 CE + Dice + 작은 boundary loss다. CE는 안정적인 다중 클래스 기준, Dice는 논/밭 픽셀 불균형 보완, boundary는 필지 경계 민감도를 높인다. Focal은 어려운 픽셀에 집중하지만 노이즈에 과민할 수 있고, Tversky는 FP/FN 비용을 조절하지만 alpha/beta 튜닝이 필요하다. 모든 조합은 `configs/model.yaml`의 weight로 켜고 끈다.

TensorBoard는 기본 활성화된다. `logging.wandb: true`로 바꾸고 로그인하면 동일한 epoch 지표를 W&B에도 기록한다.

## Docker

```bash
docker build -t farmland-segmentation .
docker run --rm --gpus all \
  -v "$PWD/../data/01.데이터:/data/01.데이터:ro" \
  -v "$PWD/configs:/workspace/configs:ro" \
  -v "$PWD/outputs:/workspace/outputs" \
  farmland-segmentation
```

## 테스트

```bash
python -m pytest -q
```

pytest가 없는 최소 환경에서도 테스트 본문은 unittest 호환이다.

```bash
python -m unittest discover -v
```

## 오류 해결

- `CRS/Transform이 없습니다`: 기존 대응 `_META.json`을 `--reference-meta`로 전달한다.
- `영상/JSON/Meta 파일명이 대응되지 않습니다`: 영상 stem, 라벨 JSON stem, `_META`를 제외한 Meta stem이 같은지 확인한다.
- `입력 밴드가 부족합니다`: `dataset.channel_indices`와 실제 밴드 수를 확인한다. 현재 항공영상은 `[1,2,3]` RGB다.
- CUDA OOM: batch size 또는 tile size를 줄이고 gradient accumulation을 늘린다.
- 체크포인트 구조 불일치: checkpoint의 model/dataset 설정과 현재 resolved config를 비교한다.
