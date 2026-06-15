# SEM 입자 정량화 — 기술 가이드

## 개요

이 프로젝트는 리튬이온 배터리 전구체 입자의 SEM (Scanning Electron Microscopy) 이미지에 대해 두 가지 독립적인 측정 파이프라인을 제공합니다.

| 스크립트 | 대상 | 측정 방법 |
|--------|--------|--------|
| `primary_measure.py` | **1차 입자** — 2차 입자 표면의 침상(acicular, 바늘 모양) 또는 판상(plate) 결정 | LSD (침상) 또는 SAM2 segmentation (판상) |
| `secondary_measure.py` | **2차 입자** — 구형 응집체 자체 | SAM2 segmentation |

---

## 패키지 구조

```
measure/
├── primary_measure.py          1차 입자 분석 CLI 진입점
├── secondary_measure.py        2차 입자 분석 CLI 진입점
│
├── configs/
│   ├── __init__.py             preset 및 경로 config 로더
│   ├── presets.yaml            분석 preset (particle_type × magnification)
│   └── paths.yaml              기본 파일 경로 (model, input, output)
│
├── core/
│   └── schema.py               Dataclass 정의 (config, 측정값, 결과)
│
├── models/
│   └── __init__.py             Ultralytics를 통한 SAM2 모델 lazy-loader
│
├── services/
│   ├── sam2_service.py         Base SAM2 service (tiling, point sampling, dedup,
│   │                           mask 측정, overlay, histogram, CSV/JSON 출력)
│   ├── primary_particle.py     1차 입자 파이프라인 (ROI 추출, LSD/SAM2 분기,
│   │                           acicular hybrid 모드, batch runner, CLI parser)
│   └── secondary_particle.py   2차 입자 파이프라인 (runner, batch 집계, CLI parser)
│
└── utils/
    ├── image.py                구(sphere) 검출, center crop, CLAHE texture
    │                           enhancement, Shi-Tomasi point sampling, tile
    │                           생성, adaptive block-size helper
    ├── iou.py                  Binary mask IoU, bounding-box IoU
    ├── lsd.py                  LSD 탐지 파이프라인, perpendicular profile
    │                           두께 측정, segment fusion
    ├── metrics.py              px→µm 변환, 통계, JSON 직렬화
    └── io.py                   입력 파일/디렉터리 수집, 출력 경로 생성
```

---

## 스케일 보정 (Scale Calibration)

모든 길이 측정값은 2점 보정(two-point calibration)을 통해 픽셀에서 마이크로미터로 변환됩니다.

```
µm_value = px_value × (scale_um / scale_pixels)
```

| 배율 (Magnification) | 보정값 | px/µm |
|---------------|-------------|-------|
| **20k** | 147 px = 1 µm | 147.0 |
| **50k** | 371 px = 1 µm | 371.0 |

`--magnification`을 지정하면 preset에 의해 자동으로 적용됩니다. 비표준 배율의 경우 `--scale_pixels`와 `--scale_um`으로 직접 지정하세요.

---

## 공통 전처리 (두 파이프라인 모두 적용)

### 이미지 정규화 (Image Normalisation)

분석 전 모든 이미지는 고정된 크기로 정규화됩니다.

1. **Resize** → 2048 × 1636 px (bilinear interpolation)
2. **하단 100 px 제거** → 최종 크기 **2048 × 1536 px**

이 과정은 SEM 이미지 하단의 데이터 표시줄을 제거하고, 원본 해상도와 관계없이 일정한 pixel grid를 보장합니다.

### ROI 추출 (ROI Extraction)

정규화 후, 배경을 제외하고 입자 표면에 집중하기 위해 분석 대상 영역(Region of Interest)을 잘라냅니다.

#### 구 자동 검출 (primary 20k, `auto_detect_sphere=True`)

전체 2차 입자가 화면에 보이는 경우 사용합니다.

1. Grayscale 변환 → Gaussian blur (21×21) → Otsu threshold
2. Foreground 비율 > 50%이면 반전 (어두운 배경 처리)
3. Morphological close (kernel 15, 3회) → open (kernel 15, 2회)
4. External contour 탐색 → 면적 기준 최대 contour 선택 (전체 프레임의 ≥ 2%)
5. Minimum enclosing circle 피팅 → 중심 (cx, cy) 및 반지름 r 계산
6. Cap ROI: `y ∈ [cy − r, cy − r + 2r × sphere_cap_fraction]`, `x ∈ [cx − r, cx + r]`
   — 기본 `sphere_cap_fraction = 0.65`는 구 직경의 상단 65%를 캡처
7. 검출 실패 시 center crop으로 fallback

#### Center crop (primary 50k, `auto_center_crop=True`)

입자 표면만 화면을 채우는 경우 사용합니다.

```
margin_x = W × (1 − crop_ratio) / 2
margin_y = H × (1 − crop_ratio) / 2
ROI = image[margin_y : H − margin_y, margin_x : W − margin_x]
```

기본값: 50k에서 `crop_ratio = 0.85`, 20k sphere 모드에서 `0.60`.

#### 수동 ROI

`--no-auto_center_crop` 설정 시 `--roi_x_min`, `--roi_y_min`, `--roi_x_max`, `--roi_y_max`로 직접 지정합니다.

---

## 1차 입자 파이프라인 (Primary Particle Pipeline)

### 측정 모드 선택

| `--particle_type` | `--measure_mode` | 모델 필요 여부 |
|-------------------|-----------------|----------------|
| `acicular` | `lsd` (기본값) | 불필요 |
| `plate` | `sam2` (기본값) | 필요 |

`measure_mode`는 preset에 의해 자동 설정됩니다. `--measure_mode`로 직접 지정할 수도 있습니다.

---

### 침상 모드 — LSD Pipeline

LSD (Line Segment Detector)는 신경망 없이 grayscale 이미지에서 직접 침상의 길이와 수직 두께를 측정합니다. 처리 시간은 일반적으로 < 2초입니다.

#### Step 1 — 대비 향상 (Contrast Enhancement)

```
CLAHE(clipLimit=2.0, tileGridSize=8×8)  →  GaussianBlur(3×3)
```

CLAHE (Contrast-Limited Adaptive Histogram Equalization)는 ROI 전체의 조명 불균일을 보정합니다. 이후 Gaussian blur로 엣지 검출 전 픽셀 수준의 노이즈를 제거합니다.

**Debug 이미지:** `lsd_01_preprocessed.png`

#### Step 2 — Thresholding

`--lsd_adaptive_thresh`로 두 가지 방식 중 선택합니다.

**Otsu (기본값):** 흐릿한(blurred) 이미지에 Otsu 방법으로 단일 전역 threshold를 계산합니다. Threshold 이상의 픽셀이 밝은 영역(침상 몸체)으로 분류됩니다.

**Adaptive Gaussian (`--lsd_adaptive_thresh`):** 로컬 이웃 평균에서 상수 C=4를 뺀 값으로 픽셀별 threshold를 계산합니다. Block size는 ROI 크기에서 자동 결정됩니다: `block_size = max(11, ⌊min(H, W) / 12⌋)` (홀수로 올림). 프레임 전체에 걸쳐 조명이 크게 변하는 경우에 유용합니다.

두 방식 모두 다음을 생성합니다.
- Perpendicular profile scanning에 사용되는 binary mask
- Debug 이미지

**Debug 이미지:** `lsd_02_otsu_thresh.png` 또는 `lsd_02_adaptive_thresh.png`

#### Step 3 — LSD 원본 탐지

```python
cv2.createLineSegmentDetector(0).detect(arr_blur)
```

OpenCV의 LSD 구현 (Grompone von Gioi et al. 2010)은 gradient 방향으로 정렬된 pixel chain을 추적하여 line segment를 탐지합니다. 각 segment의 끝점 (x1, y1, x2, y2)과 추정 선폭(width)을 출력합니다.

**일반적인 출력:** ROI당 1,000–3,000개 segment.

**Debug 이미지:** `lsd_03_raw_detections.png` (노란색 선)

#### Step 4 — 길이 및 Aspect-Ratio 필터

| 기준 | Threshold | 이유 |
|-----------|-----------|-----------|
| 길이 < 20 px | 제거 | 서브픽셀 노이즈 및 결정립 경계 |
| `lsd_width / length ≥ ar_loose` | 제거 | 짧고 뭉툭한 segment는 침상이 아님; `ar_loose = min(acicular_threshold + 0.20, 0.65)` |

**Debug 이미지:** `lsd_04_after_filter.png` (청록색 선)

#### Step 5 — 중복 제거 (Deduplication)

하나의 침상에서 여러 개의 겹치는 LSD segment가 생성될 수 있습니다 (엣지별, 구간별). Segment를 길이 내림차순으로 정렬 후 greedy 방식으로 수락합니다.

```
for each candidate C (longest first):
    for each already-accepted segment P:
        if dist(centre_C, centre_P) < 12 px  AND  |angle_C − angle_P| < 25°:
            reject C as duplicate
            break
    else:
        accept C
```

물리적 침상 하나당 최대 하나의 대표 segment만 유지합니다.

**Debug 이미지:** `lsd_05_after_dedup.png` (주황색 선)

#### Step 6 — Segment Fusion (선택, `--lsd_fuse_segments`)

침상이 여러 개의 비연속 segment로 분리된 경우(예: 중간 부분이 가려진 경우), 이 단계에서 하나의 긴 segment로 합칩니다.

**Algorithm — union-find on collinear neighbours:**

두 segment i와 j는 다음 세 조건이 모두 충족될 때 합쳐집니다.

1. **각도 유사성:** `min(|αᵢ − αⱼ|, 180° − |αᵢ − αⱼ|) ≤ 10°`
2. **수직 근접성:** segment i의 축에 수직 방향으로 중점 간 거리 `< 8 px` (같은 선상에 있음을 보장, 동일 침상의 평행 엣지와 구분)
3. **축 방향 근접성:** j의 끝점을 i의 축에 투영했을 때, 구간이 겹치거나 간격 `< 15 px`

연결된 구성요소(connected component)는 길이 가중 평균 방향으로 모든 끝점을 투영하여 극값을 새 끝점으로 취합니다.

**Debug 이미지:** `lsd_06_after_fusion.png` (자홍색 선)

#### Step 7 — 수직 두께 측정 (Perpendicular Thickness Measurement)

수락된 각 segment에 대해, segment 축을 따라 7개의 균등 간격 위치 (t = 0.2, 0.25, …, 0.8)에서 수직 방향으로 intensity profile을 샘플링하여 침상 두께를 측정합니다.

**각 샘플 위치에서:**

1. Segment에 수직 방향으로, 중심에서 `2 × max(15, 0.5 × px_per_µm)` 픽셀 길이의 scan line 계산
2. Scan line을 따라 흐릿한(blurred) grayscale 이미지 샘플링
3. Otsu threshold (또는 미리 계산된 adaptive binary)로 profile 이진화
4. 이진화된 profile에서 연속적인 밝은 구간(bright run) 탐색
5. 중점이 scan line 중심에 가장 가까운 밝은 구간 선택 — 인접 침상이 아닌 측정 대상 침상을 선택하기 위함
6. 선택된 구간의 너비 = 해당 위치의 두께 추정값

최종 두께는 7개 추정값의 **median** (이상치 및 가려진 위치에 강건함).

**Perpendicular offset** (LSD 엣지 라인에서 밝은 영역 중심까지의 부호 있는 거리)도 계산되며, mask 사각형을 탐지된 엣지에서 물리적 침상 몸체 중심으로 이동하는 데 사용됩니다.

두께 < 2 px인 segment는 신뢰할 수 있는 검출 한계 이하로 제거됩니다.

#### Step 8 — 분류 및 Masking

```
aspect_ratio = thickness_px / length_px
```

| AR | 분류 |
|----|----------|
| < `acicular_threshold` (기본값 0.40) | `acicular` |
| ≥ `acicular_threshold` | `plate` |
| mask 면적 < `area_threshold` | `fragment` |

`--particle_type acicular`인 경우 plate 분류 segment가 제거되고, 그 반대도 마찬가지입니다.

침상 몸체 중심에 위치한 oriented bounding box의 네 모서리로 binary mask 사각형을 구성합니다. `float_area_threshold` 이하의 면적을 가진 mask와 ROI 엣지에서 `int_bboxEdgeMargin` 픽셀 이내에 걸친 mask는 제거됩니다 (불완전한 입자).

#### 침상당 측정값

| 필드 | 설명 |
|-------|-------------|
| `float_thicknessPx` / `float_thicknessUm` | Median 수직 너비 |
| `float_longAxisPx` / `float_longAxisUm` | LSD segment 길이 |
| `float_aspectRatio` | thickness / length |
| `float_minRectAngle` | Segment 방향각 (degrees) |
| `int_maskArea` | 사각형 mask의 픽셀 수 |
| `float_centroidX/Y` | Mask의 기하학적 중심 |

---

### 판상 모드 — SAM2 Pipeline

판상 입자는 형태가 너무 불규칙하여 LSD를 사용하기 어렵습니다. 대신 SAM2 (Segment Anything Model 2)를 사용하여 픽셀 수준의 정밀한 mask를 생성합니다.

#### Step 1 — Tiling

SAM2의 추론 해상도를 초과하는 이미지를 처리하기 위해 ROI를 겹치는 정사각형 타일로 분할합니다.

```
tile_size (기본값 192 px)  ×  stride (기본값 96 px)
```

엣지 타일은 ROI 경계까지 확장하고, 중복 타일 위치는 제거됩니다.

#### Step 2 — Interest-Point Sampling (Shi-Tomasi)

각 타일에서 후보 point prompt를 추출합니다.

1. **Texture enhancement** — CLAHE + Gaussian blur + Sobel gradient magnitude + Laplacian + morphological blackhat을 단일 enhanced grayscale 채널로 혼합
2. Enhanced 타일에서 **Shi-Tomasi corner detection** (`cv2.goodFeaturesToTrack`), 최대 `points_per_tile`개 (preset에 따라 기본값 80–150)
3. 요청된 수보다 적은 point가 탐지되면, enhanced 이미지를 Otsu-threshold 처리하고 contour centroid를 추가 후보로 사용
4. 최종 후보 집합: `point_min_distance` px 이상 간격으로 `points_per_tile`개까지 가장 강한 코너

#### Step 3 — SAM2 Batch Inference

Point를 `point_batch_size`개 (기본값 32) 단위로 SAM2에 전송합니다. 각 point에 foreground 라벨(label = 1)을 부여합니다. SAM2는 각 point에 대한 binary mask를 반환합니다.

각 raw mask는 `mask_binarize_threshold` (기본값 0.0, 즉 양수 logit → foreground)에서 이진화됩니다.

#### Step 4 — 타일 수준 필터링

SAM2가 반환한 각 mask에 대해:

1. **최소 면적:** `mask.sum() < int_minValidMaskArea`이면 제거
2. **Contour 추출:** 가장 큰 external contour 탐색; 없으면 제거
3. **타일 엣지 제외:** Bounding box가 타일 경계에서 `int_tileEdgeMargin` px 이내이면 제거 (타일 엣지에 잘린 입자는 인접 타일에서 측정)
4. **Bbox IoU 중복 제거:** 기존 수락된 mask와의 bounding-box IoU ≥ `float_bboxDedupIou` (기본값 0.85)이면 제거 — 비용이 큰 pixel 비교 전 빠른 사전 필터

#### Step 5 — ROI 수준 중복 제거

타일 mask를 ROI 좌표계로 재샘플링합니다. 각 후보와 기존 수락된 모든 mask 간의 pixel 수준 IoU를 계산합니다. IoU ≥ `float_dedupIou` (기본값 0.60)인 후보는 중복으로 제거합니다.

#### Step 6 — Mask 정제 (Morphology)

`int_maskMorphKernelSize > 1`이면 수락된 binary mask를 후처리합니다.

1. Morphological **open** (`int_maskMorphOpenIterations`회) — 작은 돌출부 제거
2. Morphological **close** (`int_maskMorphCloseIterations`회) — 내부 홀 채우기

#### Step 7 — minAreaRect를 통한 측정

정제된 mask의 가장 큰 contour에 `cv2.minAreaRect` (최소 면적 oriented bounding rectangle)을 피팅합니다.

```
thickness_px = min(rect_width, rect_height)
long_axis_px = max(rect_width, rect_height)
aspect_ratio = thickness_px / long_axis_px
```

5개 미만의 점을 가진 contour는 axis-aligned bounding box를 대신 사용합니다.

Mask를 가로지르는 가장 긴 연속 수평/수직 span도 기록됩니다 (`int_longestHorizontal`, `int_longestVertical`).

#### Step 8 — 분류

| 조건 | 분류 |
|-----------|----------|
| `mask_area < area_threshold` | 제외 (1차입자로 분류하지 않음) |
| `aspect_ratio < acicular_threshold` (0.40) | `acicular` |
| 그 외 | `plate` |

`--particle_type`이 `acicular`/`plate`로 설정되면 해당 분류만 유지됩니다. `active` 등 형태 무관 타입은 모두 유지됩니다.

#### Step 9 — Convex Hull 최종화

분류·필터링을 통과한 마스크는 convex hull을 적용해 최종 mask로 사용하고, hull 적용 후 좌표로 재측정합니다 (`--small`/`--active` 공통).

#### Acicular hybrid 모드 (`--particle_mode acicular`)

SAM2로 침상 입자를 측정하는 경우(LSD 비사용 모드), OpenCV 사전 필터로 SAM2 호출 횟수를 줄입니다.

1. CLAHE + adaptive threshold → erosion → `cv2.connectedComponentsWithStats`
2. 각 blob에서 image moment 고유값 계산: `AR = √(λ_min / λ_max)` (공분산 행렬의 고유값 비율 ≈ (단축 / 장축)²)
3. `AR < acicular_ar_screen`인 blob 유지 (충분히 길쭉한 후보)
4. Blob centroid를 SAM2 point prompt로 전송

후보가 3개 미만이면 표준 tiled point-prompt 모드로 fallback합니다.

---

## 2차 입자 파이프라인 (Secondary Particle Pipeline)

2차 입자는 구형 응집체입니다. 파이프라인은 1차 파이프라인의 SAM2 분기(tiling → Shi-Tomasi → SAM2 → dedup → morphology)와 동일하지만 측정 및 분류 방식이 다릅니다.

### 측정값

| 측정량 | Algorithm |
|----------|-----------|
| `float_eqDiameterUm` | Equivalent circle diameter: `2 × sqrt(mask_area / π)` → µm 변환 |
| `float_sphericity` | Wadell 2D isoperimetric ratio: `4π × area / perimeter²` (1.0 = 원, < 1.0 = 불규칙) |
| `float_bboxWidthUm`, `float_bboxHeightUm` | Bounding-box 크기 (µm) |

Sphericity 측정은 `cv2.arcLength`로 계산한 contour perimeter를 사용합니다. Particle 분류에만 계산되며 (fragment 제외), [0, 1] 범위로 제한됩니다.

### 분류

| 조건 | 분류 |
|-----------|----------|
| `mask_area < area_threshold` (기본값 1500 px²) | `fragment` |
| 그 외 | `particle` |

2차 입자 분석에서는 acicular/plate 구분이 없습니다.

---

## 출력 파일

### 1차 파이프라인 출력

| 파일 | 내용 |
|------|---------|
| `01_input.png` | 정규화된 입력 이미지 (2048×1536) |
| `02_input_roi.png` | 분석에 사용된 ROI |
| `03_overlay_roi.png` | Segmentation overlay (mask + contour) |
| `04_overlay_full.png` | ROI 사각형이 강조된 전체 이미지 |
| `05_opencv_candidates.png` | LSD 최종 선분: 초록=acicular, 주황=plate (LSD 모드) |
| `06_sphere_detection.png` | 구 중심 + cap ROI 시각화 (20k만 생성) |
| `lsd_01_preprocessed.png` | CLAHE + blur 결과 |
| `lsd_02_otsu_thresh.png` | Otsu binary mask |
| `lsd_02_adaptive_thresh.png` | Adaptive binary mask (`--lsd_adaptive_thresh` 시) |
| `lsd_03_raw_detections.png` | 모든 LSD segment, 노란색 |
| `lsd_04_after_filter.png` | 길이/AR 필터 후, 청록색 |
| `lsd_05_after_dedup.png` | 중복 제거 후, 주황색 |
| `lsd_06_after_fusion.png` | Segment fusion 후, 자홍색 (`--lsd_fuse_segments` 시) |
| `objects.csv` | 탐지된 모든 입자 |
| `acicular.csv` / `plate.csv` | 대상 유형 입자만 분리 |
| `thickness_dist.png` | 두께 분포 histogram |
| `summary.json` | 통계 및 분석 설정 (roi_density 포함) |
| `objects.json` | 입자별 상세 측정값 |
| `debug.json` | 타일, point, mask 디버그 정보 |
| `acicular_masks/` `plate_masks/` | 개별 mask PNG (`--save_mask_imgs` 시) |

### 2차 파이프라인 출력

| 파일 | 내용 |
|------|---------|
| `01_input.png` – `04_overlay_full.png` | 1차 파이프라인과 동일 |
| `objects.csv` | sphericity 포함 탐지된 모든 객체 |
| `particles.csv` | Particle 분류 객체만 |
| `particle_dist.png` | 평균 크기 분포 histogram |
| `sphericity_dist.png` | Sphericity 분포 histogram |
| `summary.json` | sphericity 및 크기 통계 포함 |
| `particle_masks/` `fragment_masks/` | 개별 mask PNG |

### Batch 모드 추가 출력

| 파일 | 내용 |
|------|---------|
| `<IMG_ID>/img_id_summary.json` | 그룹별 집계 통계 |
| `batch_summary.json` | 전체 배치 총계 및 평균 |

---

## 명령줄 사용법 (Command-Line Reference)

### 1차 입자 분석

```bash
python primary_measure.py \
  --input <경로>                   # 이미지 파일 또는 디렉터리
  --particle_type acicular|plate   # preset에 필요
  --magnification 20k|50k          # preset에 필요
  [--output_dir <경로>]            # 기본값: out_primary_<timestamp>
  [--config configs/paths.yaml]    # 선택적 경로 기본값 파일
```

#### 주요 override 옵션

```bash
# 비표준 스케일 바 (이미지에서 직접 측정)
--scale_pixels 206 --scale_um 1

# Otsu 대신 Adaptive thresholding 사용 (조명 불균일 시)
--lsd_adaptive_thresh

# 분절된 침상 탐지 합치기
--lsd_fuse_segments

# Acicular/plate 경계 조정 (기본값 0.40)
--acicular_threshold 0.35

# Particle로 인정되는 최소 mask 면적
--area_threshold 50

# SAM2 모델 경로 (plate 모드 또는 acicular SAM2 모드)
--model model/sam2.1_hiera_base_plus.pt
--model_cfg model/sam2.1_hiera_t.yaml
```

#### LSD Threshold 및 Fusion 옵션

```bash
# Adaptive threshold + segment fusion (조명 불균일 / 분절 침상에 권장)
python primary_measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k \
  --lsd_adaptive_thresh \
  --lsd_fuse_segments

# Otsu threshold + segment fusion
python primary_measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k \
  --no-lsd_adaptive_thresh \
  --lsd_fuse_segments

# Adaptive threshold, fusion 없음
python primary_measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k \
  --lsd_adaptive_thresh \
  --no-lsd_fuse_segments
```

#### 사용 예시

```bash
# 침상, 50k 표면 클로즈업 — 모델 불필요
python primary_measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k

# 침상, 20k 구 검출 포함
python primary_measure.py \
  --input sem_20k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --output_dir out_acicular_20k

# 판상, 50k — SAM2 필요
python primary_measure.py \
  --input sem_plate_50k.jpg \
  --particle_type plate \
  --magnification 50k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml

# 배치 처리
python primary_measure.py \
  --input ./samples/ \
  --particle_type acicular \
  --magnification 50k \
  --output_dir out_batch
```

### 2차 입자 분석

```bash
python secondary_measure.py \
  --input <경로>                   # 이미지 파일 또는 디렉터리
  [--output_dir <경로>]
  [--model model/sam2.1_hiera_base_plus.pt]
  [--model_cfg model/sam2.1_hiera_t.yaml]
  [--small_particle]               # 50k scale 사용 (371 px/µm, 기본값: 20k 147 px/µm)
  [--scale_pixels 371]             # 직접 지정 시 --small_particle보다 우선
  [--scale_um 1]
  [--area_threshold 1500]          # Particle로 인정되는 최소 px²
  [--config configs/paths.yaml]
```

#### 사용 예시

```bash
python secondary_measure.py \
  --input sem_secondary.jpg \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml \
  --output_dir out_secondary
```

### `configs/paths.yaml` — 기본 경로 설정

매번 실행 시 모델 및 디렉터리 경로를 반복 입력하는 번거로움을 줄입니다. CLI 인자가 항상 최우선입니다.

```yaml
input: "img/sample.jpg"
output_dir: ""                         # 비어 있으면 자동 timestamp 이름 사용
model: "model/sam2.1_hiera_base_plus.pt"
model_cfg: "model/sam2.1_hiera_t.yaml"
device: ""                             # 비어 있으면 자동 (cpu / cuda)
```

---

## Preset 파라미터 참조표

`--particle_type` × `--magnification`으로 preset이 선택됩니다. 개별 파라미터는 CLI에서 언제든 override할 수 있습니다.

| 파라미터 | acicular/20k | acicular/50k | plate/20k | plate/50k |
|-----------|-------------|-------------|----------|----------|
| `measure_mode` | lsd | lsd | sam2 | sam2 |
| `scale_pixels` | 147 | 371 | 147 | 371 |
| `scale_um` | 1 | 1 | 1 | 1 |
| `particle_mode` | acicular | acicular | auto | auto |
| `auto_detect_sphere` | true | false | true | false |
| `sphere_cap_fraction` | 0.65 | — | 0.65 | — |
| `center_crop_ratio` | 0.60 | 0.85 | 0.60 | 0.85 |
| `tile_size` | 192 | 192 | 192 | 192 |
| `stride` | 96 | 96 | 96 | 96 |
| `points_per_tile` | 120 | 150 | 100 | 100 |
| `point_min_distance` | 8 | 5 | 10 | 8 |
| `area_threshold` | 80 | 20 | 300 | 150 |

---

## 환경 설정

```bash
# conda 환경 활성화
conda activate measure

# 프로젝트 루트에서 실행
cd ~/Desktop/Projects/measure
python primary_measure.py --help
python secondary_measure.py --help
```

`measure` conda 환경 구성: Python 3.11, opencv-python, numpy, matplotlib, pyyaml, ultralytics (SAM2), torch, torchvision.

---

## 알고리즘 결정 트리

```
입력 SEM 이미지
       │
       ▼
Resize → 2048×1636, 하단 100 px 제거 → 2048×1536
       │
       ├─── primary_measure.py ──────────────────────────────────┐
       │                                                          │
       │    ROI 추출                                              │
       │      20k → sphere detect → cap crop                     │
       │      50k → center crop (85%)                            │
       │                                                          │
       │    particle_type=acicular                                │
       │    measure_mode=lsd                                      │
       │      CLAHE → blur → Otsu/Adaptive threshold              │
       │      → LSD → 길이/AR 필터 → dedup                       │
       │      → [optional fusion]                                 │
       │      → perpendicular profile → 두께 (7개 median)         │
       │      → AR = thickness/length → acicular/plate 분류       │
       │      → 침상당 사각형 mask 생성                            │
       │      → density = white_px / total_px                     │
       │                                                          │
       │    particle_type=plate                                   │
       │    measure_mode=sam2                                     │
       │      Tiling → Shi-Tomasi points → SAM2 batch inference   │
       │      → IoU dedup → morphology → minAreaRect              │
       │      → AR = 단축/장축 → acicular/plate 분류              │
       │      → density = Otsu binary의 white_px / total_px       │
       │                                                          │
       │    출력: thickness_um, long_axis_um, AR, density 등      │
       │                                                          │
       └─── secondary_measure.py ────────────────────────────────┘
                                                                  │
            ROI 추출 (수동 또는 기본 전체 이미지)                  │
                                                                  │
            Tiling → Shi-Tomasi points → SAM2 batch inference     │
            → IoU dedup → morphology                              │
            → 최장 H/V span → AR = min/max span                   │
            → sphericity = 4π·area / perimeter²                   │
            → 면적 기준 particle/fragment 분류                     │
                                                                  │
            출력: size_um, AR, sphericity per particle            ▼
```
