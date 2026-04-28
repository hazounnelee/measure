# Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `core.py` (~2200 lines) and `measure.py` (~2270 lines) into a clean package with single-responsibility modules, making the codebase navigable and maintainable.

**Architecture:** Dataclasses live in `core/schema.py`; pure algorithmic functions in `utils/`; preset config in `configs/presets.yaml`; SAM2 model loader in `models/`; the service class that assembles everything in `services/primary_particle.py`; CLI entry point in `primary_measure.py`. No new functionality — pure structural migration.

**Tech Stack:** Python 3.x, OpenCV (`cv2`), NumPy, PyYAML, ultralytics (optional/lazy), matplotlib

---

## File Map

| New file | Responsibility | Source |
|---|---|---|
| `core/__init__.py` | re-exports from schema | new |
| `core/schema.py` | all dataclasses | core.py:109-174, measure.py:248-300 |
| `configs/__init__.py` | `get_analysis_preset()` | measure.py:230-239 |
| `configs/presets.yaml` | DICT_PRESETS as YAML | measure.py:166-227 |
| `data/__init__.py` | placeholder | new |
| `models/__init__.py` | lazy SAM2 loader, `load_sam2_model()` | core.py:39-44 |
| `utils/__init__.py` | re-exports | new |
| `utils/metrics.py` | unit conversion, stats helpers, json_default | core.py:177-215, measure.py:99-107 |
| `utils/iou.py` | `calculate_binary_iou`, `calculate_box_iou` | core.py:378-427 |
| `utils/image.py` | CLAHE enhance, Shi-Tomasi points, tiling, sphere detect, center crop | core.py:219-375, measure.py:327-490 |
| `utils/lsd.py` | LSD constants, `measure_perpendicular_thickness`, `detect_acicular_lsd` | measure.py:146-151, 881-1141 |
| `utils/io.py` | `collect_input_groups`, `build_image_output_dir`, `iter_chunks` | core.py:429-477, 1550-1660 |
| `services/__init__.py` | re-exports | new |
| `services/sam2_service.py` | `Sam2AspectRatioService` base class | core.py:577-1548 |
| `services/primary_particle.py` | `PrimaryParticleService`, `run_primary_particle_analysis` | measure.py:308-1984 |
| `primary_measure.py` | CLI arg-parse + call service | measure.py:1986-2265 |

**Deleted after Task 13:** `core.py`, `measure.py`

---

## Task 1: Directory skeleton

**Files:**
- Create: `core/__init__.py`
- Create: `configs/__init__.py`
- Create: `data/__init__.py`
- Create: `models/__init__.py`
- Create: `utils/__init__.py`
- Create: `services/__init__.py`

- [ ] **Step 1: Create all directories and empty `__init__.py` files**

```bash
mkdir -p core configs data models utils services
touch core/__init__.py configs/__init__.py data/__init__.py models/__init__.py utils/__init__.py services/__init__.py
```

- [ ] **Step 2: Verify structure**

```bash
find . -name "__init__.py" | sort
```
Expected output includes all 6 paths above.

- [ ] **Step 3: Commit**

```bash
git add core/ configs/ data/ models/ utils/ services/
git commit -m "chore: create package directory skeleton"
```

---

## Task 2: `core/schema.py` — all dataclasses

**Files:**
- Create: `core/schema.py`
- Modify: `core/__init__.py`

- [ ] **Step 1: Create `core/schema.py`**

```python
from __future__ import annotations
import typing as tp
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sam2AspectRatioConfig:
    path_input: Path = Path(".")
    path_outputDir: Path = Path("out")
    path_modelConfig: Path = Path("model/sam2.1_hiera_t.yaml")
    path_modelWeights: Path = Path("model/sam2.1_hiera_base_plus.pt")
    int_roiXMin: int = 0
    int_roiYMin: int = 0
    int_roiXMax: int = 1024
    int_roiYMax: int = 768
    int_bboxEdgeMargin: int = 8
    int_tileEdgeMargin: int = 8
    float_particleAreaThreshold: float = 1500.0
    float_maskBinarizeThreshold: float = 0.0
    int_minValidMaskArea: int = 1
    int_maskMorphKernelSize: int = 0
    int_maskMorphOpenIterations: int = 0
    int_maskMorphCloseIterations: int = 0
    int_imgSize: int = 1536
    int_tileSize: int = 512
    int_stride: int = 256
    int_pointsPerTile: int = 80
    int_pointMinDistance: int = 14
    float_pointQualityLevel: float = 0.03
    int_pointBatchSize: int = 32
    float_dedupIou: float = 0.60
    float_bboxDedupIou: float = 0.85
    bool_usePointPrompts: bool = True
    bool_smallParticle: bool = False
    float_scalePixels: float = 276.0
    float_scaleMicrometers: float = 50.0
    str_device: str = "cpu"
    bool_retinaMasks: bool = True
    bool_saveIndividualMasks: bool = True


@dataclass
class ObjectMeasurement:
    int_index: int
    str_category: str
    int_maskArea: int
    float_confidence: tp.Optional[float]
    int_bboxX: int
    int_bboxY: int
    int_bboxWidth: int
    int_bboxHeight: int
    float_bboxWidthUm: float
    float_bboxHeightUm: float
    float_centroidX: float
    float_centroidY: float
    int_longestHorizontal: int
    int_longestVertical: int
    float_longestHorizontalUm: float
    float_longestVerticalUm: float
    float_aspectRatioWH: float


@dataclass
class Sam2AspectRatioResult:
    list_objects: tp.List[ObjectMeasurement]
    dict_summary: tp.Dict[str, tp.Any]


@dataclass
class PrimaryParticleConfig(Sam2AspectRatioConfig):
    float_acicularThreshold: float = 0.40
    bool_autoCenterCrop: bool = True
    float_centerCropRatio: float = 0.60
    int_targetParticleCount: int = 10
    str_particleMode: str = "auto"
    bool_autoDetectSphere: bool = False
    float_sphereCapFraction: float = 0.45
    str_particleType: str = "unknown"
    str_magnification: str = "unknown"
    str_measureMode: str = "sam2"


@dataclass
class PrimaryParticleMeasurement:
    int_index: int
    str_category: str
    int_maskArea: int
    float_confidence: tp.Optional[float]
    int_bboxX: int
    int_bboxY: int
    int_bboxWidth: int
    int_bboxHeight: int
    float_centroidX: float
    float_centroidY: float
    float_thicknessPx: float
    float_longAxisPx: float
    float_minRectAngle: float
    float_thicknessUm: float
    float_longAxisUm: float
    float_aspectRatio: float
    int_longestHorizontal: int
    int_longestVertical: int
    float_longestHorizontalUm: float
    float_longestVerticalUm: float


@dataclass
class PrimaryParticleResult:
    list_objects: tp.List[PrimaryParticleMeasurement]
    dict_summary: tp.Dict[str, tp.Any]
```

- [ ] **Step 2: Update `core/__init__.py`**

```python
from core.schema import (
    Sam2AspectRatioConfig,
    ObjectMeasurement,
    Sam2AspectRatioResult,
    PrimaryParticleConfig,
    PrimaryParticleMeasurement,
    PrimaryParticleResult,
)

__all__ = [
    "Sam2AspectRatioConfig",
    "ObjectMeasurement",
    "Sam2AspectRatioResult",
    "PrimaryParticleConfig",
    "PrimaryParticleMeasurement",
    "PrimaryParticleResult",
]
```

- [ ] **Step 3: Smoke test**

```bash
python -c "from core import PrimaryParticleConfig, PrimaryParticleMeasurement; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add core/
git commit -m "feat: add core/schema.py with all dataclasses"
```

---

## Task 3: `configs/` — YAML presets

**Files:**
- Create: `configs/presets.yaml`
- Modify: `configs/__init__.py`

- [ ] **Step 1: Create `configs/presets.yaml`**

```yaml
acicular:
  20k:
    scale_pixels: 276.0
    scale_um: 50.0
    particle_mode: acicular
    measure_mode: lsd
    auto_detect_sphere: true
    sphere_cap_fraction: 0.65
    auto_center_crop: true
    center_crop_ratio: 0.60
    tile_size: 192
    stride: 96
    points_per_tile: 120
    point_min_distance: 8
    area_threshold: 80.0
  50k:
    scale_pixels: 184.0
    scale_um: 10.0
    particle_mode: acicular
    measure_mode: lsd
    auto_detect_sphere: false
    auto_center_crop: true
    center_crop_ratio: 0.85
    tile_size: 192
    stride: 96
    points_per_tile: 150
    point_min_distance: 5
    area_threshold: 20.0
plate:
  20k:
    scale_pixels: 276.0
    scale_um: 50.0
    particle_mode: auto
    auto_detect_sphere: true
    sphere_cap_fraction: 0.65
    auto_center_crop: true
    center_crop_ratio: 0.60
    tile_size: 192
    stride: 96
    points_per_tile: 100
    point_min_distance: 10
    area_threshold: 300.0
  50k:
    scale_pixels: 184.0
    scale_um: 10.0
    particle_mode: auto
    auto_detect_sphere: false
    auto_center_crop: true
    center_crop_ratio: 0.85
    tile_size: 192
    stride: 96
    points_per_tile: 100
    point_min_distance: 8
    area_threshold: 150.0
```

- [ ] **Step 2: Create `configs/__init__.py`**

```python
from __future__ import annotations
import typing as tp
from pathlib import Path
import yaml

_PRESETS_PATH = Path(__file__).parent / "presets.yaml"
_PRESETS: tp.Optional[tp.Dict[str, tp.Any]] = None


def _load() -> tp.Dict[str, tp.Any]:
    global _PRESETS
    if _PRESETS is None:
        with _PRESETS_PATH.open(encoding="utf-8") as f:
            _PRESETS = yaml.safe_load(f)
    return _PRESETS


def get_analysis_preset(
    str_particleType: str,
    str_magnification: str,
) -> tp.Dict[str, tp.Any]:
    """Return preset dict for particle_type × magnification, or {} if not found."""
    data = _load()
    return dict(data.get(str_particleType, {}).get(str_magnification, {}))
```

- [ ] **Step 3: Verify PyYAML is available and preset loads correctly**

```bash
python -c "
from configs import get_analysis_preset
p = get_analysis_preset('acicular', '20k')
assert p['measure_mode'] == 'lsd'
assert p['scale_pixels'] == 276.0
p2 = get_analysis_preset('plate', '50k')
assert p2['particle_mode'] == 'auto'
print('OK', p)
"
```
Expected: prints `OK` then the acicular/20k dict.

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "feat: add configs/presets.yaml and loader"
```

---

## Task 4: `models/__init__.py` — lazy SAM2 loader

**Files:**
- Modify: `models/__init__.py`

- [ ] **Step 1: Write `models/__init__.py`**

```python
from __future__ import annotations
import typing as tp

try:
    from ultralytics import SAM as SAM
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    SAM = None  # type: ignore[assignment,misc]
    _ULTRALYTICS_AVAILABLE = False


def load_sam2_model(str_weights_path: str) -> tp.Any:
    """Load and return a SAM2 model instance.

    Raises ImportError if ultralytics is not installed.
    """
    if not _ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "ultralytics 패키지가 없습니다. SAM2 모드에는 ultralytics가 필요합니다."
        )
    return SAM(str_weights_path)
```

- [ ] **Step 2: Verify import works without ultralytics**

```bash
python -c "from models import _ULTRALYTICS_AVAILABLE; print('available:', _ULTRALYTICS_AVAILABLE)"
```
Expected: prints `available: True` or `available: False` without crashing.

- [ ] **Step 3: Commit**

```bash
git add models/
git commit -m "feat: add models/__init__.py with lazy SAM2 loader"
```

---

## Task 5: `utils/metrics.py` — unit conversion and stats

**Files:**
- Create: `utils/metrics.py`

- [ ] **Step 1: Create `utils/metrics.py`**

```python
from __future__ import annotations
import typing as tp
import numpy as np


def normalize_image_to_uint8(arr_img: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 [0, 255]."""
    arr_f = arr_img.astype(np.float32)
    float_min = float(arr_f.min())
    float_max = float(arr_f.max())
    if float_max - float_min < 1e-6:
        return np.zeros_like(arr_img, dtype=np.uint8)
    arr_norm = (arr_f - float_min) / (float_max - float_min) * 255.0
    return arr_norm.astype(np.uint8)


def convert_pixels_to_micrometers(
    float_pixels: float,
    float_scalePixels: float,
    float_scaleMicrometers: float,
) -> float:
    """Convert pixel length to micrometers using scale bar calibration."""
    if float_scalePixels <= 0:
        return 0.0
    return float_pixels * (float_scaleMicrometers / float_scalePixels)


def calculate_mean_from_optional_values(
    list_values: tp.List[tp.Optional[float]],
) -> tp.Optional[float]:
    """Return mean of non-None values, or None if list is empty."""
    valid = [v for v in list_values if v is not None]
    return float(np.mean(valid)) if valid else None


def calculate_percentage(
    int_part: int,
    int_total: int,
) -> tp.Optional[float]:
    """Return part/total as percentage (0-100), or None if total is 0."""
    if int_total == 0:
        return None
    return round(100.0 * int_part / int_total, 2)


def json_default(obj: tp.Any) -> tp.Any:
    """Custom JSON default: convert numpy scalar/array to Python native type."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from utils.metrics import convert_pixels_to_micrometers, json_default
import numpy as np
assert abs(convert_pixels_to_micrometers(206.0, 206.0, 5.0) - 5.0) < 1e-9
assert json_default(np.float32(1.5)) == 1.5
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Update `utils/__init__.py`**

```python
from utils.metrics import (
    normalize_image_to_uint8,
    convert_pixels_to_micrometers,
    calculate_mean_from_optional_values,
    calculate_percentage,
    json_default,
)
```

- [ ] **Step 4: Commit**

```bash
git add utils/metrics.py utils/__init__.py
git commit -m "feat: add utils/metrics.py"
```

---

## Task 6: `utils/iou.py` — IoU helpers

**Files:**
- Create: `utils/iou.py`

- [ ] **Step 1: Create `utils/iou.py`**

Copy the exact implementations from `core.py` lines 378–427.

```python
from __future__ import annotations
import numpy as np


def calculate_binary_iou(arr_maskA: np.ndarray, arr_maskB: np.ndarray) -> float:
    """Pixel-level IoU between two binary masks."""
    bool_inter = np.logical_and(arr_maskA > 0, arr_maskB > 0)
    bool_union = np.logical_or(arr_maskA > 0, arr_maskB > 0)
    int_union = int(bool_union.sum())
    if int_union == 0:
        return 0.0
    return float(bool_inter.sum()) / float(int_union)


def calculate_box_iou(
    tpl_boxA: tuple[int, int, int, int],
    tpl_boxB: tuple[int, int, int, int],
) -> float:
    """Bounding-box IoU. Boxes are (x, y, w, h)."""
    int_ax, int_ay, int_aw, int_ah = tpl_boxA
    int_bx, int_by, int_bw, int_bh = tpl_boxB
    int_interX1 = max(int_ax, int_bx)
    int_interY1 = max(int_ay, int_by)
    int_interX2 = min(int_ax + int_aw, int_bx + int_bw)
    int_interY2 = min(int_ay + int_ah, int_by + int_bh)
    int_interW = max(0, int_interX2 - int_interX1)
    int_interH = max(0, int_interY2 - int_interY1)
    int_interArea = int_interW * int_interH
    int_aArea = int_aw * int_ah
    int_bArea = int_bw * int_bh
    int_unionArea = int_aArea + int_bArea - int_interArea
    if int_unionArea == 0:
        return 0.0
    return float(int_interArea) / float(int_unionArea)
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
import numpy as np
from utils.iou import calculate_binary_iou, calculate_box_iou
a = np.array([[1,1,0],[1,1,0],[0,0,0]])
b = np.array([[0,1,1],[0,1,1],[0,0,0]])
assert abs(calculate_binary_iou(a, b) - 2/6) < 1e-9
assert calculate_box_iou((0,0,2,2),(1,1,2,2)) == 1/7
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Update `utils/__init__.py`** — append:

```python
from utils.iou import calculate_binary_iou, calculate_box_iou
```

- [ ] **Step 4: Commit**

```bash
git add utils/iou.py utils/__init__.py
git commit -m "feat: add utils/iou.py"
```

---

## Task 7: `utils/image.py` — image processing helpers

**Files:**
- Create: `utils/image.py`

- [ ] **Step 1: Create `utils/image.py`**

Copy implementations from `core.py` (enhance_image_texture:271, sample_interest_points:299, create_processing_tiles:219) and extract sphere/crop logic from `measure.py` (327-490) into pure functions with explicit parameters.

```python
from __future__ import annotations
import typing as tp
import cv2
import numpy as np


def create_processing_tiles(
    int_x1: int,
    int_y1: int,
    int_x2: int,
    int_y2: int,
    int_tileSize: int,
    int_stride: int,
) -> tp.List[tp.Tuple[int, int, int, int]]:
    """Divide ROI into overlapping tiles. Returns list of (x1, y1, x2, y2)."""
    list_tiles: tp.List[tp.Tuple[int, int, int, int]] = []
    int_roiW = int_x2 - int_x1
    int_roiH = int_y2 - int_y1
    if int_roiW <= 0 or int_roiH <= 0:
        return list_tiles
    int_tileW = min(int_tileSize, int_roiW)
    int_tileH = min(int_tileSize, int_roiH)
    int_y = 0
    while int_y < int_roiH:
        int_x = 0
        while int_x < int_roiW:
            int_tx1 = int_x1 + int_x
            int_ty1 = int_y1 + int_y
            int_tx2 = min(int_x2, int_tx1 + int_tileW)
            int_ty2 = min(int_y2, int_ty1 + int_tileH)
            list_tiles.append((int_tx1, int_ty1, int_tx2, int_ty2))
            int_x += int_stride
            if int_x + int_tileW > int_roiW and int_x < int_roiW:
                int_x = max(0, int_roiW - int_tileW)
                int_tx1 = int_x1 + int_x
                int_ty1 = int_y1 + int_y
                int_tx2 = min(int_x2, int_tx1 + int_tileW)
                int_ty2 = min(int_y2, int_ty1 + int_tileH)
                list_tiles.append((int_tx1, int_ty1, int_tx2, int_ty2))
                break
        int_y += int_stride
        if int_y + int_tileH > int_roiH and int_y < int_roiH:
            int_y = max(0, int_roiH - int_tileH)
            int_x = 0
            while int_x < int_roiW:
                int_tx1 = int_x1 + int_x
                int_ty1 = int_y1 + int_y
                int_tx2 = min(int_x2, int_tx1 + int_tileW)
                int_ty2 = min(int_y2, int_ty1 + int_tileH)
                list_tiles.append((int_tx1, int_ty1, int_tx2, int_ty2))
                int_x += int_stride
                if int_x + int_tileW > int_roiW and int_x < int_roiW:
                    int_x = max(0, int_roiW - int_tileW)
                    list_tiles.append((
                        int_x1 + int_x, int_y1 + int_y,
                        int_x2, int_y2,
                    ))
                    break
            break
    # deduplicate while preserving order
    seen: tp.Set[tp.Tuple[int, int, int, int]] = set()
    list_dedup: tp.List[tp.Tuple[int, int, int, int]] = []
    for tpl in list_tiles:
        if tpl not in seen:
            seen.add(tpl)
            list_dedup.append(tpl)
    return list_dedup


def enhance_image_texture(arr_tileGray: np.ndarray) -> np.ndarray:
    """CLAHE + gradient + Laplacian texture enhancement for point detection."""
    obj_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    arr_eq = obj_clahe.apply(arr_tileGray)
    arr_blur = cv2.GaussianBlur(arr_eq, (3, 3), 0)
    arr_grad = cv2.Sobel(arr_blur, cv2.CV_32F, 1, 0) ** 2
    arr_grad += cv2.Sobel(arr_blur, cv2.CV_32F, 0, 1) ** 2
    arr_grad = np.sqrt(arr_grad).astype(np.uint8)
    arr_lap = np.abs(cv2.Laplacian(arr_blur, cv2.CV_32F)).astype(np.uint8)
    arr_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    arr_blackhat = cv2.morphologyEx(arr_eq, cv2.MORPH_BLACKHAT, arr_kernel)
    arr_enhanced = cv2.addWeighted(arr_eq, 0.4, arr_grad, 0.3, 0)
    arr_enhanced = cv2.addWeighted(arr_enhanced, 0.8, arr_lap, 0.1, 0)
    arr_enhanced = cv2.addWeighted(arr_enhanced, 0.9, arr_blackhat, 0.1, 0)
    return arr_enhanced


def sample_interest_points(
    arr_tileGray: np.ndarray,
    int_maxPoints: int,
    int_minDist: int,
    float_qualityLevel: float,
) -> np.ndarray:
    """Shi-Tomasi corner detection on a tile. Returns (N, 2) float32 array of (x, y)."""
    arr_enhanced = enhance_image_texture(arr_tileGray)
    arr_corners = cv2.goodFeaturesToTrack(
        arr_enhanced,
        maxCorners=int_maxPoints,
        qualityLevel=float_qualityLevel,
        minDistance=float(int_minDist),
    )
    if arr_corners is not None:
        return arr_corners.reshape(-1, 2)
    # fallback: contour centroids
    _, arr_thresh = cv2.threshold(arr_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    list_cnts, _ = cv2.findContours(arr_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_pts: tp.List[tp.List[float]] = []
    for cnt in list_cnts:
        obj_m = cv2.moments(cnt)
        if obj_m["m00"] > 0:
            list_pts.append([obj_m["m10"] / obj_m["m00"], obj_m["m01"] / obj_m["m00"]])
    if list_pts:
        return np.array(list_pts, dtype=np.float32)
    int_h, int_w = arr_tileGray.shape[:2]
    return np.array([[int_w / 2.0, int_h / 2.0]], dtype=np.float32)


def detect_sphere_roi(
    arr_image_bgr: np.ndarray,
    float_cap_fraction: float = 0.65,
    int_morph_kernel: int = 15,
    float_min_radius_ratio: float = 0.15,
) -> tp.Optional[tp.Tuple[tp.Tuple[int, int, int, int], np.ndarray]]:
    """Detect spherical secondary particle, return cap ROI coords + debug mask.

    Returns:
        ((x1, y1, x2, y2), arr_debug_mask) or None if detection fails.
    """
    arr_gray = cv2.cvtColor(arr_image_bgr, cv2.COLOR_BGR2GRAY)
    int_h, int_w = arr_gray.shape[:2]
    arr_blur = cv2.GaussianBlur(arr_gray, (21, 21), 0)
    _, arr_thresh = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(arr_thresh.sum()) / (255.0 * int_h * int_w) > 0.5:
        arr_thresh = cv2.bitwise_not(arr_thresh)
    int_k = int_morph_kernel
    arr_kernel = np.ones((int_k, int_k), np.uint8)
    arr_closed = cv2.morphologyEx(arr_thresh, cv2.MORPH_CLOSE, arr_kernel, iterations=3)
    arr_opened = cv2.morphologyEx(arr_closed, cv2.MORPH_OPEN, arr_kernel, iterations=2)
    list_cnts, _ = cv2.findContours(arr_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not list_cnts:
        return None
    arr_cnt = max(list_cnts, key=cv2.contourArea)
    if float(cv2.contourArea(arr_cnt)) < int_h * int_w * 0.02:
        return None
    (float_cx, float_cy), float_r = cv2.minEnclosingCircle(arr_cnt)
    int_cx, int_cy, int_r = int(float_cx), int(float_cy), int(float_r)
    if int_r < int(min(int_h, int_w) * float_min_radius_ratio):
        return None
    float_cap = float(np.clip(float_cap_fraction, 0.1, 1.0))
    int_y1 = max(0, int_cy - int_r)
    int_y2 = min(int_h, int_y1 + int(int_r * 2 * float_cap))
    int_x1 = max(0, int_cx - int_r)
    int_x2 = min(int_w, int_cx + int_r)
    if int_x2 <= int_x1 or int_y2 <= int_y1:
        return None
    arr_debug = np.zeros((int_h, int_w), dtype=np.uint8)
    cv2.circle(arr_debug, (int_cx, int_cy), int_r, 255, 2)
    cv2.rectangle(arr_debug, (int_x1, int_y1), (int_x2, int_y2), 128, 2)
    print(
        f"[sphere-detect] 구 검출: center=({int_cx},{int_cy}) r={int_r}px  "
        f"cap ROI=({int_x1},{int_y1})-({int_x2},{int_y2})",
        flush=True,
    )
    return (int_x1, int_y1, int_x2, int_y2), arr_debug


def compute_center_roi(
    int_h: int,
    int_w: int,
    float_crop_ratio: float,
) -> tp.Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) for a centered crop of the given ratio."""
    float_ratio = float(np.clip(float_crop_ratio, 0.1, 1.0))
    int_xm = int(int_w * (1.0 - float_ratio) / 2.0)
    int_ym = int(int_h * (1.0 - float_ratio) / 2.0)
    return max(0, int_xm), max(0, int_ym), min(int_w, int_w - int_xm), min(int_h, int_h - int_ym)
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from utils.image import compute_center_roi, create_processing_tiles
x0, y0, x1, y1 = compute_center_roi(100, 200, 0.5)
assert x0 == 50 and y0 == 25 and x1 == 150 and y1 == 75
tiles = create_processing_tiles(0, 0, 100, 100, 64, 32)
assert len(tiles) > 0
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Update `utils/__init__.py`** — append:

```python
from utils.image import (
    create_processing_tiles,
    enhance_image_texture,
    sample_interest_points,
    detect_sphere_roi,
    compute_center_roi,
)
```

- [ ] **Step 4: Commit**

```bash
git add utils/image.py utils/__init__.py
git commit -m "feat: add utils/image.py"
```

---

## Task 8: `utils/lsd.py` — LSD detection and perpendicular thickness

**Files:**
- Create: `utils/lsd.py`

- [ ] **Step 1: Create `utils/lsd.py`**

Extract constants and methods from `measure.py` (lines 146–151, 881–1141). `detect_acicular_lsd` replaces `analyze_with_lsd` — it receives explicit params instead of `self`.

```python
from __future__ import annotations
import typing as tp
import cv2
import numpy as np

from core.schema import PrimaryParticleMeasurement
from utils.metrics import convert_pixels_to_micrometers

# LSD tuning constants
CONST_LSD_MIN_LENGTH_PX: int = 20
CONST_LSD_DEDUP_DIST_PX: int = 12
CONST_LSD_DEDUP_ANGLE_DEG: float = 25.0
CONST_LSD_PERP_N_SAMPLES: int = 7


def measure_perpendicular_thickness(
    arr_gray: np.ndarray,
    float_otsu_thresh: float,
    float_x1: float,
    float_y1: float,
    float_x2: float,
    float_y2: float,
    float_px_per_um: float,
) -> float:
    """Scan perpendicular to a line segment and return the needle width in pixels.

    Uses global Otsu threshold and picks the bright region closest to the scan
    center to avoid spanning multiple needles in dense images.

    Returns 0.0 if no valid sample found.
    """
    float_dx = float_x2 - float_x1
    float_dy = float_y2 - float_y1
    float_length = float(np.sqrt(float_dx ** 2 + float_dy ** 2))
    if float_length < 1.0:
        return 0.0
    float_ux = float_dx / float_length
    float_uy = float_dy / float_length
    float_px = -float_uy
    float_py = float_ux
    int_roiH, int_roiW = arr_gray.shape[:2]
    int_half_scan = max(15, int(0.5 * float_px_per_um))
    int_center = int_half_scan
    arr_scan = np.arange(-int_half_scan, int_half_scan + 1, dtype=np.float32)
    list_widths: tp.List[float] = []
    for float_t in np.linspace(0.2, 0.8, CONST_LSD_PERP_N_SAMPLES):
        float_sx = float_x1 + float_t * float_dx
        float_sy = float_y1 + float_t * float_dy
        arr_xs = np.clip(float_sx + float_px * arr_scan, 0, int_roiW - 1).astype(np.int32)
        arr_ys = np.clip(float_sy + float_py * arr_scan, 0, int_roiH - 1).astype(np.int32)
        arr_profile = arr_gray[arr_ys, arr_xs].astype(np.float32)
        arr_above = arr_profile > float_otsu_thresh
        if not arr_above.any():
            continue
        list_regions: tp.List[tp.Tuple[int, int]] = []
        bool_in = False
        int_start = 0
        for int_k, bool_v in enumerate(arr_above.tolist()):
            if bool_v and not bool_in:
                bool_in = True
                int_start = int_k
            elif not bool_v and bool_in:
                bool_in = False
                list_regions.append((int_start, int_k - 1))
        if bool_in:
            list_regions.append((int_start, len(arr_above) - 1))
        if not list_regions:
            continue

        def _dist(tpl: tp.Tuple[int, int]) -> int:
            return min(abs(tpl[0] - int_center), abs(tpl[1] - int_center))

        tpl_best = min(list_regions, key=_dist)
        int_width = tpl_best[1] - tpl_best[0] + 1
        if int_width > 1:
            list_widths.append(float(int_width))
    return float(np.median(list_widths)) if list_widths else 0.0


def _is_bbox_near_edge(
    int_bx: int, int_by: int, int_bw: int, int_bh: int,
    int_roiW: int, int_roiH: int, int_margin: int,
) -> bool:
    return (
        int_bx < int_margin
        or int_by < int_margin
        or (int_bx + int_bw) > (int_roiW - int_margin)
        or (int_by + int_bh) > (int_roiH - int_margin)
    )


def detect_acicular_lsd(
    arr_roi_gray: np.ndarray,
    arr_roi_bgr: np.ndarray,
    float_acicular_threshold: float,
    str_particle_type: str,
    float_scale_pixels: float,
    float_scale_um: float,
    int_edge_margin: int = 8,
) -> tp.Tuple[tp.List[PrimaryParticleMeasurement], tp.List[np.ndarray], np.ndarray]:
    """Detect acicular particles with LSD and measure thickness via perpendicular profile.

    Args:
        arr_roi_gray: Grayscale ROI image.
        arr_roi_bgr: BGR ROI image (for debug visualization).
        float_acicular_threshold: AR < this → acicular.
        str_particle_type: "acicular" or "plate" (wrong-shape candidates are dropped).
        float_scale_pixels: Scale bar length in pixels.
        float_scale_um: Scale bar length in micrometers.
        int_edge_margin: Pixels from ROI edge to discard.

    Returns:
        (list_measurements, list_masks, arr_debug_bgr)
    """
    int_roiH, int_roiW = arr_roi_gray.shape[:2]
    float_px_per_um = float_scale_pixels / max(float_scale_um, 1e-9)

    obj_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_eq = obj_clahe.apply(arr_roi_gray)
    arr_blur = cv2.GaussianBlur(arr_eq, (3, 3), 0)

    float_otsu_thresh, _ = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    obj_lsd = cv2.createLineSegmentDetector(0)
    arr_lines, arr_widths, _, _ = obj_lsd.detect(arr_blur)

    list_objects: tp.List[PrimaryParticleMeasurement] = []
    list_masks: tp.List[np.ndarray] = []

    if arr_lines is None:
        return list_objects, list_masks, arr_roi_bgr.copy()

    float_ar_loose = min(float_acicular_threshold + 0.20, 0.65)
    list_cands: tp.List[tp.Dict[str, float]] = []
    for int_i, arr_line in enumerate(arr_lines):
        float_x1, float_y1, float_x2, float_y2 = arr_line[0]
        float_len = float(np.sqrt((float_x2 - float_x1) ** 2 + (float_y2 - float_y1) ** 2))
        if float_len < CONST_LSD_MIN_LENGTH_PX:
            continue
        float_lsd_w = float(arr_widths[int_i][0]) if arr_widths is not None else 5.0
        if float_len > 0 and float_lsd_w / float_len >= float_ar_loose:
            continue
        float_angle = float(np.degrees(np.arctan2(float_y2 - float_y1, float_x2 - float_x1)) % 180)
        list_cands.append({
            "x1": float_x1, "y1": float_y1, "x2": float_x2, "y2": float_y2,
            "length": float_len, "angle": float_angle,
        })

    list_cands.sort(key=lambda d: d["length"], reverse=True)
    list_accepted: tp.List[tp.Dict[str, float]] = []
    for dict_c in list_cands:
        float_cx = (dict_c["x1"] + dict_c["x2"]) / 2.0
        float_cy = (dict_c["y1"] + dict_c["y2"]) / 2.0
        bool_dup = False
        for dict_p in list_accepted:
            float_pcx = (dict_p["x1"] + dict_p["x2"]) / 2.0
            float_pcy = (dict_p["y1"] + dict_p["y2"]) / 2.0
            float_dist = float(np.sqrt((float_cx - float_pcx) ** 2 + (float_cy - float_pcy) ** 2))
            float_adiff = abs(dict_c["angle"] - dict_p["angle"])
            float_adiff = min(float_adiff, 180.0 - float_adiff)
            if float_dist < CONST_LSD_DEDUP_DIST_PX and float_adiff < CONST_LSD_DEDUP_ANGLE_DEG:
                bool_dup = True
                break
        if not bool_dup:
            list_accepted.append(dict_c)

    print(
        f"[LSD] 원본 {len(arr_lines)}개 → 필터 {len(list_cands)}개 "
        f"→ 중복제거 {len(list_accepted)}개",
        flush=True,
    )

    arr_debug = arr_roi_bgr.copy()

    for int_idx, dict_c in enumerate(list_accepted):
        float_x1 = dict_c["x1"]
        float_y1 = dict_c["y1"]
        float_x2 = dict_c["x2"]
        float_y2 = dict_c["y2"]
        float_len = dict_c["length"]

        float_thickness = measure_perpendicular_thickness(
            arr_blur, float_otsu_thresh,
            float_x1, float_y1, float_x2, float_y2,
            float_px_per_um,
        )
        if float_thickness < 2.0:
            continue

        float_ar = float_thickness / float_len

        if float_ar < float_acicular_threshold:
            str_category = "acicular" if str_particle_type != "plate" else "fragment"
        else:
            str_category = "plate" if str_particle_type != "acicular" else "fragment"

        if str_particle_type in ("acicular", "plate") and str_category == "fragment":
            continue

        float_ux = (float_x2 - float_x1) / max(float_len, 1.0)
        float_uy = (float_y2 - float_y1) / max(float_len, 1.0)
        float_half_t = float_thickness / 2.0
        float_px_dir = -float_uy
        float_py_dir = float_ux

        arr_corners = np.float32([
            [float_x1 - float_half_t * float_px_dir, float_y1 - float_half_t * float_py_dir],
            [float_x1 + float_half_t * float_px_dir, float_y1 + float_half_t * float_py_dir],
            [float_x2 + float_half_t * float_px_dir, float_y2 + float_half_t * float_py_dir],
            [float_x2 - float_half_t * float_px_dir, float_y2 - float_half_t * float_py_dir],
        ])

        int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_corners.astype(np.int32))
        int_bx = max(0, int_bx)
        int_by = max(0, int_by)
        int_bw = min(int_bw, int_roiW - int_bx)
        int_bh = min(int_bh, int_roiH - int_by)

        if _is_bbox_near_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH, int_edge_margin):
            continue

        arr_mask = np.zeros((int_roiH, int_roiW), dtype=np.uint8)
        cv2.fillPoly(arr_mask, [arr_corners.astype(np.int32).reshape(-1, 1, 2)], 1)

        float_mcx = (float_x1 + float_x2) / 2.0
        float_mcy = (float_y1 + float_y2) / 2.0

        list_objects.append(PrimaryParticleMeasurement(
            int_index=int_idx,
            str_category=str_category,
            int_maskArea=int(arr_mask.sum()),
            float_confidence=None,
            int_bboxX=int_bx,
            int_bboxY=int_by,
            int_bboxWidth=int_bw,
            int_bboxHeight=int_bh,
            float_centroidX=float_mcx,
            float_centroidY=float_mcy,
            float_thicknessPx=float_thickness,
            float_longAxisPx=float_len,
            float_minRectAngle=dict_c["angle"],
            float_thicknessUm=convert_pixels_to_micrometers(float_thickness, float_scale_pixels, float_scale_um),
            float_longAxisUm=convert_pixels_to_micrometers(float_len, float_scale_pixels, float_scale_um),
            float_aspectRatio=float_ar,
            int_longestHorizontal=int_bw,
            int_longestVertical=int_bh,
            float_longestHorizontalUm=convert_pixels_to_micrometers(float(int_bw), float_scale_pixels, float_scale_um),
            float_longestVerticalUm=convert_pixels_to_micrometers(float(int_bh), float_scale_pixels, float_scale_um),
        ))
        list_masks.append(arr_mask)

        cv2.line(arr_debug,
                 (int(float_x1), int(float_y1)), (int(float_x2), int(float_y2)),
                 (0, 255, 0) if str_category == "acicular" else (0, 128, 255), 1)

    print(f"[LSD] → 최종 {len(list_objects)}개", flush=True)
    return list_objects, list_masks, arr_debug
```

- [ ] **Step 2: Smoke test — import only (full functional test happens in Task 13)**

```bash
python -c "from utils.lsd import detect_acicular_lsd, measure_perpendicular_thickness; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Update `utils/__init__.py`** — append:

```python
from utils.lsd import detect_acicular_lsd, measure_perpendicular_thickness
```

- [ ] **Step 4: Commit**

```bash
git add utils/lsd.py utils/__init__.py
git commit -m "feat: add utils/lsd.py"
```

---

## Task 9: `utils/io.py` — file I/O helpers

**Files:**
- Create: `utils/io.py`

- [ ] **Step 1: Create `utils/io.py`**

Copy from `core.py` lines 429–477 (`iter_chunks`) and 1550–1610 (`collect_input_groups`, `build_image_output_dir`).

```python
from __future__ import annotations
import typing as tp
from pathlib import Path

CONST_SUPPORTED_IMAGE_SUFFIXES: tp.Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
)


def iter_chunks(lst: tp.List[tp.Any], int_n: int) -> tp.Iterator[tp.List[tp.Any]]:
    """Yield successive n-sized chunks from lst."""
    for int_i in range(0, len(lst), int_n):
        yield lst[int_i: int_i + int_n]


def collect_input_groups(
    path_input: Path,
) -> tp.List[tp.Tuple[str, tp.List[Path]]]:
    """Collect image groups from a file or directory.

    Single file → one group (group_id = stem).
    Directory   → one group per IMG_ID subfolder, or one group for flat images.

    Returns:
        List of (str_groupId, list_imagePaths).
    """
    if path_input.is_file():
        if path_input.suffix.lower() not in CONST_SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"지원하지 않는 이미지 형식: {path_input.suffix}")
        if not path_input.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path_input}")
        return [(path_input.stem, [path_input])]

    if not path_input.exists():
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {path_input}")

    # Subdirectory per IMG_ID
    list_subdirs = sorted([p for p in path_input.iterdir() if p.is_dir()])
    if list_subdirs:
        list_groups: tp.List[tp.Tuple[str, tp.List[Path]]] = []
        for path_sub in list_subdirs:
            list_imgs = sorted([
                p for p in path_sub.iterdir()
                if p.is_file() and p.suffix.lower() in CONST_SUPPORTED_IMAGE_SUFFIXES
            ])
            if list_imgs:
                list_groups.append((path_sub.name, list_imgs))
        if list_groups:
            return list_groups

    # Flat directory
    list_imgs = sorted([
        p for p in path_input.iterdir()
        if p.is_file() and p.suffix.lower() in CONST_SUPPORTED_IMAGE_SUFFIXES
    ])
    if not list_imgs:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path_input}")
    return [("batch", list_imgs)]


def build_image_output_dir(
    path_outputRoot: Path,
    str_groupId: str,
    path_image: Path,
    bool_isBatch: bool,
) -> Path:
    """Return per-image output directory path."""
    if bool_isBatch:
        return path_outputRoot / str_groupId / path_image.stem
    return path_outputRoot
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from utils.io import collect_input_groups, iter_chunks
from pathlib import Path
chunks = list(iter_chunks([1,2,3,4,5], 2))
assert chunks == [[1,2],[3,4],[5]]
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Update `utils/__init__.py`** — append:

```python
from utils.io import collect_input_groups, build_image_output_dir, iter_chunks
```

- [ ] **Step 4: Commit**

```bash
git add utils/io.py utils/__init__.py
git commit -m "feat: add utils/io.py"
```

---

## Task 10: `services/sam2_service.py` — SAM2 base service

**Files:**
- Create: `services/sam2_service.py`

- [ ] **Step 1: Create `services/sam2_service.py`**

Copy `Sam2AspectRatioService` from `core.py` lines 577–1548. Replace the old lazy-import block and `collect_input_groups`/`build_image_output_dir` calls with imports from `models` and `utils`.

Key substitutions vs. original `core.py`:
- `from ultralytics import SAM` → `from models import load_sam2_model`
- `self.obj_model = SAM(...)` → `self.obj_model = load_sam2_model(...)`
- `convert_pixels_to_micrometers(px, self.obj_config.float_scalePixels, ...)` → import from `utils.metrics`
- `calculate_binary_iou` / `calculate_box_iou` → import from `utils.iou`
- `create_processing_tiles` / `enhance_image_texture` / `sample_interest_points` → import from `utils.image`
- `iter_chunks` → import from `utils.io`
- Dataclasses → import from `core.schema`

The class body (methods) is copied verbatim from `core.py` — no logic changes.

```python
from __future__ import annotations
# ... (copy full class from core.py:577-1548, updating imports as above)
```

> **Note for implementer:** The full class is ~970 lines. Copy it exactly from `core.py` starting at line 577 up to (but not including) the standalone functions after line 1548. Replace the import block at the top of the file as described above.

- [ ] **Step 2: Smoke test — import only**

```bash
python -c "from services.sam2_service import Sam2AspectRatioService; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Update `services/__init__.py`**

```python
from services.sam2_service import Sam2AspectRatioService
```

- [ ] **Step 4: Commit**

```bash
git add services/
git commit -m "feat: add services/sam2_service.py"
```

---

## Task 11: `services/primary_particle.py` — primary particle service

**Files:**
- Create: `services/primary_particle.py`

- [ ] **Step 1: Create `services/primary_particle.py`**

Copy `PrimaryParticleService` from `measure.py` lines 308–1984. Replace all references:
- Dataclasses → `from core.schema import ...`
- `get_analysis_preset` → `from configs import get_analysis_preset`
- `_json_default` → `from utils.metrics import json_default` (use `default=json_default`)
- `collect_input_groups` / `build_image_output_dir` → `from utils.io import ...`
- `detect_sphere_roi` / `compute_center_roi` → `from utils.image import ...`
- `detect_acicular_lsd` → `from utils.lsd import detect_acicular_lsd`
- `Sam2AspectRatioService` → `from services.sam2_service import Sam2AspectRatioService`
- The `analyze_with_lsd` method is **removed** (replaced by `detect_acicular_lsd` from utils)
- The `_measure_perpendicular_thickness` method is **removed** (moved to utils/lsd.py)
- In `detect_sphere_and_extract_cap` → call `detect_sphere_roi(arr_image_bgr, self.obj_primary_config.float_sphereCapFraction)` and return result
- In `compute_center_roi` → call `compute_center_roi(int_h, int_w, self.obj_primary_config.float_centerCropRatio)`
- In `process_primary()` LSD branch → call `detect_acicular_lsd(arr_roi_gray, arr_roi_bgr, ..., float_scale_pixels=self.obj_config.float_scalePixels, float_scale_um=self.obj_config.float_scaleMicrometers, ...)`

Also copy `build_primary_img_id_summary`, `build_primary_batch_summary`, `run_primary_particle_analysis`, and `build_primary_arg_parser` from measure.py (lines 1716–1984).

- [ ] **Step 2: Smoke test — import only**

```bash
python -c "from services.primary_particle import PrimaryParticleService; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Update `services/__init__.py`** — append:

```python
from services.primary_particle import PrimaryParticleService, run_primary_particle_analysis
```

- [ ] **Step 4: Commit**

```bash
git add services/primary_particle.py services/__init__.py
git commit -m "feat: add services/primary_particle.py"
```

---

## Task 12: `primary_measure.py` — CLI entry point

**Files:**
- Create: `primary_measure.py`

- [ ] **Step 1: Create `primary_measure.py`**

Copy only `main()` and `build_primary_arg_parser()` logic from `measure.py` (lines 1986–2265), updating imports to pull from `services`.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1차 입자 두께 측정 CLI 진입점."""
import json
import time
import sys
from pathlib import Path

from services.primary_particle import run_primary_particle_analysis, build_primary_arg_parser
from utils.metrics import json_default


def main() -> None:
    obj_parser = build_primary_arg_parser()
    obj_args = obj_parser.parse_args()
    float_start = time.perf_counter()

    dict_summary = run_primary_particle_analysis(
        str_input=obj_args.input,
        str_outputDir=obj_args.output_dir,
        str_model=obj_args.model,
        str_modelCfg=obj_args.model_cfg,
        str_device=obj_args.device,
        int_roiXMin=obj_args.roi_x_min,
        int_roiYMin=obj_args.roi_y_min,
        int_roiXMax=obj_args.roi_x_max,
        int_roiYMax=obj_args.roi_y_max,
        float_acicularThreshold=obj_args.acicular_threshold,
        float_areaThreshold=obj_args.area_threshold,
        int_targetParticleCount=obj_args.target_particle_count,
        float_scalePixels=obj_args.scale_pixels,
        float_scaleUm=obj_args.scale_um,
        int_imgSize=obj_args.imgsz,
        int_tileSize=obj_args.tile_size,
        int_stride=obj_args.stride,
        int_pointsPerTile=obj_args.points_per_tile,
        int_pointMinDistance=obj_args.point_min_distance,
        float_pointQualityLevel=obj_args.point_quality_level,
        int_pointBatchSize=obj_args.point_batch_size,
        float_dedupIou=obj_args.dedup_iou,
        float_bboxDedupIou=obj_args.bbox_dedup_iou,
        bool_usePointPrompts=obj_args.use_point_prompts,
        bool_autoCenterCrop=obj_args.auto_center_crop,
        float_centerCropRatio=obj_args.center_crop_ratio,
        bool_saveIndividualMasks=obj_args.save_individual_masks,
        bool_retinaMasks=obj_args.retina_masks,
        str_particleType=obj_args.particle_type or "unknown",
        str_magnification=obj_args.magnification or "unknown",
        str_particleMode=obj_args.particle_mode,
        bool_autoDetectSphere=obj_args.auto_detect_sphere,
        float_sphereCapFraction=obj_args.sphere_cap_fraction,
        str_measureMode=obj_args.measure_mode,
    )

    print("===== 1차 입자 분석 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2, default=json_default))
    print(f"Elapsed time: {time.perf_counter() - float_start:.4f} seconds")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — help output (no images needed)**

```bash
python primary_measure.py --help 2>&1 | head -5
```
Expected: shows `usage: primary_measure.py ...`

- [ ] **Step 3: Commit**

```bash
git add primary_measure.py
git commit -m "feat: add primary_measure.py CLI entry point"
```

---

## Task 13: Integration smoke test

**Files:** none created

- [ ] **Step 1: Run against saved test image (20k)**

```bash
python primary_measure.py \
  --particle_type acicular \
  --magnification 20k \
  --scale_pixels 206 --scale_um 5 \
  --input out_img6_20k/01_input.png \
  --output_dir out_pkg_test_20k
```
Expected output contains `"num_acicular": 78` (or similar) and `"measure_mode": "lsd"` with no traceback.

- [ ] **Step 2: Run against saved test image (50k)**

```bash
python primary_measure.py \
  --particle_type acicular \
  --magnification 50k \
  --scale_pixels 311 --scale_um 2 \
  --input out_img7_50k/01_input.png \
  --output_dir out_pkg_test_50k
```
Expected output contains `"num_acicular": 217` (or similar) and no traceback.

- [ ] **Step 3: Confirm output directory was created**

```bash
ls out_pkg_test_20k/
```
Expected: `01_input.png`, `summary.json`, `objects.json`, `acicular.csv`, etc.

---

## Task 14: Delete old files and final commit

**Files:**
- Delete: `core.py`
- Delete: `measure.py`

- [ ] **Step 1: Delete old monolithic files**

```bash
git rm core.py measure.py
```

- [ ] **Step 2: Verify nothing imports from the deleted files**

```bash
python -c "import primary_measure" 2>&1
```
Expected: no output (clean import).

- [ ] **Step 3: Update `.gitignore` to exclude `out_pkg_test_*/`**

Add `out_pkg_test_*/` to `.gitignore`.

- [ ] **Step 4: Final commit**

```bash
git add .gitignore
git commit -m "refactor: complete package restructure — replace core.py/measure.py with package layout"
```

---

## Self-Review

**Spec coverage:**
- `core/schema.py` ✓ Task 2
- `configs/presets.yaml` ✓ Task 3
- `data/__init__.py` placeholder ✓ Task 1
- `models/__init__.py` lazy SAM2 ✓ Task 4
- `utils/metrics.py` ✓ Task 5
- `utils/iou.py` ✓ Task 6
- `utils/image.py` ✓ Task 7
- `utils/lsd.py` ✓ Task 8
- `utils/io.py` ✓ Task 9
- `services/sam2_service.py` ✓ Task 10
- `services/primary_particle.py` ✓ Task 11
- `primary_measure.py` ✓ Task 12
- Delete `core.py`/`measure.py` ✓ Task 14

**Task 10 note:** The instruction says copy the body verbatim — the implementer must open `core.py` and copy lines 577–1548 in full. This avoids re-writing ~970 lines in the plan and the risk of transcription errors.

**Task 11 note:** Same — copy `measure.py` lines 308–1984. The substitutions listed are the only changes needed.

**Type consistency:** `PrimaryParticleMeasurement` defined in Task 2 (`core/schema.py`) and used in Task 8 (`utils/lsd.py`) and Task 11 (`services/primary_particle.py`) — consistent. `json_default` defined in Task 5 and used in Task 12 — consistent.
