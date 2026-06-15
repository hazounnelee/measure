# SEM Particle Quantification — Technical Guide

## Overview

This project provides two separate measurement pipelines for SEM (Scanning Electron Microscopy) images of lithium-ion battery precursor particles:

| Script | Target | Method |
|--------|--------|--------|
| `primary_measure.py` | **Primary particles** — acicular (needle-like) or plate-like crystals on the surface of the secondary particle | LSD (acicular) or SAM2 segmentation (plate) |
| `secondary_measure.py` | **Secondary particles** — the spherical agglomerate itself | SAM2 segmentation |

---

## Package Structure

```
quantification_primary/
├── primary_measure.py          Primary particle CLI entry point
├── secondary_measure.py        Secondary particle CLI entry point
│
├── configs/
│   ├── __init__.py             Preset and paths config loaders
│   ├── presets.yaml            Analysis presets (particle_type × magnification)
│   └── paths.yaml              Default file paths (model, input, output)
│
├── core/
│   └── schema.py               Dataclass definitions (configs, measurements, results)
│
├── models/
│   └── __init__.py             SAM2 model lazy-loader via Ultralytics
│
├── services/
│   ├── sam2_service.py         Base SAM2 service (tiling, point sampling, dedup,
│   │                           mask measurement, overlay, histogram, CSV/JSON output)
│   ├── primary_particle.py     Primary particle pipeline (ROI, LSD/SAM2 branching,
│   │                           acicular hybrid mode, batch runner, CLI parser)
│   └── secondary_particle.py   Secondary particle pipeline (runner, batch aggregation,
│                               CLI parser)
│
└── utils/
    ├── image.py                Sphere detection, center crop, CLAHE texture
    │                           enhancement, Shi-Tomasi point sampling, tile
    │                           generation, adaptive block-size helper
    ├── iou.py                  Binary mask IoU, bounding-box IoU
    ├── lsd.py                  LSD detection pipeline, perpendicular profile
    │                           thickness, segment fusion
    ├── metrics.py              px→µm conversion, statistics, JSON serialization
    └── io.py                   Input file/directory collection, output path builder
```

---

## Scale Calibration

All length measurements are converted from pixels to micrometers using a two-point calibration:

```
µm_value = px_value × (scale_um / scale_pixels)
```

| Magnification | Calibration | px/µm |
|---------------|-------------|-------|
| **20k** | 147 px = 1 µm | 147.0 |
| **50k** | 371 px = 1 µm | 371.0 |

These values are baked into the presets and applied automatically when `--magnification` is specified. For non-standard magnifications, override with `--scale_pixels` and `--scale_um`.

---

## Common Preprocessing (Both Pipelines)

### Image Normalisation

Every image is normalised to a fixed canvas before any analysis:

1. **Resize** to 2048 × 1636 px (bilinear interpolation)
2. **Crop bottom 100 px** → final shape **2048 × 1536 px**

This removes the SEM data bar at the bottom and guarantees a consistent pixel grid across all images regardless of original export resolution.

### ROI Extraction

After normalisation a Region of Interest (ROI) is cropped to exclude background and focus the detector on the particle surface.

#### Sphere auto-detection (primary 20k, `auto_detect_sphere=True`)

Used when the entire secondary particle is visible in the frame.

1. Convert to grayscale → Gaussian blur (21×21) → Otsu threshold
2. If foreground fraction > 50 %, invert (dark-background convention)
3. Morphological close (kernel 15, 3 iterations) → open (kernel 15, 2 iterations)
4. Find external contours → select the largest by area (must be ≥ 2 % of frame)
5. Fit minimum enclosing circle to get centre (cx, cy) and radius r
6. Cap ROI: `y ∈ [cy − r, cy − r + 2r × sphere_cap_fraction]`, `x ∈ [cx − r, cx + r]`
   — default `sphere_cap_fraction = 0.65` captures the top 65 % of the sphere diameter
7. Falls back to center crop if detection fails

#### Center crop (primary 50k, `auto_center_crop=True`)

Used when only the surface of a particle fills the frame.

```
margin_x = W × (1 − crop_ratio) / 2
margin_y = H × (1 − crop_ratio) / 2
ROI = image[margin_y : H − margin_y, margin_x : W − margin_x]
```

Default `crop_ratio = 0.85` for 50k, `0.60` for 20k sphere mode.

#### Manual ROI

Override with `--roi_x_min`, `--roi_y_min`, `--roi_x_max`, `--roi_y_max` when `--no-auto_center_crop` is set.

---

## Primary Particle Pipeline

### Mode selection

| `--particle_type` | `--measure_mode` | Model required |
|-------------------|-----------------|----------------|
| `acicular` | `lsd` (default) | No |
| `plate` | `sam2` (default) | Yes |

The preset sets `measure_mode` automatically. It can be overridden with `--measure_mode`.

---

### Acicular Mode — LSD Pipeline

Line Segment Detector (LSD) directly measures needle length and perpendicular thickness from the grayscale image without a neural network. Processing time is typically < 2 s.

#### Step 1 — Contrast enhancement

```
CLAHE(clipLimit=2.0, tileGridSize=8×8)  →  GaussianBlur(3×3)
```

CLAHE (Contrast-Limited Adaptive Histogram Equalization) corrects uneven illumination across the ROI. The subsequent Gaussian blur suppresses pixel-level noise before edge detection.

**Debug image:** `lsd_01_preprocessed.png`

#### Step 2 — Thresholding

Two options controlled by `--lsd_adaptive_thresh`:

**Otsu (default):** A single global threshold is computed via Otsu's method on the blurred image. Pixels above the threshold are classified as bright (needle body).

**Adaptive Gaussian (`--lsd_adaptive_thresh`):** A per-pixel threshold is computed from the local neighbourhood mean minus a constant C=4. The block size is derived from the ROI dimensions: `block_size = max(11, ⌊min(H, W) / 12⌋)`, rounded up to the nearest odd number. Useful when illumination varies significantly across the frame.

Both variants produce:
- A binary mask used for perpendicular profile scanning
- A debug image

**Debug image:** `lsd_02_otsu_thresh.png` or `lsd_02_adaptive_thresh.png`

#### Step 3 — LSD raw detection

```python
cv2.createLineSegmentDetector(0).detect(arr_blur)
```

OpenCV's LSD implementation (Grompone von Gioi et al. 2010) detects line segments by tracing gradient-aligned pixel chains. It outputs segment endpoints (x1, y1, x2, y2) and an estimated line width for each segment.

**Typical output:** 1,000–3,000 segments per ROI.

**Debug image:** `lsd_03_raw_detections.png` (yellow lines)

#### Step 4 — Length and aspect-ratio filter

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Length < 20 px | discard | Sub-pixel noise and grain boundaries |
| `lsd_width / length ≥ ar_loose` | discard | Short, stubby segments cannot be needles; `ar_loose = min(acicular_threshold + 0.20, 0.65)` |

**Debug image:** `lsd_04_after_filter.png` (cyan lines)

#### Step 5 — Deduplication

A single needle often produces multiple overlapping LSD segments (one per edge, one per sub-section). Segments are sorted longest-first, then greedily accepted:

```
for each candidate C (longest first):
    for each already-accepted segment P:
        if dist(centre_C, centre_P) < 12 px  AND  |angle_C − angle_P| < 25°:
            reject C as duplicate
            break
    else:
        accept C
```

This keeps at most one representative segment per physical needle.

**Debug image:** `lsd_05_after_dedup.png` (orange lines)

#### Step 6 — Segment fusion (optional, `--lsd_fuse_segments`)

When a needle is fragmented into multiple non-overlapping segments (e.g. occluded in the middle), this step merges them into a single long segment.

**Algorithm — union-find on collinear neighbours:**

Two segments i and j are fused if all three conditions hold:

1. **Angle similarity:** `min(|αᵢ − αⱼ|, 180° − |αᵢ − αⱼ|) ≤ 10°`
2. **Perpendicular proximity:** signed distance between midpoints perpendicular to segment i's axis `< 8 px` (ensures they lie on the same line, not on parallel edges of the same needle)
3. **Axial proximity:** projecting j's endpoints onto i's axis, the intervals overlap or the gap between them `< 15 px`

Connected components are merged by projecting all endpoints onto the length-weighted average direction and taking the extremes as the new endpoints.

**Debug image:** `lsd_06_after_fusion.png` (magenta lines)

#### Step 7 — Perpendicular thickness measurement

For each accepted segment, the needle thickness is measured by sampling intensity profiles perpendicular to the segment axis at 7 evenly spaced positions (t = 0.2, 0.25, …, 0.8 along the segment).

**At each sample position:**

1. Compute a scan line of length `2 × max(15, 0.5 × px_per_µm)` pixels centred on the segment, oriented perpendicular to it
2. Sample the blurred grayscale image along this scan line
3. Binarise the profile against the Otsu threshold (or look up the precomputed adaptive binary)
4. Find contiguous bright runs in the binarised profile
5. Select the bright run whose midpoint is closest to the scan-line centre — this selects the needle being measured, not an adjacent neighbour
6. Width of the selected run = thickness estimate at this position

The final thickness is the **median** of the 7 estimates (robust to occluded or noisy positions).

A **perpendicular offset** (signed distance from the LSD edge to the bright-region centre) is also computed and used to shift the mask rectangle from the detected edge line onto the physical needle body centre.

Segments with thickness < 2 px are discarded (below reliable detection limit).

#### Step 8 — Classification and masking

```
aspect_ratio = thickness_px / length_px
```

| AR | Category |
|----|----------|
| < `acicular_threshold` (default 0.40) | `acicular` |
| ≥ `acicular_threshold` | `plate` |
| mask area < `area_threshold` | `fragment` |

When `--particle_type acicular`, plate-category segments are dropped and vice versa.

A binary mask rectangle is constructed from the four corners of the oriented bounding box centred on the needle body. Masks whose area falls below `float_area_threshold` are discarded. Masks whose bounding box touches the ROI edge within `int_bboxEdgeMargin` pixels are also discarded (incomplete particles).

#### Measured quantities (per needle)

| Field | Description |
|-------|-------------|
| `float_thicknessPx` / `float_thicknessUm` | Median perpendicular width |
| `float_longAxisPx` / `float_longAxisUm` | LSD segment length |
| `float_aspectRatio` | thickness / length |
| `float_minRectAngle` | Segment orientation (degrees) |
| `int_maskArea` | Pixel count of the rectangular mask |
| `float_centroidX/Y` | Geometric centre of the mask |

---

### Plate Mode — SAM2 Pipeline

Plate-like particles are too irregular in shape for LSD. SAM2 (Segment Anything Model 2) is used instead to produce pixel-accurate masks.

#### Step 1 — Tiling

The ROI is divided into overlapping square tiles to handle images wider than SAM2's inference resolution:

```
tile_size (default 192 px)  ×  stride (default 96 px)
```

Edge tiles are extended to the ROI boundary to avoid leaving gaps. Duplicate tile positions are deduplicated.

#### Step 2 — Interest-point sampling (Shi-Tomasi)

For each tile, candidate point prompts are extracted:

1. **Texture enhancement** — CLAHE + Gaussian blur + Sobel gradient magnitude + Laplacian + morphological blackhat, blended into a single enhanced grayscale channel
2. **Shi-Tomasi corner detection** (`cv2.goodFeaturesToTrack`) on the enhanced tile, up to `points_per_tile` (default 80–150 depending on preset)
3. If fewer than the requested points are found, the enhanced image is Otsu-thresholded and contour centroids are used as additional candidates
4. Final candidate set: up to `points_per_tile` strongest corners spaced ≥ `point_min_distance` px apart

#### Step 3 — SAM2 batch inference

Points are sent to SAM2 in batches of `point_batch_size` (default 32). Each point is labelled as foreground (label = 1). SAM2 returns a binary mask for each point.

Each raw mask is binarised at `mask_binarize_threshold` (default 0.0, i.e. any positive logit → foreground).

#### Step 4 — Tile-level filtering

For each mask returned by SAM2:

1. **Minimum area:** discard if `mask.sum() < int_minValidMaskArea`
2. **Contour extraction:** find the largest external contour; discard if none
3. **Tile-edge exclusion:** discard if the bounding box falls within `int_tileEdgeMargin` px of the tile boundary (particles cut by the tile edge are measured by a neighbouring tile)
4. **Bbox IoU deduplication:** discard if bounding-box IoU with any already-accepted mask ≥ `float_bboxDedupIou` (default 0.85) — fast pre-filter before the expensive pixel comparison

#### Step 5 — ROI-level deduplication

Tile masks are resampled into ROI coordinates. Pixel-level IoU is computed between each candidate and all already-accepted masks. Candidates with IoU ≥ `float_dedupIou` (default 0.60) are discarded as duplicates.

#### Step 6 — Mask refinement (morphology)

If `int_maskMorphKernelSize > 1`, the accepted binary mask is post-processed:

1. Morphological **open** (`int_maskMorphOpenIterations` iterations) — removes small spurs
2. Morphological **close** (`int_maskMorphCloseIterations` iterations) — fills interior holes

#### Step 7 — Measurement via minAreaRect

The refined mask's largest contour is fitted with `cv2.minAreaRect`, a minimum-area oriented bounding rectangle:

```
thickness_px = min(rect_width, rect_height)
long_axis_px = max(rect_width, rect_height)
aspect_ratio = thickness_px / long_axis_px
```

For contours with fewer than 5 points, the axis-aligned bounding box is used instead.

Additionally, the longest contiguous horizontal and vertical spans across the mask are recorded (`int_longestHorizontal`, `int_longestVertical`).

#### Step 8 — Classification

| Condition | Category |
|-----------|----------|
| `mask_area < area_threshold` | excluded (not classified as a primary particle) |
| `aspect_ratio < acicular_threshold` (0.40) | `acicular` |
| otherwise | `plate` |

When `--particle_type` is `acicular`/`plate`, only the matching category is kept. Shape-agnostic types such as `active` keep everything.

#### Step 9 — Convex hull finalization

Masks that pass classification/filtering are replaced by their convex hull, and re-measured from the hull-applied mask (shared by `--small`/`--active`).

#### Acicular hybrid mode (`--particle_mode acicular`)

For acicular particles measured with SAM2 (non-LSD mode), an OpenCV pre-filter reduces the number of SAM2 calls:

1. CLAHE + adaptive threshold → erosion → `cv2.connectedComponentsWithStats`
2. For each blob, compute image moment eigenvalues: `AR = √(λ_min / λ_max)` (eigenvalue ratio of the covariance matrix ≈ (short axis / long axis)²)
3. Keep blobs with `AR < acicular_ar_screen` (elongated enough to be candidates)
4. Blob centroids are sent as SAM2 point prompts

If fewer than 3 candidates are found, the pipeline falls back to the standard tiled point-prompt mode.

---

## Secondary Particle Pipeline

Secondary particles are the spherical agglomerates. The pipeline is identical to the SAM2 branch of the primary pipeline (tiling → Shi-Tomasi → SAM2 → dedup → morphology) but the measurement and classification differ.

### Measurement

| Quantity | Algorithm |
|----------|-----------|
| `float_eqDiameterUm` | Equivalent circle diameter: `2 × sqrt(mask_area / π)` converted to µm |
| `float_sphericity` | Wadell 2D isoperimetric ratio: `4π × area / perimeter²` (1.0 = circle, < 1.0 = irregular) |
| `float_bboxWidthUm`, `float_bboxHeightUm` | Bounding-box dimensions in µm |

The sphericity metric uses the contour perimeter computed by `cv2.arcLength`. It is only computed for particles (not fragments) and is clamped to [0, 1].

### Classification

| Condition | Category |
|-----------|----------|
| `mask_area < area_threshold` (default 1500 px²) | `fragment` |
| otherwise | `particle` |

There is no acicular/plate distinction in secondary mode.

---

## Outputs

### Primary pipeline outputs

| File | Content |
|------|---------|
| `01_input.png` | Normalised input image (2048×1536) |
| `02_input_roi.png` | Cropped ROI used for analysis |
| `03_overlay_roi.png` | Segmentation overlay (mask + contour) |
| `04_overlay_full.png` | Full image with ROI rectangle highlighted |
| `05_opencv_candidates.png` | LSD final lines: green=acicular, orange=plate (LSD mode) |
| `06_sphere_detection.png` | Sphere centre + cap ROI visualisation (20k only) |
| `lsd_01_preprocessed.png` | CLAHE + blur result |
| `lsd_02_otsu_thresh.png` | Otsu binary mask |
| `lsd_02_adaptive_thresh.png` | Adaptive binary mask (when `--lsd_adaptive_thresh`) |
| `lsd_03_raw_detections.png` | All LSD segments, yellow |
| `lsd_04_after_filter.png` | After length/AR filter, cyan |
| `lsd_05_after_dedup.png` | After deduplication, orange |
| `lsd_06_after_fusion.png` | After segment fusion, magenta (when `--lsd_fuse_segments`) |
| `objects.csv` | All detected particles |
| `acicular.csv` / `plate.csv` | Target-type particles only |
| `thickness_dist.png` | Thickness distribution histogram |
| `summary.json` | Statistics and analysis settings |
| `objects.json` | Per-particle measurements |
| `debug.json` | Tile, point, and mask debug information |
| `acicular_masks/` `plate_masks/` | Individual mask PNGs (when `--save_mask_imgs`) |

### Secondary pipeline outputs

| File | Content |
|------|---------|
| `01_input.png` – `04_overlay_full.png` | Same as primary |
| `objects.csv` | All detected objects with sphericity |
| `particles.csv` | Particle-class objects only |
| `particle_dist.png` | Mean size distribution histogram |
| `sphericity_dist.png` | Sphericity distribution histogram |
| `summary.json` | Includes sphericity and size statistics |
| `particle_masks/` `fragment_masks/` | Individual mask PNGs |

### Batch mode outputs (additional)

| File | Content |
|------|---------|
| `<IMG_ID>/img_id_summary.json` | Per-group aggregated statistics |
| `batch_summary.json` | Cross-group totals and means |

---

## Command-Line Reference

### Primary particle analysis

```bash
python primary_measure.py \
  --input <path>                   # image file or directory
  --particle_type acicular|plate   # required for preset
  --magnification 20k|50k          # required for preset
  [--output_dir <path>]            # default: out_primary_<timestamp>
  [--config configs/paths.yaml]    # optional path defaults file
```

#### Common overrides

```bash
# Non-standard scale bar (measured manually from image)
--scale_pixels 206 --scale_um 1

# Use adaptive thresholding instead of Otsu (uneven illumination)
--lsd_adaptive_thresh

# Merge fragmented needle detections
--lsd_fuse_segments

# Relax acicular/plate boundary (default 0.40)
--acicular_threshold 0.35

# Minimum mask area to count as a particle
--area_threshold 50

# SAM2 model paths (plate mode or acicular SAM2 mode)
--model model/sam2.1_hiera_base_plus.pt
--model_cfg model/sam2.1_hiera_t.yaml
```

#### Examples

```bash
# Acicular, 50k surface close-up — no model needed
python primary_measure.py \
  --input sem_50k.jpg \
  --particle_type acicular \
  --magnification 50k

# Acicular, 20k with sphere detection
python primary_measure.py \
  --input sem_20k.jpg \
  --particle_type acicular \
  --magnification 20k \
  --output_dir out_acicular_20k

# Plate, 50k — SAM2 required
python primary_measure.py \
  --input sem_plate_50k.jpg \
  --particle_type plate \
  --magnification 50k \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml

# Batch processing
python primary_measure.py \
  --input ./samples/ \
  --particle_type acicular \
  --magnification 50k \
  --output_dir out_batch
```

### Secondary particle analysis

```bash
python secondary_measure.py \
  --input <path>                   # image file or directory
  [--output_dir <path>]
  [--model model/sam2.1_hiera_base_plus.pt]
  [--model_cfg model/sam2.1_hiera_t.yaml]
  [--small_particle]               # use 50k scale (371 px/µm) instead of 20k (147 px/µm)
  [--area_threshold 1500]          # px² minimum to count as a particle
  [--config configs/paths.yaml]
```

#### Example

```bash
python secondary_measure.py \
  --input sem_secondary.jpg \
  --model model/sam2.1_hiera_base_plus.pt \
  --model_cfg model/sam2.1_hiera_t.yaml \
  --output_dir out_secondary
```

### `configs/paths.yaml` — default paths

Avoids repeating model and directory paths on every invocation. CLI arguments always take priority.

```yaml
input: "img/sample.jpg"
output_dir: ""                         # empty = auto timestamp name
model: "model/sam2.1_hiera_base_plus.pt"
model_cfg: "model/sam2.1_hiera_t.yaml"
device: ""                             # empty = auto (cpu / cuda)
```

---

## Preset Parameter Reference

`--particle_type` × `--magnification` selects a preset. Individual parameters can still be overridden from the CLI.

| Parameter | acicular/20k | acicular/50k | plate/20k | plate/50k |
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

## Environment Setup

```bash
# Create and activate conda environment
conda activate measure

# Run (from project root)
cd ~/Desktop/Projects/quantification_primary
python primary_measure.py --help
python secondary_measure.py --help
```

The `measure` conda environment contains: Python 3.11, opencv-python, numpy, matplotlib, pyyaml, ultralytics (SAM2), torch, torchvision.

---

## Algorithm Decision Tree

```
Input SEM image
       │
       ▼
Resize → 2048×1636, crop bottom 100 px → 2048×1536
       │
       ├─── primary_measure.py ──────────────────────────────────┐
       │                                                          │
       │    ROI extraction                                        │
       │      20k → sphere detect → cap crop                     │
       │      50k → center crop (85%)                            │
       │                                                          │
       │    particle_type=acicular                                │
       │    measure_mode=lsd                                      │
       │      CLAHE → blur → Otsu/Adaptive threshold              │
       │      → LSD → length/AR filter → dedup                   │
       │      → [optional fusion]                                 │
       │      → perpendicular profile → thickness (median of 7)  │
       │      → AR = thickness/length → classify acicular/plate   │
       │      → rectangular mask per needle                       │
       │                                                          │
       │    particle_type=plate                                   │
       │    measure_mode=sam2                                     │
       │      Tiling → Shi-Tomasi points → SAM2 batch inference   │
       │      → IoU dedup → morphology → minAreaRect              │
       │      → AR = short/long axis → classify acicular/plate    │
       │                                                          │
       │    Output: thickness_um, long_axis_um, AR per particle   │
       │                                                          │
       └─── secondary_measure.py ────────────────────────────────┘
                                                                  │
            ROI extraction (manual or default full image)         │
                                                                  │
            Tiling → Shi-Tomasi points → SAM2 batch inference     │
            → IoU dedup → morphology                              │
            → longest H/V span → AR = min/max span                │
            → sphericity = 4π·area / perimeter²                   │
            → classify particle/fragment by area                  │
                                                                  │
            Output: size_um, AR, sphericity per particle          ▼
```
