from __future__ import annotations
import typing as tp
import cv2
import numpy as np

from core.schema import PrimaryParticleMeasurement
from utils.metrics import convert_pixels_to_micrometers

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
        float_acicular_threshold: AR < this -> acicular.
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
