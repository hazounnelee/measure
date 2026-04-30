from __future__ import annotations
import typing as tp
import cv2
import numpy as np

from core.schema import PrimaryParticleMeasurement
from utils.metrics import convert_pixels_to_micrometers
from utils.image import compute_adaptive_block_size

CONST_LSD_MIN_LENGTH_PX: int = 20
CONST_LSD_DEDUP_DIST_PX: int = 12
CONST_LSD_DEDUP_ANGLE_DEG: float = 25.0
CONST_LSD_PERP_N_SAMPLES: int = 7

CONST_LSD_FUSE_ANGLE_DEG: float = 10.0
CONST_LSD_FUSE_OVERLAP_RATIO: float = 0.3   # intersection / min(area_i, area_j)


def measure_perpendicular_thickness(
    arr_gray: np.ndarray,
    float_thresh: float,
    float_x1: float,
    float_y1: float,
    float_x2: float,
    float_y2: float,
    float_px_per_um: float,
    arr_binary: tp.Optional[np.ndarray] = None,
) -> tp.Tuple[float, float]:
    """Scan perpendicular to a line segment; return (needle_width_px, perp_offset_px).

    When arr_binary is provided (adaptive threshold case) pixel membership is
    looked up directly from the binary image instead of thresholding the
    grey-level profile with float_thresh.
    Returns (0.0, 0.0) when no valid sample is found.
    """
    float_dx = float_x2 - float_x1
    float_dy = float_y2 - float_y1
    float_length = float(np.sqrt(float_dx ** 2 + float_dy ** 2))
    if float_length < 1.0:
        return 0.0, 0.0
    float_ux = float_dx / float_length
    float_uy = float_dy / float_length
    float_px = -float_uy
    float_py = float_ux
    int_roiH, int_roiW = arr_gray.shape[:2]
    int_half_scan = max(15, int(0.5 * float_px_per_um))
    int_center = int_half_scan
    arr_scan = np.arange(-int_half_scan, int_half_scan + 1, dtype=np.float32)
    list_widths: tp.List[float] = []
    list_offsets: tp.List[float] = []
    for float_t in np.linspace(0.2, 0.8, CONST_LSD_PERP_N_SAMPLES):
        float_sx = float_x1 + float_t * float_dx
        float_sy = float_y1 + float_t * float_dy
        arr_xs = np.clip(float_sx + float_px * arr_scan, 0, int_roiW - 1).astype(np.int32)
        arr_ys = np.clip(float_sy + float_py * arr_scan, 0, int_roiH - 1).astype(np.int32)
        if arr_binary is not None:
            arr_above = arr_binary[arr_ys, arr_xs] > 0
        else:
            arr_profile = arr_gray[arr_ys, arr_xs].astype(np.float32)
            arr_above = arr_profile > float_thresh
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
            list_offsets.append((tpl_best[0] + tpl_best[1]) / 2.0 - int_center)
    if not list_widths:
        return 0.0, 0.0
    return float(np.median(list_widths)), float(np.median(list_offsets))


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


def _bbox_corners(dict_cand: tp.Dict[str, float]) -> np.ndarray:
    """Return 4 corners of the oriented bounding box for a line segment candidate."""
    float_cx = (dict_cand["x1"] + dict_cand["x2"]) / 2.0
    float_cy = (dict_cand["y1"] + dict_cand["y2"]) / 2.0
    float_length = max(dict_cand["length"], 1.0)
    float_width = max(dict_cand.get("width", 3.0), 3.0)
    # cv2.boxPoints expects angle in OpenCV convention (clockwise from x-axis)
    float_angle_cv = -dict_cand["angle"]
    return cv2.boxPoints(((float_cx, float_cy), (float_length, float_width), float_angle_cv)).astype(np.float32)


def _fuse_contour_boxes(
    list_cands: tp.List[tp.Dict[str, float]],
    float_angle_tol_deg: float,
    float_overlap_ratio: float,
) -> tp.List[tp.Dict[str, float]]:
    """Fuse oriented bounding boxes using union-find.

    Two boxes are fused when:
      1. Angle difference < float_angle_tol_deg (mod 180°)
      2. intersection_area / min(area_i, area_j) >= float_overlap_ratio
    """
    n = len(list_cands)
    if n <= 1:
        return list(list_cands)

    list_corners = [_bbox_corners(c) for c in list_cands]
    list_areas = [
        max(c["length"], 1.0) * max(c.get("width", 3.0), 3.0)
        for c in list_cands
    ]

    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ci = list_cands[i]
        for j in range(i + 1, n):
            cj = list_cands[j]

            # 1. Angle check
            float_adiff = abs(ci["angle"] - cj["angle"])
            float_adiff = min(float_adiff, 180.0 - float_adiff)
            if float_adiff > float_angle_tol_deg:
                continue

            # 2. Bbox overlap
            _ret, arr_inter = cv2.intersectConvexConvex(list_corners[i], list_corners[j])
            if arr_inter is None or len(arr_inter) == 0:
                continue
            float_area_inter = float(cv2.contourArea(arr_inter))
            if float_area_inter <= 0.0:
                continue
            if float_area_inter / max(min(list_areas[i], list_areas[j]), 1.0) < float_overlap_ratio:
                continue

            pi, pj = _find(i), _find(j)
            if pi != pj:
                parent[pi] = pj

    groups: tp.Dict[int, tp.List[int]] = {}
    for i in range(n):
        root = _find(i)
        groups.setdefault(root, []).append(i)

    list_fused: tp.List[tp.Dict[str, float]] = []
    for list_idx in groups.values():
        if len(list_idx) == 1:
            list_fused.append(list_cands[list_idx[0]])
            continue

        # Principal direction: length-weighted average, sign-aligned to first
        c0 = list_cands[list_idx[0]]
        float_l0 = max(c0["length"], 1.0)
        float_ux_ref = (c0["x2"] - c0["x1"]) / float_l0
        float_uy_ref = (c0["y2"] - c0["y1"]) / float_l0

        float_sum_ux, float_sum_uy = 0.0, 0.0
        for k in list_idx:
            ck = list_cands[k]
            float_l = max(ck["length"], 1.0)
            float_ukx = (ck["x2"] - ck["x1"]) / float_l
            float_uky = (ck["y2"] - ck["y1"]) / float_l
            if float_ukx * float_ux_ref + float_uky * float_uy_ref < 0:
                float_ukx, float_uky = -float_ukx, -float_uky
            float_sum_ux += float_ukx * float_l
            float_sum_uy += float_uky * float_l

        float_norm = float(np.sqrt(float_sum_ux ** 2 + float_sum_uy ** 2))
        if float_norm < 1e-9:
            list_fused.append(list_cands[list_idx[0]])
            continue
        float_fux = float_sum_ux / float_norm
        float_fuy = float_sum_uy / float_norm

        float_ref_x = sum(
            (list_cands[k]["x1"] + list_cands[k]["x2"]) / 2.0 for k in list_idx
        ) / len(list_idx)
        float_ref_y = sum(
            (list_cands[k]["y1"] + list_cands[k]["y2"]) / 2.0 for k in list_idx
        ) / len(list_idx)

        # Project all 4 corners of each bbox onto principal axis; take extremes
        float_t_min = float("inf")
        float_t_max = float("-inf")
        for k in list_idx:
            for float_ex, float_ey in list_corners[k]:
                float_t = (float_ex - float_ref_x) * float_fux + (float_ey - float_ref_y) * float_fuy
                float_t_min = min(float_t_min, float_t)
                float_t_max = max(float_t_max, float_t)

        float_nx1 = float_ref_x + float_t_min * float_fux
        float_ny1 = float_ref_y + float_t_min * float_fuy
        float_nx2 = float_ref_x + float_t_max * float_fux
        float_ny2 = float_ref_y + float_t_max * float_fuy
        float_new_len = float(np.sqrt((float_nx2 - float_nx1) ** 2 + (float_ny2 - float_ny1) ** 2))
        float_new_angle = float(
            np.degrees(np.arctan2(float_ny2 - float_ny1, float_nx2 - float_nx1)) % 180
        )
        float_avg_width = sum(list_cands[k].get("width", 3.0) for k in list_idx) / len(list_idx)
        list_fused.append({
            "x1": float_nx1, "y1": float_ny1,
            "x2": float_nx2, "y2": float_ny2,
            "length": float_new_len,
            "angle": float_new_angle,
            "width": float_avg_width,
        })

    return list_fused


def detect_acicular_lsd(
    arr_roi_gray: np.ndarray,
    arr_roi_bgr: np.ndarray,
    float_acicular_threshold: float,
    str_particle_type: str,
    float_scale_pixels: float,
    float_scale_um: float,
    int_edge_margin: int = 8,
    float_area_threshold: float = 0.0,
    bool_adaptive_thresh: bool = False,
    bool_fuse_segments: bool = False,
) -> tp.Tuple[
    tp.List[PrimaryParticleMeasurement],
    tp.List[np.ndarray],
    np.ndarray,
    tp.Dict[str, np.ndarray],
    float,
]:
    """Detect acicular particles with LSD and measure thickness via perpendicular profile.

    Args:
        arr_roi_gray: Grayscale ROI image.
        arr_roi_bgr: BGR ROI image (for debug visualization).
        float_acicular_threshold: AR < this -> acicular.
        str_particle_type: "acicular" or "plate" (wrong-shape candidates are dropped).
        float_scale_pixels: Scale bar length in pixels.
        float_scale_um: Scale bar length in micrometers.
        int_edge_margin: Pixels from ROI edge to discard.
        float_area_threshold: Minimum mask area in pixels²; smaller masks are dropped.
        bool_adaptive_thresh: Use adaptive (Gaussian) threshold instead of Otsu for
            both the step-2 visualization and the perpendicular profile scan.
        bool_fuse_segments: Fuse oriented bounding boxes that have similar direction
            (Δangle < 10°) and significant overlap (intersection/min_area ≥ 0.3).
            Produces an extra lsd_06_after_fusion debug image.

    Returns:
        (list_measurements, list_masks, arr_debug_bgr, dict_step_images)
    """
    int_roiH, int_roiW = arr_roi_gray.shape[:2]
    float_px_per_um = float_scale_pixels / max(float_scale_um, 1e-9)

    obj_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_eq = obj_clahe.apply(arr_roi_gray)
    arr_blur = cv2.GaussianBlur(arr_eq, (3, 3), 0)

    # --- thresholding ---
    if bool_adaptive_thresh:
        arr_thresh_binary = cv2.adaptiveThreshold(
            arr_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            compute_adaptive_block_size(int_roiH, int_roiW, 12), 4,
        )
        float_otsu_thresh = 0.0
        arr_binary_for_profile: tp.Optional[np.ndarray] = arr_thresh_binary
        str_thresh_key = "lsd_02_adaptive_thresh"
    else:
        float_otsu_thresh, arr_thresh_binary = cv2.threshold(
            arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr_binary_for_profile = None
        str_thresh_key = "lsd_02_otsu_thresh"

    # Density: fraction of bright (white) pixels in the binary image.
    # More black area (background) → lower density.
    float_density = float((arr_thresh_binary > 0).sum()) / (int_roiH * int_roiW)

    obj_lsd = cv2.createLineSegmentDetector(0)
    arr_lines, arr_widths, _, _ = obj_lsd.detect(arr_blur)

    dict_steps: tp.Dict[str, np.ndarray] = {
        "lsd_01_preprocessed": cv2.cvtColor(arr_blur, cv2.COLOR_GRAY2BGR),
        str_thresh_key: cv2.cvtColor(arr_thresh_binary, cv2.COLOR_GRAY2BGR),
    }

    list_objects: tp.List[PrimaryParticleMeasurement] = []
    list_masks: tp.List[np.ndarray] = []

    if arr_lines is None:
        dict_steps["lsd_03_raw_detections"] = arr_roi_bgr.copy()
        dict_steps["lsd_04_after_filter"] = arr_roi_bgr.copy()
        dict_steps["lsd_05_after_dedup"] = arr_roi_bgr.copy()
        return list_objects, list_masks, arr_roi_bgr.copy(), dict_steps, float_density

    # step 3: raw LSD detections (yellow)
    arr_step_raw = arr_roi_bgr.copy()
    for arr_line in arr_lines:
        fx1, fy1, fx2, fy2 = arr_line[0]
        cv2.line(arr_step_raw, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 255), 1)
    dict_steps["lsd_03_raw_detections"] = arr_step_raw

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
            "length": float_len, "angle": float_angle, "width": float_lsd_w,
        })

    # step 4: after length/AR filter (cyan)
    arr_step_filtered = arr_roi_bgr.copy()
    for dict_c in list_cands:
        cv2.line(arr_step_filtered,
                 (int(dict_c["x1"]), int(dict_c["y1"])),
                 (int(dict_c["x2"]), int(dict_c["y2"])),
                 (255, 255, 0), 1)
    dict_steps["lsd_04_after_filter"] = arr_step_filtered

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

    # step 5: after deduplication (orange)
    arr_step_deduped = arr_roi_bgr.copy()
    for dict_c in list_accepted:
        cv2.line(arr_step_deduped,
                 (int(dict_c["x1"]), int(dict_c["y1"])),
                 (int(dict_c["x2"]), int(dict_c["y2"])),
                 (0, 165, 255), 1)
    dict_steps["lsd_05_after_dedup"] = arr_step_deduped

    # --- optional contour box fusion ---
    int_before_fuse = len(list_accepted)
    if bool_fuse_segments and list_accepted:
        list_accepted = _fuse_contour_boxes(
            list_accepted,
            float_angle_tol_deg=CONST_LSD_FUSE_ANGLE_DEG,
            float_overlap_ratio=CONST_LSD_FUSE_OVERLAP_RATIO,
        )
        list_accepted.sort(key=lambda d: d["length"], reverse=True)

        arr_step_fused = arr_roi_bgr.copy()
        for dict_c in list_accepted:
            cv2.line(arr_step_fused,
                     (int(dict_c["x1"]), int(dict_c["y1"])),
                     (int(dict_c["x2"]), int(dict_c["y2"])),
                     (255, 0, 255), 1)
        dict_steps["lsd_06_after_fusion"] = arr_step_fused

    print(
        f"[LSD] 원본 {len(arr_lines)}개 → 필터 {len(list_cands)}개 "
        f"→ 중복제거 {int_before_fuse}개"
        + (f" → 융합 {len(list_accepted)}개" if bool_fuse_segments else ""),
        flush=True,
    )

    arr_debug = arr_roi_bgr.copy()

    for int_idx, dict_c in enumerate(list_accepted):
        float_x1 = dict_c["x1"]
        float_y1 = dict_c["y1"]
        float_x2 = dict_c["x2"]
        float_y2 = dict_c["y2"]
        float_len = dict_c["length"]

        float_thickness, float_offset = measure_perpendicular_thickness(
            arr_blur, float_otsu_thresh,
            float_x1, float_y1, float_x2, float_y2,
            float_px_per_um,
            arr_binary=arr_binary_for_profile,
        )
        if float_thickness < 2.0:
            continue

        float_ar = float_thickness / float_len

        str_category = "acicular" if float_ar < float_acicular_threshold else "plate"

        if str_particle_type in ("acicular", "plate") and str_category != str_particle_type:
            continue

        float_ux = (float_x2 - float_x1) / max(float_len, 1.0)
        float_uy = (float_y2 - float_y1) / max(float_len, 1.0)
        float_half_t = float_thickness / 2.0
        float_px_dir = -float_uy
        float_py_dir = float_ux

        float_nx1 = float_x1 + float_offset * float_px_dir
        float_ny1 = float_y1 + float_offset * float_py_dir
        float_nx2 = float_x2 + float_offset * float_px_dir
        float_ny2 = float_y2 + float_offset * float_py_dir

        arr_corners = np.float32([
            [float_nx1 - float_half_t * float_px_dir, float_ny1 - float_half_t * float_py_dir],
            [float_nx1 + float_half_t * float_px_dir, float_ny1 + float_half_t * float_py_dir],
            [float_nx2 + float_half_t * float_px_dir, float_ny2 + float_half_t * float_py_dir],
            [float_nx2 - float_half_t * float_px_dir, float_ny2 - float_half_t * float_py_dir],
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

        if float_area_threshold > 0.0 and float(arr_mask.sum()) < float_area_threshold:
            continue

        float_mcx = (float_nx1 + float_nx2) / 2.0
        float_mcy = (float_ny1 + float_ny2) / 2.0

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
                 (int(float_nx1), int(float_ny1)), (int(float_nx2), int(float_ny2)),
                 (0, 255, 0) if str_category == "acicular" else (0, 128, 255), 1)

    print(f"[LSD] → 최종 {len(list_objects)}개  density={float_density:.3f}", flush=True)
    return list_objects, list_masks, arr_debug, dict_steps, float_density
