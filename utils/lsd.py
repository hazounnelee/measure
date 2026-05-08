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
        arr_pad = np.empty(len(arr_above) + 2, dtype=np.int8)
        arr_pad[0] = 0
        arr_pad[1:-1] = arr_above.astype(np.int8)
        arr_pad[-1] = 0
        arr_diff = np.diff(arr_pad)
        arr_starts = np.where(arr_diff == 1)[0]
        arr_ends = np.where(arr_diff == -1)[0] - 1
        if arr_starts.size == 0:
            continue
        list_regions = list(zip(arr_starts.tolist(), arr_ends.tolist()))

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
        int_bx <= int_margin
        or int_by <= int_margin
        or (int_bx + int_bw) >= (int_roiW - int_margin)
        or (int_by + int_bh) >= (int_roiH - int_margin)
    )



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
    int_min_length_px: int = CONST_LSD_MIN_LENGTH_PX,
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
    int_total_px = int_roiH * int_roiW
    float_density = float((arr_thresh_binary > 0).sum()) / int_total_px if int_total_px > 0 else 0.0

    obj_lsd = cv2.createLineSegmentDetector(0)
    arr_lines, _, _, _ = obj_lsd.detect(arr_blur)

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

    list_cands: tp.List[tp.Dict[str, float]] = []
    for arr_line in arr_lines:
        float_x1, float_y1, float_x2, float_y2 = arr_line[0]
        float_len = float(np.sqrt((float_x2 - float_x1) ** 2 + (float_y2 - float_y1) ** 2))
        if float_len < int_min_length_px:
            continue
        float_angle = float(np.degrees(np.arctan2(float_y2 - float_y1, float_x2 - float_x1)) % 180)
        list_cands.append({
            "x1": float_x1, "y1": float_y1, "x2": float_x2, "y2": float_y2,
            "length": float_len, "angle": float_angle,
        })

    # step 4: after length filter (cyan)
    arr_step_filtered = arr_roi_bgr.copy()
    for dict_c in list_cands:
        cv2.line(arr_step_filtered,
                 (int(dict_c["x1"]), int(dict_c["y1"])),
                 (int(dict_c["x2"]), int(dict_c["y2"])),
                 (255, 255, 0), 1)
    dict_steps["lsd_04_after_filter"] = arr_step_filtered

    list_cands.sort(key=lambda d: d["length"], reverse=True)
    int_n_cands = len(list_cands)
    arr_cx = np.array([(d["x1"] + d["x2"]) * 0.5 for d in list_cands], dtype=np.float32)
    arr_cy = np.array([(d["y1"] + d["y2"]) * 0.5 for d in list_cands], dtype=np.float32)
    arr_ang = np.array([d["angle"] for d in list_cands], dtype=np.float32)
    list_accepted_idx: tp.List[int] = []
    arr_acc_cx = np.empty(int_n_cands, dtype=np.float32)
    arr_acc_cy = np.empty(int_n_cands, dtype=np.float32)
    arr_acc_ang = np.empty(int_n_cands, dtype=np.float32)
    int_n_acc = 0
    for int_i in range(int_n_cands):
        if int_n_acc > 0:
            arr_d = np.sqrt((arr_cx[int_i] - arr_acc_cx[:int_n_acc]) ** 2
                            + (arr_cy[int_i] - arr_acc_cy[:int_n_acc]) ** 2)
            arr_ad = np.abs(arr_ang[int_i] - arr_acc_ang[:int_n_acc])
            arr_ad = np.minimum(arr_ad, 180.0 - arr_ad)
            if np.any((arr_d < CONST_LSD_DEDUP_DIST_PX) & (arr_ad < CONST_LSD_DEDUP_ANGLE_DEG)):
                continue
        arr_acc_cx[int_n_acc] = arr_cx[int_i]
        arr_acc_cy[int_n_acc] = arr_cy[int_i]
        arr_acc_ang[int_n_acc] = arr_ang[int_i]
        int_n_acc += 1
        list_accepted_idx.append(int_i)
    list_accepted = [list_cands[int_i] for int_i in list_accepted_idx]

    # step 5: after deduplication (orange)
    arr_step_deduped = arr_roi_bgr.copy()
    for dict_c in list_accepted:
        cv2.line(arr_step_deduped,
                 (int(dict_c["x1"]), int(dict_c["y1"])),
                 (int(dict_c["x2"]), int(dict_c["y2"])),
                 (0, 165, 255), 1)
    dict_steps["lsd_05_after_dedup"] = arr_step_deduped

    print(
        f"[LSD] 원본 {len(arr_lines)}개 → 필터 {len(list_cands)}개 "
        f"→ 중복제거 {len(list_accepted)}개",
        flush=True,
    )

    arr_debug = arr_roi_bgr.copy()

    def _process_segment(
        int_idx: int,
        dict_c: tp.Dict[str, float],
    ) -> tp.Optional[tp.Tuple[PrimaryParticleMeasurement, np.ndarray]]:
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
            return None
        if float_len < 1.0:
            return None

        float_ar = float_thickness / float_len
        str_category = "acicular" if float_ar < float_acicular_threshold else "plate"
        if str_particle_type in ("acicular", "plate") and str_category != str_particle_type:
            return None

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
        int_x2 = int_bx + int_bw
        int_y2 = int_by + int_bh
        int_bx = max(0, int_bx)
        int_by = max(0, int_by)
        int_bw = max(0, min(int_x2, int_roiW) - int_bx)
        int_bh = max(0, min(int_y2, int_roiH) - int_by)

        if _is_bbox_near_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH, int_edge_margin):
            return None

        arr_mask = np.zeros((int_roiH, int_roiW), dtype=np.uint8)
        cv2.fillPoly(arr_mask, [arr_corners.astype(np.int32).reshape(-1, 1, 2)], 1)

        if float_area_threshold > 0.0 and float(arr_mask.sum()) < float_area_threshold:
            return None

        float_mcx = (float_nx1 + float_nx2) / 2.0
        float_mcy = (float_ny1 + float_ny2) / 2.0

        return (PrimaryParticleMeasurement(
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
        ), arr_mask)

    list_raw = [_process_segment(int_idx, dict_c)
                for int_idx, dict_c in enumerate(list_accepted)]

    for tpl_result in list_raw:
        if tpl_result is None:
            continue
        obj_m, arr_mask = tpl_result
        list_objects.append(obj_m)
        list_masks.append(arr_mask)
        tpl_color = (0, 255, 0) if obj_m.str_category == "acicular" else (0, 128, 255)
        # 직사각형 윤곽선 (두께 포함)
        float_angle_rad = np.radians(obj_m.float_minRectAngle)
        float_cos = np.cos(float_angle_rad)
        float_sin = np.sin(float_angle_rad)
        float_half_l = obj_m.float_longAxisPx / 2.0
        float_half_t = obj_m.float_thicknessPx / 2.0
        float_cx = obj_m.float_centroidX
        float_cy = obj_m.float_centroidY
        arr_rect_pts = np.array([
            [float_cx - float_cos * float_half_l + float_sin * float_half_t,
             float_cy - float_sin * float_half_l - float_cos * float_half_t],
            [float_cx + float_cos * float_half_l + float_sin * float_half_t,
             float_cy + float_sin * float_half_l - float_cos * float_half_t],
            [float_cx + float_cos * float_half_l - float_sin * float_half_t,
             float_cy + float_sin * float_half_l + float_cos * float_half_t],
            [float_cx - float_cos * float_half_l - float_sin * float_half_t,
             float_cy - float_sin * float_half_l + float_cos * float_half_t],
        ], dtype=np.int32)
        cv2.polylines(arr_debug, [arr_rect_pts.reshape(-1, 1, 2)], True, tpl_color, 1)
        # 장축 중심선
        int_ax1 = int(round(float_cx - float_cos * float_half_l))
        int_ay1 = int(round(float_cy - float_sin * float_half_l))
        int_ax2 = int(round(float_cx + float_cos * float_half_l))
        int_ay2 = int(round(float_cy + float_sin * float_half_l))
        cv2.line(arr_debug, (int_ax1, int_ay1), (int_ax2, int_ay2), tpl_color, 1)

    print(f"[LSD] → 최종 {len(list_objects)}개  density={float_density:.3f}", flush=True)
    return list_objects, list_masks, arr_debug, dict_steps, float_density
