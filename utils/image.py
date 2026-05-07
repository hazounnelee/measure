from __future__ import annotations
import typing as tp
import cv2
import numpy as np


def draw_label_no_overlap(
    arr_img: np.ndarray,
    list_lines: tp.List[str],
    int_anchorX: int,
    int_anchorY: int,
    tpl_color: tp.Tuple[int, int, int],
    list_placedRects: tp.List[tp.Tuple[int, int, int, int]],
    float_fontScale: float = 0.5,
) -> None:
    """Draw a multi-line label near anchor without overlapping already-placed labels.

    Tries 8 candidate positions around the anchor. Draws a dark outline under the
    colored text for readability. Appends the placed rect to list_placedRects.
    """
    int_font = cv2.FONT_HERSHEY_SIMPLEX
    int_lineH, int_maxW = 0, 0
    for str_line in list_lines:
        (int_tw, int_th), int_bl = cv2.getTextSize(str_line, int_font, float_fontScale, 1)
        int_maxW = max(int_maxW, int_tw)
        int_lineH = max(int_lineH, int_th + int_bl)

    int_gap = 2
    int_totalH = int_lineH * len(list_lines) + int_gap * (len(list_lines) - 1)
    int_pad = 4
    int_imgH, int_imgW = arr_img.shape[:2]

    list_candidates = [
        (int_anchorX - int_maxW // 2, int_anchorY - int_totalH - int_pad),
        (int_anchorX + int_pad, int_anchorY - int_totalH // 2),
        (int_anchorX - int_maxW // 2, int_anchorY + int_pad),
        (int_anchorX - int_maxW - int_pad, int_anchorY - int_totalH // 2),
        (int_anchorX + int_pad, int_anchorY - int_totalH - int_pad),
        (int_anchorX - int_maxW - int_pad, int_anchorY - int_totalH - int_pad),
        (int_anchorX + int_pad, int_anchorY + int_pad),
        (int_anchorX - int_maxW - int_pad, int_anchorY + int_pad),
    ]

    def _no_overlap(int_tx: int, int_ty: int) -> bool:
        tpl_r = (int_tx, int_ty, int_tx + int_maxW, int_ty + int_totalH)
        return all(
            tpl_r[2] < r[0] or tpl_r[0] > r[2] or tpl_r[3] < r[1] or tpl_r[1] > r[3]
            for r in list_placedRects
        )

    int_tx, int_ty = list_candidates[0]
    for int_cx, int_cy in list_candidates:
        int_cx = int(np.clip(int_cx, 0, max(0, int_imgW - int_maxW)))
        int_cy = int(np.clip(int_cy, 0, max(0, int_imgH - int_totalH)))
        if _no_overlap(int_cx, int_cy):
            int_tx, int_ty = int_cx, int_cy
            break

    int_tx = int(np.clip(int_tx, 0, max(0, int_imgW - int_maxW)))
    int_ty = int(np.clip(int_ty, 0, max(0, int_imgH - int_totalH)))
    list_placedRects.append((int_tx, int_ty, int_tx + int_maxW, int_ty + int_totalH))

    for int_i, str_line in enumerate(list_lines):
        int_y = int_ty + int_lineH * (int_i + 1) + int_gap * int_i
        cv2.putText(arr_img, str_line, (int_tx, int_y), int_font,
                    float_fontScale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(arr_img, str_line, (int_tx, int_y), int_font,
                    float_fontScale, tpl_color, 1, cv2.LINE_AA)


def compute_adaptive_block_size(
    int_h: int,
    int_w: int,
    int_divisor: int,
    int_max: int = 0,
) -> int:
    """Return an odd block size for cv2.adaptiveThreshold scaled to image dimensions."""
    int_bs = max(11, int(min(int_h, int_w) / int_divisor))
    if int_max > 0:
        int_bs = min(int_bs, int_max)
    return int_bs if int_bs % 2 == 1 else int_bs + 1


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
    int_stride = max(1, int_stride)
    int_tileW = min(max(1, int_tileSize), int_roiW)
    int_tileH = min(max(1, int_tileSize), int_roiH)
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
                list_tiles.append((
                    int_x1 + int_x, int_y1 + int_y,
                    int_x2, min(int_y2, int_y1 + int_y + int_tileH),
                ))
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
    arr_grad = cv2.normalize(np.sqrt(arr_grad), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    arr_lap = cv2.normalize(np.abs(cv2.Laplacian(arr_blur, cv2.CV_32F)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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

    Returns ((x1, y1, x2, y2), arr_debug_mask) or None if detection fails.
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
    int_y_sphere_top = int_cy - int_r
    int_y1 = max(0, int_y_sphere_top)
    int_y2 = min(int_h, int_y_sphere_top + int(int_r * 2 * float_cap))
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
