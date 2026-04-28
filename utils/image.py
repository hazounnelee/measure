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
