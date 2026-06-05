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




def _find_fg_mask(arr_tileGray: np.ndarray) -> np.ndarray:
    """Otsu threshold + morph cleanup → foreground (bright particles) binary mask.

    SEM secondary particle images always have bright particles on dark background,
    so THRESH_BINARY keeps bright=foreground without inversion.
    """
    arr_blur = cv2.GaussianBlur(arr_tileGray, (5, 5), 0)
    _, arr_fg = cv2.threshold(arr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    arr_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    arr_fg = cv2.morphologyEx(arr_fg, cv2.MORPH_CLOSE, arr_k, iterations=2)
    arr_fg = cv2.morphologyEx(arr_fg, cv2.MORPH_OPEN,  arr_k, iterations=1)
    return arr_fg


def find_dist_transform_peaks(
    arr_blob: np.ndarray,
    int_min_peak_dist: int,
    int_max_peaks: int = 200,
) -> tp.List[tp.Tuple[int, int]]:
    """All local maxima of the distance transform within a blob.

    Uses dilation-based non-maximum suppression with radius ``int_min_peak_dist``.
    Returns at most ``int_max_peaks`` peaks as (x, y) tuples.
    """
    arr_dist = cv2.distanceTransform(arr_blob.astype(np.uint8), cv2.DIST_L2, 5)
    float_max = float(arr_dist.max())
    if float_max == 0:
        return []
    int_ks = 2 * int_min_peak_dist + 1
    arr_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int_ks, int_ks))
    arr_dilated = cv2.dilate(arr_dist, arr_kernel)
    arr_peak = (arr_dist == arr_dilated) & (arr_dist > 0.25 * float_max) & (arr_blob > 0)
    arr_coords = np.column_stack(np.where(arr_peak))  # (row, col)
    if arr_coords.shape[0] == 0:
        return []
    # Sort by distance value descending, keep top N
    arr_vals = arr_dist[arr_coords[:, 0], arr_coords[:, 1]]
    arr_order = np.argsort(-arr_vals)[:int_max_peaks]
    return [(int(arr_coords[i, 1]), int(arr_coords[i, 0])) for i in arr_order]


def detect_hct_prompts(
    arr_tileGray: np.ndarray,
    int_minDist: int = 14,
    int_numNeg: int = 3,
    int_minArea: int = 1500,
    float_dist_thresh: float = 0.45,
) -> tp.Tuple[
    tp.List[np.ndarray],
    tp.List[tp.Tuple[int, int]],
    tp.List[tp.Tuple[int, int]],
    tp.List[tp.Tuple[int, int, int]],
]:
    """Hough Circle Transform → SAM2 positive/negative prompts.

    HCT는 closed ring이 없어도 gradient 투표로 원의 중심을 찾으며,
    겹침/부분 가려짐에도 강건하다.

    1. HCT → 각 원의 중심 (x, y) → SAM2 포지티브 프롬프트
    2. HCT 실패 시 fallback: Canny contour fill + 거리변환 peak
    3. HCT 미커버 전경 블롭 → 거리변환 peak 추가 (fragment 등)
    4. 배경에서 네거티브 프롬프트 샘플링

    Returns:
        isolated_masks — 항상 빈 리스트 (모두 SAM2 경로로 처리)
        pos_points     — 입자 중심 (x, y)
        neg_points     — 배경 (x, y)
    """
    int_h, int_w = arr_tileGray.shape[:2]
    arr_k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ── 1. Hough Circle Transform ──────────────────────────────────────────
    # 반지름 범위: minArea → min_r, 타일 단변의 1/3 → max_r
    int_min_r = max(8, int(np.sqrt(int_minArea / np.pi) * 0.8))
    int_max_r = max(int_min_r + 10, min(int_h, int_w) // 4)
    # HCT minDist: 입자 간 최소 중심 간격 ≈ 1.5 × 최소 반지름
    int_hough_min_dist = max(int_minDist, int(int_min_r * 1.5))

    arr_blur = cv2.GaussianBlur(arr_tileGray, (9, 9), 0)
    arr_circles = cv2.HoughCircles(
        arr_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=float(int_hough_min_dist),
        param1=70,
        param2=35,
        minRadius=int_min_r,
        maxRadius=int_max_r,
    )

    # Otsu threshold to reject circle centers that fall in dark background gaps
    int_otsu_val, _ = cv2.threshold(
        cv2.GaussianBlur(arr_tileGray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    list_pos: tp.List[tp.Tuple[int, int]] = []
    list_hct_circles: tp.List[tp.Tuple[int, int, int]] = []
    arr_circles_mask = np.zeros((int_h, int_w), dtype=np.uint8)  # HCT 원이 커버하는 영역
    if arr_circles is not None:
        margin = int_minDist // 2
        for (float_x, float_y, float_r) in arr_circles[0]:
            int_x, int_y = int(round(float_x)), int(round(float_y))
            if margin <= int_x < int_w - margin and margin <= int_y < int_h - margin:
                # Verify circle center lands in a bright (particle) region, not a dark gap
                int_sr = max(3, int(float_r * 0.25))
                arr_patch = arr_tileGray[
                    max(0, int_y - int_sr): min(int_h, int_y + int_sr),
                    max(0, int_x - int_sr): min(int_w, int_x + int_sr),
                ]
                if arr_patch.size == 0 or float(arr_patch.mean()) < float(int_otsu_val) * 0.85:
                    continue  # dark gap between particles — skip
                list_pos.append((int_x, int_y))
                list_hct_circles.append((int_x, int_y, int(round(float_r))))
                cv2.circle(arr_circles_mask, (int_x, int_y), int(round(float_r)), 255, -1)

    # ── 2. Fallback: contour RETR_CCOMP hole fill + dist transform peaks ──
    if not list_pos:
        arr_edges = cv2.Canny(arr_blur, 30, 90)
        arr_k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        arr_closed = cv2.morphologyEx(arr_edges, cv2.MORPH_CLOSE, arr_k7, iterations=3)
        # 내부 구멍(particle interior)을 RETR_CCOMP hole로 추출
        arr_filled = np.zeros((int_h, int_w), dtype=np.uint8)
        list_cnts, arr_hier = cv2.findContours(arr_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if arr_hier is not None:
            for int_i, cnt in enumerate(list_cnts):
                if arr_hier[0][int_i][3] != -1 and cv2.contourArea(cnt) >= int_minArea:
                    cv2.drawContours(arr_filled, [cnt], 0, 255, -1)
        list_pos = find_dist_transform_peaks(
            arr_filled if arr_filled.sum() > 0 else _find_fg_mask(arr_tileGray),
            int_minDist,
        )

    # ── 3. HCT 미커버 전경 블롭 → fragment/비원형 입자 프롬프트 추가 ─────
    arr_fg = _find_fg_mask(arr_tileGray)
    arr_uncovered = cv2.bitwise_and(arr_fg, cv2.bitwise_not(arr_circles_mask))
    # 3×3 opening: 1px 노이즈만 제거, 작은 fragment 보존 (기존 7×7은 너무 많이 지웠음)
    arr_k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    arr_uncovered = cv2.morphologyEx(arr_uncovered, cv2.MORPH_OPEN, arr_k3, iterations=1)
    int_n_uc, arr_uc_labels = cv2.connectedComponents(arr_uncovered)
    arr_uc_dist = cv2.distanceTransform(arr_uncovered, cv2.DIST_L2, 5)
    for int_lbl in range(1, int_n_uc):
        arr_blob = (arr_uc_labels == int_lbl).astype(np.uint8)
        if arr_blob.sum() < int_minArea // 8:  # 기준 완화: ~190px (기존 375px)
            continue
        # fragment는 부분적으로 어두울 수 있어 0.75× Otsu로 완화 (기존 1.0×)
        if float(arr_tileGray[arr_blob > 0].mean()) < float(int_otsu_val) * 0.75:
            continue
        int_peak = int(np.argmax(arr_uc_dist * arr_blob.astype(np.float32)))
        int_py, int_px = divmod(int_peak, int_w)
        list_pos.append((int_px, int_py))

    # ── 4. 네거티브: 배경 영역에서 균등 샘플링 ────────────────────────────
    arr_bg_eroded = cv2.erode(cv2.bitwise_not(arr_fg), arr_k5, iterations=4)
    arr_bg_coords = np.column_stack(np.where(arr_bg_eroded > 0))
    list_neg: tp.List[tp.Tuple[int, int]] = []
    if arr_bg_coords.shape[0] > 0 and int_numNeg > 0:
        arr_idx = np.linspace(0, arr_bg_coords.shape[0] - 1, int_numNeg, dtype=int)
        for idx in arr_idx:
            list_neg.append((int(arr_bg_coords[idx, 1]), int(arr_bg_coords[idx, 0])))

    return [], list_pos, list_neg, list_hct_circles


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
