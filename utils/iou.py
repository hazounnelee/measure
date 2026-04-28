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
    tpl_boxA: tuple,
    tpl_boxB: tuple,
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
