from __future__ import annotations
import dataclasses
import typing as tp
import cv2
import numpy as np

from core.schema import PrimaryParticleMeasurement
from utils.metrics import convert_pixels_to_micrometers

CONST_FUSE_ANGLE_DEG: float = 15.0
CONST_FUSE_OVERLAP_RATIO: float = 0.7


def fuse_contours(
    list_objects: tp.List[PrimaryParticleMeasurement],
    list_masks: tp.List[np.ndarray],
    float_acicular_threshold: float,
    str_particle_type: str,
    float_scale_pixels: float,
    float_scale_um: float,
    float_angle_tol_deg: float = CONST_FUSE_ANGLE_DEG,
    float_overlap_ratio: float = CONST_FUSE_OVERLAP_RATIO,
) -> tp.Tuple[tp.List[PrimaryParticleMeasurement], tp.List[np.ndarray]]:
    """Fuse contours (2D masks) that share long-axis direction and overlap significantly.

    Two masks are fused when:
      1. |angle_i - angle_j| (mod 180°) < float_angle_tol_deg
      2. intersection_pixels / min(area_i, area_j) >= float_overlap_ratio
    Union-find handles chains of 3+ overlapping masks.
    """
    n = len(list_objects)
    if n <= 1:
        return list_objects, list_masks

    arr_areas = np.array([float(m.sum()) for m in list_masks], dtype=np.float64)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # 벡터화된 bbox overlap 행렬 사전 계산
    arr_bx1 = np.array([o.int_bboxX for o in list_objects], dtype=np.int32)
    arr_by1 = np.array([o.int_bboxY for o in list_objects], dtype=np.int32)
    arr_bx2 = arr_bx1 + np.array([o.int_bboxWidth for o in list_objects], dtype=np.int32)
    arr_by2 = arr_by1 + np.array([o.int_bboxHeight for o in list_objects], dtype=np.int32)
    arr_angles = np.array([o.float_minRectAngle for o in list_objects], dtype=np.float32)

    arr_bbox_ok = (
        (arr_bx2[:, None] > arr_bx1[None, :]) & (arr_bx2[None, :] > arr_bx1[:, None]) &
        (arr_by2[:, None] > arr_by1[None, :]) & (arr_by2[None, :] > arr_by1[:, None])
    )

    arr_adiff = np.abs(arr_angles[:, None] - arr_angles[None, :])
    arr_adiff = np.minimum(arr_adiff, 180.0 - arr_adiff)
    arr_angle_ok = arr_adiff < float_angle_tol_deg

    arr_candidate = np.triu(arr_bbox_ok & arr_angle_ok, k=1)
    list_pairs = list(zip(*np.where(arr_candidate)))

    for int_i, int_j in list_pairs:
        float_inter = float((list_masks[int_i] & list_masks[int_j]).sum())
        if float_inter <= 0.0:
            continue
        if float_inter / max(min(arr_areas[int_i], arr_areas[int_j]), 1.0) < float_overlap_ratio:
            continue
        pi, pj = _find(int_i), _find(int_j)
        if pi != pj:
            parent[pi] = pj

    groups: tp.Dict[int, tp.List[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    list_new_objects: tp.List[PrimaryParticleMeasurement] = []
    list_new_masks: tp.List[np.ndarray] = []

    for int_new_idx, list_idx in enumerate(groups.values()):
        if len(list_idx) == 1:
            list_new_objects.append(dataclasses.replace(list_objects[list_idx[0]], int_index=int_new_idx))
            list_new_masks.append(list_masks[list_idx[0]])
            continue

        arr_merged = list_masks[list_idx[0]].copy()
        for k in list_idx[1:]:
            arr_merged = cv2.bitwise_or(arr_merged, list_masks[k])

        list_cnts, _ = cv2.findContours(arr_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_cnts:
            continue
        arr_cnt = max(list_cnts, key=cv2.contourArea)

        rect = cv2.minAreaRect(arr_cnt)
        (float_cx, float_cy), (float_rw, float_rh), _ = rect
        float_long = max(float_rw, float_rh)
        float_short = min(float_rw, float_rh)
        if float_long < 1.0:
            continue

        arr_bpts = cv2.boxPoints(rect)
        float_d01 = float(np.linalg.norm(arr_bpts[1] - arr_bpts[0]))
        float_d12 = float(np.linalg.norm(arr_bpts[2] - arr_bpts[1]))
        arr_vec = arr_bpts[1] - arr_bpts[0] if float_d01 >= float_d12 else arr_bpts[2] - arr_bpts[1]
        float_angle = float(np.degrees(np.arctan2(float(arr_vec[1]), float(arr_vec[0]))) % 180)

        float_ar = float_short / max(float_long, 1.0)
        str_category = "acicular" if float_ar < float_acicular_threshold else "plate"
        if str_particle_type in ("acicular", "plate") and str_category != str_particle_type:
            continue

        int_imgH, int_imgW = arr_merged.shape[:2]
        arr_rect_mask = np.zeros(arr_merged.shape, dtype=np.uint8)
        cv2.fillPoly(arr_rect_mask, [arr_bpts.astype(np.int32)], 1)

        int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_bpts.astype(np.int32))
        int_bx = max(0, int_bx)
        int_by = max(0, int_by)
        int_bw = max(0, min(int_bw, int_imgW - int_bx))
        int_bh = max(0, min(int_bh, int_imgH - int_by))
        list_new_objects.append(PrimaryParticleMeasurement(
            int_index=int_new_idx,
            str_category=str_category,
            int_maskArea=int(arr_rect_mask.sum()),
            float_confidence=None,
            int_bboxX=int_bx,
            int_bboxY=int_by,
            int_bboxWidth=int_bw,
            int_bboxHeight=int_bh,
            float_centroidX=float_cx,
            float_centroidY=float_cy,
            float_thicknessPx=float_short,
            float_longAxisPx=float_long,
            float_minRectAngle=float_angle,
            float_thicknessUm=convert_pixels_to_micrometers(float_short, float_scale_pixels, float_scale_um),
            float_longAxisUm=convert_pixels_to_micrometers(float_long, float_scale_pixels, float_scale_um),
            float_aspectRatio=float_ar,
            int_longestHorizontal=int_bw,
            int_longestVertical=int_bh,
            float_longestHorizontalUm=convert_pixels_to_micrometers(float(int_bw), float_scale_pixels, float_scale_um),
            float_longestVerticalUm=convert_pixels_to_micrometers(float(int_bh), float_scale_pixels, float_scale_um),
        ))
        list_new_masks.append(arr_rect_mask)

    return list_new_objects, list_new_masks
