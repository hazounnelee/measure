from __future__ import annotations
import dataclasses
import typing as tp
import cv2
import numpy as np

from core.schema import PrimaryParticleMeasurement
from utils.metrics import convert_pixels_to_micrometers

CONST_FUSE_ANGLE_DEG: float = 10.0
CONST_FUSE_OVERLAP_RATIO: float = 0.4


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

    list_areas = [float(m.sum()) for m in list_masks]
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            float_adiff = abs(list_objects[i].float_minRectAngle - list_objects[j].float_minRectAngle)
            float_adiff = min(float_adiff, 180.0 - float_adiff)
            if float_adiff > float_angle_tol_deg:
                continue
            float_inter = float((list_masks[i] & list_masks[j]).sum())
            if float_inter <= 0.0:
                continue
            if float_inter / max(min(list_areas[i], list_areas[j]), 1.0) < float_overlap_ratio:
                continue
            pi, pj = _find(i), _find(j)
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

        arr_bpts = cv2.boxPoints(rect)
        float_d01 = float(np.linalg.norm(arr_bpts[1] - arr_bpts[0]))
        float_d12 = float(np.linalg.norm(arr_bpts[2] - arr_bpts[1]))
        arr_vec = arr_bpts[1] - arr_bpts[0] if float_d01 >= float_d12 else arr_bpts[2] - arr_bpts[1]
        float_angle = float(np.degrees(np.arctan2(float(arr_vec[1]), float(arr_vec[0]))) % 180)

        float_ar = float_short / max(float_long, 1.0)
        str_category = "acicular" if float_ar < float_acicular_threshold else "plate"
        if str_particle_type in ("acicular", "plate") and str_category != str_particle_type:
            continue

        int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_merged)
        list_new_objects.append(PrimaryParticleMeasurement(
            int_index=int_new_idx,
            str_category=str_category,
            int_maskArea=int(arr_merged.sum()),
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
        list_new_masks.append(arr_merged)

    return list_new_objects, list_new_masks
