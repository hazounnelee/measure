from __future__ import annotations
import dataclasses
import typing as tp
import cv2
import numpy as np

from core.schema import PrimaryParticleMeasurement
from utils.metrics import convert_pixels_to_micrometers

CONST_FUSE_ANGLE_DEG: float = 15.0
CONST_FUSE_OVERLAP_RATIO: float = 0.7

# advanced_fuse 전용
CONST_FUSE_CONTAINMENT_THRESHOLD: float = 0.70   # 작은 마스크가 큰 마스크에 70% 이상 포함 → 드롭
CONST_FUSE_LONG_AXIS_THRESHOLD: float = 0.70     # 중심 변위의 장축 방향 성분 비율 >= 이 값 → 끝-끝
CONST_FUSE_SHORT_OVERLAP_THRESHOLD: float = 0.70  # 단축 방향 투영 겹침 / min(t_i,t_j)


def _proj_overlap(
    float_c1: float, float_half1: float,
    float_c2: float, float_half2: float,
) -> float:
    """두 1D 구간 [c-half, c+half]의 겹침 길이."""
    return max(0.0, min(float_c1 + float_half1, float_c2 + float_half2)
               - max(float_c1 - float_half1, float_c2 - float_half2))


def fuse_contours(
    list_objects: tp.List[PrimaryParticleMeasurement],
    list_masks: tp.List[np.ndarray],
    float_acicular_threshold: float,
    str_particle_type: str,
    float_scale_pixels: float,
    float_scale_um: float,
    float_angle_tol_deg: float = CONST_FUSE_ANGLE_DEG,
    float_overlap_ratio: float = CONST_FUSE_OVERLAP_RATIO,
    bool_advanced: bool = False,
) -> tp.Tuple[tp.List[PrimaryParticleMeasurement], tp.List[np.ndarray]]:
    """Fuse contours (2D masks).

    --fuse (bool_advanced=False):
        각도차 < angle_tol AND intersection/min_area >= overlap_ratio → 합침

    --advanced_fuse (bool_advanced=True):
        조건 1 — 포함 필터: intersection/min_area >= 0.70 → 작은 마스크 드롭
        조건 2 — 끝-끝 체인:
            각도차 < angle_tol
            AND 단축 방향 투영 겹침 / min(t_i,t_j) >= 0.70  (단면이 충분히 겹침)
            AND 장축 방향 투영 겹침 > 0                       (장축이 실제로 닿음)
            → 합침 (두께는 장축 길이 가중 평균)
    """
    n = len(list_objects)
    if n <= 1:
        return list_objects, list_masks

    arr_areas = np.array([float(m.sum()) for m in list_masks], dtype=np.float64)
    arr_thickness = np.array([o.float_thicknessPx for o in list_objects], dtype=np.float64)
    arr_longaxis = np.array([o.float_longAxisPx for o in list_objects], dtype=np.float64)
    arr_angles = np.array([o.float_minRectAngle for o in list_objects], dtype=np.float32)
    arr_cx = np.array([o.float_centroidX for o in list_objects], dtype=np.float64)
    arr_cy = np.array([o.float_centroidY for o in list_objects], dtype=np.float64)

    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # bbox 겹침 + 각도 유사 후보 쌍 사전 계산
    arr_bx1 = np.array([o.int_bboxX for o in list_objects], dtype=np.int32)
    arr_by1 = np.array([o.int_bboxY for o in list_objects], dtype=np.int32)
    arr_bx2 = arr_bx1 + np.array([o.int_bboxWidth for o in list_objects], dtype=np.int32)
    arr_by2 = arr_by1 + np.array([o.int_bboxHeight for o in list_objects], dtype=np.int32)

    arr_bbox_ok = (
        (arr_bx2[:, None] > arr_bx1[None, :]) & (arr_bx2[None, :] > arr_bx1[:, None]) &
        (arr_by2[:, None] > arr_by1[None, :]) & (arr_by2[None, :] > arr_by1[:, None])
    )
    arr_adiff = np.abs(arr_angles[:, None] - arr_angles[None, :])
    arr_adiff = np.minimum(arr_adiff, 180.0 - arr_adiff)
    arr_angle_ok = arr_adiff < float_angle_tol_deg
    arr_candidate = np.triu(arr_bbox_ok & arr_angle_ok, k=1)
    list_pairs = list(zip(*np.where(arr_candidate)))

    set_drop: tp.Set[int] = set()

    for int_i, int_j in list_pairs:
        if int_i in set_drop or int_j in set_drop:
            continue

        if bool_advanced:
            # ── 조건 1: 포함 필터 ──────────────────────────────────────────
            float_inter = float((list_masks[int_i] & list_masks[int_j]).sum())
            if float_inter <= 0.0:
                continue
            float_area_small = min(arr_areas[int_i], arr_areas[int_j])
            float_area_large = max(arr_areas[int_i], arr_areas[int_j])
            float_containment = float_inter / max(float_area_small, 1.0)
            # 크기가 비슷한 두 마스크(비율 > 0.7)는 포함 관계가 아님
            bool_size_asymmetric = float_area_small / max(float_area_large, 1.0) <= 0.7
            if float_containment >= CONST_FUSE_CONTAINMENT_THRESHOLD and bool_size_asymmetric:
                int_small = int_i if arr_areas[int_i] <= arr_areas[int_j] else int_j
                set_drop.add(int_small)
                continue

            # ── 조건 2: 끝-끝 체인 ────────────────────────────────────────
            float_avg_rad = np.radians(
                (float(arr_angles[int_i]) + float(arr_angles[int_j])) / 2.0)
            float_cos = float(np.cos(float_avg_rad))
            float_sin = float(np.sin(float_avg_rad))

            # 2-1. 중심 변위가 장축 방향인지 (끝-끝 배열)
            float_dx = arr_cx[int_j] - arr_cx[int_i]
            float_dy = arr_cy[int_j] - arr_cy[int_i]
            float_dist = float(np.sqrt(float_dx ** 2 + float_dy ** 2))
            if float_dist > 1.0:
                float_d_long = abs(float_dx * float_cos + float_dy * float_sin)
                if float_d_long / float_dist < CONST_FUSE_LONG_AXIS_THRESHOLD:
                    continue  # 단축 방향으로 나란히 붙은 경우 → 합치지 않음

            # 2-2. 장축 방향 투영 겹침 > 0 (실제로 닿아 있음)
            float_long_ci = arr_cx[int_i] * float_cos + arr_cy[int_i] * float_sin
            float_long_cj = arr_cx[int_j] * float_cos + arr_cy[int_j] * float_sin
            float_long_overlap = _proj_overlap(
                float_long_ci, arr_longaxis[int_i] / 2.0,
                float_long_cj, arr_longaxis[int_j] / 2.0,
            )
            if float_long_overlap <= 0.0:
                continue

            # 2-3. 단축 방향 단면 겹침 (두 마스크의 단면이 정렬됨)
            float_short_ci = -arr_cx[int_i] * float_sin + arr_cy[int_i] * float_cos
            float_short_cj = -arr_cx[int_j] * float_sin + arr_cy[int_j] * float_cos
            float_short_overlap = _proj_overlap(
                float_short_ci, arr_thickness[int_i] / 2.0,
                float_short_cj, arr_thickness[int_j] / 2.0,
            )
            float_min_t = min(arr_thickness[int_i], arr_thickness[int_j])
            if float_short_overlap / max(float_min_t, 1.0) < CONST_FUSE_SHORT_OVERLAP_THRESHOLD:
                continue

        else:
            # ── basic --fuse ───────────────────────────────────────────────
            float_inter = float((list_masks[int_i] & list_masks[int_j]).sum())
            if float_inter <= 0.0:
                continue
            float_smaller_area = min(arr_areas[int_i], arr_areas[int_j])
            if float_inter / max(float_smaller_area, 1.0) < float_overlap_ratio:
                continue

        pi, pj = _find(int_i), _find(int_j)
        if pi != pj:
            parent[pi] = pj

    groups: tp.Dict[int, tp.List[int]] = {}
    for i in range(n):
        if i in set_drop:
            continue
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
        (float_cx_new, float_cy_new), (float_rw, float_rh), _ = rect
        float_long = max(float_rw, float_rh)
        float_short_rect = min(float_rw, float_rh)
        if float_long < 1.0:
            continue

        arr_bpts = cv2.boxPoints(rect)
        float_d01 = float(np.linalg.norm(arr_bpts[1] - arr_bpts[0]))
        float_d12 = float(np.linalg.norm(arr_bpts[2] - arr_bpts[1]))
        arr_vec = arr_bpts[1] - arr_bpts[0] if float_d01 >= float_d12 else arr_bpts[2] - arr_bpts[1]
        float_angle = float(np.degrees(np.arctan2(float(arr_vec[1]), float(arr_vec[0]))) % 180)

        # 두께: 장축 길이 가중 평균 (advanced) / minAreaRect 단축 (basic)
        if bool_advanced:
            float_weights = sum(arr_longaxis[k] for k in list_idx)
            float_thickness_new = (
                sum(arr_thickness[k] * arr_longaxis[k] for k in list_idx)
                / max(float_weights, 1.0)
            )
        else:
            float_thickness_new = float_short_rect

        float_ar = float_thickness_new / max(float_long, 1.0)
        str_category = "acicular" if float_ar < float_acicular_threshold else "plate"
        if str_particle_type in ("acicular", "plate") and str_category != str_particle_type:
            continue

        int_imgH, int_imgW = arr_merged.shape[:2]
        arr_rect_mask = np.zeros(arr_merged.shape, dtype=np.uint8)
        cv2.fillPoly(arr_rect_mask, [arr_bpts.astype(np.int32)], 1)

        int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_bpts.astype(np.int32))
        int_x2 = int_bx + int_bw
        int_y2 = int_by + int_bh
        int_bx = max(0, int_bx)
        int_by = max(0, int_by)
        int_bw = max(0, min(int_x2, int_imgW) - int_bx)
        int_bh = max(0, min(int_y2, int_imgH) - int_by)

        list_new_objects.append(PrimaryParticleMeasurement(
            int_index=int_new_idx,
            str_category=str_category,
            int_maskArea=int(arr_rect_mask.sum()),
            float_confidence=None,
            int_bboxX=int_bx,
            int_bboxY=int_by,
            int_bboxWidth=int_bw,
            int_bboxHeight=int_bh,
            float_centroidX=float_cx_new,
            float_centroidY=float_cy_new,
            float_thicknessPx=float_thickness_new,
            float_longAxisPx=float_long,
            float_minRectAngle=float_angle,
            float_thicknessUm=convert_pixels_to_micrometers(float_thickness_new, float_scale_pixels, float_scale_um),
            float_longAxisUm=convert_pixels_to_micrometers(float_long, float_scale_pixels, float_scale_um),
            float_aspectRatio=float_ar,
            int_longestHorizontal=int_bw,
            int_longestVertical=int_bh,
            float_longestHorizontalUm=convert_pixels_to_micrometers(float(int_bw), float_scale_pixels, float_scale_um),
            float_longestVerticalUm=convert_pixels_to_micrometers(float(int_bh), float_scale_pixels, float_scale_um),
        ))
        list_new_masks.append(arr_rect_mask)

    return list_new_objects, list_new_masks
