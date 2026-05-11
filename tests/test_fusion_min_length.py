#!/usr/bin/env python3
"""
fuse_contours + min_length 필터 순서 검증 단위 테스트.

테스트 시나리오:
  - obj_A: longAxisPx=80  (min_length=50 통과)
  - obj_B1: longAxisPx=30, obj_B2: longAxisPx=30  (개별 탈락)
    B1/B2 마스크는 겹침 → fuse 후 longAxisPx ~60 (통과)

기대:
  no_fuse + min_length=50 → A만 = 1개
  fuse    + min_length=50 → A + fused_B = 2개
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.schema import PrimaryParticleMeasurement
from utils.contour import fuse_contours

H, W = 200, 400


def make_rod_mask(cx, cy, length, width, angle_deg):
    """회전 직사각형 마스크 생성."""
    mask = np.zeros((H, W), dtype=np.uint8)
    import cv2
    rad = np.radians(angle_deg)
    dx, dy = np.cos(rad) * length / 2, np.sin(rad) * length / 2
    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))
    cv2.line(mask, pt1, pt2, 1, width)
    return mask


def make_obj(idx, cx, cy, length, width, angle_deg,
             scale_px=100.0, scale_um=1.0):
    from utils.metrics import convert_pixels_to_micrometers
    import cv2
    mask = make_rod_mask(cx, cy, length, width, angle_deg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (rw, rh), _ = rect
    long_ax = max(rw, rh)
    short_ax = min(rw, rh)
    pts = cv2.boxPoints(rect)
    d01 = np.linalg.norm(pts[1] - pts[0])
    d12 = np.linalg.norm(pts[2] - pts[1])
    vec = pts[1] - pts[0] if d01 >= d12 else pts[2] - pts[1]
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])) % 180)
    bx, by, bw, bh = cv2.boundingRect(mask)
    ar = short_ax / max(long_ax, 1.0)
    return PrimaryParticleMeasurement(
        int_index=idx,
        str_category="acicular",
        int_maskArea=int(mask.sum()),
        float_confidence=None,
        int_bboxX=bx, int_bboxY=by, int_bboxWidth=bw, int_bboxHeight=bh,
        float_centroidX=float(rcx), float_centroidY=float(rcy),
        float_thicknessPx=float(short_ax),
        float_longAxisPx=float(long_ax),
        float_minRectAngle=angle,
        float_thicknessUm=convert_pixels_to_micrometers(short_ax, scale_px, scale_um),
        float_longAxisUm=convert_pixels_to_micrometers(long_ax, scale_px, scale_um),
        float_aspectRatio=ar,
        int_longestHorizontal=bw, int_longestVertical=bh,
        float_longestHorizontalUm=convert_pixels_to_micrometers(bw, scale_px, scale_um),
        float_longestVerticalUm=convert_pixels_to_micrometers(bh, scale_px, scale_um),
    )


# ── 입자 생성 ──────────────────────────────────────────────────────────────────
SCALE_PX, SCALE_UM = 100.0, 1.0

# A: longAxisPx ~80, passes min_length=50
obj_A = make_obj(0, cx=80,  cy=100, length=80, width=8, angle_deg=45,
                 scale_px=SCALE_PX, scale_um=SCALE_UM)

# B1, B2: shortaxis ~30, overlapping masks
obj_B1 = make_obj(1, cx=260, cy=80,  length=30, width=8, angle_deg=45,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)
obj_B2 = make_obj(2, cx=260, cy=83, length=30, width=8, angle_deg=45,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)

mask_A  = make_rod_mask(cx=80,  cy=100, length=80, width=8, angle_deg=45)
mask_B1 = make_rod_mask(cx=260, cy=80,  length=30, width=8, angle_deg=45)
mask_B2 = make_rod_mask(cx=260, cy=83, length=30, width=8, angle_deg=45)

list_all_objs  = [obj_A, obj_B1, obj_B2]
list_all_masks = [mask_A, mask_B1, mask_B2]

print(f"obj_A.float_longAxisPx  = {obj_A.float_longAxisPx:.1f}px")
print(f"obj_B1.float_longAxisPx = {obj_B1.float_longAxisPx:.1f}px")
print(f"obj_B2.float_longAxisPx = {obj_B2.float_longAxisPx:.1f}px")
print(f"B1∩B2 mask overlap      = {int((mask_B1 & mask_B2).sum())} px")

MIN_LENGTH = 38

# ── Case 1: min_length 없이 전체 ─────────────────────────────────────────────
assert len(list_all_objs) == 3, "입자 3개 시작 확인"
print(f"\n[Case 1] all objects: {len(list_all_objs)}개  (기대: 3)")

# ── Case 2: no fuse + min_length ──────────────────────────────────────────────
filtered_no_fuse = [(o, m) for o, m in zip(list_all_objs, list_all_masks)
                    if o.float_longAxisPx >= MIN_LENGTH]
n_no_fuse = len(filtered_no_fuse)
print(f"[Case 2] no_fuse + min_length={MIN_LENGTH}: {n_no_fuse}개  (기대: 1 - A만)")
assert n_no_fuse == 1, f"기대 1개, 실제 {n_no_fuse}개"

# ── Case 3: fuse → min_length ─────────────────────────────────────────────────
fused_objs, fused_masks = fuse_contours(
    list_all_objs, list_all_masks,
    float_acicular_threshold=0.4,
    str_particle_type="acicular",
    float_scale_pixels=SCALE_PX,
    float_scale_um=SCALE_UM,
)
print(f"  fuse 결과: {len(list_all_objs)}개 → {len(fused_objs)}개")
for o in fused_objs:
    print(f"    idx={o.int_index}  longAxisPx={o.float_longAxisPx:.1f}")

filtered_fuse = [(o, m) for o, m in zip(fused_objs, fused_masks)
                 if o.float_longAxisPx >= MIN_LENGTH]
n_fuse = len(filtered_fuse)
print(f"[Case 3] fuse + min_length={MIN_LENGTH}: {n_fuse}개  (기대: 2 - A + fused-B)")
assert n_fuse == 2, f"기대 2개, 실제 {n_fuse}개"

# ── 핵심 검증 ────────────────────────────────────────────────────────────────
assert n_fuse > n_no_fuse, "fuse 후 min_length가 no_fuse 보다 많아야 함 (순서 검증)"

# ── Case 4: advanced_fuse — 단축 방향 겹침은 융합, 장축 끝-끝은 융합 안 함 ─────
#
# 각도 90° (수직 침상):  long_axis = (cos90°, sin90°) = (0, 1)  (y 방향)
#                        short_axis = (-sin90°, cos90°) = (-1, 0) (x 방향)
#
# D1=(100,100), D2=(102,100): displacement (+2, 0)
#   d_long = |2*cos90° + 0*sin90°| = 0  → pure short-axis → 융합 대상
#
# E1=(100,100), E2=(100,104): displacement (0, +4)
#   d_long = |0*cos90° + 4*sin90°| = 4 = dist → long_fraction=1.0 → 끝-끝 → skip



obj_D1 = make_obj(0, cx=100, cy=100, length=40, width=10, angle_deg=90,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)
obj_D2 = make_obj(1, cx=102, cy=100, length=40, width=10, angle_deg=90,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)
mask_D1 = make_rod_mask(cx=100, cy=100, length=40, width=10, angle_deg=90)
mask_D2 = make_rod_mask(cx=102, cy=100, length=40, width=10, angle_deg=90)

obj_E1 = make_obj(2, cx=100, cy=100, length=40, width=10, angle_deg=90,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)
obj_E2 = make_obj(3, cx=100, cy=112, length=40, width=10, angle_deg=90,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)
mask_E1 = make_rod_mask(cx=100, cy=100, length=40, width=10, angle_deg=90)
mask_E2 = make_rod_mask(cx=100, cy=112, length=40, width=10, angle_deg=90)

print(f"\nD1/D2 mask overlap: {int((mask_D1 & mask_D2).sum())} px  "
      f"ratio={float((mask_D1 & mask_D2).sum())/min(mask_D1.sum(), mask_D2.sum()):.2f}")
print(f"E1/E2 mask overlap: {int((mask_E1 & mask_E2).sum())} px  "
      f"ratio={float((mask_E1 & mask_E2).sum())/min(mask_E1.sum(), mask_E2.sum()):.2f}")

# D1+D2: 단축 방향 나란히 (dx=2) → advanced_fuse는 합치지 않음 (끝-끝 아님)
adv_D, _ = fuse_contours(
    [obj_D1, obj_D2], [mask_D1, mask_D2],
    float_acicular_threshold=0.4,
    str_particle_type="acicular",
    float_scale_pixels=SCALE_PX,
    float_scale_um=SCALE_UM,
    bool_advanced=True,
)
print(f"[Case 4a] D1+D2 (단축 방향 나란히) advanced_fuse: 2 → {len(adv_D)}개  (기대 2)")
assert len(adv_D) == 2, f"단축 방향 나란히는 advanced_fuse에서 합치지 않아야 함, 실제 {len(adv_D)}개"

# E1+E2: 장축 방향 끝-끝 → advanced_fuse에서 합침
adv_E, _ = fuse_contours(
    [obj_E1, obj_E2], [mask_E1, mask_E2],
    float_acicular_threshold=0.4,
    str_particle_type="acicular",
    float_scale_pixels=SCALE_PX,
    float_scale_um=SCALE_UM,
    bool_advanced=True,
)
print(f"[Case 4b] E1+E2 (장축 끝-끝 체인) advanced_fuse: 2 → {len(adv_E)}개  (기대 1)")
assert len(adv_E) == 1, f"장축 끝-끝 체인은 advanced_fuse에서 합쳐야 함, 실제 {len(adv_E)}개"

# E1+E2: --fuse → 1개 (끝-끝도 융합)
naive_E, _ = fuse_contours(
    [obj_E1, obj_E2], [mask_E1, mask_E2],
    float_acicular_threshold=0.4,
    str_particle_type="acicular",
    float_scale_pixels=SCALE_PX,
    float_scale_um=SCALE_UM,
)
print(f"[Case 4c] E1+E2  --fuse: 2 → {len(naive_E)}개  (기대 1)")
assert len(naive_E) == 1, f"--fuse 는 끝-끝도 융합해야 함, 실제 {len(naive_E)}개"

# ── Case 5: containment filtering ────────────────────────────────────────────
# F_big: 큰 침상, F_small: F_big 안에 90%+ 포함된 작은 침상 (동일 각도, 동일 위치)
# → advanced_fuse 에서 F_small은 필터링됨 (1개 남음)
# → --fuse 에서는 합쳐서 1개 (union)

from utils.contour import CONST_FUSE_CONTAINMENT_THRESHOLD

obj_Fb = make_obj(0, cx=150, cy=150, length=60, width=12, angle_deg=90,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)
obj_Fs = make_obj(1, cx=150, cy=150, length=20, width=8,  angle_deg=90,
                  scale_px=SCALE_PX, scale_um=SCALE_UM)
mask_Fb = make_rod_mask(cx=150, cy=150, length=60, width=12, angle_deg=90)
mask_Fs = make_rod_mask(cx=150, cy=150, length=20, width=8,  angle_deg=90)

inter_Fbs = float((mask_Fb & mask_Fs).sum())
containment_Fs = inter_Fbs / max(float(mask_Fs.sum()), 1.0)
print(f"\nF_big area={mask_Fb.sum()}  F_small area={mask_Fs.sum()}")
print(f"containment of F_small inside F_big = {containment_Fs:.3f}  (기대 > {CONST_FUSE_CONTAINMENT_THRESHOLD})")
assert containment_Fs >= CONST_FUSE_CONTAINMENT_THRESHOLD, \
    f"테스트 설계 오류: F_small이 F_big에 충분히 포함되지 않음 ({containment_Fs:.3f})"

# advanced_fuse: F_small은 필터링 → 1개 (F_big만 남음)
adv_F, _ = fuse_contours(
    [obj_Fb, obj_Fs], [mask_Fb, mask_Fs],
    float_acicular_threshold=0.4,
    str_particle_type="acicular",
    float_scale_pixels=SCALE_PX,
    float_scale_um=SCALE_UM,
    bool_advanced=True,
)
print(f"[Case 5a] F_big+F_small advanced_fuse: 2 → {len(adv_F)}개  (기대 1, F_small 필터링)")
assert len(adv_F) == 1, f"containment 필터링으로 F_small이 제거돼야 함, 실제 {len(adv_F)}개"
assert adv_F[0].float_longAxisPx >= obj_Fb.float_longAxisPx * 0.9, "남은 마스크는 큰 침상이어야 함"

# --fuse: F_small도 합쳐서 1개 (union → 큰 침상과 거의 동일)
naive_F, _ = fuse_contours(
    [obj_Fb, obj_Fs], [mask_Fb, mask_Fs],
    float_acicular_threshold=0.4,
    str_particle_type="acicular",
    float_scale_pixels=SCALE_PX,
    float_scale_um=SCALE_UM,
)
print(f"[Case 5b] F_big+F_small --fuse: 2 → {len(naive_F)}개  (기대 1, union 융합)")
assert len(naive_F) == 1, f"--fuse 는 포함 관계도 융합해야 함, 실제 {len(naive_F)}개"

print("\n✓ 모든 assertions 통과: min_length / advanced_fuse 방향 / advanced_fuse containment 필터링")
