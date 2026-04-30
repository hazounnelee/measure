from __future__ import annotations
import argparse
import csv
import json
import sys
import typing as tp
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.schema import (
    PrimaryParticleConfig,
    PrimaryParticleMeasurement,
    PrimaryParticleResult,
)
from configs import get_analysis_preset
from utils.metrics import convert_pixels_to_micrometers, calculate_mean_from_optional_values, calculate_percentage, json_default
from utils.image import detect_sphere_roi, compute_center_roi, compute_adaptive_block_size, draw_label_no_overlap
from utils.lsd import detect_acicular_lsd
from utils.contour import fuse_contours
from utils.io import collect_input_groups, build_image_output_dir
from services.sam2_service import Sam2AspectRatioService


# =========================================================
# 1차 입자 분석 전용 상수
# =========================================================

# 침상(acicular) / 판상(plate) 분류 기준
CONST_ACICULAR_THRESHOLD: float = 0.40

# 유효 1차 입자 최소 면적 (미만이면 fragment)
CONST_PRIMARY_PARTICLE_AREA_THRESHOLD: float = 200.0

# 자동 중앙 crop 비율 (이미지 중앙의 이 비율 영역을 사용)
CONST_CENTER_CROP_RATIO: float = 0.60

# 침상+판상 입자 목표 수 (미달 시 경고)
CONST_TARGET_PARTICLE_COUNT: int = 10

# ---- 구형(sphere) 2차 입자 자동 검출 ----
CONST_SPHERE_CAP_FRACTION: float = 0.45
CONST_SPHERE_MORPH_KERNEL: int = 15
CONST_SPHERE_MIN_RADIUS_RATIO: float = 0.15

# 1차 입자용 SAM2 추론 파라미터 (더 촘촘한 타일/포인트)
CONST_PRIMARY_TILE_SIZE: int = 256
CONST_PRIMARY_TILE_STRIDE: int = 128
CONST_PRIMARY_POINTS_PER_TILE: int = 120
CONST_PRIMARY_POINT_MIN_DISTANCE: int = 8

# ---- 침상 hybrid mode (OpenCV → SAM2 box prompt) ----
CONST_ACICULAR_ADAPT_BLOCK_SIZE: int = 35
CONST_ACICULAR_ADAPT_C: int = 4
CONST_ACICULAR_CANDIDATE_MIN_AREA: float = 60.0
CONST_ACICULAR_CANDIDATE_MAX_AREA: float = 25000.0
CONST_ACICULAR_CANDIDATE_AR_SCREEN: float = 0.60
CONST_ACICULAR_BBOX_PAD_RATIO: float = 0.08
CONST_ACICULAR_BOX_PROMPT_BATCH: int = 16
CONST_ACICULAR_FALLBACK_THRESHOLD: int = 3

# Sam2AspectRatioConfig 기본값에서 가져온 공통 상수
CONST_SCALE_PIXELS: float = 147.0
CONST_SCALE_MICROMETERS: float = 1.0
CONST_BBOX_EDGE_MARGIN: int = 8
CONST_TILE_EDGE_MARGIN: int = 8
CONST_ROI_X_MIN: int = 0
CONST_ROI_Y_MIN: int = 0
CONST_ROI_X_MAX: int = 1024
CONST_ROI_Y_MAX: int = 768
CONST_MASK_BINARIZE_THRESHOLD: float = 0.0
CONST_MIN_VALID_MASK_AREA: int = 1
CONST_MASK_MORPH_KERNEL_SIZE: int = 0
CONST_MASK_MORPH_OPEN_ITERATIONS: int = 0
CONST_MASK_MORPH_CLOSE_ITERATIONS: int = 0
CONST_DEFAULT_IMAGE_SIZE: int = 1536
CONST_DEFAULT_RETINA_MASKS: bool = True
CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS: bool = True
CONST_DEFAULT_POINT_QUALITY_LEVEL: float = 0.03
CONST_DEFAULT_POINT_BATCH_SIZE: int = 32
CONST_DEFAULT_DEDUP_IOU: float = 0.60
CONST_DEFAULT_BBOX_DEDUP_IOU: float = 0.85
CONST_DEFAULT_USE_POINT_PROMPTS: bool = True

# iou 모듈 (predict_with_box_prompts, _merge_mask_results 등에서 사용)
from utils.iou import calculate_binary_iou, calculate_box_iou


# =========================================================
# Service
# =========================================================


class PrimaryParticleService(Sam2AspectRatioService):
    """1차 입자 segmentation, 측정, 저장 서비스."""

    def __init__(self, obj_config: PrimaryParticleConfig) -> None:
        super().__init__(obj_config)
        self.obj_primary_config: PrimaryParticleConfig = obj_config

    def validate_inputs(self) -> None:
        """LSD 모드에서는 모델 파일 불필요 → 이미지와 config만 확인한다."""
        if self.obj_primary_config.str_measureMode == "lsd":
            if not self.obj_config.path_input.exists():
                raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {self.obj_config.path_input}")
            return
        super().validate_inputs()

    # ----------------------------------------------------------
    # ROI: 구형 2차 입자 검출 + cap 영역 추출
    # ----------------------------------------------------------

    def detect_sphere_and_extract_cap(
        self,
        arr_imageBgr: np.ndarray,
    ) -> tp.Tuple[tp.Optional[tp.Tuple[int, int, int, int]], tp.Optional[np.ndarray]]:
        result = detect_sphere_roi(
            arr_imageBgr,
            float_cap_fraction=self.obj_primary_config.float_sphereCapFraction,
        )
        if result is None:
            return None, None
        return result[0], result[1]

    # ----------------------------------------------------------
    # ROI: 자동 중앙 crop
    # ----------------------------------------------------------

    def compute_center_roi(
        self, int_h: int, int_w: int
    ) -> tp.Tuple[int, int, int, int]:
        return compute_center_roi(int_h, int_w, self.obj_primary_config.float_centerCropRatio)

    def extract_inference_roi(
        self,
        arr_imageBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Dict[str, int]]:
        """ROI를 추출한다.

        우선순위:
        1. `bool_autoDetectSphere=True` (acicular 모드) → 구 검출 후 cap ROI
        2. `bool_autoCenterCrop=True` → 중앙 crop ROI
        3. 명시적 ROI 좌표 사용
        """
        int_h, int_w = arr_imageBgr.shape[:2]

        # 구 자동 검출 (acicular 모드에서만)
        if (
            self.obj_primary_config.str_particleMode == "acicular"
            and self.obj_primary_config.bool_autoDetectSphere
        ):
            tpl_cap, arr_sphereDbg = self.detect_sphere_and_extract_cap(arr_imageBgr)
            if tpl_cap is not None:
                int_x0, int_y0, int_x1, int_y1 = tpl_cap
                arr_roiBgr = arr_imageBgr[int_y0:int_y1, int_x0:int_x1].copy()
                dict_roi: tp.Dict[str, int] = {
                    "x_min": int_x0,
                    "y_min": int_y0,
                    "x_max": int_x1,
                    "y_max": int_y1,
                    "width": int_x1 - int_x0,
                    "height": int_y1 - int_y0,
                }
                # sphere debug 이미지를 인스턴스 변수로 보관 → save 단계에서 저장
                self._arr_sphereDebug: tp.Optional[np.ndarray] = arr_sphereDbg
                return arr_roiBgr, dict_roi
            else:
                print(
                    "[sphere-detect] 구 검출 실패 → center crop fallback",
                    flush=True,
                )

        self._arr_sphereDebug = None

        if self.obj_primary_config.bool_autoCenterCrop:
            int_x0, int_y0, int_x1, int_y1 = self.compute_center_roi(int_h, int_w)
        else:
            int_x0 = max(0, min(self.obj_config.int_roiXMin, int_w))
            int_y0 = max(0, min(self.obj_config.int_roiYMin, int_h))
            int_x1 = max(int_x0, min(self.obj_config.int_roiXMax, int_w))
            int_y1 = max(int_y0, min(self.obj_config.int_roiYMax, int_h))

        if int_x1 <= int_x0 or int_y1 <= int_y0:
            raise ValueError("유효한 ROI를 만들 수 없습니다. ROI 좌표와 이미지 크기를 확인하세요.")

        arr_roiBgr = arr_imageBgr[int_y0:int_y1, int_x0:int_x1].copy()
        dict_roi = {
            "x_min": int_x0,
            "y_min": int_y0,
            "x_max": int_x1,
            "y_max": int_y1,
            "width": int_x1 - int_x0,
            "height": int_y1 - int_y0,
        }
        return arr_roiBgr, dict_roi

    # ----------------------------------------------------------
    # 측정: minAreaRect 기반 두께 + 침상/판상 분류
    # ----------------------------------------------------------

    def measure_primary_mask(
        self,
        arr_mask: np.ndarray,
        int_index: int,
        float_confidence: tp.Optional[float],
    ) -> tp.Optional[PrimaryParticleMeasurement]:
        """단일 mask를 minAreaRect로 측정하고 침상/판상/fragment로 분류한다.

        Args:
            arr_mask: ROI 좌표계 기준 binary mask.
            int_index: 마스크 인덱스 (식별용).
            float_confidence: SAM2 confidence score. 없으면 None.

        Returns:
            유효하면 PrimaryParticleMeasurement, 아니면 None.
        """
        arr_refined = self.refine_mask_for_area(arr_mask)
        int_maskArea = int(arr_refined.sum())
        if int_maskArea < self.obj_config.int_minValidMaskArea:
            return None

        arr_contour = self.extract_largest_contour(arr_refined)
        if arr_contour is None:
            return None

        int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_contour)
        int_roiH, int_roiW = arr_refined.shape[:2]
        if self.is_bbox_near_roi_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
            return None

        # centroid
        obj_moments = cv2.moments(arr_contour)
        if obj_moments["m00"] > 0.0:
            float_cx = float(obj_moments["m10"] / obj_moments["m00"])
            float_cy = float(obj_moments["m01"] / obj_moments["m00"])
        else:
            float_cx = float(int_bx + int_bw / 2.0)
            float_cy = float(int_by + int_bh / 2.0)

        # minAreaRect: 회전 보정된 단/장축 측정
        # contour 점이 5개 이상일 때만 사용 가능
        if len(arr_contour) >= 5:
            tpl_rect = cv2.minAreaRect(arr_contour)
            (_, _), (float_rectW, float_rectH), float_rectAngle = tpl_rect
            float_thicknessPx = float(min(float_rectW, float_rectH))
            float_longAxisPx = float(max(float_rectW, float_rectH))
        else:
            float_thicknessPx = float(min(int_bw, int_bh))
            float_longAxisPx = float(max(int_bw, int_bh))
            float_rectAngle = 0.0

        # 장축이 0에 가까우면 skip
        if float_longAxisPx < 1.0:
            return None

        float_aspectRatio = float_thicknessPx / float_longAxisPx

        # H/V span (참조용)
        int_horizontal = min(
            self.get_longest_span(arr_refined, bool_horizontal=True), int_bw)
        int_vertical = min(
            self.get_longest_span(arr_refined, bool_horizontal=False), int_bh)

        # 분류: 면적 미달은 fragment, 그 외는 AR 기준으로 acicular/plate 분류
        if int_maskArea < int(round(self.obj_config.float_particleAreaThreshold)):
            str_category = "fragment"
        elif float_aspectRatio < self.obj_primary_config.float_acicularThreshold:
            str_category = "acicular"
        else:
            str_category = "plate"

        return PrimaryParticleMeasurement(
            int_index=int_index,
            str_category=str_category,
            int_maskArea=int_maskArea,
            float_confidence=float_confidence,
            int_bboxX=int_bx,
            int_bboxY=int_by,
            int_bboxWidth=int_bw,
            int_bboxHeight=int_bh,
            float_centroidX=float_cx,
            float_centroidY=float_cy,
            float_thicknessPx=float_thicknessPx,
            float_longAxisPx=float_longAxisPx,
            float_minRectAngle=float(float_rectAngle),
            float_thicknessUm=convert_pixels_to_micrometers(float_thicknessPx, self.obj_config.float_scalePixels, self.obj_config.float_scaleMicrometers),
            float_longAxisUm=convert_pixels_to_micrometers(float_longAxisPx, self.obj_config.float_scalePixels, self.obj_config.float_scaleMicrometers),
            float_aspectRatio=float_aspectRatio,
            int_longestHorizontal=int_horizontal,
            int_longestVertical=int_vertical,
            float_longestHorizontalUm=convert_pixels_to_micrometers(float(int_horizontal), self.obj_config.float_scalePixels, self.obj_config.float_scaleMicrometers),
            float_longestVerticalUm=convert_pixels_to_micrometers(float(int_vertical), self.obj_config.float_scalePixels, self.obj_config.float_scaleMicrometers),
        )

    # ----------------------------------------------------------
    # 침상 hybrid: OpenCV 후보 탐지
    # ----------------------------------------------------------

    def detect_acicular_candidates_opencv(
        self,
        arr_roiGray: np.ndarray,
        float_arScreen: float = CONST_ACICULAR_CANDIDATE_AR_SCREEN,
        float_minArea: float = CONST_ACICULAR_CANDIDATE_MIN_AREA,
        float_maxArea: float = CONST_ACICULAR_CANDIDATE_MAX_AREA,
        int_maxCandidates: int = 500,
        bool_arScreen: bool = False,
    ) -> tp.Tuple[tp.List[tp.Tuple[int, int, int, int]], np.ndarray]:
        """connectedComponents + moments 기반으로 침상 후보 bbox를 탐지한다.

        Adaptive threshold 후 erosion으로 인접 입자를 분리하고,
        connectedComponentsWithStats로 개별 blob을 추출한 뒤,
        image moments의 공분산 고유값으로 elongation을 계산한다
        (minAreaRect/fitEllipse보다 픽셀 분포를 직접 반영해 정확).

        Returns:
            (list_bboxes, arr_debugMask):
            - list_bboxes: 패딩이 적용된 (x1, y1, x2, y2) 후보 bbox 목록.
            - arr_debugMask: erosion 후 이진 이미지 (시각화·저장용).
        """
        int_roiH, int_roiW = arr_roiGray.shape[:2]

        # 1. 대비 정규화
        obj_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        arr_eq = obj_clahe.apply(arr_roiGray)
        arr_blur = cv2.GaussianBlur(arr_eq, (3, 3), 0)

        arr_thresh = cv2.adaptiveThreshold(
            arr_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            compute_adaptive_block_size(int_roiH, int_roiW, 25, CONST_ACICULAR_ADAPT_BLOCK_SIZE),
            CONST_ACICULAR_ADAPT_C,
        )
        if float(arr_thresh.sum()) / (255.0 * int_roiH * int_roiW) > 0.55:
            arr_thresh = cv2.bitwise_not(arr_thresh)

        # 3. Erosion — 서로 붙어있는 입자 경계를 끊어 개별 blob으로 분리
        arr_kernelE = np.ones((2, 2), np.uint8)
        arr_eroded = cv2.erode(arr_thresh, arr_kernelE, iterations=2)

        # 4. connectedComponentsWithStats → 개별 blob 추출
        int_numLabels, arr_labels, arr_stats, arr_centroids = cv2.connectedComponentsWithStats(
            arr_eroded, connectivity=8)

        # 5. 각 blob에 대해 면적 필터 + elongation 계산 (image moments 기반)
        list_bboxes: tp.List[tp.Tuple[int, int, int, int]] = []
        arr_debugMask = np.zeros_like(arr_eroded)

        for int_lbl in range(1, int_numLabels):  # 0 = 배경
            int_area = int(arr_stats[int_lbl, cv2.CC_STAT_AREA])
            if int_area < float_minArea or int_area > float_maxArea:
                continue

            int_bx = int(arr_stats[int_lbl, cv2.CC_STAT_LEFT])
            int_by = int(arr_stats[int_lbl, cv2.CC_STAT_TOP])
            int_bw = int(arr_stats[int_lbl, cv2.CC_STAT_WIDTH])
            int_bh = int(arr_stats[int_lbl, cv2.CC_STAT_HEIGHT])

            # elongation: image moments의 공분산 행렬 고유값 비율
            # lambda_min / lambda_max = (단축/장축)² ≈ AR²
            arr_compMask = (arr_labels == int_lbl).astype(np.uint8)
            dict_m = cv2.moments(arr_compMask)
            float_m00 = dict_m["m00"]
            if float_m00 < 1.0:
                continue

            float_mu20 = dict_m["mu20"] / float_m00
            float_mu02 = dict_m["mu02"] / float_m00
            float_mu11 = dict_m["mu11"] / float_m00
            float_disc = float(np.sqrt(max(0.0,
                (float_mu20 - float_mu02) ** 2 + 4.0 * float_mu11 ** 2)))
            float_l1 = (float_mu20 + float_mu02 + float_disc) / 2.0  # 장축 분산
            float_l2 = (float_mu20 + float_mu02 - float_disc) / 2.0  # 단축 분산

            # AR = sqrt(l2/l1): 1에 가까울수록 원형, 0에 가까울수록 길쭉
            float_ar = float(np.sqrt(max(0.0, float_l2) / max(1e-9, float_l1)))
            if bool_arScreen and float_ar >= float_arScreen:
                continue  # 충분히 길쭉하지 않음

            if self.is_bbox_near_roi_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
                continue

            # bbox 패딩
            int_padX = max(2, int(int_bw * CONST_ACICULAR_BBOX_PAD_RATIO))
            int_padY = max(2, int(int_bh * CONST_ACICULAR_BBOX_PAD_RATIO))
            int_x1 = max(0, int_bx - int_padX)
            int_y1 = max(0, int_by - int_padY)
            int_x2 = min(int_roiW, int_bx + int_bw + int_padX)
            int_y2 = min(int_roiH, int_by + int_bh + int_padY)
            list_bboxes.append((int_x1, int_y1, int_x2, int_y2))
            arr_debugMask[arr_labels == int_lbl] = 255

            if len(list_bboxes) >= int_maxCandidates:
                break

        return list_bboxes, arr_debugMask

    # ----------------------------------------------------------
    # 침상 hybrid: SAM2 box prompt 추론
    # ----------------------------------------------------------

    def predict_with_box_prompts(
        self,
        arr_roi: np.ndarray,
        list_boxes: tp.List[tp.Tuple[int, int, int, int]],
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any]]:
        """SAM2 bounding box prompt 방식으로 ROI 전체에 대해 mask를 추론한다.

        Args:
            arr_roi: ROI BGR 이미지.
            list_boxes: (x1, y1, x2, y2) 형식의 bbox 목록.

        Returns:
            (arr_masks, arr_scores, dict_debug) — predict_tiled_point_prompts와 동일 형식.
        """
        if self.obj_model is None:
            self.initialize_model()

        int_roiH, int_roiW = arr_roi.shape[:2]

        dict_predictCommon: tp.Dict[str, tp.Any] = {
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictCommon["device"] = self.obj_config.str_device

        list_keptMasks: tp.List[np.ndarray] = []
        list_keptScores: tp.List[tp.Optional[float]] = []
        list_keptBboxes: tp.List[tp.Tuple[int, int, int, int]] = []
        int_acceptedCount = 0
        int_bboxDedupRejected = 0

        int_bs = max(1, CONST_ACICULAR_BOX_PROMPT_BATCH)
        for int_start in range(0, len(list_boxes), int_bs):
            list_batch = list_boxes[int_start:int_start + int_bs]
            list_bboxForSam = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in list_batch]

            try:
                list_results = self.obj_model(  # type: ignore[misc]
                    source=arr_roi,
                    bboxes=list_bboxForSam,
                    **dict_predictCommon,
                )
            except Exception as exc:
                print(f"[WARN] box prompt batch 실패 (skip): {exc}", flush=True)
                continue

            if not list_results:
                continue

            obj_result = list_results[0]
            if obj_result.masks is None or obj_result.masks.data is None:
                continue

            arr_batchMasks = obj_result.masks.data.detach().cpu().numpy()
            arr_batchScores = None
            if obj_result.boxes is not None and obj_result.boxes.conf is not None:
                arr_batchScores = obj_result.boxes.conf.detach().cpu().numpy()

            for int_mi, arr_rawMask in enumerate(arr_batchMasks):
                arr_mask = (
                    arr_rawMask > self.obj_config.float_maskBinarizeThreshold
                ).astype(np.uint8)
                if int(arr_mask.sum()) < self.obj_config.int_minValidMaskArea:
                    continue

                arr_contour = self.extract_largest_contour(arr_mask)
                if arr_contour is None:
                    continue

                int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_contour)
                if self.is_bbox_near_roi_edge(
                        int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
                    continue

                tuple_box = (int_bx, int_by, int_bw, int_bh)
                bool_bboxDup = any(
                    calculate_box_iou(tuple_box, prev) >= self.obj_config.float_bboxDedupIou
                    for prev in list_keptBboxes
                )
                if bool_bboxDup:
                    int_bboxDedupRejected += 1
                    continue

                bool_maskDup = any(
                    calculate_binary_iou(arr_mask, prev) >= self.obj_config.float_dedupIou
                    for prev in list_keptMasks
                )
                if bool_maskDup:
                    continue

                int_acceptedCount += 1
                list_keptMasks.append(arr_mask)
                list_keptBboxes.append(tuple_box)
                float_score: tp.Optional[float] = None
                if arr_batchScores is not None and int_mi < len(arr_batchScores):
                    float_score = float(arr_batchScores[int_mi])
                list_keptScores.append(float_score)

        arr_masks = (
            np.stack(list_keptMasks, axis=0).astype(np.uint8)
            if list_keptMasks
            else np.empty((0, int_roiH, int_roiW), dtype=np.uint8)
        )
        arr_scores: tp.Optional[np.ndarray] = None
        if list_keptScores:
            arr_scores = np.array(
                [np.nan if x is None else float(x) for x in list_keptScores],
                dtype=np.float32,
            )
        dict_debug: tp.Dict[str, tp.Any] = {
            "num_tiles": 1,
            "num_candidate_boxes": len(list_boxes),
            "num_candidate_points": 0,
            "num_accepted_masks": int_acceptedCount,
            "num_bbox_dedup_rejected": int_bboxDedupRejected,
        }
        return arr_masks, arr_scores, dict_debug

    def _merge_mask_results(
        self,
        arr_masksA: np.ndarray,
        arr_scoresA: tp.Optional[np.ndarray],
        arr_masksB: np.ndarray,
        arr_scoresB: tp.Optional[np.ndarray],
        int_roiH: int,
        int_roiW: int,
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray]]:
        """두 mask 배열을 IoU dedup하며 병합한다 (A 우선)."""
        list_masks: tp.List[np.ndarray] = list(arr_masksA) if arr_masksA.shape[0] > 0 else []
        list_scores: tp.List[tp.Optional[float]] = (
            [float(x) if not np.isnan(x) else None for x in arr_scoresA]
            if arr_scoresA is not None else [None] * len(list_masks)
        )

        for int_i, arr_m in enumerate(arr_masksB):
            if any(
                calculate_binary_iou(arr_m, prev) >= self.obj_config.float_dedupIou
                for prev in list_masks
            ):
                continue
            list_masks.append(arr_m)
            float_sb: tp.Optional[float] = None
            if arr_scoresB is not None and int_i < len(arr_scoresB):
                float_sb = float(arr_scoresB[int_i]) if not np.isnan(arr_scoresB[int_i]) else None
            list_scores.append(float_sb)

        arr_merged = (
            np.stack(list_masks, axis=0).astype(np.uint8)
            if list_masks
            else np.empty((0, int_roiH, int_roiW), dtype=np.uint8)
        )
        arr_mergedScores: tp.Optional[np.ndarray] = None
        if list_scores:
            arr_mergedScores = np.array(
                [np.nan if x is None else float(x) for x in list_scores],
                dtype=np.float32,
            )
        return arr_merged, arr_mergedScores

    # ----------------------------------------------------------
    # 침상 hybrid: 메인 파이프라인
    # ----------------------------------------------------------

    def _extract_blob_centroids(
        self,
        list_bboxes: tp.List[tp.Tuple[int, int, int, int]],
    ) -> tp.List[tp.Tuple[int, int]]:
        """bbox 목록에서 centroid 좌표 목록을 계산한다."""
        return [
            (int_x1 + (int_x2 - int_x1) // 2, int_y1 + (int_y2 - int_y1) // 2)
            for (int_x1, int_y1, int_x2, int_y2) in list_bboxes
        ]

    def predict_with_guided_point_prompts(
        self,
        arr_roi: np.ndarray,
        list_points: tp.List[tp.Tuple[int, int]],
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any]]:
        """OpenCV 검출 centroid를 SAM2 point prompt로 전달한다.

        box prompt는 작은 입자에서 SAM2가 박스 전체를 채우는 과분할을 일으키지만,
        point prompt는 입자 중심만 알려주어 SAM2가 자연스럽게 경계를 찾는다.
        """
        if self.obj_model is None:
            self.initialize_model()

        int_roiH, int_roiW = arr_roi.shape[:2]

        dict_predictCommon: tp.Dict[str, tp.Any] = {
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictCommon["device"] = self.obj_config.str_device

        list_keptMasks: tp.List[np.ndarray] = []
        list_keptScores: tp.List[tp.Optional[float]] = []
        list_keptBboxes: tp.List[tp.Tuple[int, int, int, int]] = []
        int_acceptedCount = 0
        int_bboxDedupRejected = 0

        int_bs = max(1, self.obj_config.int_pointBatchSize)
        for int_start in range(0, len(list_points), int_bs):
            list_chunk = list_points[int_start:int_start + int_bs]
            try:
                list_results = self.obj_model(  # type: ignore[misc]
                    source=arr_roi,
                    points=[[int_px, int_py] for (int_px, int_py) in list_chunk],
                    labels=[1] * len(list_chunk),
                    **dict_predictCommon,
                )
            except Exception as exc:
                print(f"[WARN] guided point prompt batch 실패: {exc}", flush=True)
                continue

            if not list_results:
                continue
            obj_result = list_results[0]
            if obj_result.masks is None or obj_result.masks.data is None:
                continue

            arr_batchMasks = obj_result.masks.data.detach().cpu().numpy()
            arr_batchScores = None
            if obj_result.boxes is not None and obj_result.boxes.conf is not None:
                arr_batchScores = obj_result.boxes.conf.detach().cpu().numpy()

            for int_mi, arr_rawMask in enumerate(arr_batchMasks):
                arr_mask = (arr_rawMask > self.obj_config.float_maskBinarizeThreshold).astype(np.uint8)
                if int(arr_mask.sum()) < self.obj_config.int_minValidMaskArea:
                    continue
                arr_contour = self.extract_largest_contour(arr_mask)
                if arr_contour is None:
                    continue
                int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_contour)
                if self.is_bbox_near_roi_edge(int_bx, int_by, int_bw, int_bh, int_roiW, int_roiH):
                    continue
                tuple_box = (int_bx, int_by, int_bw, int_bh)
                if any(calculate_box_iou(tuple_box, prev) >= self.obj_config.float_bboxDedupIou
                       for prev in list_keptBboxes):
                    int_bboxDedupRejected += 1
                    continue
                if any(calculate_binary_iou(arr_mask, prev) >= self.obj_config.float_dedupIou
                       for prev in list_keptMasks):
                    continue
                int_acceptedCount += 1
                list_keptMasks.append(arr_mask)
                list_keptBboxes.append(tuple_box)
                float_sc: tp.Optional[float] = None
                if arr_batchScores is not None and int_mi < len(arr_batchScores):
                    float_sc = float(arr_batchScores[int_mi])
                list_keptScores.append(float_sc)

        arr_masks = (
            np.stack(list_keptMasks, axis=0).astype(np.uint8)
            if list_keptMasks else np.empty((0, int_roiH, int_roiW), dtype=np.uint8)
        )
        arr_scores: tp.Optional[np.ndarray] = None
        if list_keptScores:
            arr_scores = np.array(
                [np.nan if x is None else float(x) for x in list_keptScores],
                dtype=np.float32,
            )
        return arr_masks, arr_scores, {
            "num_tiles": 1,
            "num_guided_points": len(list_points),
            "num_candidate_points": len(list_points),
            "num_accepted_masks": int_acceptedCount,
            "num_bbox_dedup_rejected": int_bboxDedupRejected,
        }

    def process_acicular_hybrid(
        self,
        arr_inputRoiBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any],
                  tp.Optional[np.ndarray]]:
        """OpenCV centroid → SAM2 guided point prompt hybrid 파이프라인.

        OpenCV로 침상 후보 blob의 centroid를 추출하고,
        이를 SAM2 point prompt로 전달해 정밀 mask를 얻는다.
        box prompt 대신 point prompt를 쓰는 이유: 작은 입자에서 box는
        SAM2가 박스 전체를 채우는 과분할을 일으키지만 point는 그렇지 않다.

        Returns:
            (arr_masks, arr_scores, dict_debug, arr_opencvDebugMask)
        """
        arr_roiGray = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
        int_roiH, int_roiW = arr_inputRoiBgr.shape[:2]

        float_arScreen = min(
            CONST_ACICULAR_CANDIDATE_AR_SCREEN,
            self.obj_primary_config.float_acicularThreshold + 0.25,
        )
        list_bboxes, arr_opencvMask = self.detect_acicular_candidates_opencv(
            arr_roiGray,
            float_arScreen=float_arScreen,
            float_minArea=max(8.0, self.obj_config.float_particleAreaThreshold * 0.5),
            bool_arScreen=self.obj_primary_config.bool_arScreen,
        )

        print(
            f"[acicular-hybrid] OpenCV 후보 침상: {len(list_bboxes)}개", flush=True)

        if len(list_bboxes) >= CONST_ACICULAR_FALLBACK_THRESHOLD:
            # OpenCV centroid → SAM2 guided point prompts
            list_centroids = self._extract_blob_centroids(list_bboxes)
            arr_masks, arr_scores, dict_debug = self.predict_with_guided_point_prompts(
                arr_inputRoiBgr, list_centroids)
            dict_debug["opencv_candidates"] = len(list_bboxes)
            dict_debug["mode"] = "opencv_guided_points"

            # 결과 부족 시 tiled point prompt로 보완
            if len(arr_masks) < CONST_ACICULAR_FALLBACK_THRESHOLD:
                print("[acicular-hybrid] guided point 결과 부족 → tiled point 보완",
                      flush=True)
                arr_ptMasks, arr_ptScores, dict_ptDebug = self.predict_tiled_point_prompts(
                    arr_inputRoiBgr)
                arr_masks, arr_scores = self._merge_mask_results(
                    arr_masks, arr_scores, arr_ptMasks, arr_ptScores,
                    int_roiH=int_roiH, int_roiW=int_roiW,
                )
                dict_debug["fallback_point_masks"] = int(arr_ptMasks.shape[0])
                dict_debug["mode"] = "opencv_guided+tiled_fallback"
        else:
            print("[acicular-hybrid] OpenCV 후보 부족 → tiled point fallback",
                  flush=True)
            arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(
                arr_inputRoiBgr)
            dict_debug["opencv_candidates"] = len(list_bboxes)
            dict_debug["mode"] = "tiled_point_fallback"
            arr_opencvMask = None

        return arr_masks, arr_scores, dict_debug, arr_opencvMask

    # ----------------------------------------------------------
    # 시각화
    # ----------------------------------------------------------

    def create_primary_overlay(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[PrimaryParticleMeasurement],
        list_masks: tp.List[np.ndarray],
        float_density: tp.Optional[float] = None,
    ) -> np.ndarray:
        """침상(파랑)/판상(초록)/fragment(주황) 색으로 overlay를 생성한다."""
        # BGR 색상 (mask fill / contour+label)
        dict_fill = {
            "acicular": (230, 80,  80),
            "plate":    (60,  220, 60),
            "fragment": (0,   165, 255),
        }
        dict_edge = {
            "acicular": (180, 40,  40),
            "plate":    (0,   180, 0),
            "fragment": (0,   120, 200),
        }

        int_h, int_w = arr_imageBgr.shape[:2]
        arr_overlay = cv2.resize(arr_imageBgr, (int_w * 2, int_h * 2), interpolation=cv2.INTER_LINEAR)

        for obj_m, arr_mask in zip(list_objects, list_masks):
            arr_mask2 = cv2.resize(arr_mask, (int_w * 2, int_h * 2), interpolation=cv2.INTER_NEAREST)
            tpl_c = dict_fill.get(obj_m.str_category, (128, 128, 128))
            arr_overlay[arr_mask2 > 0] = (
                arr_overlay[arr_mask2 > 0].astype(np.float32) * 0.5
                + np.array(tpl_c, dtype=np.float32) * 0.5
            ).astype(np.uint8)

        list_placedRects: tp.List[tp.Tuple[int, int, int, int]] = []
        for obj_m, arr_mask in zip(list_objects, list_masks):
            arr_contour = self.extract_largest_contour(arr_mask)
            if arr_contour is None:
                continue
            arr_contour2 = (arr_contour * 2).astype(np.int32)
            tpl_ec = dict_edge.get(obj_m.str_category, (128, 128, 128))
            cv2.drawContours(arr_overlay, [arr_contour2], -1, tpl_ec, 1)

            if obj_m.str_category in ("acicular", "plate"):
                int_cx2 = int(round(obj_m.float_centroidX * 2))
                int_cy2 = int(round(obj_m.float_centroidY * 2))
                draw_label_no_overlap(
                    arr_overlay,
                    [f"{obj_m.float_thicknessUm:.2f}um"],
                    int_cx2, int_cy2,
                    tpl_ec,
                    list_placedRects,
                )

        # --- 하단 통계 바 ---
        list_t = [o.float_thicknessUm for o in list_objects if o.str_category in ("acicular", "plate")]
        int_count = len(list_t)
        str_stats = (
            f"N={int_count}"
            f"  mean={np.mean(list_t):.3f}um" if list_t else f"N={int_count}"
        )
        if list_t:
            str_stats = (
                f"N={int_count}"
                f"  mean={np.mean(list_t):.3f}um"
                f"  median={float(np.median(list_t)):.3f}um"
                f"  std={float(np.std(list_t)):.3f}um"
            )
        if float_density is not None:
            str_stats += f"  density={float_density:.3f}"

        int_ow = arr_overlay.shape[1]
        float_scale = int_ow / 1800.0
        int_font = cv2.FONT_HERSHEY_SIMPLEX
        int_thickness_txt = max(1, int(round(float_scale * 1.5)))
        (int_tw, int_th), int_bl = cv2.getTextSize(str_stats, int_font, float_scale, int_thickness_txt)
        int_bar_h = int_th + int_bl + int(16 * float_scale)
        arr_bar = np.zeros((int_bar_h, int_ow, 3), dtype=np.uint8)
        int_ty = int_th + int(8 * float_scale)
        cv2.putText(arr_bar, str_stats, (int(8 * float_scale), int_ty),
                    int_font, float_scale, (220, 220, 220), int_thickness_txt, cv2.LINE_AA)
        arr_overlay = np.vstack([arr_overlay, arr_bar])

        return arr_overlay

    # ----------------------------------------------------------
    # 통계
    # ----------------------------------------------------------

    def build_primary_summary(
        self,
        list_objects: tp.List[PrimaryParticleMeasurement],
    ) -> tp.Dict[str, tp.Any]:
        """대상 입자 유형의 두께·종횡비 통계를 포함한 summary dict 를 생성한다."""

        str_type = self.obj_primary_config.str_particleType

        def _stats(
            list_vals: tp.List[float],
        ) -> tp.Dict[str, tp.Optional[float]]:
            if not list_vals:
                return {"mean": None, "median": None, "std": None, "min": None, "max": None}
            arr = np.array(list_vals, dtype=np.float32)
            return {
                "mean":   float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std":    float(np.std(arr)),
                "min":    float(np.min(arr)),
                "max":    float(np.max(arr)),
            }

        float_um_per_px = convert_pixels_to_micrometers(
            1.0,
            float_scalePixels=self.obj_config.float_scalePixels,
            float_scaleMicrometers=self.obj_config.float_scaleMicrometers,
        )

        list_thicknessUm = [o.float_thicknessUm for o in list_objects]
        list_longAxisUm  = [o.float_longAxisUm  for o in list_objects]
        list_aspectRatio = [o.float_aspectRatio  for o in list_objects]

        return {
            "input_path": str(self.obj_config.path_input),
            "output_dir": str(self.obj_config.path_outputDir),
            "model_weights_path": str(self.obj_config.path_modelWeights),
            "particle_type": str_type,
            "magnification": str(self.obj_primary_config.str_magnification),
            "scale_pixels": float(self.obj_config.float_scalePixels),
            "scale_micrometers": float(self.obj_config.float_scaleMicrometers),
            "micrometers_per_pixel": float(float_um_per_px),
            "acicular_threshold": float(self.obj_primary_config.float_acicularThreshold),
            "particle_mode": str(self.obj_primary_config.str_particleMode),
            "auto_center_crop": bool(self.obj_primary_config.bool_autoCenterCrop),
            "center_crop_ratio": float(self.obj_primary_config.float_centerCropRatio),
            "particle_area_threshold": float(self.obj_config.float_particleAreaThreshold),
            "num_total_objects": len(list_objects),
            f"num_{str_type}": len(list_objects),
            f"{str_type}_thickness_um": _stats(list_thicknessUm),
            f"{str_type}_long_axis_um": _stats(list_longAxisUm),
            f"{str_type}_aspect_ratio": _stats(list_aspectRatio),
            "all_primary_thickness_um": _stats(list_thicknessUm),
        }

    # ----------------------------------------------------------
    # 저장
    # ----------------------------------------------------------

    def save_thickness_histogram(
        self,
        list_objects: tp.List[PrimaryParticleMeasurement],
        path_output: Path,
    ) -> None:
        """침상/판상별 두께 분포 histogram 을 PNG로 저장한다."""
        list_ac = [o.float_thicknessUm for o in list_objects if o.str_category == "acicular"]
        list_pl = [o.float_thicknessUm for o in list_objects if o.str_category == "plate"]

        str_lot = self.obj_config.path_input.resolve().parent.name or "UnknownLot"
        obj_fig, obj_ax = plt.subplots(figsize=(10, 6), dpi=100)
        try:
            obj_ax.set_title(f"{str_lot} — Primary Particle Thickness", fontsize=18)
            obj_ax.set_xlabel("Thickness (µm)", fontsize=14)
            obj_ax.set_ylabel("Count", fontsize=14)
            obj_ax.tick_params(labelsize=12)

            bool_hasData = False
            for list_vals, str_label, str_color in [
                (list_ac, "Acicular", "#5588ff"),
                (list_pl, "Plate",    "#44cc44"),
            ]:
                if list_vals:
                    bool_hasData = True
                    arr_v = np.array(list_vals, dtype=np.float32)
                    int_bins = int(np.clip(np.sqrt(len(arr_v)), 5, 20))
                    float_mean = float(np.mean(arr_v))
                    obj_ax.hist(
                        arr_v, bins=int_bins, alpha=0.65,
                        label=str_label, color=str_color,
                        edgecolor="#333333", linewidth=0.8,
                    )
                    obj_ax.axvline(float_mean, linestyle="--", linewidth=1.5, color=str_color)
                    float_ymax = obj_ax.get_ylim()[1]
                    obj_ax.text(
                        float_mean, float_ymax * 0.95,
                        f"  {str_label[:2]} mean: {float_mean:.3f} µm",
                        color=str_color, fontsize=11, va="top",
                    )

            if bool_hasData:
                obj_ax.legend(fontsize=12)
                obj_ax.grid(axis="y", linestyle="--", alpha=0.3)
            else:
                obj_ax.text(
                    0.5, 0.5, "No primary particle data",
                    ha="center", va="center",
                    transform=obj_ax.transAxes, fontsize=13, color="#666666",
                )

            obj_fig.tight_layout()
            obj_fig.savefig(str(path_output), bbox_inches="tight")
        finally:
            plt.close(obj_fig)

    def save_primary_outputs(
        self,
        arr_inputBgr: np.ndarray,
        arr_inputRoiBgr: np.ndarray,
        arr_overlayRoi: np.ndarray,
        list_objects: tp.List[PrimaryParticleMeasurement],
        list_masks: tp.List[np.ndarray],
        dict_summary: tp.Dict[str, tp.Any],
        dict_roi: tp.Dict[str, int],
        dict_debug: tp.Dict[str, tp.Any],
        arr_opencvDebugMask: tp.Optional[np.ndarray] = None,
        dict_lsdSteps: tp.Optional[tp.Dict[str, np.ndarray]] = None,
    ) -> None:
        """이미지, CSV, JSON, histogram 등 1차 입자 분석 산출물을 저장한다."""
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)

        # overlay_full: 원본 이미지 위에 ROI overlay를 덮는다
        arr_overlayFull = arr_inputBgr.copy()
        int_roiW = dict_roi["x_max"] - dict_roi["x_min"]
        int_roiH = dict_roi["y_max"] - dict_roi["y_min"]
        arr_overlayRoiSmall = cv2.resize(arr_overlayRoi, (int_roiW, int_roiH), interpolation=cv2.INTER_LINEAR)
        arr_overlayFull[
            dict_roi["y_min"]:dict_roi["y_max"],
            dict_roi["x_min"]:dict_roi["x_max"],
        ] = arr_overlayRoiSmall
        cv2.rectangle(
            arr_overlayFull,
            (dict_roi["x_min"], dict_roi["y_min"]),
            (dict_roi["x_max"], dict_roi["y_max"]),
            (255, 255, 0), 2,
        )

        cv2.imwrite(str(self.obj_config.path_outputDir / "01_input.png"), arr_inputBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir / "02_input_roi.png"), arr_inputRoiBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir / "03_overlay_roi.png"), arr_overlayRoi)
        cv2.imwrite(str(self.obj_config.path_outputDir / "04_overlay_full.png"), arr_overlayFull)

        # OpenCV/LSD debug image
        if arr_opencvDebugMask is not None:
            cv2.imwrite(
                str(self.obj_config.path_outputDir / "05_opencv_candidates.png"),
                arr_opencvDebugMask,
            )

        # LSD step-by-step images (lsd_01..05)
        if dict_lsdSteps:
            for str_name, arr_step in dict_lsdSteps.items():
                cv2.imwrite(
                    str(self.obj_config.path_outputDir / f"{str_name}.png"),
                    arr_step,
                )

        # Sphere detection debug (자동 구 검출 모드에서만 생성)
        arr_sphereDbg: tp.Optional[np.ndarray] = getattr(self, "_arr_sphereDebug", None)
        if arr_sphereDbg is not None:
            cv2.imwrite(
                str(self.obj_config.path_outputDir / "06_sphere_detection.png"),
                arr_sphereDbg,
            )

        # objects.csv (전체)
        path_csvAll = self.obj_config.path_outputDir / "objects.csv"
        with path_csvAll.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_objects:
                obj_writer = csv.DictWriter(
                    obj_f, fieldnames=list(asdict(list_objects[0]).keys()))
                obj_writer.writeheader()
                for obj_m in list_objects:
                    obj_writer.writerow(asdict(obj_m))

        # target-type CSV (acicular.csv or plate.csv, never both)
        str_type = self.obj_primary_config.str_particleType
        if str_type in ("acicular", "plate"):
            list_rows = [asdict(o) for o in list_objects if o.str_category == str_type]
            path_csv_type = self.obj_config.path_outputDir / f"{str_type}.csv"
            with path_csv_type.open("w", newline="", encoding="utf-8-sig") as obj_f:
                if list_rows:
                    obj_writer = csv.DictWriter(
                        obj_f, fieldnames=list(list_rows[0].keys()))
                    obj_writer.writeheader()
                    for dict_row in list_rows:
                        obj_writer.writerow(dict_row)

        # thickness 분포 histogram
        self.save_thickness_histogram(
            list_objects,
            self.obj_config.path_outputDir / "thickness_dist.png",
        )

        # JSON
        with (self.obj_config.path_outputDir / "summary.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2,
                      default=json_default)
        with (self.obj_config.path_outputDir / "objects.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(
                [asdict(o) for o in list_objects], obj_f, ensure_ascii=False, indent=2,
                default=json_default)
        with (self.obj_config.path_outputDir / "debug.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(dict_debug, obj_f, ensure_ascii=False, indent=2,
                      default=json_default)

        if not self.obj_config.bool_saveIndividualMasks:
            return

        # 개별 mask png (acicular_masks / plate_masks / fragment_masks)
        for str_cat in ("acicular", "plate", "fragment"):
            (self.obj_config.path_outputDir / f"{str_cat}_masks").mkdir(
                parents=True, exist_ok=True)

        for obj_m, arr_mask in zip(list_objects, list_masks):
            str_fname = f"{obj_m.str_category}_{obj_m.int_index:04d}.png"
            path_maskDir = self.obj_config.path_outputDir / f"{obj_m.str_category}_masks"
            cv2.imwrite(str(path_maskDir / str_fname), arr_mask.astype(np.uint8) * 255)

    # ----------------------------------------------------------
    # 메인 파이프라인
    # ----------------------------------------------------------

    def process_primary(self) -> PrimaryParticleResult:
        """단일 이미지에 대한 1차 입자 분석 파이프라인을 실행한다."""
        arr_inputBgr = self.load_image_bgr()
        arr_inputRoiBgr, dict_roi = self.extract_inference_roi(arr_inputBgr)

        arr_opencvDebugMask: tp.Optional[np.ndarray] = None
        dict_debug: tp.Dict[str, tp.Any] = {}

        # ── LSD 직접 측정 모드 (SAM2 불필요) ────────────────────
        if self.obj_primary_config.str_measureMode == "lsd":
            arr_roiGray = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
            list_objects, list_validMasks, arr_opencvDebugMask, dict_lsdSteps, float_roiDensity = detect_acicular_lsd(
                arr_roiGray,
                arr_inputRoiBgr,
                float_acicular_threshold=self.obj_primary_config.float_acicularThreshold,
                str_particle_type=self.obj_primary_config.str_particleType,
                float_scale_pixels=self.obj_config.float_scalePixels,
                float_scale_um=self.obj_config.float_scaleMicrometers,
                int_edge_margin=self.obj_config.int_bboxEdgeMargin,
                float_area_threshold=self.obj_config.float_particleAreaThreshold,
                bool_adaptive_thresh=self.obj_primary_config.bool_lsdAdaptiveThresh,
                int_min_length_px=self.obj_primary_config.int_lsdMinLengthPx,
            )

            if self.obj_primary_config.bool_fuseContours and list_objects:
                int_before = len(list_objects)
                list_objects, list_validMasks = fuse_contours(
                    list_objects, list_validMasks,
                    float_acicular_threshold=self.obj_primary_config.float_acicularThreshold,
                    str_particle_type=self.obj_primary_config.str_particleType,
                    float_scale_pixels=self.obj_config.float_scalePixels,
                    float_scale_um=self.obj_config.float_scaleMicrometers,
                )
                print(f"[fuse] {int_before}개 → {len(list_objects)}개", flush=True)

            int_primaryCount = sum(
                1 for o in list_objects if o.str_category in ("acicular", "plate"))
            if int_primaryCount < self.obj_primary_config.int_targetParticleCount:
                print(
                    f"[WARNING] 침상+판상 입자 수({int_primaryCount})가 목표치"
                    f"({self.obj_primary_config.int_targetParticleCount}) 미만입니다.",
                    flush=True,
                )

            arr_overlay = self.create_primary_overlay(arr_inputRoiBgr, list_objects, list_validMasks, float_density=float_roiDensity)
            dict_summary = self.build_primary_summary(list_objects)
            dict_summary["roi"] = dict_roi
            dict_summary["measure_mode"] = "lsd"
            dict_summary["particle_mode"] = self.obj_primary_config.str_particleMode
            dict_summary["roi_density"] = round(float_roiDensity, 4)

            self.save_primary_outputs(
                arr_inputBgr, arr_inputRoiBgr, arr_overlay,
                list_objects, list_validMasks,
                dict_summary, dict_roi, dict_debug,
                arr_opencvDebugMask=arr_opencvDebugMask,
                dict_lsdSteps=dict_lsdSteps,
            )
            return PrimaryParticleResult(list_objects=list_objects, dict_summary=dict_summary)

        # ── SAM2 모드 (기존 파이프라인) ─────────────────────────
        if self.obj_primary_config.str_particleMode == "acicular":
            arr_masks, arr_scores, dict_debug, arr_opencvDebugMask = (
                self.process_acicular_hybrid(arr_inputRoiBgr))
            dict_debug["particle_mode"] = "acicular"
        else:
            arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(arr_inputRoiBgr)
            dict_debug["particle_mode"] = "auto"

        list_objects = []
        list_validMasks = []

        for int_index, arr_mask in enumerate(arr_masks):
            float_conf: tp.Optional[float] = None
            if arr_scores is not None and int_index < len(arr_scores):
                float_conf = float(arr_scores[int_index])

            obj_m = self.measure_primary_mask(arr_mask, int_index, float_conf)
            if obj_m is None:
                continue
            str_target = self.obj_primary_config.str_particleType
            if obj_m.str_category == "fragment":
                continue
            if str_target in ("acicular", "plate") and obj_m.str_category != str_target:
                continue
            list_objects.append(obj_m)
            list_validMasks.append(self.refine_mask_for_area(arr_mask).astype(np.uint8))

        int_primaryCount = sum(
            1 for o in list_objects if o.str_category in ("acicular", "plate"))
        if int_primaryCount < self.obj_primary_config.int_targetParticleCount:
            print(
                f"[WARNING] 침상+판상 입자 수({int_primaryCount})가 목표치"
                f"({self.obj_primary_config.int_targetParticleCount}) 미만입니다. "
                "ROI 조정 또는 파라미터 변경을 고려하세요.",
                flush=True,
            )

        arr_overlay = self.create_primary_overlay(arr_inputRoiBgr, list_objects, list_validMasks)
        dict_summary = self.build_primary_summary(list_objects)
        arr_roiGray = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
        _, arr_roiBinary = cv2.threshold(arr_roiGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        dict_summary["roi"] = dict_roi
        dict_summary["measure_mode"] = "sam2"
        dict_summary["particle_mode"] = self.obj_primary_config.str_particleMode
        dict_summary["roi_density"] = round(float((arr_roiBinary > 0).sum()) / arr_roiBinary.size, 4)
        dict_summary["num_tiles"] = dict_debug.get("num_tiles")
        dict_summary["num_candidate_points"] = dict_debug.get("num_candidate_points")
        dict_summary["num_accepted_masks"] = dict_debug.get("num_accepted_masks")
        dict_summary["num_bbox_dedup_rejected"] = dict_debug.get("num_bbox_dedup_rejected")
        if "opencv_candidates" in dict_debug:
            dict_summary["opencv_candidates"] = dict_debug["opencv_candidates"]

        self.save_primary_outputs(
            arr_inputBgr, arr_inputRoiBgr, arr_overlay,
            list_objects, list_validMasks,
            dict_summary, dict_roi, dict_debug,
            arr_opencvDebugMask=arr_opencvDebugMask,
        )

        return PrimaryParticleResult(list_objects=list_objects, dict_summary=dict_summary)

    def _mean_stat(self, str_key: str, str_stat: str, list_fileSummaries: tp.List[tp.Dict[str, tp.Any]]) -> tp.Optional[float]:
        list_vals = [
            d[str_key][str_stat]
            for d in list_fileSummaries
            if isinstance(d.get(str_key), dict)
            and d[str_key].get(str_stat) is not None
        ]
        return calculate_mean_from_optional_values(list_vals)

    @classmethod
    def _create_config(
        cls,
        str_groupId: str,
        path_image: Path,
        path_outputRoot: Path,
        bool_isBatch: bool,
        str_modelConfig: str,
        str_modelWeights: str,
        int_roiXMin: int,
        int_roiYMin: int,
        int_roiXMax: int,
        int_roiYMax: int,
        int_bboxEdgeMargin: int,
        int_tileEdgeMargin: int,
        float_particleAreaThreshold: float,
        float_maskBinarizeThreshold: float,
        int_minValidMaskArea: int,
        int_maskMorphKernelSize: int,
        int_maskMorphOpenIterations: int,
        int_maskMorphCloseIterations: int,
        int_imgSize: int,
        int_tileSize: int,
        int_stride: int,
        int_pointsPerTile: int,
        int_pointMinDistance: int,
        float_pointQualityLevel: float,
        int_pointBatchSize: int,
        float_dedupIou: float,
        float_bboxDedupIou: float,
        bool_usePointPrompts: bool,
        float_scalePixels: float,
        float_scaleMicrometers: float,
        str_device: tp.Optional[str],
        bool_retinaMasks: bool,
        bool_saveIndividualMasks: bool,
        float_acicularThreshold: float,
        bool_autoCenterCrop: bool,
        float_centerCropRatio: float,
        int_targetParticleCount: int,
        str_particleMode: str,
        bool_autoDetectSphere: bool,
        float_sphereCapFraction: float,
        str_particleType: str,
        str_magnification: str,
        str_measureMode: str,
        bool_lsdAdaptiveThresh: bool,
        bool_arScreen: bool,
        int_lsdMinLengthPx: int,
        bool_fuseContours: bool,
    ) -> PrimaryParticleConfig:
        return PrimaryParticleConfig(
            path_input=path_image,
            path_outputDir=build_image_output_dir(
                path_outputRoot, str_groupId, path_image, bool_isBatch),
            path_modelConfig=Path(str_modelConfig),
            path_modelWeights=Path(str_modelWeights),
            int_roiXMin=int_roiXMin,
            int_roiYMin=int_roiYMin,
            int_roiXMax=int_roiXMax,
            int_roiYMax=int_roiYMax,
            int_bboxEdgeMargin=int_bboxEdgeMargin,
            int_tileEdgeMargin=int_tileEdgeMargin,
            float_particleAreaThreshold=float_particleAreaThreshold,
            float_maskBinarizeThreshold=float_maskBinarizeThreshold,
            int_minValidMaskArea=int_minValidMaskArea,
            int_maskMorphKernelSize=int_maskMorphKernelSize,
            int_maskMorphOpenIterations=int_maskMorphOpenIterations,
            int_maskMorphCloseIterations=int_maskMorphCloseIterations,
            int_imgSize=int_imgSize,
            int_tileSize=int_tileSize,
            int_stride=int_stride,
            int_pointsPerTile=int_pointsPerTile,
            int_pointMinDistance=int_pointMinDistance,
            float_pointQualityLevel=float_pointQualityLevel,
            int_pointBatchSize=int_pointBatchSize,
            float_dedupIou=float_dedupIou,
            float_bboxDedupIou=float_bboxDedupIou,
            bool_usePointPrompts=bool_usePointPrompts,
            bool_smallParticle=False,
            float_scalePixels=float_scalePixels,
            float_scaleMicrometers=float_scaleMicrometers,
            str_device=str_device,
            bool_retinaMasks=bool_retinaMasks,
            bool_saveIndividualMasks=bool_saveIndividualMasks,
            float_acicularThreshold=float_acicularThreshold,
            bool_autoCenterCrop=bool_autoCenterCrop,
            float_centerCropRatio=float_centerCropRatio,
            int_targetParticleCount=int_targetParticleCount,
            str_particleMode=str_particleMode,
            bool_autoDetectSphere=bool_autoDetectSphere,
            float_sphereCapFraction=float_sphereCapFraction,
            str_particleType=str_particleType,
            str_magnification=str_magnification,
            str_measureMode=str_measureMode,
            bool_lsdAdaptiveThresh=bool_lsdAdaptiveThresh,
            bool_arScreen=bool_arScreen,
            int_lsdMinLengthPx=int_lsdMinLengthPx,
            bool_fuseContours=bool_fuseContours,
        )


# =========================================================
# 배치 집계
# =========================================================


def build_primary_img_id_summary(
    str_imgId: str,
    path_outputRoot: Path,
    list_fileSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    """동일 IMG_ID 그룹의 1차 입자 summary 들을 집계한다."""

    def _mean_stat(str_key: str, str_stat: str) -> tp.Optional[float]:
        list_vals = [
            d[str_key][str_stat]
            for d in list_fileSummaries
            if isinstance(d.get(str_key), dict)
            and d[str_key].get(str_stat) is not None
        ]
        return calculate_mean_from_optional_values(list_vals)

    return {
        "img_id": str_imgId,
        "output_dir": str(path_outputRoot / str_imgId),
        "num_images": len(list_fileSummaries),
        "num_total_objects": int(
            sum(d.get("num_total_objects", 0) for d in list_fileSummaries)),
        "num_acicular": int(
            sum(d.get("num_acicular", 0) for d in list_fileSummaries)),
        "num_plate": int(
            sum(d.get("num_plate", 0) for d in list_fileSummaries)),
        "num_fragment": int(
            sum(d.get("num_fragment", 0) for d in list_fileSummaries)),
        "acicular_thickness_um_mean": _mean_stat("acicular_thickness_um", "mean"),
        "acicular_long_axis_um_mean": _mean_stat("acicular_long_axis_um", "mean"),
        "acicular_aspect_ratio_mean": _mean_stat("acicular_aspect_ratio", "mean"),
        "plate_thickness_um_mean": _mean_stat("plate_thickness_um", "mean"),
        "plate_long_axis_um_mean": _mean_stat("plate_long_axis_um", "mean"),
        "plate_aspect_ratio_mean": _mean_stat("plate_aspect_ratio", "mean"),
        "all_primary_thickness_um_mean": _mean_stat("all_primary_thickness_um", "mean"),
        "roi_density_mean": calculate_mean_from_optional_values(
            d.get("roi_density") for d in list_fileSummaries),
        "files": list_fileSummaries,
    }


def build_primary_batch_summary(
    path_input: Path,
    path_outputDir: Path,
    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    """1차 입자 배치 전체 통합 summary 를 생성한다."""
    return {
        "input_path": str(path_input),
        "output_dir": str(path_outputDir),
        "num_img_ids": len(list_groupSummaries),
        "num_images": int(
            sum(d.get("num_images", 0) for d in list_groupSummaries)),
        "num_total_objects": int(
            sum(d.get("num_total_objects", 0) for d in list_groupSummaries)),
        "num_acicular": int(
            sum(d.get("num_acicular", 0) for d in list_groupSummaries)),
        "num_plate": int(
            sum(d.get("num_plate", 0) for d in list_groupSummaries)),
        "num_fragment": int(
            sum(d.get("num_fragment", 0) for d in list_groupSummaries)),
        "acicular_thickness_um_mean": calculate_mean_from_optional_values(
            d.get("acicular_thickness_um_mean") for d in list_groupSummaries),
        "plate_thickness_um_mean": calculate_mean_from_optional_values(
            d.get("plate_thickness_um_mean") for d in list_groupSummaries),
        "all_primary_thickness_um_mean": calculate_mean_from_optional_values(
            d.get("all_primary_thickness_um_mean") for d in list_groupSummaries),
        "roi_density_mean": calculate_mean_from_optional_values(
            d.get("roi_density_mean") for d in list_groupSummaries),
        "img_ids": list_groupSummaries,
    }


# =========================================================
# 최상위 실행 함수
# =========================================================


def run_primary_particle_analysis(
    str_input: str,
    str_outputDir: str,
    str_modelConfig: str,
    str_modelWeights: str,
    # center crop
    bool_autoCenterCrop: bool = True,
    float_centerCropRatio: float = CONST_CENTER_CROP_RATIO,
    # ROI (auto_center_crop=False 일 때 사용)
    int_roiXMin: int = CONST_ROI_X_MIN,
    int_roiYMin: int = CONST_ROI_Y_MIN,
    int_roiXMax: int = CONST_ROI_X_MAX,
    int_roiYMax: int = CONST_ROI_Y_MAX,
    int_bboxEdgeMargin: int = CONST_BBOX_EDGE_MARGIN,
    int_tileEdgeMargin: int = CONST_TILE_EDGE_MARGIN,
    # 분류
    float_acicularThreshold: float = CONST_ACICULAR_THRESHOLD,
    float_particleAreaThreshold: float = CONST_PRIMARY_PARTICLE_AREA_THRESHOLD,
    int_targetParticleCount: int = CONST_TARGET_PARTICLE_COUNT,
    # mask 후처리
    float_maskBinarizeThreshold: float = CONST_MASK_BINARIZE_THRESHOLD,
    int_minValidMaskArea: int = CONST_MIN_VALID_MASK_AREA,
    int_maskMorphKernelSize: int = CONST_MASK_MORPH_KERNEL_SIZE,
    int_maskMorphOpenIterations: int = CONST_MASK_MORPH_OPEN_ITERATIONS,
    int_maskMorphCloseIterations: int = CONST_MASK_MORPH_CLOSE_ITERATIONS,
    # SAM2 추론
    int_imgSize: int = CONST_DEFAULT_IMAGE_SIZE,
    int_tileSize: int = CONST_PRIMARY_TILE_SIZE,
    int_stride: int = CONST_PRIMARY_TILE_STRIDE,
    int_pointsPerTile: int = CONST_PRIMARY_POINTS_PER_TILE,
    int_pointMinDistance: int = CONST_PRIMARY_POINT_MIN_DISTANCE,
    float_pointQualityLevel: float = CONST_DEFAULT_POINT_QUALITY_LEVEL,
    int_pointBatchSize: int = CONST_DEFAULT_POINT_BATCH_SIZE,
    float_dedupIou: float = CONST_DEFAULT_DEDUP_IOU,
    float_bboxDedupIou: float = CONST_DEFAULT_BBOX_DEDUP_IOU,
    bool_usePointPrompts: bool = CONST_DEFAULT_USE_POINT_PROMPTS,
    # 스케일
    float_scalePixels: float = CONST_SCALE_PIXELS,
    float_scaleMicrometers: float = CONST_SCALE_MICROMETERS,
    # 기타
    str_device: tp.Optional[str] = None,
    bool_retinaMasks: bool = CONST_DEFAULT_RETINA_MASKS,
    bool_saveIndividualMasks: bool = CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
    str_particleMode: str = "auto",
    bool_autoDetectSphere: bool = False,
    float_sphereCapFraction: float = CONST_SPHERE_CAP_FRACTION,
    str_particleType: str = "unknown",
    str_magnification: str = "unknown",
    str_measureMode: str = "sam2",
    bool_lsdAdaptiveThresh: bool = False,
    bool_arScreen: bool = False,
    int_lsdMinLengthPx: int = 20,
    bool_fuseContours: bool = False,
) -> tp.Dict[str, tp.Any]:
    """외부에서 호출 가능한 최상위 실행 함수.

    Args:
        str_input: 단일 이미지 또는 batch root directory 경로.
        str_outputDir: 결과 저장 root directory 경로.
        str_modelConfig: SAM2 설정 파일 경로.
        str_modelWeights: SAM2 weight 파일 경로.
        (나머지 파라미터는 build_primary_arg_parser 설명 참조)

    Returns:
        단일 입력이면 단일 이미지 summary dict,
        directory 입력이면 batch summary dict.
    """
    path_input = Path(str_input)
    path_outputRoot = Path(str_outputDir)
    list_inputGroups = collect_input_groups(path_input)
    bool_isBatch = path_input.is_dir()

    def _create_config(
        str_groupId: str, path_image: Path
    ) -> PrimaryParticleConfig:
        return PrimaryParticleService._create_config(
            str_groupId=str_groupId,
            path_image=path_image,
            path_outputRoot=path_outputRoot,
            bool_isBatch=bool_isBatch,
            str_modelConfig=str_modelConfig,
            str_modelWeights=str_modelWeights,
            int_roiXMin=int_roiXMin,
            int_roiYMin=int_roiYMin,
            int_roiXMax=int_roiXMax,
            int_roiYMax=int_roiYMax,
            int_bboxEdgeMargin=int_bboxEdgeMargin,
            int_tileEdgeMargin=int_tileEdgeMargin,
            float_particleAreaThreshold=float_particleAreaThreshold,
            float_maskBinarizeThreshold=float_maskBinarizeThreshold,
            int_minValidMaskArea=int_minValidMaskArea,
            int_maskMorphKernelSize=int_maskMorphKernelSize,
            int_maskMorphOpenIterations=int_maskMorphOpenIterations,
            int_maskMorphCloseIterations=int_maskMorphCloseIterations,
            int_imgSize=int_imgSize,
            int_tileSize=int_tileSize,
            int_stride=int_stride,
            int_pointsPerTile=int_pointsPerTile,
            int_pointMinDistance=int_pointMinDistance,
            float_pointQualityLevel=float_pointQualityLevel,
            int_pointBatchSize=int_pointBatchSize,
            float_dedupIou=float_dedupIou,
            float_bboxDedupIou=float_bboxDedupIou,
            bool_usePointPrompts=bool_usePointPrompts,
            float_scalePixels=float_scalePixels,
            float_scaleMicrometers=float_scaleMicrometers,
            str_device=str_device,
            bool_retinaMasks=bool_retinaMasks,
            bool_saveIndividualMasks=bool_saveIndividualMasks,
            float_acicularThreshold=float_acicularThreshold,
            bool_autoCenterCrop=bool_autoCenterCrop,
            float_centerCropRatio=float_centerCropRatio,
            int_targetParticleCount=int_targetParticleCount,
            str_particleMode=str_particleMode,
            bool_autoDetectSphere=bool_autoDetectSphere,
            float_sphereCapFraction=float_sphereCapFraction,
            str_particleType=str_particleType,
            str_magnification=str_magnification,
            str_measureMode=str_measureMode,
            bool_lsdAdaptiveThresh=bool_lsdAdaptiveThresh,
            bool_arScreen=bool_arScreen,
            int_lsdMinLengthPx=int_lsdMinLengthPx,
            bool_fuseContours=bool_fuseContours,
        )

    # 단일 이미지
    if not bool_isBatch:
        str_groupId, list_imagePaths = list_inputGroups[0]
        print(f"[single] processing: {list_imagePaths[0].name}", flush=True)
        obj_service = PrimaryParticleService(_create_config(str_groupId, list_imagePaths[0]))
        obj_result = obj_service.process_primary()
        print(f"[single] done: {list_imagePaths[0].name}", flush=True)
        return obj_result.dict_summary

    # 배치
    path_outputRoot.mkdir(parents=True, exist_ok=True)

    str_firstGroupId, list_firstImages = list_inputGroups[0]
    print(f"[batch] init model: {list_firstImages[0].name}", flush=True)
    obj_sharedService = PrimaryParticleService(
        _create_config(str_firstGroupId, list_firstImages[0]))
    obj_sharedService.initialize_model()

    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]] = []
    int_numGroups = len(list_inputGroups)

    for int_gi, (str_groupId, list_imagePaths) in enumerate(list_inputGroups, start=1):
        print(
            f"[batch][group {int_gi}/{int_numGroups}] IMG_ID={str_groupId} "
            f"({len(list_imagePaths)} images)",
            flush=True,
        )
        list_fileSummaries: tp.List[tp.Dict[str, tp.Any]] = []
        int_numImages = len(list_imagePaths)

        for int_ii, path_image in enumerate(list_imagePaths, start=1):
            print(f"  [image {int_ii}/{int_numImages}] {path_image.name}", flush=True)
            obj_service = PrimaryParticleService(_create_config(str_groupId, path_image))
            obj_service.obj_model = obj_sharedService.obj_model
            obj_service.dict_modelConfig = dict(obj_sharedService.dict_modelConfig)
            obj_result = obj_service.process_primary()
            dict_fs = dict(obj_result.dict_summary)
            dict_fs["img_id"] = str_groupId
            dict_fs["image_name"] = path_image.name
            list_fileSummaries.append(dict_fs)

        dict_groupSummary = build_primary_img_id_summary(
            str_groupId, path_outputRoot, list_fileSummaries)
        path_groupDir = path_outputRoot / str_groupId
        path_groupDir.mkdir(parents=True, exist_ok=True)
        with (path_groupDir / "img_id_summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_groupSummary, obj_f, ensure_ascii=False, indent=2,
                      default=json_default)
        list_groupSummaries.append(dict_groupSummary)

        print(
            f"[batch][group done] {str_groupId}  "
            f"acicular={dict_groupSummary['num_acicular']}  "
            f"plate={dict_groupSummary['num_plate']}  "
            f"fragment={dict_groupSummary['num_fragment']}",
            flush=True,
        )

    dict_batchSummary = build_primary_batch_summary(
        path_input, path_outputRoot, list_groupSummaries)
    with (path_outputRoot / "batch_summary.json").open("w", encoding="utf-8") as obj_f:
        json.dump(dict_batchSummary, obj_f, ensure_ascii=False, indent=2,
                  default=json_default)

    print(
        f"[batch] done: {dict_batchSummary['num_img_ids']} groups, "
        f"{dict_batchSummary['num_images']} images  "
        f"total acicular={dict_batchSummary['num_acicular']} "
        f"plate={dict_batchSummary['num_plate']}",
        flush=True,
    )
    return dict_batchSummary


# =========================================================
# CLI
# =========================================================


def build_primary_arg_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 생성한다."""
    str_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    obj_parser = argparse.ArgumentParser(
        description=(
            "SAM2로 1차 입자를 segmentation하고 "
            "침상(acicular)/판상(plate)으로 분류하여 두께를 측정합니다."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 입출력 / 모델
    obj_parser.add_argument(
        "--input", default="img/primary_test.jpg",
        help="입력 이미지 또는 디렉터리 경로")
    obj_parser.add_argument(
        "--output_dir", default=f"out_primary_{str_ts}",
        help="결과 저장 폴더")
    obj_parser.add_argument(
        "--model_cfg", default="model/sam2.1_hiera_t.yaml",
        help="SAM2 YAML 설정 파일 경로")
    obj_parser.add_argument(
        "--model", default="model/sam2.1_hiera_base_plus.pt",
        help="SAM2 가중치 파일 경로")

    # 중앙 crop
    obj_parser.add_argument(
        "--auto_center_crop",
        action=argparse.BooleanOptionalAction, default=True,
        help="자동 중앙 crop 사용 여부. False 이면 --roi_* 파라미터로 직접 지정")
    obj_parser.add_argument(
        "--center_crop_ratio", type=float, default=CONST_CENTER_CROP_RATIO,
        help="중앙 crop 비율 (0.1 ~ 1.0). 예: 0.6 → 이미지 중앙 60%% 사용")

    # ROI (auto_center_crop=False 일 때)
    obj_parser.add_argument("--roi_x_min", type=int, default=CONST_ROI_X_MIN)
    obj_parser.add_argument("--roi_y_min", type=int, default=CONST_ROI_Y_MIN)
    obj_parser.add_argument("--roi_x_max", type=int, default=CONST_ROI_X_MAX)
    obj_parser.add_argument("--roi_y_max", type=int, default=CONST_ROI_Y_MAX)
    obj_parser.add_argument(
        "--bbox_edge_margin", type=int, default=CONST_BBOX_EDGE_MARGIN,
        help="ROI 경계 근처 bbox 제외 margin")
    obj_parser.add_argument(
        "--tile_edge_margin", type=int, default=CONST_TILE_EDGE_MARGIN,
        help="타일 경계 근처 bbox 제외 margin")

    # 분류 기준
    obj_parser.add_argument(
        "--acicular_threshold", type=float, default=CONST_ACICULAR_THRESHOLD,
        help="침상/판상 분류 aspect_ratio 임계값. "
             "aspect_ratio(= 두께/장축) < 이 값 → 침상, >= 이 값 → 판상")
    obj_parser.add_argument(
        "--area_threshold", type=float, default=CONST_PRIMARY_PARTICLE_AREA_THRESHOLD,
        help="유효 1차 입자 최소 면적 (미만이면 fragment)")
    obj_parser.add_argument(
        "--target_particle_count", type=int, default=CONST_TARGET_PARTICLE_COUNT,
        help="목표 침상+판상 입자 수 (미달 시 경고)")

    # 스케일 (SEM 이미지 배율에 맞게 설정 필요)
    obj_parser.add_argument(
        "--scale_pixels", type=float, default=CONST_SCALE_PIXELS,
        help="스케일 기준 pixel 수. 20k=147, 50k=371")
    obj_parser.add_argument(
        "--scale_um", type=float, default=CONST_SCALE_MICROMETERS,
        help="스케일 기준 µm 값. 20k=1, 50k=1 (기본 단위: 1 µm)")

    # SAM2 추론
    obj_parser.add_argument(
        "--imgsz", type=int, default=CONST_DEFAULT_IMAGE_SIZE,
        help="SAM2 추론 이미지 크기")
    obj_parser.add_argument(
        "--tile_size", type=int, default=CONST_PRIMARY_TILE_SIZE,
        help="ROI 내부 타일 크기 (1차 입자는 작게 설정)")
    obj_parser.add_argument(
        "--stride", type=int, default=CONST_PRIMARY_TILE_STRIDE,
        help="타일 stride")
    obj_parser.add_argument(
        "--points_per_tile", type=int, default=CONST_PRIMARY_POINTS_PER_TILE,
        help="각 타일에서 추출할 후보점 수")
    obj_parser.add_argument(
        "--point_min_distance", type=int, default=CONST_PRIMARY_POINT_MIN_DISTANCE,
        help="후보점 최소 거리 (pixel)")
    obj_parser.add_argument(
        "--point_quality_level", type=float, default=CONST_DEFAULT_POINT_QUALITY_LEVEL,
        help="goodFeaturesToTrack qualityLevel")
    obj_parser.add_argument(
        "--point_batch_size", type=int, default=CONST_DEFAULT_POINT_BATCH_SIZE,
        help="한 번의 SAM2 호출에 묶을 point 수")
    obj_parser.add_argument(
        "--dedup_iou", type=float, default=CONST_DEFAULT_DEDUP_IOU,
        help="mask 기준 중복 제거 IoU threshold")
    obj_parser.add_argument(
        "--bbox_dedup_iou", type=float, default=CONST_DEFAULT_BBOX_DEDUP_IOU,
        help="bbox 기준 1차 중복 제거 IoU threshold")
    obj_parser.add_argument(
        "--use_point_prompts",
        action=argparse.BooleanOptionalAction, default=True,
        help="OpenCV 후보점 기반 point prompt 추론 사용 여부")

    # Mask 후처리
    obj_parser.add_argument(
        "--mask_binarize_threshold", type=float, default=CONST_MASK_BINARIZE_THRESHOLD,
        help="SAM2 raw mask → binary mask 변환 threshold")
    obj_parser.add_argument(
        "--min_valid_mask_area", type=int, default=CONST_MIN_VALID_MASK_AREA,
        help="이 값보다 작은 mask 는 무시")
    obj_parser.add_argument(
        "--mask_morph_kernel_size", type=int, default=CONST_MASK_MORPH_KERNEL_SIZE,
        help="morphology kernel 크기. 0/1 이면 비활성화")
    obj_parser.add_argument(
        "--mask_morph_open_iterations", type=int, default=CONST_MASK_MORPH_OPEN_ITERATIONS)
    obj_parser.add_argument(
        "--mask_morph_close_iterations", type=int, default=CONST_MASK_MORPH_CLOSE_ITERATIONS)

    # 기타
    obj_parser.add_argument(
        "--retina_masks",
        action=argparse.BooleanOptionalAction, default=True)
    obj_parser.add_argument(
        "--save_mask_imgs", "--save_individual_masks",
        dest="save_mask_imgs",
        action=argparse.BooleanOptionalAction, default=True,
        help="개별 mask 이미지 저장 여부")
    obj_parser.add_argument(
        "--device", default=None,
        help="추론 device. 예: cpu, cuda:0")

    # ── 핵심 분석 모드 선택 (preset 적용됨) ──────────────────────
    obj_parser.add_argument(
        "--particle_type", default=None,
        choices=["acicular", "plate"],
        help=(
            "1차 입자 형태 (필수 권장). "
            "'acicular': 침상 입자 = 대입경 전구체 특성. "
            "'plate': 판상 입자 = 소입경 전구체 특성. "
            "지정하면 --magnification 과 함께 최적 파라미터가 자동 설정됨."
        ),
    )
    obj_parser.add_argument(
        "--magnification", default=None,
        choices=["20k", "50k"],
        help=(
            "SEM 이미지 배율. "
            "'20k' (20000배): 구형 2차 입자 전체가 화면에 들어옴. "
            "'50k' (50000배): 입자 표면 클로즈업 (위아래 잘림 가능)."
        ),
    )

    # 입자 형태 모드 (preset으로 자동 설정되지만 개별 override 가능)
    obj_parser.add_argument(
        "--particle_mode", default="auto",
        choices=["auto", "acicular"],
        help=(
            "입자 형태 처리 모드. "
            "'auto': 기존 tiled point prompt 방식. "
            "'acicular': OpenCV elongated-contour 탐지 → SAM2 box prompt hybrid 방식 "
            "(침상 입자에 특화; 탐지가 부족하면 자동으로 point prompt fallback)."
        ),
    )

    # 구형 2차 입자 자동 검출
    obj_parser.add_argument(
        "--auto_detect_sphere",
        action=argparse.BooleanOptionalAction, default=False,
        help=(
            "구형 2차 입자를 자동 검출하여 top-cap 영역을 ROI로 사용한다 (acicular 모드 전용). "
            "2차 입자 전체가 보이는 SEM 이미지(예: 5000~20000배)에 유용하다. "
            "검출 실패 시 center crop으로 fallback."
        ),
    )
    obj_parser.add_argument(
        "--sphere_cap_fraction", type=float, default=CONST_SPHERE_CAP_FRACTION,
        help=(
            "구형 2차 입자 검출 시 ROI로 사용할 cap 높이 비율 (0.1~1.0). "
            f"기본값 {CONST_SPHERE_CAP_FRACTION}: 구 지름의 상단 {int(CONST_SPHERE_CAP_FRACTION*100)}%% 영역."
        ),
    )

    # LSD 전처리 / 후처리 옵션
    obj_parser.add_argument(
        "--fuse",
        action=argparse.BooleanOptionalAction, default=False,
        help="겹치고 방향이 같은 컨투어를 하나로 융합한다 (Δangle < 10°, 겹침 ≥ 40%%). 기본값: OFF.",
    )
    obj_parser.add_argument(
        "--min_length", type=int, default=20,
        help="LSD 선분 최소 길이 (px). 이 값보다 짧은 선분은 무시된다. 기본값: 20.",
    )
    obj_parser.add_argument(
        "--lsd_adaptive_thresh",
        action=argparse.BooleanOptionalAction, default=False,
        help=(
            "LSD 전처리에서 전역 Otsu 대신 지역 Adaptive(Gaussian) threshold를 사용한다. "
            "명암 불균일 이미지에서 엣지 검출 품질을 높인다."
        ),
    )
    obj_parser.add_argument(
        "--ar_screen",
        action=argparse.BooleanOptionalAction, default=False,
        help=(
            "OpenCV 침상 후보 탐지 시 종횡비(AR) 필터를 적용한다. "
            "ON이면 AR ≥ threshold인 후보를 제외, OFF(기본)이면 모든 blob을 후보로 통과시킨다."
        ),
    )

    # 측정 방법
    obj_parser.add_argument(
        "--measure_mode", default="sam2",
        choices=["sam2", "lsd"],
        help=(
            "측정 방법. "
            "'sam2'(기본): SAM2 segmentation → minAreaRect 두께. "
            "'lsd': LSD(Line Segment Detector) + 수직 강도 프로파일 직접 측정 "
            "(SAM2 불필요, 매우 빠름, 대량 탐지). "
            "acicular 프리셋은 기본으로 'lsd'를 사용."
        ),
    )

    return obj_parser
