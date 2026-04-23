#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Primary Particle Thickness - SAM2 기반 1차 입자 침상/판상 분류 및 두께 측정
=============================================================================
2차전지 전구체 SEM 이미지(20000배 / 50000배)에서 1차 입자를 segmentation하고,
침상(acicular) / 판상(plate) 으로 분류하여 두께를 측정한다.

처리 흐름:
1. 원본 SEM 이미지 로드 → 자동 중앙 crop (또는 명시적 ROI)
2. SAM2 타일 추론으로 1차 입자 segmentation
3. cv2.minAreaRect 기반 두께(단축) / 장축 측정
4. aspect ratio 기준 침상 / 판상 / fragment 분류
5. 결과 저장 (objects.csv, acicular.csv, plate.csv, overlay, thickness histogram,
              summary.json, objects.json, debug.json, 개별 mask)
6. 배치 입력 시 IMG_ID 집계 및 batch_summary.json 생성
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import typing as tp
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from precursur_aspect_ratio_SAM2 import (
        Sam2AspectRatioConfig,
        Sam2AspectRatioService,
        convert_pixels_to_micrometers,
        calculate_mean_from_optional_values,
        calculate_percentage,
        collect_input_groups,
        build_image_output_dir,
        CONST_SCALE_PIXELS,
        CONST_SCALE_MICROMETERS,
        CONST_SMALL_PARTICLE_SCALE_PIXELS,
        CONST_SMALL_PARTICLE_SCALE_MICROMETERS,
        CONST_DEFAULT_SMALL_PARTICLE,
        CONST_BBOX_EDGE_MARGIN,
        CONST_TILE_EDGE_MARGIN,
        CONST_ROI_X_MIN,
        CONST_ROI_Y_MIN,
        CONST_ROI_X_MAX,
        CONST_ROI_Y_MAX,
        CONST_MASK_BINARIZE_THRESHOLD,
        CONST_MIN_VALID_MASK_AREA,
        CONST_MASK_MORPH_KERNEL_SIZE,
        CONST_MASK_MORPH_OPEN_ITERATIONS,
        CONST_MASK_MORPH_CLOSE_ITERATIONS,
        CONST_DEFAULT_IMAGE_SIZE,
        CONST_DEFAULT_RETINA_MASKS,
        CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
        CONST_DEFAULT_TILE_SIZE,
        CONST_DEFAULT_TILE_STRIDE,
        CONST_DEFAULT_POINTS_PER_TILE,
        CONST_DEFAULT_POINT_MIN_DISTANCE,
        CONST_DEFAULT_POINT_QUALITY_LEVEL,
        CONST_DEFAULT_POINT_BATCH_SIZE,
        CONST_DEFAULT_DEDUP_IOU,
        CONST_DEFAULT_BBOX_DEDUP_IOU,
        CONST_DEFAULT_USE_POINT_PROMPTS,
    )
except ImportError as e:
    print(
        f"[ERROR] precursur_aspect_ratio_SAM2.py 를 import할 수 없습니다: {e}\n"
        "같은 디렉터리에 있는지 확인하세요.",
        file=sys.stderr,
    )
    sys.exit(1)


# =========================================================
# 1차 입자 분석 전용 상수
# =========================================================

# 침상(acicular) / 판상(plate) 분류 기준
# aspect_ratio = thickness_px / long_axis_px  (0 < x <= 1)
# aspect_ratio <  이 값 : 침상
# aspect_ratio >= 이 값 : 판상
CONST_ACICULAR_THRESHOLD: float = 0.40

# 유효 1차 입자 최소 면적 (미만이면 fragment)
# 2차 입자 기본값(1500)보다 훨씬 작게 설정
CONST_PRIMARY_PARTICLE_AREA_THRESHOLD: float = 200.0

# 자동 중앙 crop 비율 (이미지 중앙의 이 비율 영역을 사용)
CONST_CENTER_CROP_RATIO: float = 0.60

# 침상+판상 입자 목표 수 (미달 시 경고)
CONST_TARGET_PARTICLE_COUNT: int = 10

# 1차 입자용 SAM2 추론 파라미터 (더 촘촘한 타일/포인트)
CONST_PRIMARY_TILE_SIZE: int = 256
CONST_PRIMARY_TILE_STRIDE: int = 128
CONST_PRIMARY_POINTS_PER_TILE: int = 120
CONST_PRIMARY_POINT_MIN_DISTANCE: int = 8


# =========================================================
# Config / Dataclass
# =========================================================


@dataclass
class PrimaryParticleConfig(Sam2AspectRatioConfig):
    """1차 입자 분석 전용 설정. Sam2AspectRatioConfig 를 확장한다."""

    float_acicularThreshold: float = CONST_ACICULAR_THRESHOLD
    bool_autoCenterCrop: bool = True
    float_centerCropRatio: float = CONST_CENTER_CROP_RATIO
    int_targetParticleCount: int = CONST_TARGET_PARTICLE_COUNT


@dataclass
class PrimaryParticleMeasurement:
    """1차 입자 단일 마스크 측정 결과."""

    int_index: int
    str_category: str            # "acicular" | "plate" | "fragment"
    int_maskArea: int
    float_confidence: tp.Optional[float]
    int_bboxX: int
    int_bboxY: int
    int_bboxWidth: int
    int_bboxHeight: int
    float_centroidX: float
    float_centroidY: float
    # minAreaRect 기반 단/장축 측정 (회전 보정)
    float_thicknessPx: float     # 단축 = 두께 [pixel]
    float_longAxisPx: float      # 장축 [pixel]
    float_minRectAngle: float    # 회전 각도 [degree]
    float_thicknessUm: float     # 두께 [µm]
    float_longAxisUm: float      # 장축 [µm]
    float_aspectRatio: float     # thickness / long_axis  (0 < x <= 1)
    # H/V span (참조용 — 기존 secondary 파이프라인 호환)
    int_longestHorizontal: int
    int_longestVertical: int
    float_longestHorizontalUm: float
    float_longestVerticalUm: float


@dataclass
class PrimaryParticleResult:
    """단일 이미지 처리 결과."""

    list_objects: tp.List[PrimaryParticleMeasurement]
    dict_summary: tp.Dict[str, tp.Any]


# =========================================================
# Service
# =========================================================


class PrimaryParticleService(Sam2AspectRatioService):
    """1차 입자 segmentation, 측정, 저장 서비스."""

    def __init__(self, obj_config: PrimaryParticleConfig) -> None:
        super().__init__(obj_config)
        self.obj_primary_config: PrimaryParticleConfig = obj_config

    # ----------------------------------------------------------
    # ROI: 자동 중앙 crop
    # ----------------------------------------------------------

    def compute_center_roi(
        self, int_h: int, int_w: int
    ) -> tp.Tuple[int, int, int, int]:
        """이미지 크기에서 중앙 crop ROI 좌표를 계산한다.

        Returns:
            (x0, y0, x1, y1) 형식의 ROI 좌표.
        """
        float_ratio = float(np.clip(self.obj_primary_config.float_centerCropRatio, 0.1, 1.0))
        int_xMargin = int(int_w * (1.0 - float_ratio) / 2.0)
        int_yMargin = int(int_h * (1.0 - float_ratio) / 2.0)
        int_x0 = max(0, int_xMargin)
        int_y0 = max(0, int_yMargin)
        int_x1 = min(int_w, int_w - int_xMargin)
        int_y1 = min(int_h, int_h - int_yMargin)
        return int_x0, int_y0, int_x1, int_y1

    def extract_inference_roi(
        self,
        arr_imageBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Dict[str, int]]:
        """bool_autoCenterCrop=True 이면 중앙 crop ROI를 사용하고,
        False 이면 명시적 ROI 파라미터를 그대로 사용한다.
        """
        int_h, int_w = arr_imageBgr.shape[:2]

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

        # 분류
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
            float_thicknessUm=self.convert_pixels_to_micrometers(float_thicknessPx),
            float_longAxisUm=self.convert_pixels_to_micrometers(float_longAxisPx),
            float_aspectRatio=float_aspectRatio,
            int_longestHorizontal=int_horizontal,
            int_longestVertical=int_vertical,
            float_longestHorizontalUm=self.convert_pixels_to_micrometers(float(int_horizontal)),
            float_longestVerticalUm=self.convert_pixels_to_micrometers(float(int_vertical)),
        )

    # ----------------------------------------------------------
    # 시각화
    # ----------------------------------------------------------

    def create_primary_overlay(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[PrimaryParticleMeasurement],
        list_masks: tp.List[np.ndarray],
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

        arr_overlay = arr_imageBgr.copy()

        # 반투명 색칠
        for obj_m, arr_mask in zip(list_objects, list_masks):
            tpl_c = dict_fill.get(obj_m.str_category, (128, 128, 128))
            arr_overlay[arr_mask > 0] = (
                arr_overlay[arr_mask > 0].astype(np.float32) * 0.5
                + np.array(tpl_c, dtype=np.float32) * 0.5
            ).astype(np.uint8)

        # contour + bbox + label
        for obj_m, arr_mask in zip(list_objects, list_masks):
            arr_contour = self.extract_largest_contour(arr_mask)
            if arr_contour is None:
                continue

            tpl_ec = dict_edge.get(obj_m.str_category, (128, 128, 128))
            cv2.drawContours(arr_overlay, [arr_contour], -1, tpl_ec, 1)
            cv2.rectangle(
                arr_overlay,
                (obj_m.int_bboxX, obj_m.int_bboxY),
                (
                    obj_m.int_bboxX + obj_m.int_bboxWidth,
                    obj_m.int_bboxY + obj_m.int_bboxHeight,
                ),
                tpl_ec, 1,
            )

            int_lx = obj_m.int_bboxX
            int_ly = max(14, obj_m.int_bboxY - 4)

            if obj_m.str_category == "fragment":
                str_label = f"F{obj_m.int_index} A={obj_m.int_maskArea}"
            else:
                str_prefix = "Ac" if obj_m.str_category == "acicular" else "Pl"
                str_label = (
                    f"{str_prefix}{obj_m.int_index} "
                    f"t={obj_m.float_thicknessUm:.2f}um "
                    f"AR={obj_m.float_aspectRatio:.2f}"
                )

            cv2.putText(
                arr_overlay, str_label,
                (int_lx, int_ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                tpl_ec, 1, cv2.LINE_AA,
            )

        return arr_overlay

    # ----------------------------------------------------------
    # 통계
    # ----------------------------------------------------------

    def build_primary_summary(
        self,
        list_objects: tp.List[PrimaryParticleMeasurement],
    ) -> tp.Dict[str, tp.Any]:
        """침상/판상별 두께·종횡비 통계를 포함한 summary dict 를 생성한다."""

        list_acicular = [o for o in list_objects if o.str_category == "acicular"]
        list_plate = [o for o in list_objects if o.str_category == "plate"]
        list_fragment = [o for o in list_objects if o.str_category == "fragment"]

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

        return {
            "input_path": str(self.obj_config.path_input),
            "output_dir": str(self.obj_config.path_outputDir),
            "model_weights_path": str(self.obj_config.path_modelWeights),
            "scale_pixels": float(self.obj_config.float_scalePixels),
            "scale_micrometers": float(self.obj_config.float_scaleMicrometers),
            "micrometers_per_pixel": float(float_um_per_px),
            "acicular_threshold": float(self.obj_primary_config.float_acicularThreshold),
            "auto_center_crop": bool(self.obj_primary_config.bool_autoCenterCrop),
            "center_crop_ratio": float(self.obj_primary_config.float_centerCropRatio),
            "particle_area_threshold": float(self.obj_config.float_particleAreaThreshold),
            "num_total_objects": len(list_objects),
            "num_acicular": len(list_acicular),
            "num_plate": len(list_plate),
            "num_fragment": len(list_fragment),
            "acicular_thickness_um": _stats([o.float_thicknessUm for o in list_acicular]),
            "acicular_long_axis_um": _stats([o.float_longAxisUm for o in list_acicular]),
            "acicular_aspect_ratio": _stats([o.float_aspectRatio for o in list_acicular]),
            "plate_thickness_um": _stats([o.float_thicknessUm for o in list_plate]),
            "plate_long_axis_um": _stats([o.float_longAxisUm for o in list_plate]),
            "plate_aspect_ratio": _stats([o.float_aspectRatio for o in list_plate]),
            "all_primary_thickness_um": _stats(
                [o.float_thicknessUm for o in list_acicular + list_plate]
            ),
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
                (list_ac, "Acicular (침상)", "#5588ff"),
                (list_pl, "Plate (판상)",    "#44cc44"),
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
    ) -> None:
        """이미지, CSV, JSON, histogram 등 1차 입자 분석 산출물을 저장한다."""
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)

        # overlay_full: 원본 이미지 위에 ROI overlay를 덮는다
        arr_overlayFull = arr_inputBgr.copy()
        arr_overlayFull[
            dict_roi["y_min"]:dict_roi["y_max"],
            dict_roi["x_min"]:dict_roi["x_max"],
        ] = arr_overlayRoi
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

        # objects.csv (전체)
        path_csvAll = self.obj_config.path_outputDir / "objects.csv"
        with path_csvAll.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_objects:
                obj_writer = csv.DictWriter(
                    obj_f, fieldnames=list(asdict(list_objects[0]).keys()))
                obj_writer.writeheader()
                for obj_m in list_objects:
                    obj_writer.writerow(asdict(obj_m))

        # acicular.csv / plate.csv
        for str_cat, str_fname in [("acicular", "acicular.csv"), ("plate", "plate.csv")]:
            list_rows = [asdict(o) for o in list_objects if o.str_category == str_cat]
            path_csv = self.obj_config.path_outputDir / str_fname
            with path_csv.open("w", newline="", encoding="utf-8-sig") as obj_f:
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
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)
        with (self.obj_config.path_outputDir / "objects.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(
                [asdict(o) for o in list_objects], obj_f, ensure_ascii=False, indent=2)
        with (self.obj_config.path_outputDir / "debug.json").open(
                "w", encoding="utf-8") as obj_f:
            json.dump(dict_debug, obj_f, ensure_ascii=False, indent=2)

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
        arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(arr_inputRoiBgr)

        list_objects: tp.List[PrimaryParticleMeasurement] = []
        list_validMasks: tp.List[np.ndarray] = []

        for int_index, arr_mask in enumerate(arr_masks):
            float_conf: tp.Optional[float] = None
            if arr_scores is not None and int_index < len(arr_scores):
                float_conf = float(arr_scores[int_index])

            obj_m = self.measure_primary_mask(arr_mask, int_index, float_conf)
            if obj_m is None:
                continue
            list_objects.append(obj_m)
            list_validMasks.append(self.refine_mask_for_area(arr_mask).astype(np.uint8))

        # 목표 입자 수 경고
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
        dict_summary["roi"] = dict_roi
        dict_summary["num_tiles"] = dict_debug.get("num_tiles")
        dict_summary["num_candidate_points"] = dict_debug.get("num_candidate_points")
        dict_summary["num_accepted_masks"] = dict_debug.get("num_accepted_masks")
        dict_summary["num_bbox_dedup_rejected"] = dict_debug.get("num_bbox_dedup_rejected")

        self.save_primary_outputs(
            arr_inputBgr, arr_inputRoiBgr, arr_overlay,
            list_objects, list_validMasks,
            dict_summary, dict_roi, dict_debug,
        )

        return PrimaryParticleResult(list_objects=list_objects, dict_summary=dict_summary)


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
            json.dump(dict_groupSummary, obj_f, ensure_ascii=False, indent=2)
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
        json.dump(dict_batchSummary, obj_f, ensure_ascii=False, indent=2)

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
        help="스케일 기준 pixel 수. 예: 276 (기본 = 276 px = 50 µm)")
    obj_parser.add_argument(
        "--scale_um", type=float, default=CONST_SCALE_MICROMETERS,
        help="스케일 기준 µm 값. 예: 50 (기본 = 276 px = 50 µm). "
             "소입자 스케일 예시: --scale_pixels 184 --scale_um 10")

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

    return obj_parser


def main() -> None:
    """CLI 진입점."""
    obj_args = build_primary_arg_parser().parse_args()

    dict_summary = run_primary_particle_analysis(
        str_input=obj_args.input,
        str_outputDir=obj_args.output_dir,
        str_modelConfig=obj_args.model_cfg,
        str_modelWeights=obj_args.model,
        bool_autoCenterCrop=obj_args.auto_center_crop,
        float_centerCropRatio=obj_args.center_crop_ratio,
        int_roiXMin=obj_args.roi_x_min,
        int_roiYMin=obj_args.roi_y_min,
        int_roiXMax=obj_args.roi_x_max,
        int_roiYMax=obj_args.roi_y_max,
        int_bboxEdgeMargin=obj_args.bbox_edge_margin,
        int_tileEdgeMargin=obj_args.tile_edge_margin,
        float_acicularThreshold=obj_args.acicular_threshold,
        float_particleAreaThreshold=obj_args.area_threshold,
        int_targetParticleCount=obj_args.target_particle_count,
        float_scalePixels=obj_args.scale_pixels,
        float_scaleMicrometers=obj_args.scale_um,
        float_maskBinarizeThreshold=obj_args.mask_binarize_threshold,
        int_minValidMaskArea=obj_args.min_valid_mask_area,
        int_maskMorphKernelSize=obj_args.mask_morph_kernel_size,
        int_maskMorphOpenIterations=obj_args.mask_morph_open_iterations,
        int_maskMorphCloseIterations=obj_args.mask_morph_close_iterations,
        int_imgSize=obj_args.imgsz,
        int_tileSize=obj_args.tile_size,
        int_stride=obj_args.stride,
        int_pointsPerTile=obj_args.points_per_tile,
        int_pointMinDistance=obj_args.point_min_distance,
        float_pointQualityLevel=obj_args.point_quality_level,
        int_pointBatchSize=obj_args.point_batch_size,
        float_dedupIou=obj_args.dedup_iou,
        float_bboxDedupIou=obj_args.bbox_dedup_iou,
        bool_usePointPrompts=obj_args.use_point_prompts,
        str_device=obj_args.device,
        bool_retinaMasks=obj_args.retina_masks,
        bool_saveIndividualMasks=obj_args.save_mask_imgs,
    )

    print("===== 1차 입자 분석 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import time
    float_t0 = time.time()
    main()
    print(f"Elapsed time: {time.time() - float_t0:.4f} seconds")
