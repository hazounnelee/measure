from __future__ import annotations
import csv
import math
import os
import shutil
import typing as tp
from dataclasses import asdict, replace as dataclasses_replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.schema import Sam2AspectRatioConfig, ObjectMeasurement, Sam2AspectRatioResult
from models import load_sam2_model
from utils.image import draw_label_no_overlap, create_processing_tiles, find_dist_transform_peaks, detect_hct_prompts
from utils.metrics import convert_pixels_to_micrometers, calculate_percentage, json_dump_safe
from utils.iou import calculate_binary_iou, calculate_box_iou
from utils.io import iter_chunks
from utils.histograms import (
    save_particle_distribution_histogram,
    save_sphericity_distribution_histogram,
)


CONST_PREPROCESS_WIDTH: int = 2048
CONST_PREPROCESS_HEIGHT: int = 1636
CONST_PREPROCESS_BOTTOM_CROP: int = 100
CONST_SCALE_REFERENCE_WIDTH: int = 1024  # presets.yaml scale_pixels 기준 해상도


class Sam2AspectRatioService:
    """SAM2 추론, 후처리, 결과 저장을 담당하는 서비스 클래스.

    Attributes:
        obj_config: 전체 파이프라인 설정을 담는 `Sam2AspectRatioConfig`.
        obj_model: 초기화 이후의 Ultralytics `SAM` 모델 인스턴스. 초기에는 `None`.
        dict_modelConfig: YAML 또는 대체 파싱 결과를 담는 메타데이터 dict.
    """

    def __init__(self, obj_config: Sam2AspectRatioConfig) -> None:
        """서비스 객체를 생성한다.

        Args:
            obj_config: 경로, 추론 파라미터, 후처리 파라미터를 포함한 설정 객체.
        """
        if obj_config.int_preprocessWidth != CONST_SCALE_REFERENCE_WIDTH:
            float_factor = obj_config.int_preprocessWidth / CONST_SCALE_REFERENCE_WIDTH
            obj_config = dataclasses_replace(
                obj_config,
                float_scalePixels=obj_config.float_scalePixels * float_factor,
            )
        if obj_config.float_scalePixels <= 0:
            print(
                f"[WARN] float_scalePixels={obj_config.float_scalePixels} ≤ 0: "
                "µm 환산이 불가능하여 모든 크기 지표가 0으로 출력됩니다. "
                "--scale_pixels 값을 확인하세요.",
                flush=True,
            )
        self.obj_config = obj_config
        self.obj_model: tp.Optional[tp.Any] = None
        self.dict_modelConfig: tp.Dict[str, tp.Any] = dict()

    # Maps long Hiera-style names → short Ultralytics-canonical names.
    # Used both by resolve_model_weights_path (file creation) and
    # _canonical_weights_name (pure name lookup, no filesystem side effects).
    _DICT_ALIAS_NAMES: tp.ClassVar[tp.Dict[str, str]] = {
        "sam2_hiera_tiny.pt": "sam2_t.pt",
        "sam2_hiera_small.pt": "sam2_s.pt",
        "sam2_hiera_base_plus.pt": "sam2_b.pt",
        "sam2_hiera_large.pt": "sam2_l.pt",
        "sam2.1_hiera_tiny.pt": "sam2.1_t.pt",
        "sam2.1_hiera_small.pt": "sam2.1_s.pt",
        "sam2.1_hiera_base_plus.pt": "sam2.1_b.pt",
        "sam2.1_hiera_large.pt": "sam2.1_l.pt",
    }

    # Short Ultralytics-canonical names that don't need aliasing.
    _SET_SUPPORTED_NAMES: tp.ClassVar[tp.FrozenSet[str]] = frozenset({
        "sam_h.pt", "sam_l.pt", "sam_b.pt", "mobile_sam.pt",
        "sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt",
        "sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt",
    })

    # Union: all names that Ultralytics can auto-download (skip existence check).
    _SET_AUTO_DOWNLOAD_NAMES: tp.ClassVar[tp.FrozenSet[str]] = (
        _SET_SUPPORTED_NAMES | frozenset(_DICT_ALIAS_NAMES.keys())
    )

    def validate_inputs(self) -> None:
        """필수 입력 경로들의 존재 여부를 검증한다.

        Raises:
            FileNotFoundError: 입력 이미지 또는 모델 설정 파일을 찾을 수 없을 때.
        """
        for path_item in [self.obj_config.path_input, self.obj_config.path_modelConfig]:
            if not path_item.exists():
                raise FileNotFoundError(f"필수 경로를 찾을 수 없습니다: {path_item}")

        # 모델 가중치: ultralytics 자동 다운로드 대상이면 파일 존재 체크 생략
        path_w = self.obj_config.path_modelWeights
        if path_w.name not in self._SET_AUTO_DOWNLOAD_NAMES and not path_w.exists():
            raise FileNotFoundError(f"모델 가중치를 찾을 수 없습니다: {path_w}")

    def load_model_config(self) -> None:
        """모델 설정 파일을 읽어 결과 메타데이터용 dict로 정리한다.

        Returns:
            없음. 파싱 결과는 `self.dict_modelConfig`에 저장된다.

        Notes:
            설정 파일이 정상 YAML dict이면 그대로 저장한다. YAML이 아니거나 HTML이
            들어있으면 파싱 상태와 일부 preview만 메타데이터로 남긴다.
        """
        str_rawText = self.obj_config.path_modelConfig.read_text(
            encoding="utf-8", errors="ignore")

        try:
            obj_loaded = yaml.safe_load(str_rawText)
        except yaml.YAMLError:
            obj_loaded = None

        if isinstance(obj_loaded, dict):
            self.dict_modelConfig = obj_loaded
            self.dict_modelConfig.setdefault("config_parse_status", "parsed")
            return

        str_parseStatus = "unparsed"
        if "<!DOCTYPE html>" in str_rawText[:256] or "<html" in str_rawText[:256].lower():
            str_parseStatus = "html_instead_of_yaml"

        self.dict_modelConfig = {
            "config_parse_status": str_parseStatus,
            "config_preview": str_rawText[:200].strip(),
        }

    def _canonical_weights_name(self) -> str:
        """Return the Ultralytics-canonical filename without touching the filesystem."""
        name = self.obj_config.path_modelWeights.name
        return self._DICT_ALIAS_NAMES.get(name, name)

    def resolve_model_weights_path(self) -> Path:
        """
        Ultralytics가 요구하는 SAM2 파일명 alias로 가중치 경로를 정규화.

        일부 체크포인트는 파일 내용은 정상이어도 파일명 규칙이 다르면 Ultralytics가
        지원 모델로 인식하지 못하므로, 필요한 경우 alias 파일을 생성한다.

        Returns:
            Ultralytics가 인식 가능한 파일명으로 정규화된 weight 파일 경로.

        Raises:
            FileNotFoundError: 현재 코드가 지원하지 않는 weight 파일명일 때 발생한다.
        """
        path_weights = self.obj_config.path_modelWeights

        if path_weights.name in self._SET_SUPPORTED_NAMES:
            return path_weights

        str_aliasName = self._DICT_ALIAS_NAMES.get(path_weights.name)
        if str_aliasName is None:
            raise FileNotFoundError(
                f"{path_weights} 는 현재 ultralytics가 인식하는 SAM2 체크포인트 이름이 아닙니다."
            )

        path_aliasDir = self.obj_config.path_outputDir / "_model_alias"
        path_aliasDir.mkdir(parents=True, exist_ok=True)
        path_alias = path_aliasDir / str_aliasName

        if path_alias.exists() and path_alias.stat().st_size == path_weights.stat().st_size:
            return path_alias

        if path_alias.exists():
            path_alias.unlink()

        try:
            os.link(path_weights, path_alias)
        except OSError:
            try:
                shutil.copy2(path_weights, path_alias)
            except OSError as exc:
                raise RuntimeError(
                    f"모델 가중치를 복사할 수 없습니다: {path_weights} → {path_alias}\n"
                    f"디스크 공간 또는 권한을 확인하세요. 원인: {exc}"
                ) from exc

        return path_alias

    def initialize_model(self) -> None:
        """입력 검증과 설정 로드를 거쳐 SAM2 모델을 초기화한다.

        Returns:
            없음. 초기화된 모델은 `self.obj_model`에 저장된다.
        """
        self.validate_inputs()
        self.load_model_config()
        path_resolvedWeights = self.resolve_model_weights_path()
        self.obj_model = load_sam2_model(str(path_resolvedWeights))

    def load_image_bgr(self) -> np.ndarray:
        """입력 이미지를 OpenCV BGR 형식으로 로드한다.

        Returns:
            shape `(H, W, 3)`의 BGR `np.ndarray`.

        Raises:
            FileNotFoundError: 이미지를 읽을 수 없을 때 발생한다.
        """
        arr_image = cv2.imread(
            str(self.obj_config.path_input), cv2.IMREAD_COLOR)
        if arr_image is None:
            raise FileNotFoundError(
                f"이미지를 읽을 수 없습니다: {self.obj_config.path_input}")
        int_w = max(1, self.obj_config.int_preprocessWidth)
        int_h_raw = max(1, round(int_w * CONST_PREPROCESS_HEIGHT / CONST_PREPROCESS_WIDTH))
        int_crop = round(int_w * CONST_PREPROCESS_BOTTOM_CROP / CONST_PREPROCESS_WIDTH)
        int_h_final = max(1, int_h_raw - int_crop)
        if arr_image.shape[:2] != (int_h_raw, int_w):
            arr_image = cv2.resize(arr_image, (int_w, int_h_raw), interpolation=cv2.INTER_LINEAR)
        return arr_image[:int_h_final, :]

    def extract_inference_roi(
        self,
        arr_imageBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Dict[str, int]]:
        """전체 이미지에서 실제 추론 대상 ROI를 crop한다.

        Args:
            arr_imageBgr: 원본 BGR 이미지 배열.

        Returns:
            `(arr_roiBgr, dict_roi)` 튜플.
            - `arr_roiBgr`: ROI 영역만 잘라낸 BGR 이미지.
            - `dict_roi`: `x_min`, `y_min`, `x_max`, `y_max`, `width`, `height`
              키를 가지는 ROI 메타데이터 dict.

        Raises:
            ValueError: ROI 설정이 이미지 범위와 교차하지 않아 유효한 crop을 만들 수
                없을 때 발생한다.
        """
        int_h, int_w = arr_imageBgr.shape[:2]

        int_x0 = max(0, min(self.obj_config.int_roiXMin, int_w))
        int_y0 = max(0, min(self.obj_config.int_roiYMin, int_h))
        int_x1 = max(int_x0, min(self.obj_config.int_roiXMax, int_w))
        int_y1 = max(int_y0, min(self.obj_config.int_roiYMax, int_h))

        if int_x1 <= int_x0 or int_y1 <= int_y0:
            raise ValueError(
                "유효한 ROI를 만들 수 없습니다. ROI 좌표와 입력 이미지 크기를 확인하세요."
            )

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

    def predict_tiled_point_prompts(
        self,
        arr_inputBgr: np.ndarray,
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray], tp.Dict[str, tp.Any]]:
        """ROI 이미지를 타일 단위로 SAM2 추론한다.

        Args:
            arr_inputBgr: ROI crop 이후의 BGR 이미지. shape은 일반적으로 `(H, W, 3)`.

        Returns:
            `(arr_masks, arr_scores, dict_debug)` 튜플.
            - `arr_masks`: ROI 좌표계 기준 binary mask 배열. shape은
              `(N, H, W)`이며, mask가 없으면 빈 배열이다.
            - `arr_scores`: 각 mask의 confidence score 배열. score가 없으면 `None`.
            - `dict_debug`: tile 수, 후보점 수, 중복 제거 수, 각 tile/point 정보를 담는
              디버그 dict.

        Notes:
            `bool_usePointPrompts=True`이면 OpenCV 후보점을 추출해 batch point prompt로
            SAM2를 호출하고, `False`이면 tile 전체에 대해 자동 분할을 수행한다.
        """
        if self.obj_model is None:
            self.initialize_model()

        int_roiHeight, int_roiWidth = arr_inputBgr.shape[:2]
        arr_inputGray = cv2.cvtColor(arr_inputBgr, cv2.COLOR_BGR2GRAY)
        list_tiles = create_processing_tiles(
            0,
            0,
            int_roiWidth,
            int_roiHeight,
            int_tileSize=self.obj_config.int_tileSize,
            int_stride=self.obj_config.int_stride,
        )

        dict_predictCommon: tp.Dict[str, tp.Any] = {
            "imgsz": self.obj_config.int_imgSize,
            "retina_masks": self.obj_config.bool_retinaMasks,
            "verbose": False,
        }
        if self.obj_config.str_device:
            dict_predictCommon["device"] = self.obj_config.str_device

        list_keptMasks = list()
        list_keptScores = list()
        list_keptBboxes: tp.List[tp.Tuple[int, int, int, int]] = list()
        list_debugTiles = list()
        list_debugPoints = list()
        list_debugHctCircles: tp.List[tp.Dict[str, tp.Any]] = list()
        list_debugCcContours: tp.List[np.ndarray] = list()
        int_candidateCount = 0
        int_acceptedCount = 0
        int_bboxDedupRejected = 0

        for int_tileIdx, (int_tx1, int_ty1, int_tx2, int_ty2) in enumerate(list_tiles):
            arr_tileBgr = arr_inputBgr[int_ty1:int_ty2, int_tx1:int_tx2].copy()
            list_posPoints: tp.List[tp.Tuple[int, int]] = []
            list_negPoints: tp.List[tp.Tuple[int, int]] = []

            if self.obj_config.bool_usePointPrompts:
                arr_tileGray = arr_inputGray[int_ty1:int_ty2, int_tx1:int_tx2].copy()
                list_isolatedMasks: tp.List[np.ndarray] = []
                try:
                    list_isolatedMasks, list_posPoints, list_negPoints, dict_hctInfo = detect_hct_prompts(
                        arr_tileGray=arr_tileGray,
                        int_minDist=self.obj_config.int_pointMinDistance,
                        int_numNeg=self.obj_config.int_numNegativePoints,
                        int_minArea=int(self.obj_config.float_particleAreaThreshold),
                    )
                    for (int_cx, int_cy, int_cr) in dict_hctInfo["hct_circles"]:
                        list_debugHctCircles.append({
                            "center_roi": [int_tx1 + int_cx, int_ty1 + int_cy],
                            "radius": int_cr,
                        })
                    int_numHctPos = dict_hctInfo["num_hct_pos"]
                    for cnt in dict_hctInfo["cc_contours"]:
                        cnt_roi = cnt.copy()
                        cnt_roi[:, 0, 0] += int_tx1
                        cnt_roi[:, 0, 1] += int_ty1
                        list_debugCcContours.append(cnt_roi)
                except Exception as exc:
                    print(f"[WARN] tile {int_tileIdx} 포인트 추출 실패 (skip): {exc}", flush=True)

                # ── isolated 마스크: OpenCV 직접 수용 ──────────────────────────
                for arr_tileMask in list_isolatedMasks:
                    if int(arr_tileMask.sum()) < self.obj_config.int_minValidMaskArea:
                        continue
                    arr_tileContour = self.extract_largest_contour(arr_tileMask)
                    if arr_tileContour is None:
                        continue
                    int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_tileContour)
                    int_tileH, int_tileW = arr_tileMask.shape[:2]
                    if self.is_bbox_near_edge(int_bx, int_by, int_bw, int_bh,
                                              int_tileW, int_tileH,
                                              self.obj_config.int_tileEdgeMargin):
                        continue
                    tuple_globalBox = (int_tx1 + int_bx, int_ty1 + int_by, int_bw, int_bh)
                    if any(calculate_box_iou(tuple_globalBox, b) >= self.obj_config.float_bboxDedupIou
                           for b in list_keptBboxes):
                        int_bboxDedupRejected += 1
                        continue
                    arr_roiMask = np.zeros((int_roiHeight, int_roiWidth), dtype=np.uint8)
                    arr_roiMask[int_ty1:int_ty2, int_tx1:int_tx2] = arr_tileMask
                    if any(calculate_binary_iou(arr_roiMask, m) >= self.obj_config.float_dedupIou
                           for m in list_keptMasks):
                        continue
                    int_acceptedCount += 1
                    list_keptMasks.append(arr_roiMask)
                    list_keptBboxes.append(tuple_globalBox)
                    list_keptScores.append(None)

                for int_posIdx, (int_px, int_py) in enumerate(list_posPoints):
                    int_candidateCount += 1
                    list_debugPoints.append(
                        {
                            "tile_index": int_tileIdx,
                            "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                            "point_xy_tile": [int(int_px), int(int_py)],
                            "point_xy_roi": [int_tx1 + int(int_px), int_ty1 + int(int_py)],
                            "label": 1,
                            "source": "hct" if int_posIdx < int_numHctPos else "cc",
                        }
                    )
                for int_px, int_py in list_negPoints:
                    list_debugPoints.append(
                        {
                            "tile_index": int_tileIdx,
                            "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                            "point_xy_tile": [int(int_px), int(int_py)],
                            "point_xy_roi": [int_tx1 + int(int_px), int_ty1 + int(int_py)],
                            "label": 0,
                        }
                    )

            list_debugTiles.append(
                {
                    "tile_index": int_tileIdx,
                    "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                    "num_points": len(list_posPoints),
                    "num_negative_points": len(list_negPoints),
                    "use_point_prompts": bool(self.obj_config.bool_usePointPrompts),
                }
            )

            # 포지티브 포인트를 배치로 묶어 SAM2 호출 (1배치 = N마스크)
            list_promptBatches = (
                list(iter_chunks(list_posPoints, max(1, self.obj_config.int_pointBatchSize)))
                if self.obj_config.bool_usePointPrompts else [None]
            )

            for list_batch in list_promptBatches:
                try:
                    if self.obj_config.bool_usePointPrompts:
                        list_results = self.obj_model(  # type: ignore[misc]
                            source=arr_tileBgr,
                            points=[[int(px), int(py)] for px, py in list_batch],
                            labels=[1] * len(list_batch),
                            **dict_predictCommon,
                        )
                    else:
                        list_results = self.obj_model(  # type: ignore[misc]
                            source=arr_tileBgr,
                            **dict_predictCommon,
                        )
                except Exception as exc:
                    print(f"[WARN] tile {int_tileIdx} 추론 실패 (skip): {exc}", flush=True)
                    continue
                if not list_results:
                    continue

                obj_result = list_results[0]
                if obj_result.masks is None or obj_result.masks.data is None:
                    continue

                arr_tileMasks = obj_result.masks.data.detach().cpu().numpy()
                arr_tileScores = None
                if obj_result.boxes is not None and obj_result.boxes.conf is not None:
                    arr_tileScores = obj_result.boxes.conf.detach().cpu().numpy()

                for int_maskIdx, arr_tm in enumerate(arr_tileMasks):
                    arr_tileMask = (
                        arr_tm > self.obj_config.float_maskBinarizeThreshold).astype(np.uint8)
                    if int(arr_tileMask.sum()) < self.obj_config.int_minValidMaskArea:
                        continue

                    arr_tileContour = self.extract_largest_contour(
                        arr_tileMask)
                    if arr_tileContour is None:
                        continue
                    int_bx, int_by, int_bw, int_bh = cv2.boundingRect(
                        arr_tileContour)
                    int_tileHeight, int_tileWidth = arr_tileMask.shape[:2]
                    if self.is_bbox_near_edge(
                        int_x=int_bx,
                        int_y=int_by,
                        int_w=int_bw,
                        int_h=int_bh,
                        int_width=int_tileWidth,
                        int_height=int_tileHeight,
                        int_margin=self.obj_config.int_tileEdgeMargin,
                    ):
                        continue

                    tuple_globalBox = (
                        int_tx1 + int_bx,
                        int_ty1 + int_by,
                        int_bw,
                        int_bh,
                    )
                    bool_bboxDup = False
                    for tuple_prevBox in list_keptBboxes:
                        if calculate_box_iou(tuple_prevBox, tuple_globalBox) >= self.obj_config.float_bboxDedupIou:
                            bool_bboxDup = True
                            break
                    if bool_bboxDup:
                        int_bboxDedupRejected += 1
                        continue

                    arr_roiMask = np.zeros(
                        (int_roiHeight, int_roiWidth), dtype=np.uint8)
                    arr_roiMask[int_ty1:int_ty2,
                                int_tx1:int_tx2] = arr_tileMask

                    bool_isDup = False
                    float_new_area = float(arr_roiMask.sum())
                    for arr_prevMask in list_keptMasks:
                        if calculate_binary_iou(arr_prevMask, arr_roiMask) >= self.obj_config.float_dedupIou:
                            bool_isDup = True
                            break
                        # 포함 관계 체크: 작은 마스크가 큰 마스크에 75%+ 포함되면 제거
                        float_inter = float((arr_prevMask & arr_roiMask).sum())
                        if float_inter > 0:
                            float_small_area = min(float(arr_prevMask.sum()), float_new_area)
                            if float_inter / max(float_small_area, 1.0) >= 0.75:
                                bool_isDup = True
                                break
                    if bool_isDup:
                        continue

                    int_acceptedCount += 1
                    list_keptMasks.append(arr_roiMask)
                    list_keptBboxes.append(tuple_globalBox)
                    if arr_tileScores is not None and int_maskIdx < len(arr_tileScores):
                        float_s = float(arr_tileScores[int_maskIdx])
                        list_keptScores.append(None if math.isnan(float_s) else float_s)
                    else:
                        list_keptScores.append(None)

        arr_masks = (
            np.stack(list_keptMasks, axis=0).astype(np.uint8)
            if list_keptMasks
            else np.empty((0, int_roiHeight, int_roiWidth), dtype=np.uint8)
        )
        arr_scores = None
        if list_keptScores:
            arr_scores = np.array(
                [np.nan if x is None else float(x) for x in list_keptScores],
                dtype=np.float32,
            )
        dict_debug = {
            "num_tiles": len(list_tiles),
            "num_candidate_points": int_candidateCount,
            "num_accepted_masks": int_acceptedCount,
            "num_bbox_dedup_rejected": int_bboxDedupRejected,
            "tiles": list_debugTiles,
            "candidate_points": list_debugPoints,
            "hct_circles": list_debugHctCircles,
            "cc_contours": list_debugCcContours,
        }
        return arr_masks, arr_scores, dict_debug

    # ── Post-processing helpers ────────────────────────────────────────────────

    @staticmethod
    def _smooth_mask(arr_mask: np.ndarray) -> np.ndarray:
        """마스크 노이즈 제거: 모폴로지 스무딩 + 최대 연결 컴포넌트만 유지.

        리아스식 해안 같은 들쭉날쭉한 경계를 정리하고, 한 마스크 안에 흩어진
        작은 조각을 제거해 하나의 연결된 덩어리로 만든다.
        """
        arr_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        arr_m = cv2.morphologyEx(arr_mask, cv2.MORPH_CLOSE, arr_k, iterations=2)
        arr_m = cv2.morphologyEx(arr_m, cv2.MORPH_OPEN, arr_k, iterations=1)
        # 최대 connected component만 유지
        list_cnts, _ = cv2.findContours(arr_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_cnts:
            return arr_mask
        arr_out = np.zeros_like(arr_m)
        cv2.drawContours(arr_out, [max(list_cnts, key=cv2.contourArea)], 0, 1, -1)
        return arr_out

    @staticmethod
    def _fit_particle_circle(arr_mask: np.ndarray) -> tp.Optional[tp.Tuple[float, float, float]]:
        """가시 영역 컨투어의 convex hull에 Kasa 원 피팅을 수행한다.

        여러 개의 노치가 있어도 hull 전체를 사용하므로 안정적.
        Returns (cx, cy, r) or None if fitting fails / sanity check fails.
        """
        list_cnts, _ = cv2.findContours(arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not list_cnts:
            return None
        arr_cnt = max(list_cnts, key=cv2.contourArea)
        float_area = float(cv2.contourArea(arr_cnt))
        if float_area == 0:
            return None

        arr_hull = cv2.convexHull(arr_cnt)
        float_hull_area = float(cv2.contourArea(arr_hull))
        float_solidity = float_area / max(float_hull_area, 1.0)
        # solidity < 0.50: 심하게 불규칙(fragment) → skip
        # 상한은 두지 않음: gain > 1.02 체크가 자연 상한 역할
        # (solidity > 1/1.02 ≈ 0.98 이면 gain < 1.02 → 어차피 복원 안 됨)
        if float_solidity < 0.50:
            return None

        # Kasa 최소제곱 원 피팅 (hull 포인트만 사용)
        arr_pts = arr_hull.reshape(-1, 2).astype(np.float64)
        arr_z = arr_pts[:, 0] ** 2 + arr_pts[:, 1] ** 2
        arr_A = np.column_stack([2.0 * arr_pts[:, 0], 2.0 * arr_pts[:, 1], np.ones(len(arr_pts))])
        arr_res, _, _, _ = np.linalg.lstsq(arr_A, arr_z, rcond=None)
        float_cx = float(arr_res[0])
        float_cy = float(arr_res[1])
        float_r2 = float(arr_res[2]) + float_cx ** 2 + float_cy ** 2
        if float_r2 <= 0:
            return None
        float_r = float(np.sqrt(float_r2))

        float_r_exp = float(np.sqrt(float_area / np.pi))
        if float_r < float_r_exp * 0.7 or float_r > float_r_exp * 1.8:
            return None
        if np.pi * float_r ** 2 > float_area * 4.0:
            return None

        # 원형도 품질 체크: hull 포인트들이 피팅된 원에 얼마나 잘 맞는가
        # 타원형 입자의 경우 각 포인트의 원 중심까지 거리 편차가 크다
        arr_dists = np.sqrt((arr_pts[:, 0] - float_cx) ** 2 + (arr_pts[:, 1] - float_cy) ** 2)
        float_cv = float(np.std(arr_dists)) / max(float_r, 1.0)  # 변동계수
        if float_cv > 0.10:  # 10% 이상 편차 → 원이 아님(타원 등) → 복원 불필요
            return None

        return float_cx, float_cy, float_r

    @staticmethod
    def _hull_mask(arr_mask: np.ndarray) -> np.ndarray:
        """particle 마스크의 convex hull을 채워서 반환한다."""
        list_cnts, _ = cv2.findContours(arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_cnts:
            return arr_mask
        arr_cnt = max(list_cnts, key=cv2.contourArea)
        arr_hull = cv2.convexHull(arr_cnt)
        arr_out = np.zeros_like(arr_mask)
        cv2.fillPoly(arr_out, [arr_hull], 1)
        return arr_out


    @staticmethod
    def _split_peanut_mask(
        arr_mask: np.ndarray,
        float_ar_thresh: float = 0.60,
        int_min_peak_dist: int = 8,
    ) -> tp.List[np.ndarray]:
        """땅콩/덤벨 형태의 마스크를 두 개의 마스크로 분리한다.

        minAreaRect 종횡비가 float_ar_thresh 미만이고 거리변환에 두 개의 독립적인
        피크가 존재할 때만 분리하며, 그 외에는 [원본]을 반환한다.
        """
        list_cnts, _ = cv2.findContours(arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_cnts:
            return [arr_mask]
        arr_cnt = max(list_cnts, key=cv2.contourArea)
        _, (float_w, float_h), _ = cv2.minAreaRect(arr_cnt)
        if float_w == 0 or float_h == 0:
            return [arr_mask]
        float_ar = min(float_w, float_h) / max(float_w, float_h)
        if float_ar >= float_ar_thresh:
            return [arr_mask]

        list_peaks = find_dist_transform_peaks(arr_mask, int_min_peak_dist, int_max_peaks=2)
        if len(list_peaks) < 2:
            return [arr_mask]

        int_h, int_w = arr_mask.shape[:2]
        arr_markers = np.zeros((int_h, int_w), dtype=np.int32)
        arr_markers[arr_mask == 0] = 1
        arr_markers[list_peaks[0][1], list_peaks[0][0]] = 2
        arr_markers[list_peaks[1][1], list_peaks[1][0]] = 3
        arr_bgr = cv2.cvtColor((arr_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.watershed(arr_bgr, arr_markers)

        arr_m1 = ((arr_markers == 2) & (arr_mask > 0)).astype(np.uint8)
        arr_m2 = ((arr_markers == 3) & (arr_mask > 0)).astype(np.uint8)
        if arr_m1.sum() == 0 or arr_m2.sum() == 0:
            return [arr_mask]
        return [arr_m1, arr_m2]

    def _postprocess_masks(
        self,
        arr_masks: np.ndarray,
        arr_scores: tp.Optional[np.ndarray],
    ) -> tp.Tuple[np.ndarray, tp.Optional[np.ndarray]]:
        """마스크 후처리: 스무딩 → 땅콩 분리.

        ⑤ 마스크 스무딩 + 최대 컴포넌트 유지 (리아스식 경계 제거)
        ③ 땅콩 분리
        ② 가려진 입자 보정은 분류 후 particle에만 적용 (process()에서 수행)
        """
        list_out_masks: tp.List[np.ndarray] = []
        list_out_scores: tp.List[tp.Optional[float]] = []

        for int_i, arr_mask in enumerate(arr_masks):
            float_score = None
            if arr_scores is not None and int_i < len(arr_scores):
                float_score = None if math.isnan(float(arr_scores[int_i])) else float(arr_scores[int_i])

            # ⑤ 스무딩: 노이즈 제거 + 최대 컴포넌트
            arr_mask = self._smooth_mask(arr_mask)
            if arr_mask.sum() == 0:
                continue

            # ③ 땅콩 분리
            list_split = self._split_peanut_mask(arr_mask)

            for arr_part in list_split:
                list_out_masks.append(arr_part)
                list_out_scores.append(float_score)

        if not list_out_masks:
            int_h, int_w = (arr_masks.shape[1], arr_masks.shape[2]) if arr_masks.ndim == 3 else (0, 0)
            return np.empty((0, int_h, int_w), dtype=np.uint8), None

        arr_out = np.stack(list_out_masks, axis=0).astype(np.uint8)
        arr_out_scores = np.array(
            [np.nan if s is None else s for s in list_out_scores], dtype=np.float32
        ) if list_out_scores else None
        return arr_out, arr_out_scores

    def refine_mask_for_area(self, arr_mask: np.ndarray) -> np.ndarray:
        """
        면적 계산 전 binary mask를 후처리한다.

        Args:
            arr_mask: 입력 binary 또는 binary-like mask. 0 초과값을 foreground로 본다.

        Returns:
            morphology가 적용된 `uint8` binary mask.

        Notes:
            area threshold 자체뿐 아니라 이 함수의 morphology 설정도 최종 area 값에
            직접 영향을 준다.
        """
        arr_maskUint8 = (arr_mask > 0).astype(np.uint8)

        int_kernelSize = self.obj_config.int_maskMorphKernelSize
        if int_kernelSize <= 1:
            return arr_maskUint8

        arr_kernel = np.ones((int_kernelSize, int_kernelSize), dtype=np.uint8)
        arr_refined = arr_maskUint8

        if self.obj_config.int_maskMorphOpenIterations > 0:
            arr_refined = cv2.morphologyEx(
                arr_refined,
                cv2.MORPH_OPEN,
                arr_kernel,
                iterations=self.obj_config.int_maskMorphOpenIterations,
            )

        if self.obj_config.int_maskMorphCloseIterations > 0:
            arr_refined = cv2.morphologyEx(
                arr_refined,
                cv2.MORPH_CLOSE,
                arr_kernel,
                iterations=self.obj_config.int_maskMorphCloseIterations,
            )

        return arr_refined

    @staticmethod
    def get_longest_span(arr_mask: np.ndarray, bool_horizontal: bool) -> int:
        """마스크 내부의 가장 긴 가로/세로 span 길이를 계산한다.

        Args:
            arr_mask: 2차원 binary mask.
            bool_horizontal: `True`이면 가로 방향 span, `False`이면 세로 방향 span을
                계산한다.

        Returns:
            foreground 픽셀이 존재하는 한 줄에서의 최대 span 길이(pixel).
        """
        arr_scan = arr_mask if bool_horizontal else arr_mask.T
        int_longest = 0
        for arr_line in arr_scan:
            arr_indices = np.flatnonzero(arr_line)
            if arr_indices.size == 0:
                continue
            int_span = int(arr_indices[-1] - arr_indices[0] + 1)
            if int_span > int_longest:
                int_longest = int_span
        return int_longest

    @staticmethod
    def extract_largest_contour(arr_mask: np.ndarray) -> tp.Optional[np.ndarray]:
        """외곽 contour 중 면적이 가장 큰 contour를 반환한다.

        Args:
            arr_mask: contour를 찾을 binary mask.

        Returns:
            가장 큰 contour의 OpenCV contour 배열. contour가 없으면 `None`.
        """
        list_contours, _ = cv2.findContours(
            arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not list_contours:
            return None
        return max(list_contours, key=cv2.contourArea)

    @staticmethod
    def is_bbox_near_edge(
        int_x: int,
        int_y: int,
        int_w: int,
        int_h: int,
        int_width: int,
        int_height: int,
        int_margin: int,
    ) -> bool:
        """bbox가 영역 경계에 너무 가까운지 판정한다.

        Args:
            int_x: bbox 좌상단 x.
            int_y: bbox 좌상단 y.
            int_w: bbox width.
            int_h: bbox height.
            int_width: bbox가 놓인 기준 영역의 전체 width.
            int_height: bbox가 놓인 기준 영역의 전체 height.
            int_margin: 경계와의 최소 허용 거리.

        Returns:
            bbox의 어느 한 변이라도 기준 영역 경계에서 `int_margin` 이내면 `True`.
        """
        int_margin = max(0, int_margin)
        int_right = int_x + int_w
        int_bottom = int_y + int_h
        return (
            int_x <= int_margin
            or int_y <= int_margin
            or int_right >= (int_width - int_margin)
            or int_bottom >= (int_height - int_margin)
        )

    def is_bbox_near_roi_edge(
        self,
        int_x: int,
        int_y: int,
        int_w: int,
        int_h: int,
        int_roiWidth: int,
        int_roiHeight: int,
    ) -> bool:
        """bbox가 ROI 경계에 너무 가까운지 판정한다.

        Args:
            int_x: ROI 좌표계 기준 bbox x.
            int_y: ROI 좌표계 기준 bbox y.
            int_w: bbox width.
            int_h: bbox height.
            int_roiWidth: ROI 전체 width.
            int_roiHeight: ROI 전체 height.

        Returns:
            ROI 경계와의 거리가 `obj_config.int_bboxEdgeMargin` 이내이면 `True`.
        """
        return self.is_bbox_near_edge(
            int_x=int_x,
            int_y=int_y,
            int_w=int_w,
            int_h=int_h,
            int_width=int_roiWidth,
            int_height=int_roiHeight,
            int_margin=self.obj_config.int_bboxEdgeMargin,
        )

    def convert_pixels_to_micrometers(self, float_pixels: float) -> float:
        """현재 config의 scale 기준으로 픽셀 길이를 um로 환산한다.

        Args:
            float_pixels: pixel 단위 길이 값.

        Returns:
            현재 설정된 scale(`float_scalePixels`, `float_scaleMicrometers`) 기준의
            micrometer 길이.
        """
        return convert_pixels_to_micrometers(
            float_pixels=float_pixels,
            float_scalePixels=self.obj_config.float_scalePixels,
            float_scaleMicrometers=self.obj_config.float_scaleMicrometers,
        )

    def measure_mask(
        self,
        arr_mask: np.ndarray,
        int_index: int,
        float_confidence: tp.Optional[float],
        bool_convexHullSphericity: bool = False,
    ) -> tp.Optional[ObjectMeasurement]:
        """단일 mask의 측정값을 계산해 `ObjectMeasurement`로 변환한다.

        Args:
            arr_mask: ROI 좌표계 기준 binary mask.
            int_index: 현재 mask의 인덱스. 결과 식별용으로 그대로 저장된다.
            float_confidence: SAM2가 제공한 confidence score. 없으면 `None`.

        Returns:
            유효한 객체이면 `ObjectMeasurement`, 너무 작거나 contour가 없거나 ROI 경계에
            너무 가까우면 `None`.
        """
        arr_refinedMask = self.refine_mask_for_area(arr_mask)
        int_maskArea = int(arr_refinedMask.sum())
        if int_maskArea < self.obj_config.int_minValidMaskArea:
            return None

        arr_contour = self.extract_largest_contour(arr_refinedMask)
        if arr_contour is None:
            return None

        int_x, int_y, int_w, int_h = cv2.boundingRect(arr_contour)
        int_roiHeight, int_roiWidth = arr_refinedMask.shape[:2]
        if self.is_bbox_near_roi_edge(int_x, int_y, int_w, int_h, int_roiWidth, int_roiHeight):
            return None

        obj_moments = cv2.moments(arr_contour)
        if obj_moments["m00"] > 0.0:
            float_cx = float(obj_moments["m10"] / obj_moments["m00"])
            float_cy = float(obj_moments["m01"] / obj_moments["m00"])
        else:
            float_cx = float(int_x + int_w / 2.0)
            float_cy = float(int_y + int_h / 2.0)

        int_horizontal = min(self.get_longest_span(arr_refinedMask, bool_horizontal=True), int_w)
        int_vertical = min(self.get_longest_span(arr_refinedMask, bool_horizontal=False), int_h)

        str_category = (
            "particle"
            if int_maskArea >= int(round(self.obj_config.float_particleAreaThreshold))
            else "fragment"
        )

        float_eqDiameterPx = 2.0 * math.sqrt(int_maskArea / math.pi)
        float_eqDiameterUm = self.convert_pixels_to_micrometers(float_eqDiameterPx)

        float_sphericity = None
        float_sphericity_prime = None
        if str_category == "particle":
            # S: 4πA/P² — A와 P를 같은 볼록 폴리곤에서 계산해야
            # 이소페리메트릭 부등식(S≤1)이 항상 성립.
            # convexHull로 계단 노이즈를 제거하고 A/P 모두 hull 폴리곤 기준으로 통일.
            arr_hull_cnt = cv2.convexHull(arr_contour)
            float_hull_area = float(cv2.contourArea(arr_hull_cnt))
            float_perimeter = float(cv2.arcLength(arr_hull_cnt, closed=True))
            if float_perimeter > 0.0 and float_hull_area > 0.0:
                float_sphericity = float(
                    (4.0 * np.pi * float_hull_area) / (float_perimeter ** 2)
                )

            # S': fitEllipse 기반 b/a — 장단축비 (aspect ratio)
            if len(arr_contour) >= 5:
                _, (float_ew, float_eh), _ = cv2.fitEllipse(arr_contour)
                float_fit_a = max(float_ew, float_eh) / 2.0
                float_fit_b = min(float_ew, float_eh) / 2.0
                if float_fit_a > 0:
                    float_sphericity_prime = min(1.0, float_fit_b / float_fit_a)

        return ObjectMeasurement(
            int_index=int_index,
            str_category=str_category,
            int_maskArea=int_maskArea,
            float_confidence=float_confidence,
            int_bboxX=int(int_x),
            int_bboxY=int(int_y),
            int_bboxWidth=int(int_w),
            int_bboxHeight=int(int_h),
            float_bboxWidthUm=self.convert_pixels_to_micrometers(float(int_w)),
            float_bboxHeightUm=self.convert_pixels_to_micrometers(float(int_h)),
            float_centroidX=float_cx,
            float_centroidY=float_cy,
            int_longestHorizontal=int_horizontal,
            int_longestVertical=int_vertical,
            float_longestHorizontalUm=self.convert_pixels_to_micrometers(float(int_horizontal)),
            float_longestVerticalUm=self.convert_pixels_to_micrometers(float(int_vertical)),
            float_eqDiameterUm=float_eqDiameterUm,
            float_sphericity=float_sphericity,
            float_sphericity_prime=float_sphericity_prime,
        )

    def create_overlay(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
        str_show: str = "both",  # "both" | "S" | "Sp"
    ) -> np.ndarray:
        """객체 마스크와 라벨을 원본 ROI 이미지 위에 시각화한다.

        Args:
            arr_imageBgr: ROI 이미지.
            list_objects: 시각화할 객체 측정 결과 리스트.
            list_masks: 각 객체에 대응하는 ROI 좌표계 binary mask 리스트.

        Returns:
            mask overlay, contour, bbox, 텍스트 라벨이 그려진 BGR 이미지.
        """
        int_h, int_w = arr_imageBgr.shape[:2]
        arr_overlay = cv2.resize(arr_imageBgr, (int_w * 2, int_h * 2), interpolation=cv2.INTER_LINEAR)

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            arr_mask2 = cv2.resize(arr_mask, (int_w * 2, int_h * 2), interpolation=cv2.INTER_NEAREST)
            tpl_color = (60, 220, 60) if obj_measurement.str_category == "particle" else (0, 165, 255)
            arr_overlay[arr_mask2 > 0] = (
                arr_overlay[arr_mask2 > 0].astype(np.float32) * 0.55
                + np.array(tpl_color, dtype=np.float32) * 0.45
            ).astype(np.uint8)

        list_placedRects: tp.List[tp.Tuple[int, int, int, int]] = []
        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            arr_contour = self.extract_largest_contour(arr_mask)
            if arr_contour is None:
                continue

            arr_contour2 = (arr_contour * 2).astype(np.int32)
            tpl_color = (0, 255, 0) if obj_measurement.str_category == "particle" else (0, 140, 255)
            cv2.drawContours(arr_overlay, [arr_contour2], -1, tpl_color, 1)

            int_cx2 = int(round(obj_measurement.float_centroidX * 2))
            int_cy2 = int(round(obj_measurement.float_centroidY * 2))

            if obj_measurement.str_category == "particle":
                list_lines = []
                if obj_measurement.float_eqDiameterUm is not None:
                    list_lines.append(f"d={obj_measurement.float_eqDiameterUm:.1f}µm")
                if str_show in ("both", "S") and obj_measurement.float_sphericity is not None:
                    list_lines.append(f"S={obj_measurement.float_sphericity:.2f}")
                if str_show in ("both", "Sp") and obj_measurement.float_sphericity_prime is not None:
                    list_lines.append(f"S'={obj_measurement.float_sphericity_prime:.2f}")
                if list_lines:
                    draw_label_no_overlap(
                        arr_overlay, list_lines, int_cx2, int_cy2, tpl_color, list_placedRects)

        return arr_overlay

    def draw_eq_circles_clean(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
    ) -> np.ndarray:
        """마스크 + 등가원만 그린 클린 이미지 (레이블 없음)."""
        int_h, int_w = arr_imageBgr.shape[:2]
        arr_out = cv2.resize(arr_imageBgr, (int_w * 2, int_h * 2), interpolation=cv2.INTER_LINEAR)
        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            arr_mask2 = cv2.resize(arr_mask, (int_w * 2, int_h * 2), interpolation=cv2.INTER_NEAREST)
            tpl_fill = (60, 220, 60) if obj_measurement.str_category == "particle" else (0, 165, 255)
            arr_out[arr_mask2 > 0] = (
                arr_out[arr_mask2 > 0].astype(np.float32) * 0.55
                + np.array(tpl_fill, dtype=np.float32) * 0.45
            ).astype(np.uint8)
        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            arr_cnt = self.extract_largest_contour(arr_mask)
            if arr_cnt is None:
                continue
            tpl_color = (0, 255, 0) if obj_measurement.str_category == "particle" else (0, 140, 255)
            cv2.drawContours(arr_out, [(arr_cnt * 2).astype(np.int32)], -1, tpl_color, 1)
        return arr_out

    def build_summary(self, list_objects: tp.List[ObjectMeasurement]) -> tp.Dict[str, tp.Any]:
        """단일 이미지 처리 결과에 대한 요약 통계를 생성한다.

        Args:
            list_objects: 유효성 검사를 통과한 객체 측정 결과 리스트.

        Returns:
            JSON 저장에 바로 사용할 수 있는 요약 dict. 설정값, 집계 개수, 비율,
            종횡비 통계가 포함된다.
        """
        list_particles = [
            obj_item for obj_item in list_objects if obj_item.str_category == "particle"]
        list_fragments = [
            obj_item for obj_item in list_objects if obj_item.str_category == "fragment"]
        int_totalObjects = len(list_objects)
        int_particleCount = len(list_particles)
        int_fragmentCount = len(list_fragments)
        list_particleSphs = [
            obj_item.float_sphericity
            for obj_item in list_particles
            if obj_item.float_sphericity is not None
        ]
        list_particleSphs_prime = [
            obj_item.float_sphericity_prime
            for obj_item in list_particles
            if obj_item.float_sphericity_prime is not None
        ]
        if self.obj_config.bool_useEqDiameter:
            list_particleSizes = [obj_item.float_eqDiameterUm for obj_item in list_particles]
        else:
            list_particleSizes = [
                (obj_item.float_longestHorizontalUm + obj_item.float_longestVerticalUm) / 2.0
                for obj_item in list_particles
            ]
        float_micrometersPerPixel = convert_pixels_to_micrometers(
            float_pixels=1.0,
            float_scalePixels=self.obj_config.float_scalePixels,
            float_scaleMicrometers=self.obj_config.float_scaleMicrometers,
        )

        dict_summary: tp.Dict[str, tp.Any] = {
            "input_path": str(self.obj_config.path_input),
            "output_dir": str(self.obj_config.path_outputDir),
            "model_config_path": str(self.obj_config.path_modelConfig),
            "model_config_parse_status": self.dict_modelConfig.get("config_parse_status"),
            "model_weights_path": str(self.obj_config.path_modelWeights),
            "model_weights_resolved_name": (
                None if self.obj_config.bool_useOpenCV
                else self._canonical_weights_name()
            ),
            "model_name": (
                "opencv" if self.obj_config.bool_useOpenCV
                else self.dict_modelConfig.get("model", self.obj_config.path_modelWeights.stem)
            ),
            "scale_pixels": float(self.obj_config.float_scalePixels),
            "scale_micrometers": float(self.obj_config.float_scaleMicrometers),
            "micrometers_per_pixel": float(float_micrometersPerPixel),
            "bbox_edge_margin": int(self.obj_config.int_bboxEdgeMargin),
            "tile_edge_margin": int(self.obj_config.int_tileEdgeMargin),
            "tile_size": int(self.obj_config.int_tileSize),
            "stride": int(self.obj_config.int_stride),
            "points_per_tile": int(self.obj_config.int_pointsPerTile),
            "point_min_distance": int(self.obj_config.int_pointMinDistance),
            "point_quality_level": float(self.obj_config.float_pointQualityLevel),
            "point_batch_size": int(self.obj_config.int_pointBatchSize),
            "num_negative_points": int(self.obj_config.int_numNegativePoints),
            "dedup_iou": float(self.obj_config.float_dedupIou),
            "bbox_dedup_iou": float(self.obj_config.float_bboxDedupIou),
            "use_point_prompts": bool(self.obj_config.bool_usePointPrompts),
            "particle_area_threshold": float(self.obj_config.float_particleAreaThreshold),
            "mask_binarize_threshold": float(self.obj_config.float_maskBinarizeThreshold),
            "min_valid_mask_area": int(self.obj_config.int_minValidMaskArea),
            "mask_morph_kernel_size": int(self.obj_config.int_maskMorphKernelSize),
            "mask_morph_open_iterations": int(self.obj_config.int_maskMorphOpenIterations),
            "mask_morph_close_iterations": int(self.obj_config.int_maskMorphCloseIterations),
            "num_total_objects": int_totalObjects,
            "num_particles": int_particleCount,
            "num_fragments": int_fragmentCount,
            "fragment_count": int_fragmentCount,
            "total_object_count": int_totalObjects,
            "normal_particle_count": int_particleCount,
            "fine_particle_count": int_fragmentCount,
            "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
            "particle_sphericity_mean": None,
            "particle_sphericity_median": None,
            "particle_sphericity_std": None,
            "particle_sphericity_min": None,
            "particle_sphericity_max": None,
            "particle_sphericity_prime_mean": None,
            "particle_sphericity_prime_median": None,
            "particle_sphericity_prime_std": None,
            "particle_sphericity_prime_min": None,
            "particle_sphericity_prime_max": None,
            "particle_mean_size_um": None,
            "particle_size_median_um": None,
            "particle_size_std_um": None,
            "particle_size_min_um": None,
            "particle_size_max_um": None,
            "particle_sphericity_raw": [],
            "particle_sphericity_prime_raw": [],
            "particle_size_um_raw": [],
        }

        if list_particleSphs:
            arr_sphs = np.array(list_particleSphs, dtype=np.float32)
            dict_summary.update({
                "particle_sphericity_mean": float(np.mean(arr_sphs)),
                "particle_sphericity_median": float(np.median(arr_sphs)),
                "particle_sphericity_std": float(np.std(arr_sphs)),
                "particle_sphericity_min": float(np.min(arr_sphs)),
                "particle_sphericity_max": float(np.max(arr_sphs)),
                "particle_sphericity_raw": [float(v) for v in list_particleSphs],
            })

        if list_particleSphs_prime:
            arr_sphs_prime = np.array(list_particleSphs_prime, dtype=np.float32)
            dict_summary.update({
                "particle_sphericity_prime_mean": float(np.mean(arr_sphs_prime)),
                "particle_sphericity_prime_median": float(np.median(arr_sphs_prime)),
                "particle_sphericity_prime_std": float(np.std(arr_sphs_prime)),
                "particle_sphericity_prime_min": float(np.min(arr_sphs_prime)),
                "particle_sphericity_prime_max": float(np.max(arr_sphs_prime)),
                "particle_sphericity_prime_raw": [float(v) for v in list_particleSphs_prime],
            })

        if list_particleSizes:
            arr_sizes = np.array(list_particleSizes, dtype=np.float32)
            dict_summary.update({
                "particle_mean_size_um": float(np.mean(arr_sizes)),
                "particle_size_median_um": float(np.median(arr_sizes)),
                "particle_size_std_um": float(np.std(arr_sizes, ddof=1)) if len(arr_sizes) >= 2 else None,
                "particle_size_min_um": float(np.min(arr_sizes)),
                "particle_size_max_um": float(np.max(arr_sizes)),
                "particle_size_um_raw": [float(v) for v in list_particleSizes],
            })

        return dict_summary

    @staticmethod
    def _append_stats_bar(
        arr_img: np.ndarray,
        dict_summary: tp.Dict[str, tp.Any],
    ) -> np.ndarray:
        """오버레이 이미지 하단에 정량화 지표 텍스트 바를 붙인다."""
        int_n_particle = dict_summary.get("num_particles", 0) or 0
        int_n_fragment = dict_summary.get("num_fragments", 0) or 0
        float_fine = dict_summary.get("fine_particle_ratio_percent")
        float_size = dict_summary.get("particle_mean_size_um")
        float_sph = dict_summary.get("particle_sphericity_mean")

        str_fine = f"{float_fine:.1f}%" if float_fine is not None else "N/A"
        str_size = f"{float_size:.3f}µm" if float_size is not None else "N/A"
        str_sph = f"{float_sph:.3f}" if float_sph is not None else "N/A"

        str_stats = (
            f"Particle={int_n_particle}  Fragment={int_n_fragment}"
            f"  Fine%={str_fine}  MeanSize={str_size}  Sphericity={str_sph}"
        )

        int_ow = arr_img.shape[1]
        float_scale = int_ow / 1800.0
        int_font = cv2.FONT_HERSHEY_SIMPLEX
        int_thick = max(1, int(round(float_scale * 1.5)))
        (_, int_th), int_bl = cv2.getTextSize(str_stats, int_font, float_scale, int_thick)
        int_bar_h = int_th + int_bl + int(16 * float_scale)
        arr_bar = np.zeros((int_bar_h, int_ow, 3), dtype=np.uint8)
        cv2.putText(arr_bar, str_stats, (int(8 * float_scale), int_th + int(8 * float_scale)),
                    int_font, float_scale, (220, 220, 220), int_thick, cv2.LINE_AA)
        return np.vstack([arr_img, arr_bar])

    def save_outputs(
        self,
        arr_inputBgr: np.ndarray,
        arr_inputRoiBgr: np.ndarray,
        arr_overlayRoi: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
        dict_summary: tp.Dict[str, tp.Any],
        dict_roi: tp.Dict[str, int],
        dict_debug: tp.Dict[str, tp.Any],
        arr_raw_masks: tp.Optional[np.ndarray] = None,
        arr_restoration_viz: tp.Optional[np.ndarray] = None,
        arr_brightness_filter_viz: tp.Optional[np.ndarray] = None,
    ) -> None:
        """이미지, CSV, JSON, histogram 등 최종 산출물을 저장한다.

        Args:
            arr_inputBgr: 원본 입력 이미지.
            arr_inputRoiBgr: 추론에 사용된 ROI 이미지.
            arr_overlayRoi: ROI 위에 객체 시각화를 그린 이미지.
            list_objects: 저장할 객체 측정 결과 리스트.
            list_masks: 각 객체에 대응하는 ROI 좌표계 binary mask 리스트.
            dict_summary: summary.json에 저장할 요약 dict.
            dict_roi: ROI 좌표 및 크기 정보 dict.
            dict_debug: 디버그용 tile/point/mask 정보 dict.

        Returns:
            없음. output directory 아래에 png/csv/json 파일들을 기록한다.
        """
        self.obj_config.path_outputDir.mkdir(parents=True, exist_ok=True)

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
            (255, 255, 0),
            2,
        )

        cv2.imwrite(str(self.obj_config.path_outputDir / "input_roi.png"), arr_inputRoiBgr)

        if list_objects:
            arr_eq = self.draw_eq_circles_clean(arr_inputRoiBgr, list_objects, list_masks)
            cv2.imwrite(str(self.obj_config.path_outputDir / "classified.png"), arr_eq)

        arr_overlay_with_stats = self._append_stats_bar(arr_overlayRoi, dict_summary)
        cv2.imwrite(str(self.obj_config.path_outputDir / "overlay_roi.png"), arr_overlay_with_stats)

        with (self.obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
            json_dump_safe(dict_summary, obj_f)

        if not self.obj_config.bool_debug:
            return

        # ── 디버그 모드 전용 ───────────────────────────────────────────────────

        cv2.imwrite(str(self.obj_config.path_outputDir / "input.png"), arr_inputBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir / "overlay.png"), arr_overlayFull)

        arr_overlay_S  = self.create_overlay(arr_inputRoiBgr, list_objects, list_masks, str_show="S")
        arr_overlay_Sp = self.create_overlay(arr_inputRoiBgr, list_objects, list_masks, str_show="Sp")
        cv2.imwrite(str(self.obj_config.path_outputDir / "overlay_S.png"),
                    self._append_stats_bar(arr_overlay_S, dict_summary))
        cv2.imwrite(str(self.obj_config.path_outputDir / "overlay_Sp.png"),
                    self._append_stats_bar(arr_overlay_Sp, dict_summary))

        # 파이프라인 단계별 이미지
        list_tiles = dict_debug.get("tiles", [])
        if list_tiles:
            arr_tiles_viz = arr_inputRoiBgr.copy()
            for dict_t in list_tiles:
                int_tx1, int_ty1, int_tx2, int_ty2 = dict_t["tile_xyxy"]
                cv2.rectangle(arr_tiles_viz, (int_tx1, int_ty1), (int_tx2, int_ty2),
                              (200, 200, 0), 1)
            cv2.imwrite(str(self.obj_config.path_outputDir / "tiles.png"), arr_tiles_viz)

        # ── 프롬프트 계산 과정 디버그 이미지 (4장) ──────────────────────────
        list_hct_circles = dict_debug.get("hct_circles", [])
        list_all_pts = dict_debug.get("candidate_points", [])
        list_hct_pts = [p for p in list_all_pts if p.get("label", 1) == 1 and p.get("source") == "hct"]
        list_cc_pts = [p for p in list_all_pts if p.get("label", 1) == 1 and p.get("source") == "cc"]
        list_pos_pts = [p for p in list_all_pts if p.get("label", 1) == 1]
        list_neg_pts = [p for p in list_all_pts if p.get("label", 0) == 0]
        list_cc_contours = dict_debug.get("cc_contours", [])

        arr_gray_dbg = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
        arr_blur_dbg = cv2.GaussianBlur(arr_gray_dbg, (5, 5), 0)
        _, arr_binary_dbg = cv2.threshold(
            arr_blur_dbg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        arr_binary_bgr = cv2.cvtColor(arr_binary_dbg, cv2.COLOR_GRAY2BGR)
        arr_binary_inv_bgr = cv2.cvtColor(
            cv2.bitwise_not(arr_binary_dbg), cv2.COLOR_GRAY2BGR)

        # 1. HCT 원 + HCT positive points → 원본 ROI
        if list_hct_circles or list_hct_pts:
            arr_v1 = arr_inputRoiBgr.copy()
            for dict_c in list_hct_circles:
                int_cx, int_cy = dict_c["center_roi"]
                cv2.circle(arr_v1, (int_cx, int_cy), dict_c["radius"], (0, 255, 0), 1)
            for dict_pt in list_hct_pts:
                int_px = int(dict_pt["point_xy_roi"][0])
                int_py = int(dict_pt["point_xy_roi"][1])
                cv2.circle(arr_v1, (int_px, int_py), 3, (0, 255, 255), -1)
            cv2.imwrite(str(self.obj_config.path_outputDir / "prompt_1_hct.png"), arr_v1)

        # 2. HCT points + CC points + CC 컨투어 → Otsu 이진화
        if list_hct_pts or list_cc_pts:
            arr_v2 = arr_binary_bgr.copy()
            if list_cc_contours:
                cv2.drawContours(arr_v2, list_cc_contours, -1, (0, 180, 0), 1)
            for dict_pt in list_hct_pts:
                int_px = int(dict_pt["point_xy_roi"][0])
                int_py = int(dict_pt["point_xy_roi"][1])
                cv2.circle(arr_v2, (int_px, int_py), 3, (0, 255, 255), -1)
            for dict_pt in list_cc_pts:
                int_px = int(dict_pt["point_xy_roi"][0])
                int_py = int(dict_pt["point_xy_roi"][1])
                cv2.circle(arr_v2, (int_px, int_py), 3, (255, 128, 0), -1)
            cv2.imwrite(str(self.obj_config.path_outputDir / "prompt_2_cc.png"), arr_v2)

        # 3. 모든 positive points → 반전 이진화
        if list_pos_pts:
            arr_v3 = arr_binary_inv_bgr.copy()
            for dict_pt in list_pos_pts:
                int_px = int(dict_pt["point_xy_roi"][0])
                int_py = int(dict_pt["point_xy_roi"][1])
                cv2.circle(arr_v3, (int_px, int_py), 3, (0, 255, 255), -1)
            cv2.imwrite(str(self.obj_config.path_outputDir / "prompt_3_pos.png"), arr_v3)

        # 4. 모든 positive + negative points → 반전 이진화
        if list_pos_pts or list_neg_pts:
            arr_v4 = arr_binary_inv_bgr.copy()
            for dict_pt in list_pos_pts:
                int_px = int(dict_pt["point_xy_roi"][0])
                int_py = int(dict_pt["point_xy_roi"][1])
                cv2.circle(arr_v4, (int_px, int_py), 3, (0, 255, 255), -1)
            for dict_pt in list_neg_pts:
                int_px = int(dict_pt["point_xy_roi"][0])
                int_py = int(dict_pt["point_xy_roi"][1])
                cv2.circle(arr_v4, (int_px, int_py), 3, (255, 255, 0), -1)
            cv2.imwrite(str(self.obj_config.path_outputDir / "prompt_4_all.png"), arr_v4)

        if arr_raw_masks is not None and len(arr_raw_masks) > 0:
            arr_raw_viz = arr_inputRoiBgr.copy()
            for int_i, arr_m in enumerate(arr_raw_masks):
                int_hue = (int_i * 37) % 180
                tpl_c = cv2.cvtColor(
                    np.array([[[int_hue, 200, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                )[0, 0].tolist()
                arr_bool = arr_m.astype(bool)
                arr_raw_viz[arr_bool] = (
                    arr_raw_viz[arr_bool].astype(np.float32) * 0.5
                    + np.array(tpl_c, dtype=np.float32) * 0.5
                ).astype(np.uint8)
            cv2.imwrite(str(self.obj_config.path_outputDir / "masks_raw.png"), arr_raw_viz)

        if arr_restoration_viz is not None:
            cv2.imwrite(str(self.obj_config.path_outputDir / "restoration.png"), arr_restoration_viz)

        if arr_brightness_filter_viz is not None:
            cv2.imwrite(str(self.obj_config.path_outputDir / "brightness_filter.png"),
                        arr_brightness_filter_viz)

        path_csvAll = self.obj_config.path_outputDir / "objects.csv"
        with path_csvAll.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_objects:
                obj_writer = csv.DictWriter(
                    obj_f, fieldnames=list(asdict(list_objects[0]).keys()))
                obj_writer.writeheader()
                for obj_measurement in list_objects:
                    obj_writer.writerow(asdict(obj_measurement))

        list_particleRows = [asdict(
            obj_item) for obj_item in list_objects if obj_item.str_category == "particle"]
        path_csvParticle = self.obj_config.path_outputDir / "particles.csv"
        with path_csvParticle.open("w", newline="", encoding="utf-8-sig") as obj_f:
            if list_particleRows:
                obj_writer = csv.DictWriter(
                    obj_f, fieldnames=list(list_particleRows[0].keys()))
                obj_writer.writeheader()
                for dict_row in list_particleRows:
                    obj_writer.writerow(dict_row)

        try:
            save_particle_distribution_histogram(
                path_particlesCsv=path_csvParticle,
                path_outputImage=self.obj_config.path_outputDir / "size_dist.png",
                path_inputImage=self.obj_config.path_input,
            )
        except Exception as exc:
            print(f"[WARN] size_dist.png 저장 실패: {exc}", flush=True)

        try:
            save_sphericity_distribution_histogram(
                path_particlesCsv=path_csvParticle,
                path_outputImage=self.obj_config.path_outputDir / "sph_dist.png",
                path_inputImage=self.obj_config.path_input,
            )
        except Exception as exc:
            print(f"[WARN] sph_dist.png 저장 실패: {exc}", flush=True)

        with (self.obj_config.path_outputDir / "objects.json").open("w", encoding="utf-8") as obj_f:
            json_dump_safe([asdict(obj_item) for obj_item in list_objects], obj_f)

        with (self.obj_config.path_outputDir / "debug.json").open("w", encoding="utf-8") as obj_f:
            json_dump_safe(dict_debug, obj_f)

        if not self.obj_config.bool_saveIndividualMasks:
            return

        path_particleMaskDir = self.obj_config.path_outputDir / "particle_masks"
        path_fragmentMaskDir = self.obj_config.path_outputDir / "fragment_masks"
        path_particleMaskDir.mkdir(parents=True, exist_ok=True)
        path_fragmentMaskDir.mkdir(parents=True, exist_ok=True)

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            path_targetDir = (
                path_particleMaskDir
                if obj_measurement.str_category == "particle"
                else path_fragmentMaskDir
            )
            str_fileName = f"{obj_measurement.str_category}_{obj_measurement.int_index:04d}.png"
            cv2.imwrite(str(path_targetDir / str_fileName),
                        arr_mask.astype(np.uint8) * 255)

    def process_opencv(self) -> Sam2AspectRatioResult:
        """OpenCV CLAHE+Otsu+Watershed 기반 세그멘테이션 파이프라인.

        구형도는 convex hull 둘레로 계산해 컨투어 픽셀화 노이즈를 억제한다.
        """
        arr_inputBgr = self.load_image_bgr()
        arr_inputRoiBgr, dict_roi = self.extract_inference_roi(arr_inputBgr)

        # ── 전처리: CLAHE → Otsu ─────────────────────────────────────────────
        arr_gray = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
        obj_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr_clahe = obj_clahe.apply(arr_gray)
        _, arr_binary = cv2.threshold(arr_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Auto-invert when background dominates (>55% white) so watershed
        # always operates on foreground=particles, consistent with other
        # detection functions.
        if arr_binary.size > 0 and float((arr_binary > 0).sum()) / arr_binary.size > 0.55:
            arr_binary = cv2.bitwise_not(arr_binary)

        int_k = self.obj_config.int_maskMorphKernelSize
        if int_k > 1:
            arr_kernel = np.ones((int_k, int_k), dtype=np.uint8)
            if self.obj_config.int_maskMorphOpenIterations > 0:
                arr_binary = cv2.morphologyEx(
                    arr_binary, cv2.MORPH_OPEN, arr_kernel,
                    iterations=self.obj_config.int_maskMorphOpenIterations)
            if self.obj_config.int_maskMorphCloseIterations > 0:
                arr_binary = cv2.morphologyEx(
                    arr_binary, cv2.MORPH_CLOSE, arr_kernel,
                    iterations=self.obj_config.int_maskMorphCloseIterations)

        # ── Watershed: distance transform → seed → 분리 ──────────────────────
        arr_dist = cv2.distanceTransform(arr_binary, cv2.DIST_L2, 5)
        _, arr_sure_fg = cv2.threshold(arr_dist, 0.6 * arr_dist.max(), 255, cv2.THRESH_BINARY)
        arr_sure_fg = arr_sure_fg.astype(np.uint8)
        arr_sure_bg = cv2.dilate(arr_binary, np.ones((3, 3), dtype=np.uint8), iterations=3)
        arr_unknown = cv2.subtract(arr_sure_bg, arr_sure_fg)

        int_n, arr_markers = cv2.connectedComponents(arr_sure_fg)
        arr_markers = arr_markers + 1       # 1 = 확실한 배경, 2+ = 각 입자
        arr_markers[arr_unknown == 255] = 0  # 0 = watershed가 결정할 경계
        cv2.watershed(arr_inputRoiBgr.copy(), arr_markers)

        # ── 마스크 추출 + 측정 (convex hull 구형도) ──────────────────────────
        list_objects: tp.List[ObjectMeasurement] = []
        list_validMasks: tp.List[np.ndarray] = []
        list_rawMasks: tp.List[np.ndarray] = []
        int_index = 0
        for int_label in range(2, int_n + 1):
            arr_mask = ((arr_markers == int_label) & (arr_binary > 0)).astype(np.uint8)
            list_rawMasks.append(arr_mask)
            obj_measurement = self.measure_mask(
                arr_mask, int_index=int_index, float_confidence=None,
                bool_convexHullSphericity=True,
            )
            if obj_measurement is None:
                continue
            list_objects.append(obj_measurement)
            list_validMasks.append(self.refine_mask_for_area(arr_mask).astype(np.uint8))
            int_index += 1

        int_h_roi, int_w_roi = arr_inputRoiBgr.shape[:2]
        arr_raw_masks = (
            np.stack(list_rawMasks, axis=0)
            if list_rawMasks
            else np.empty((0, int_h_roi, int_w_roi), dtype=np.uint8)
        )
        arr_overlay = self.create_overlay(arr_inputRoiBgr, list_objects, list_validMasks)
        dict_summary = self.build_summary(list_objects)
        dict_summary["roi"] = dict_roi
        dict_summary["measure_mode"] = "opencv"
        dict_debug: tp.Dict[str, tp.Any] = {
            "measure_mode": "opencv",
            "num_watershed_labels": int_n - 1,
        }
        self.save_outputs(
            arr_inputBgr, arr_inputRoiBgr, arr_overlay,
            list_objects, list_validMasks, dict_summary, dict_roi, dict_debug,
            arr_raw_masks=arr_raw_masks,
        )

        if self.obj_config.bool_debug:
            cv2.imwrite(str(self.obj_config.path_outputDir / "clahe.png"), arr_clahe)
            cv2.imwrite(str(self.obj_config.path_outputDir / "binary.png"), arr_binary)
            arr_dist_viz = cv2.normalize(arr_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(
                str(self.obj_config.path_outputDir / "dist.png"),
                cv2.applyColorMap(arr_dist_viz, cv2.COLORMAP_JET),
            )
            arr_ws_viz = arr_inputRoiBgr.copy()
            arr_ws_viz[arr_markers == -1] = [0, 0, 255]
            cv2.imwrite(str(self.obj_config.path_outputDir / "watershed.png"), arr_ws_viz)

        return Sam2AspectRatioResult(list_objects=list_objects, dict_summary=dict_summary)

    def process(self) -> Sam2AspectRatioResult:
        """단일 이미지에 대한 전체 파이프라인을 실행한다.

        Returns:
            측정 결과 리스트와 summary dict를 포함하는 `Sam2AspectRatioResult`.
        """
        if self.obj_config.bool_useOpenCV:
            return self.process_opencv()

        arr_inputBgr = self.load_image_bgr()
        arr_inputRoiBgr, dict_roi = self.extract_inference_roi(arr_inputBgr)
        arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(
            arr_inputRoiBgr)

        # ② 가려진 입자 보정  ③ 땅콩 분리
        arr_masks, arr_scores = self._postprocess_masks(arr_masks, arr_scores)

        # 포함 관계 펀치: 작은 마스크가 큰 마스크에 97%+ 포함되면 큰 마스크에서 제거
        # → 이후 밝기 필터는 펀치된 마스크(입자 테두리 영역) 기준으로 판단
        if len(arr_masks) > 1:
            arr_masks_areas = np.array([m.sum() for m in arr_masks], dtype=np.int64)
            arr_idx_sorted = np.argsort(arr_masks_areas)[::-1]
            arr_masks_punched = [m.copy() for m in arr_masks]
            # bbox 선계산: (x1, y1, x2, y2)
            list_punch_bboxes = []
            for m in arr_masks:
                mb = m.astype(bool)
                if mb.any():
                    rows = np.where(mb.any(axis=1))[0]
                    cols = np.where(mb.any(axis=0))[0]
                    list_punch_bboxes.append((int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])))
                else:
                    list_punch_bboxes.append((0, 0, 0, 0))
            for int_i_pos in range(len(arr_idx_sorted) - 1):
                int_i = arr_idx_sorted[int_i_pos]
                ix1, iy1, ix2, iy2 = list_punch_bboxes[int_i]
                for int_j_pos in range(int_i_pos + 1, len(arr_idx_sorted)):
                    int_j = arr_idx_sorted[int_j_pos]
                    int_area_j = int(arr_masks_areas[int_j])
                    if int_area_j == 0:
                        continue
                    jx1, jy1, jx2, jy2 = list_punch_bboxes[int_j]
                    if ix2 < jx1 or jx2 < ix1 or iy2 < jy1 or jy2 < iy1:
                        continue
                    int_overlap = int(
                        (arr_masks[int_i].astype(bool) & arr_masks[int_j].astype(bool)).sum()
                    )
                    if int_overlap / int_area_j >= 0.97:
                        arr_masks_punched[int_i] = (
                            arr_masks_punched[int_i].astype(bool)
                            & ~arr_masks[int_j].astype(bool)
                        ).astype(arr_masks[int_i].dtype)
            arr_masks = arr_masks_punched

        list_objects: tp.List[ObjectMeasurement] = []
        list_validMasks: tp.List[np.ndarray] = []

        arr_restoration_viz = arr_inputRoiBgr.copy()

        # 밝기 필터 기준: Otsu 임계값의 절반 미만 평균 밝기 → 배경으로 간주
        arr_gray_roi = cv2.cvtColor(arr_inputRoiBgr, cv2.COLOR_BGR2GRAY)
        int_otsu_global, _ = cv2.threshold(
            arr_gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        float_brightness_thresh = float(int_otsu_global) * 0.5

        # 밝기 필터 디버그 오버레이: 통과(녹)·제거(적) 마스크와 Otsu 배수 표시
        int_h_viz, int_w_viz = arr_inputRoiBgr.shape[:2]
        arr_bf_viz = cv2.resize(arr_inputRoiBgr,
                                (int_w_viz * 2, int_h_viz * 2),
                                interpolation=cv2.INTER_LINEAR)
        list_bf_placed: tp.List[tp.Tuple[int, int, int, int]] = []
        for arr_m in arr_masks:
            arr_mb = arr_m.astype(bool)
            if not arr_mb.any():
                continue
            float_mb = float(arr_gray_roi[arr_mb].mean())
            float_ratio = float_mb / float(int_otsu_global) if int_otsu_global > 0 else 0.0
            bool_pass = float_mb >= float_brightness_thresh
            tpl_tint = (40, 200, 40) if bool_pass else (40, 40, 220)
            arr_m2 = cv2.resize(arr_m, (int_w_viz * 2, int_h_viz * 2),
                                interpolation=cv2.INTER_NEAREST)
            arr_bf_viz[arr_m2 > 0] = (
                arr_bf_viz[arr_m2 > 0].astype(np.float32) * 0.55
                + np.array(tpl_tint, dtype=np.float32) * 0.45
            ).astype(np.uint8)
            cnts_bf, _ = cv2.findContours(arr_m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_bf:
                cv2.drawContours(arr_bf_viz, cnts_bf, -1, tpl_tint, 1)
                M = cv2.moments(max(cnts_bf, key=cv2.contourArea))
                if M["m00"] > 0:
                    int_cx = int(M["m10"] / M["m00"])
                    int_cy = int(M["m01"] / M["m00"])
                    draw_label_no_overlap(
                        arr_bf_viz, [f"{float_ratio:.2f}x"],
                        int_cx, int_cy, tpl_tint, list_bf_placed)
        for int_index, arr_mask in enumerate(arr_masks):
            float_confidence = None
            if arr_scores is not None and int_index < len(arr_scores):
                float_s = float(arr_scores[int_index])
                float_confidence = None if math.isnan(float_s) else float_s

            # 밝기 필터: 마스크 영역 평균 밝기가 Otsu×0.5 미만이면 배경으로 제거
            arr_mask_bool = arr_mask.astype(bool)
            if arr_mask_bool.any():
                float_mean_brightness = float(arr_gray_roi[arr_mask_bool].mean())
                if float_mean_brightness < float_brightness_thresh:
                    continue

            # 원본 마스크 측정 — S(구형도)만 이 값 사용
            obj_measurement_orig = self.measure_mask(
                arr_mask, int_index=int_index, float_confidence=float_confidence)
            if obj_measurement_orig is None:
                continue
            float_sphericity_orig = obj_measurement_orig.float_sphericity

            # 솔리디티 계산 + SEM 직선 아티팩트에 의한 절단 감지
            list_cnts_s, _ = cv2.findContours(
                arr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            float_solidity = 1.0
            bool_has_line_cut = False
            if list_cnts_s:
                arr_cnt_s = max(list_cnts_s, key=cv2.contourArea)
                float_cnt_area = float(cv2.contourArea(arr_cnt_s))
                float_hull_area = float(cv2.contourArea(cv2.convexHull(arr_cnt_s)))
                float_solidity = float_cnt_area / max(float_hull_area, 1.0)

                int_mask_h, int_mask_w = arr_mask.shape[:2]
                # 조건 1: 수평/수직 직선 구간 15px 이상 존재
                arr_approx = cv2.approxPolyDP(arr_cnt_s, epsilon=2.0, closed=True)
                for int_k in range(len(arr_approx)):
                    pt1 = arr_approx[int_k][0]
                    pt2 = arr_approx[(int_k + 1) % len(arr_approx)][0]
                    float_dx = float(pt2[0] - pt1[0])
                    float_dy = float(pt2[1] - pt1[1])
                    float_seg_len = math.hypot(float_dx, float_dy)
                    if float_seg_len < 15.0:
                        continue
                    float_angle = math.degrees(math.atan2(abs(float_dy), abs(float_dx)))
                    if float_angle < 5.0 or float_angle > 85.0:
                        bool_has_line_cut = True
                        break
                # 조건 2: ROI 경계 접촉
                if not bool_has_line_cut:
                    int_bx, int_by, int_bw, int_bh = cv2.boundingRect(arr_cnt_s)
                    if (int_bx == 0 or int_by == 0
                            or int_bx + int_bw >= int_mask_w
                            or int_by + int_bh >= int_mask_h):
                        bool_has_line_cut = True

            if float_solidity < 0.97 or bool_has_line_cut:
                result = self._fit_particle_circle(arr_mask)
                if result is not None:
                    float_cx, float_cy, float_r = result
                    arr_circle = np.zeros_like(arr_mask)
                    cv2.circle(arr_circle,
                               (int(round(float_cx)), int(round(float_cy))),
                               int(round(float_r)), 1, -1)
                    arr_notch = arr_circle.astype(bool) & ~arr_mask.astype(bool)
                    arr_bright = arr_notch & (arr_gray_roi >= int(int_otsu_global) * 3 // 4)
                    if arr_bright.sum() > 50:
                        arr_mask = (arr_mask.astype(bool) | arr_bright).astype(arr_mask.dtype)

                    cv2.circle(arr_restoration_viz,
                               (int(round(float_cx)), int(round(float_cy))),
                               int(round(float_r)), (0, 200, 255), 1)
                    arr_restoration_viz[arr_bright.astype(bool)] = (255, 80, 0)

            # 모든 마스크에 hull 적용 → hull 기준 면적으로 분류
            arr_mask = self._hull_mask(arr_mask)

            obj_measurement = self.measure_mask(
                arr_mask, int_index=int_index, float_confidence=float_confidence,
                bool_convexHullSphericity=True)
            if obj_measurement is None:
                continue

            # 면적·크기·S'·분류는 hull 기준, S는 원본 컨투어 기준 유지
            obj_measurement = dataclasses_replace(
                obj_measurement,
                float_sphericity=float_sphericity_orig,
            )

            list_objects.append(obj_measurement)
            list_validMasks.append(
                self.refine_mask_for_area(arr_mask).astype(np.uint8))

        # hull 마스크 97%+ 겹침 → 합집합 병합
        int_n_hull = len(list_validMasks)
        if int_n_hull > 1:
            arr_hull_areas = np.array([m.sum() for m in list_validMasks], dtype=np.int64)
            # bbox 선계산: (x1, y1, x2, y2)
            list_hull_bboxes = []
            for m in list_validMasks:
                mb = m.astype(bool)
                if mb.any():
                    rows = np.where(mb.any(axis=1))[0]
                    cols = np.where(mb.any(axis=0))[0]
                    list_hull_bboxes.append((int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])))
                else:
                    list_hull_bboxes.append((0, 0, 0, 0))
            # union-find
            list_parent = list(range(int_n_hull))

            def _find(x: int) -> int:
                while list_parent[x] != x:
                    list_parent[x] = list_parent[list_parent[x]]
                    x = list_parent[x]
                return x

            for int_i in range(int_n_hull - 1):
                ix1, iy1, ix2, iy2 = list_hull_bboxes[int_i]
                for int_j in range(int_i + 1, int_n_hull):
                    int_area_min = int(min(arr_hull_areas[int_i], arr_hull_areas[int_j]))
                    if int_area_min == 0:
                        continue
                    jx1, jy1, jx2, jy2 = list_hull_bboxes[int_j]
                    if ix2 < jx1 or jx2 < ix1 or iy2 < jy1 or jy2 < iy1:
                        continue
                    int_overlap = int(
                        (list_validMasks[int_i].astype(bool)
                         & list_validMasks[int_j].astype(bool)).sum()
                    )
                    if int_overlap / int_area_min >= 0.97:
                        list_parent[_find(int_i)] = _find(int_j)

            dict_groups: tp.Dict[int, tp.List[int]] = {}
            for int_i in range(int_n_hull):
                dict_groups.setdefault(_find(int_i), []).append(int_i)

            list_objects_new: tp.List[ObjectMeasurement] = []
            list_validMasks_new: tp.List[np.ndarray] = []
            for list_idx in dict_groups.values():
                if len(list_idx) == 1:
                    list_objects_new.append(list_objects[list_idx[0]])
                    list_validMasks_new.append(list_validMasks[list_idx[0]])
                else:
                    arr_union = list_validMasks[list_idx[0]].copy()
                    for int_k in list_idx[1:]:
                        arr_union = (
                            arr_union.astype(bool) | list_validMasks[int_k].astype(bool)
                        ).astype(arr_union.dtype)
                    int_lead = max(list_idx, key=lambda k: arr_hull_areas[k])
                    obj_merged = self.measure_mask(
                        arr_union,
                        int_index=list_objects[int_lead].int_index,
                        float_confidence=list_objects[int_lead].float_confidence,
                        bool_convexHullSphericity=True,
                    )
                    if obj_merged is not None:
                        list_objects_new.append(obj_merged)
                        list_validMasks_new.append(
                            self.refine_mask_for_area(arr_union).astype(np.uint8))
            list_objects = list_objects_new
            list_validMasks = list_validMasks_new

        arr_overlay = self.create_overlay(
            arr_inputRoiBgr, list_objects, list_validMasks)
        dict_summary = self.build_summary(list_objects)
        dict_summary["roi"] = dict_roi
        dict_summary["num_tiles"] = dict_debug.get("num_tiles")
        dict_summary["num_candidate_points"] = dict_debug.get(
            "num_candidate_points")
        dict_summary["num_accepted_masks"] = dict_debug.get(
            "num_accepted_masks")
        dict_summary["num_bbox_dedup_rejected"] = dict_debug.get(
            "num_bbox_dedup_rejected")
        self.save_outputs(
            arr_inputBgr,
            arr_inputRoiBgr,
            arr_overlay,
            list_objects,
            list_validMasks,
            dict_summary,
            dict_roi,
            dict_debug,
            arr_raw_masks=arr_masks,
            arr_restoration_viz=arr_restoration_viz,
            arr_brightness_filter_viz=arr_bf_viz,
        )

        return Sam2AspectRatioResult(
            list_objects=list_objects,
            dict_summary=dict_summary,
        )
