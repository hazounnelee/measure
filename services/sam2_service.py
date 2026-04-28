from __future__ import annotations
import csv
import json
import os
import shutil
import typing as tp
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.schema import Sam2AspectRatioConfig, ObjectMeasurement, Sam2AspectRatioResult
from models import load_sam2_model
from utils.metrics import convert_pixels_to_micrometers, calculate_mean_from_optional_values, calculate_percentage, normalize_image_to_uint8
from utils.iou import calculate_binary_iou, calculate_box_iou
from utils.image import create_processing_tiles, enhance_image_texture, sample_interest_points
from utils.io import iter_chunks, collect_input_groups, build_image_output_dir


def load_particle_mean_sizes_from_csv(path_particlesCsv: Path) -> np.ndarray:
    """`particles.csv`에서 particle 평균 크기(um)를 읽어온다.

    Args:
        path_particlesCsv: `particles.csv` 파일 경로. UTF-8 BOM(`utf-8-sig`)로 읽는다.

    Returns:
        각 row의 `(float_longestHorizontalUm + float_longestVerticalUm) / 2` 값을 담은
        `np.float32` 1차원 배열. 파일이 없거나 유효한 row가 없으면 빈 배열을 반환한다.
    """
    if not path_particlesCsv.exists():
        return np.array([], dtype=np.float32)

    list_meanSizes: tp.List[float] = []
    with path_particlesCsv.open("r", newline="", encoding="utf-8-sig") as obj_f:
        obj_reader = csv.DictReader(obj_f)
        for dict_row in obj_reader:
            try:
                float_horizontal = float(dict_row["float_longestHorizontalUm"])
                float_vertical = float(dict_row["float_longestVerticalUm"])
            except (KeyError, TypeError, ValueError):
                continue
            list_meanSizes.append((float_horizontal + float_vertical) / 2.0)

    if not list_meanSizes:
        return np.array([], dtype=np.float32)
    return np.array(list_meanSizes, dtype=np.float32)


def get_lot_number_from_input_path(path_inputImage: Path) -> str:
    """입력 이미지 경로에서 lot 번호를 추출한다.

    Args:
        path_inputImage: 원본 입력 이미지 경로.

    Returns:
        절대경로 기준 이미지 파일의 바로 위 directory 이름. 비어 있으면
        `"UnknownLot"`을 반환한다.
    """
    try:
        path_resolved = path_inputImage.resolve()
    except OSError:
        path_resolved = path_inputImage
    str_lotNumber = path_resolved.parent.name.strip()
    return str_lotNumber if str_lotNumber else "UnknownLot"


def save_particle_distribution_histogram(
    path_particlesCsv: Path,
    path_outputImage: Path,
    path_inputImage: Path,
) -> None:
    """particle 크기 분포 histogram을 `png` 파일로 저장한다.

    Args:
        path_particlesCsv: particle 측정 결과 CSV 경로. 평균 크기 계산의 데이터 소스다.
        path_outputImage: 저장할 histogram 이미지 경로. 일반적으로 `particle_dist.png`.
        path_inputImage: lot 번호 추출에 사용할 원본 이미지 경로.

    Returns:
        없음. `matplotlib`의 headless backend(`Agg`)를 사용해 histogram 이미지를
        디스크에 저장한다.

    Notes:
        - x축 값은 micrometer 단위 particle 평균 크기다.
        - 제목은 입력 이미지의 부모 directory 이름(lot 번호)이다.
        - 평균 크기 위치에는 빨간 vertical line과 빨간 텍스트를 함께 표시한다.
        - particle 데이터가 없으면 빈 축에 안내 문구만 그린다.
    """
    arr_meanSizes = load_particle_mean_sizes_from_csv(path_particlesCsv)
    str_lotNumber = get_lot_number_from_input_path(path_inputImage)
    obj_fig, obj_ax = plt.subplots(figsize=(9.6, 6.4), dpi=100)

    try:
        obj_ax.set_title(str_lotNumber, fontsize=28)
        obj_ax.set_ylabel("Count", fontsize=20)
        obj_ax.set_xlabel("Mean of longest horizontal and vertical length (um)", fontsize=20)
        obj_ax.tick_params(axis="both", labelsize=20)

        if arr_meanSizes.size == 0:
            obj_ax.text(
                0.5,
                0.5,
                "No particle data in particles.csv",
                ha="center",
                va="center",
                fontsize=13,
                color="#666666",
                transform=obj_ax.transAxes,
            )
            obj_ax.set_xticks([])
            obj_ax.set_yticks([])
        else:
            int_numBins = int(np.clip(np.sqrt(arr_meanSizes.size), 5, 20))
            float_minValue = float(np.min(arr_meanSizes))
            float_maxValue = float(np.max(arr_meanSizes))
            float_meanValue = float(np.mean(arr_meanSizes))
            if abs(float_maxValue - float_minValue) < 1e-6:
                float_minValue -= 0.5
                float_maxValue += 0.5

            obj_ax.hist(
                arr_meanSizes,
                bins=int_numBins,
                range=(float_minValue, float_maxValue),
                color="#508cf0",
                edgecolor="#323232",
                linewidth=1.0,
            )

            obj_ax.axvline(float_meanValue, color="red", linewidth=2.0)
            float_yMax = obj_ax.get_ylim()[1]
            obj_ax.text(
                float_meanValue,
                float_yMax * 0.96,
                f"Mean: {float_meanValue:.2f} um",
                color="red",
                fontsize=24,
                ha="left",
                va="top",
            )
            obj_ax.grid(axis="y", linestyle="--", alpha=0.25)

        obj_fig.tight_layout()
        obj_fig.savefig(path_outputImage, bbox_inches="tight")
    finally:
        plt.close(obj_fig)


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
        self.obj_config = obj_config
        self.obj_model: tp.Optional[tp.Any] = None
        self.dict_modelConfig: tp.Dict[str, tp.Any] = dict()

    # ultralytics가 자동 다운로드하는 SAM2 이름 목록 (파일 미존재 시 존재 체크 생략)
    _SET_AUTO_DOWNLOAD_NAMES: tp.ClassVar[tp.FrozenSet[str]] = frozenset({
        "sam_h.pt", "sam_l.pt", "sam_b.pt", "mobile_sam.pt",
        "sam2_t.pt", "sam2_s.pt", "sam2_b.pt", "sam2_l.pt",
        "sam2.1_t.pt", "sam2.1_s.pt", "sam2.1_b.pt", "sam2.1_l.pt",
        "sam2_hiera_tiny.pt", "sam2_hiera_small.pt",
        "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt",
        "sam2.1_hiera_tiny.pt", "sam2.1_hiera_small.pt",
        "sam2.1_hiera_base_plus.pt", "sam2.1_hiera_large.pt",
    })

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
        set_supportedNames = {
            "sam_h.pt",
            "sam_l.pt",
            "sam_b.pt",
            "mobile_sam.pt",
            "sam2_t.pt",
            "sam2_s.pt",
            "sam2_b.pt",
            "sam2_l.pt",
            "sam2.1_t.pt",
            "sam2.1_s.pt",
            "sam2.1_b.pt",
            "sam2.1_l.pt",
        }
        dict_aliasNames = {
            "sam2_hiera_tiny.pt": "sam2_t.pt",
            "sam2_hiera_small.pt": "sam2_s.pt",
            "sam2_hiera_base_plus.pt": "sam2_b.pt",
            "sam2_hiera_large.pt": "sam2_l.pt",
            "sam2.1_hiera_tiny.pt": "sam2.1_t.pt",
            "sam2.1_hiera_small.pt": "sam2.1_s.pt",
            "sam2.1_hiera_base_plus.pt": "sam2.1_b.pt",
            "sam2.1_hiera_large.pt": "sam2.1_l.pt",
        }

        if path_weights.name in set_supportedNames:
            return path_weights

        str_aliasName = dict_aliasNames.get(path_weights.name)
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
            shutil.copy2(path_weights, path_alias)

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
        return arr_image

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
        int_candidateCount = 0
        int_acceptedCount = 0
        int_bboxDedupRejected = 0

        for int_tileIdx, (int_tx1, int_ty1, int_tx2, int_ty2) in enumerate(list_tiles):
            arr_tileBgr = arr_inputBgr[int_ty1:int_ty2, int_tx1:int_tx2].copy()
            list_promptBatches: tp.List[tp.Optional[tp.Sequence[tp.Tuple[int, int]]]] = [
                None]
            list_points: tp.List[tp.Tuple[int, int]] = []

            if self.obj_config.bool_usePointPrompts:
                arr_tileGray = arr_inputGray[int_ty1:int_ty2,
                                             int_tx1:int_tx2].copy()
                list_points = sample_interest_points(
                    arr_tileGray=arr_tileGray,
                    int_maxPoints=self.obj_config.int_pointsPerTile,
                    int_minDistance=self.obj_config.int_pointMinDistance,
                    float_qualityLevel=self.obj_config.float_pointQualityLevel,
                )
                list_promptBatches = list(iter_chunks(
                    list_points, self.obj_config.int_pointBatchSize))

                for int_px, int_py in list_points:
                    int_candidateCount += 1
                    list_debugPoints.append(
                        {
                            "tile_index": int_tileIdx,
                            "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                            "point_xy_tile": [int_px, int_py],
                            "point_xy_roi": [int_tx1 + int_px, int_ty1 + int_py],
                        }
                    )

            list_debugTiles.append(
                {
                    "tile_index": int_tileIdx,
                    "tile_xyxy": [int_tx1, int_ty1, int_tx2, int_ty2],
                    "num_points": len(list_points),
                    "use_point_prompts": bool(self.obj_config.bool_usePointPrompts),
                }
            )

            for list_pointChunk in list_promptBatches:
                if self.obj_config.bool_usePointPrompts:
                    if not list_pointChunk:
                        continue
                    list_results = self.obj_model(  # type: ignore[misc]
                        source=arr_tileBgr,
                        points=[[int_px, int_py]
                                for int_px, int_py in list_pointChunk],
                        labels=[1] * len(list_pointChunk),
                        **dict_predictCommon,
                    )
                else:
                    list_results = self.obj_model(  # type: ignore[misc]
                        source=arr_tileBgr,
                        **dict_predictCommon,
                    )
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
                    for arr_prevMask in list_keptMasks:
                        if calculate_binary_iou(arr_prevMask, arr_roiMask) >= self.obj_config.float_dedupIou:
                            bool_isDup = True
                            break
                    if bool_isDup:
                        continue

                    int_acceptedCount += 1
                    list_keptMasks.append(arr_roiMask)
                    list_keptBboxes.append(tuple_globalBox)
                    if arr_tileScores is not None and int_maskIdx < len(arr_tileScores):
                        list_keptScores.append(
                            float(arr_tileScores[int_maskIdx]))
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
        }
        return arr_masks, arr_scores, dict_debug

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

        # bbox가 실제 mask 외접 사각형이므로, 측정 span은 bbox 크기를 넘지 않도록 제한한다.
        int_horizontal = min(
            self.get_longest_span(arr_refinedMask, bool_horizontal=True),
            int_w,
        )
        int_vertical = min(
            self.get_longest_span(arr_refinedMask, bool_horizontal=False),
            int_h,
        )

        str_category = (
            "particle"
            if int_maskArea >= int(round(self.obj_config.float_particleAreaThreshold))
            else "fragment"
        )

        float_aspectRatio = None
        if str_category == "particle":
            int_longAxis = max(int_horizontal, int_vertical, 1)
            int_shortAxis = max(1, min(int_horizontal, int_vertical))
            float_aspectRatio = float(int_shortAxis / int_longAxis)

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
            float_aspectRatioWH=float_aspectRatio,
        )

    def create_overlay(
        self,
        arr_imageBgr: np.ndarray,
        list_objects: tp.List[ObjectMeasurement],
        list_masks: tp.List[np.ndarray],
    ) -> np.ndarray:
        """객체 마스크와 라벨을 원본 ROI 이미지 위에 시각화한다.

        Args:
            arr_imageBgr: ROI 이미지.
            list_objects: 시각화할 객체 측정 결과 리스트.
            list_masks: 각 객체에 대응하는 ROI 좌표계 binary mask 리스트.

        Returns:
            mask overlay, contour, bbox, 텍스트 라벨이 그려진 BGR 이미지.
        """
        arr_overlay = arr_imageBgr.copy()

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            tpl_color = (60, 220, 60) if obj_measurement.str_category == "particle" else (
                0, 165, 255)
            arr_overlay[arr_mask > 0] = (
                arr_overlay[arr_mask > 0].astype(np.float32) * 0.55
                + np.array(tpl_color, dtype=np.float32) * 0.45
            ).astype(np.uint8)

        for obj_measurement, arr_mask in zip(list_objects, list_masks):
            arr_contour = self.extract_largest_contour(arr_mask)
            if arr_contour is None:
                continue

            tpl_color = (0, 255, 0) if obj_measurement.str_category == "particle" else (
                0, 140, 255)
            cv2.drawContours(arr_overlay, [arr_contour], -1, tpl_color, 1)
            cv2.rectangle(
                arr_overlay,
                (obj_measurement.int_bboxX, obj_measurement.int_bboxY),
                (
                    obj_measurement.int_bboxX + obj_measurement.int_bboxWidth,
                    obj_measurement.int_bboxY + obj_measurement.int_bboxHeight,
                ),
                tpl_color,
                1,
            )

            int_labelX = obj_measurement.int_bboxX
            int_labelY = max(14, obj_measurement.int_bboxY - 4)
            if obj_measurement.str_category == "particle" and obj_measurement.float_aspectRatioWH is not None:
                str_label = (
                    f"P{obj_measurement.int_index} "
                    f"A={obj_measurement.int_maskArea} "
                    f"AR={obj_measurement.float_aspectRatioWH:.2f}"
                )
            else:
                str_label = f"F{obj_measurement.int_index} A={obj_measurement.int_maskArea}"

            cv2.putText(
                arr_overlay,
                str_label,
                (int_labelX, int_labelY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                tpl_color,
                1,
                cv2.LINE_AA,
            )

        return arr_overlay

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
        list_particleArs = [
            obj_item.float_aspectRatioWH
            for obj_item in list_particles
            if obj_item.float_aspectRatioWH is not None
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
            "model_weights_resolved_name": self.resolve_model_weights_path().name,
            "model_name": self.dict_modelConfig.get("model", self.obj_config.path_modelWeights.stem),
            "small_particle": bool(self.obj_config.bool_smallParticle),
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
            "particle_aspect_ratio_mean": None,
            "particle_aspect_ratio_median": None,
            "particle_aspect_ratio_std": None,
            "particle_aspect_ratio_min": None,
            "particle_aspect_ratio_max": None,
        }

        if list_particleArs:
            arr_particleArs = np.array(list_particleArs, dtype=np.float32)
            dict_summary.update(
                {
                    "particle_aspect_ratio_mean": float(np.mean(arr_particleArs)),
                    "particle_aspect_ratio_median": float(np.median(arr_particleArs)),
                    "particle_aspect_ratio_std": float(np.std(arr_particleArs)),
                    "particle_aspect_ratio_min": float(np.min(arr_particleArs)),
                    "particle_aspect_ratio_max": float(np.max(arr_particleArs)),
                }
            )

        return dict_summary

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
        arr_overlayFull[
            dict_roi["y_min"]:dict_roi["y_max"],
            dict_roi["x_min"]:dict_roi["x_max"],
        ] = arr_overlayRoi
        cv2.rectangle(
            arr_overlayFull,
            (dict_roi["x_min"], dict_roi["y_min"]),
            (dict_roi["x_max"], dict_roi["y_max"]),
            (255, 255, 0),
            2,
        )

        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "01_input.png"), arr_inputBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "02_input_roi.png"), arr_inputRoiBgr)
        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "03_overlay_roi.png"), arr_overlayRoi)
        cv2.imwrite(str(self.obj_config.path_outputDir /
                    "04_overlay_full.png"), arr_overlayFull)

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

        save_particle_distribution_histogram(
            path_particlesCsv=path_csvParticle,
            path_outputImage=self.obj_config.path_outputDir / "particle_dist.png",
            path_inputImage=self.obj_config.path_input,
        )

        with (self.obj_config.path_outputDir / "summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_summary, obj_f, ensure_ascii=False, indent=2)

        with (self.obj_config.path_outputDir / "objects.json").open("w", encoding="utf-8") as obj_f:
            json.dump([asdict(obj_item) for obj_item in list_objects],
                      obj_f, ensure_ascii=False, indent=2)

        with (self.obj_config.path_outputDir / "debug.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_debug, obj_f, ensure_ascii=False, indent=2)

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

    def process(self) -> Sam2AspectRatioResult:
        """단일 이미지에 대한 전체 파이프라인을 실행한다.

        Returns:
            측정 결과 리스트와 summary dict를 포함하는 `Sam2AspectRatioResult`.
        """
        arr_inputBgr = self.load_image_bgr()
        arr_inputRoiBgr, dict_roi = self.extract_inference_roi(arr_inputBgr)
        arr_masks, arr_scores, dict_debug = self.predict_tiled_point_prompts(
            arr_inputRoiBgr)

        list_objects: tp.List[ObjectMeasurement] = []
        list_validMasks: tp.List[np.ndarray] = []

        for int_index, arr_mask in enumerate(arr_masks):
            float_confidence = None
            if arr_scores is not None and int_index < len(arr_scores):
                float_confidence = float(arr_scores[int_index])

            obj_measurement = self.measure_mask(
                arr_mask, int_index=int_index, float_confidence=float_confidence)
            if obj_measurement is None:
                continue

            list_objects.append(obj_measurement)
            list_validMasks.append(
                self.refine_mask_for_area(arr_mask).astype(np.uint8))

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
        )

        return Sam2AspectRatioResult(
            list_objects=list_objects,
            dict_summary=dict_summary,
        )
