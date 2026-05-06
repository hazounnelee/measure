from __future__ import annotations
import argparse
import json
import time
import typing as tp
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from core.schema import Sam2AspectRatioConfig, Sam2AspectRatioResult
from services.sam2_service import Sam2AspectRatioService
from utils.io import collect_input_groups
from utils.metrics import calculate_mean_from_optional_values, calculate_percentage, json_default

# ── Secondary-specific constants ──────────────────────────────────────────────
CONST_PARTICLE_AREA_THRESHOLD: float = 1500.0
CONST_SCALE_PIXELS: float = 147.0          # 20k: 147 px = 1 µm
CONST_SCALE_MICROMETERS: float = 1.0
CONST_SMALL_PARTICLE_SCALE_PIXELS: float = 371.0  # 50k: 371 px = 1 µm
CONST_SMALL_PARTICLE_SCALE_MICROMETERS: float = 1.0
CONST_DEFAULT_SMALL_PARTICLE: bool = False

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
CONST_DEFAULT_TILE_SIZE: int = 512
CONST_DEFAULT_TILE_STRIDE: int = 256
CONST_DEFAULT_POINTS_PER_TILE: int = 80
CONST_DEFAULT_POINT_MIN_DISTANCE: int = 14
CONST_DEFAULT_POINT_QUALITY_LEVEL: float = 0.03
CONST_DEFAULT_POINT_BATCH_SIZE: int = 32
CONST_DEFAULT_DEDUP_IOU: float = 0.60
CONST_DEFAULT_BBOX_DEDUP_IOU: float = 0.85
CONST_DEFAULT_USE_POINT_PROMPTS: bool = True


# ── Output-dir naming (secondary uses stem+ext format for uniqueness) ──────────
def _build_image_output_dir(
    path_outputRoot: Path,
    str_groupId: str,
    path_image: Path,
    bool_isBatch: bool,
) -> Path:
    if not bool_isBatch:
        return path_outputRoot
    str_dirName = f"{path_image.stem}{path_image.suffix.lower().replace('.', '_')}"
    return path_outputRoot / str_groupId / str_dirName


# ── Batch aggregation ──────────────────────────────────────────────────────────
def _pooled_stats(list_vals: tp.List[float]) -> tp.Dict[str, tp.Optional[float]]:
    if not list_vals:
        return {"mean": None, "median": None, "std": None}
    arr = np.array(list_vals, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "std": float(np.std(arr))}


def _build_img_id_summary(
    str_imgId: str,
    path_outputRoot: Path,
    list_fileSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    int_totalObjects = int(sum(d.get("num_total_objects", 0) for d in list_fileSummaries))
    int_particleCount = int(sum(d.get("num_particles", 0) for d in list_fileSummaries))
    int_fragmentCount = int(sum(d.get("num_fragments", 0) for d in list_fileSummaries))

    list_pooled_sphs: tp.List[float] = []
    list_pooled_sizes: tp.List[float] = []
    list_fine_ratios: tp.List[float] = []
    list_times: tp.List[float] = []
    for d in list_fileSummaries:
        list_pooled_sphs.extend(d.get("particle_sphericity_raw", []))
        list_pooled_sizes.extend(d.get("particle_size_um_raw", []))
        if d.get("fine_particle_ratio_percent") is not None:
            list_fine_ratios.append(float(d["fine_particle_ratio_percent"]))
        if d.get("processing_time_sec") is not None:
            list_times.append(float(d["processing_time_sec"]))

    return {
        "img_id": str_imgId,
        "output_dir": str(path_outputRoot / str_imgId),
        "num_images": len(list_fileSummaries),
        "num_total_objects": int_totalObjects,
        "num_particles": int_particleCount,
        "num_fragments": int_fragmentCount,
        "fragment_count_total": int(sum(d.get("fragment_count", 0) for d in list_fileSummaries)),
        "total_object_count": int_totalObjects,
        "normal_particle_count": int_particleCount,
        "fine_particle_count": int_fragmentCount,
        "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
        "fine_particle_ratio_percent_stats": _pooled_stats(list_fine_ratios),
        "fragment_count_mean_per_image": calculate_mean_from_optional_values(
            float(d.get("fragment_count", 0)) for d in list_fileSummaries),
        "particle_sphericity_mean": calculate_mean_from_optional_values(
            d.get("particle_sphericity_mean") for d in list_fileSummaries),
        "particle_sphericity": _pooled_stats(list_pooled_sphs),
        "particle_mean_size_um": calculate_mean_from_optional_values(
            d.get("particle_mean_size_um") for d in list_fileSummaries),
        "particle_size_um": _pooled_stats(list_pooled_sizes),
        "particle_sphericity_raw": list_pooled_sphs,
        "particle_size_um_raw": list_pooled_sizes,
        "processing_time_sec": _pooled_stats(list_times),
        "files": list_fileSummaries,
    }


def _build_batch_summary(
    path_input: Path,
    path_outputDir: Path,
    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    int_totalObjects = int(sum(d.get("num_total_objects", 0) for d in list_groupSummaries))
    int_particleCount = int(sum(d.get("num_particles", 0) for d in list_groupSummaries))
    int_fragmentCount = int(sum(d.get("num_fragments", 0) for d in list_groupSummaries))

    list_all_sphs: tp.List[float] = []
    list_all_sizes: tp.List[float] = []
    list_all_fine_ratios: tp.List[float] = []
    list_all_times: tp.List[float] = []
    for g in list_groupSummaries:
        list_all_sphs.extend(g.get("particle_sphericity_raw", []))
        list_all_sizes.extend(g.get("particle_size_um_raw", []))
        for f in g.get("files", []):
            if f.get("fine_particle_ratio_percent") is not None:
                list_all_fine_ratios.append(float(f["fine_particle_ratio_percent"]))
            if f.get("processing_time_sec") is not None:
                list_all_times.append(float(f["processing_time_sec"]))

    return {
        "input_path": str(path_input),
        "output_dir": str(path_outputDir),
        "num_img_ids": len(list_groupSummaries),
        "num_images": int(sum(d.get("num_images", 0) for d in list_groupSummaries)),
        "num_total_objects": int_totalObjects,
        "num_particles": int_particleCount,
        "num_fragments": int_fragmentCount,
        "fragment_count_total": int(sum(d.get("fragment_count_total", 0) for d in list_groupSummaries)),
        "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
        "fine_particle_ratio_percent_stats": _pooled_stats(list_all_fine_ratios),
        "fragment_count_mean_per_img_id": calculate_mean_from_optional_values(
            d.get("fragment_count_mean_per_image") for d in list_groupSummaries),
        "particle_sphericity_mean": calculate_mean_from_optional_values(
            d.get("particle_sphericity_mean") for d in list_groupSummaries),
        "particle_sphericity": _pooled_stats(list_all_sphs),
        "particle_mean_size_um": calculate_mean_from_optional_values(
            d.get("particle_mean_size_um") for d in list_groupSummaries),
        "particle_size_um": _pooled_stats(list_all_sizes),
        "processing_time_sec": _pooled_stats(list_all_times),
        "img_ids": list_groupSummaries,
    }


# ── Main runner ────────────────────────────────────────────────────────────────
def run_secondary_particle_analysis(
    str_input: str,
    str_outputDir: str,
    str_modelConfig: str,
    str_modelWeights: str,
    int_roiXMin: int = CONST_ROI_X_MIN,
    int_roiYMin: int = CONST_ROI_Y_MIN,
    int_roiXMax: int = CONST_ROI_X_MAX,
    int_roiYMax: int = CONST_ROI_Y_MAX,
    int_bboxEdgeMargin: int = CONST_BBOX_EDGE_MARGIN,
    int_tileEdgeMargin: int = CONST_TILE_EDGE_MARGIN,
    float_particleAreaThreshold: float = CONST_PARTICLE_AREA_THRESHOLD,
    float_maskBinarizeThreshold: float = CONST_MASK_BINARIZE_THRESHOLD,
    int_minValidMaskArea: int = CONST_MIN_VALID_MASK_AREA,
    int_maskMorphKernelSize: int = CONST_MASK_MORPH_KERNEL_SIZE,
    int_maskMorphOpenIterations: int = CONST_MASK_MORPH_OPEN_ITERATIONS,
    int_maskMorphCloseIterations: int = CONST_MASK_MORPH_CLOSE_ITERATIONS,
    int_imgSize: int = CONST_DEFAULT_IMAGE_SIZE,
    int_tileSize: int = CONST_DEFAULT_TILE_SIZE,
    int_stride: int = CONST_DEFAULT_TILE_STRIDE,
    int_pointsPerTile: int = CONST_DEFAULT_POINTS_PER_TILE,
    int_pointMinDistance: int = CONST_DEFAULT_POINT_MIN_DISTANCE,
    float_pointQualityLevel: float = CONST_DEFAULT_POINT_QUALITY_LEVEL,
    int_pointBatchSize: int = CONST_DEFAULT_POINT_BATCH_SIZE,
    float_dedupIou: float = CONST_DEFAULT_DEDUP_IOU,
    float_bboxDedupIou: float = CONST_DEFAULT_BBOX_DEDUP_IOU,
    bool_usePointPrompts: bool = CONST_DEFAULT_USE_POINT_PROMPTS,
    bool_smallParticle: bool = CONST_DEFAULT_SMALL_PARTICLE,
    float_scalePixels: float = 0.0,
    float_scaleMicrometers: float = 0.0,
    str_device: tp.Optional[str] = None,
    bool_retinaMasks: bool = CONST_DEFAULT_RETINA_MASKS,
    bool_saveIndividualMasks: bool = CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
    bool_useEqDiameter: bool = True,
    int_preprocessWidth: int = 1024,
) -> tp.Dict[str, tp.Any]:
    """Run secondary particle segmentation and measurement pipeline."""
    path_input = Path(str_input)
    path_outputRoot = Path(str_outputDir)
    list_inputGroups = collect_input_groups(path_input)
    bool_isBatch = path_input.is_dir()

    # Explicit scale overrides take priority; fall back to magnification presets
    if float_scalePixels <= 0.0 or float_scaleMicrometers <= 0.0:
        float_scalePixels = CONST_SMALL_PARTICLE_SCALE_PIXELS if bool_smallParticle else CONST_SCALE_PIXELS
        float_scaleMicrometers = CONST_SMALL_PARTICLE_SCALE_MICROMETERS if bool_smallParticle else CONST_SCALE_MICROMETERS

    def _create_config(str_groupId: str, path_image: Path) -> Sam2AspectRatioConfig:
        return Sam2AspectRatioConfig(
            path_input=path_image,
            path_outputDir=_build_image_output_dir(
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
            bool_smallParticle=bool_smallParticle,
            float_scalePixels=float_scalePixels,
            float_scaleMicrometers=float_scaleMicrometers,
            str_device=str_device,
            bool_retinaMasks=bool_retinaMasks,
            bool_saveIndividualMasks=bool_saveIndividualMasks,
            bool_useEqDiameter=bool_useEqDiameter,
            int_preprocessWidth=int_preprocessWidth,
        )

    if not bool_isBatch:
        str_groupId, list_imagePaths = list_inputGroups[0]
        print(f"[single] processing: {list_imagePaths[0].name}", flush=True)
        obj_result = Sam2AspectRatioService(_create_config(str_groupId, list_imagePaths[0])).process()
        print(f"[single] done: {list_imagePaths[0].name}", flush=True)
        return obj_result.dict_summary

    path_outputRoot.mkdir(parents=True, exist_ok=True)

    str_firstGroupId, list_firstImages = list_inputGroups[0]
    print(f"[batch] init model: {list_firstImages[0].name}", flush=True)
    obj_sharedService = Sam2AspectRatioService(_create_config(str_firstGroupId, list_firstImages[0]))
    obj_sharedService.initialize_model()

    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]] = []

    for str_groupId, list_imagePaths in tqdm(list_inputGroups, desc="groups", unit="group"):
        list_fileSummaries: tp.List[tp.Dict[str, tp.Any]] = []

        for path_image in tqdm(list_imagePaths, desc=str_groupId, unit="img", leave=False):
            obj_service = Sam2AspectRatioService(_create_config(str_groupId, path_image))
            obj_service.obj_model = obj_sharedService.obj_model
            obj_service.dict_modelConfig = dict(obj_sharedService.dict_modelConfig)
            float_t0 = time.perf_counter()
            obj_result = obj_service.process()
            float_elapsed = time.perf_counter() - float_t0
            dict_fs = dict(obj_result.dict_summary)
            dict_fs["img_id"] = str_groupId
            dict_fs["image_name"] = path_image.name
            dict_fs["image_path"] = str(path_image)
            dict_fs["processing_time_sec"] = round(float_elapsed, 3)
            list_fileSummaries.append(dict_fs)

        dict_groupSummary = _build_img_id_summary(str_groupId, path_outputRoot, list_fileSummaries)
        path_groupDir = path_outputRoot / str_groupId
        path_groupDir.mkdir(parents=True, exist_ok=True)
        with (path_groupDir / "img_id_summary.json").open("w", encoding="utf-8") as obj_f:
            json.dump(dict_groupSummary, obj_f, ensure_ascii=False, indent=2, default=json_default)
        list_groupSummaries.append(dict_groupSummary)
        print(f"[batch][group done] {str_groupId}  "
              f"particles={dict_groupSummary['num_particles']}  "
              f"fragments={dict_groupSummary['num_fragments']}", flush=True)

    dict_batchSummary = _build_batch_summary(path_input, path_outputRoot, list_groupSummaries)
    with (path_outputRoot / "batch_summary.json").open("w", encoding="utf-8") as obj_f:
        json.dump(dict_batchSummary, obj_f, ensure_ascii=False, indent=2, default=json_default)
    print(f"[batch] done: {dict_batchSummary['num_img_ids']} groups, "
          f"{dict_batchSummary['num_images']} images", flush=True)
    return dict_batchSummary


# ── CLI arg parser ─────────────────────────────────────────────────────────────
def build_secondary_arg_parser() -> argparse.ArgumentParser:
    str_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    obj_parser = argparse.ArgumentParser(
        description="SAM2로 2차 입자를 분할하고 particle/fragment 분류, 종횡비, 구형도를 측정합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obj_parser.add_argument("--input", default="img/secondary_test.jpg",
                            help="입력 이미지 또는 디렉터리 경로")
    obj_parser.add_argument("--output_dir", default=f"out_secondary_{str_ts}",
                            help="결과 저장 폴더")
    obj_parser.add_argument("--model_cfg", default="model/sam2.1_hiera_t.yaml",
                            help="SAM2 YAML 설정 파일 경로")
    obj_parser.add_argument("--model", default="model/sam2.1_hiera_base_plus.pt",
                            help="SAM2 가중치 파일 경로")
    obj_parser.add_argument("--roi_x_min", type=int, default=CONST_ROI_X_MIN)
    obj_parser.add_argument("--roi_y_min", type=int, default=CONST_ROI_Y_MIN)
    obj_parser.add_argument("--roi_x_max", type=int, default=CONST_ROI_X_MAX)
    obj_parser.add_argument("--roi_y_max", type=int, default=CONST_ROI_Y_MAX)
    obj_parser.add_argument("--bbox_edge_margin", type=int, default=CONST_BBOX_EDGE_MARGIN,
                            help="ROI 경계 근처 bbox 제외 margin")
    obj_parser.add_argument("--tile_edge_margin", type=int, default=CONST_TILE_EDGE_MARGIN,
                            help="타일 경계 근처 bbox 제외 margin")
    obj_parser.add_argument("--area_threshold", type=float, default=CONST_PARTICLE_AREA_THRESHOLD,
                            help="particle / fragment 분류 면적 threshold")
    obj_parser.add_argument("--mask_binarize_threshold", type=float,
                            default=CONST_MASK_BINARIZE_THRESHOLD)
    obj_parser.add_argument("--min_valid_mask_area", type=int, default=CONST_MIN_VALID_MASK_AREA)
    obj_parser.add_argument("--mask_morph_kernel_size", type=int,
                            default=CONST_MASK_MORPH_KERNEL_SIZE)
    obj_parser.add_argument("--mask_morph_open_iterations", type=int,
                            default=CONST_MASK_MORPH_OPEN_ITERATIONS)
    obj_parser.add_argument("--mask_morph_close_iterations", type=int,
                            default=CONST_MASK_MORPH_CLOSE_ITERATIONS)
    obj_parser.add_argument("--preprocess_width", type=int, default=1024,
                            help="전처리 이미지 가로 크기 (px). 세로는 비율에 맞게 자동 계산. 기본값: 1024.")
    obj_parser.add_argument("--imgsz", type=int, default=CONST_DEFAULT_IMAGE_SIZE,
                            help="SAM2 추론 이미지 크기")
    obj_parser.add_argument("--tile_size", type=int, default=CONST_DEFAULT_TILE_SIZE)
    obj_parser.add_argument("--stride", type=int, default=CONST_DEFAULT_TILE_STRIDE)
    obj_parser.add_argument("--points_per_tile", type=int, default=CONST_DEFAULT_POINTS_PER_TILE)
    obj_parser.add_argument("--point_min_distance", type=int,
                            default=CONST_DEFAULT_POINT_MIN_DISTANCE)
    obj_parser.add_argument("--point_quality_level", type=float,
                            default=CONST_DEFAULT_POINT_QUALITY_LEVEL)
    obj_parser.add_argument("--point_batch_size", type=int, default=CONST_DEFAULT_POINT_BATCH_SIZE)
    obj_parser.add_argument("--dedup_iou", type=float, default=CONST_DEFAULT_DEDUP_IOU)
    obj_parser.add_argument("--bbox_dedup_iou", type=float, default=CONST_DEFAULT_BBOX_DEDUP_IOU)
    obj_parser.add_argument("--use_point_prompts", action=argparse.BooleanOptionalAction,
                            default=CONST_DEFAULT_USE_POINT_PROMPTS)
    obj_parser.add_argument("--small_particle", action="store_true",
                            help="50k 배율 scale 사용 (371 px = 1 µm). "
                                 "--scale_pixels/--scale_um으로 직접 지정하면 이 값보다 우선됨")
    obj_parser.add_argument("--scale_pixels", type=float, default=0.0,
                            help="스케일 기준 pixel 수 (0 = 배율 preset 사용). "
                                 "20k=147, 50k=371")
    obj_parser.add_argument("--scale_um", type=float, default=0.0,
                            help="스케일 기준 µm 값 (0 = 배율 preset 사용)")
    obj_parser.add_argument("--eq_diameter", action=argparse.BooleanOptionalAction, default=True,
                            help="크기 측정 방법: equivalent circle diameter(기본값) "
                                 "또는 --no-eq_diameter로 수평/수직 span 평균 사용")
    obj_parser.add_argument("--save_mask_imgs", "--save_individual_masks",
                            dest="save_mask_imgs",
                            action=argparse.BooleanOptionalAction, default=True)
    obj_parser.add_argument("--retina_masks", action=argparse.BooleanOptionalAction, default=True)
    obj_parser.add_argument("--device", default=None,
                            help="추론 device (예: cpu, cuda:0)")
    return obj_parser
