from __future__ import annotations
import argparse
import dataclasses
import json
import os
import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Queue

import numpy as np
from tqdm import tqdm

import cv2
from core.schema import Sam2AspectRatioConfig, Sam2AspectRatioResult
from services.sam2_service import Sam2AspectRatioService, CONST_PREPROCESS_BOTTOM_CROP, CONST_PREPROCESS_WIDTH
from utils.io import collect_input_groups
from utils.metrics import calculate_mean_from_optional_values, calculate_percentage, json_dump_safe, pooled_stats
from utils.histograms import save_secondary_batch_histograms

# ── Secondary-specific constants ──────────────────────────────────────────────
CONST_PARTICLE_AREA_THRESHOLD: float = 1500.0
CONST_SCALE_PIXELS: float = 74.0           # 20k @ 1024px: 74 px = 1 µm
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
CONST_DEFAULT_TILE_SIZE: int = 512
CONST_DEFAULT_TILE_STRIDE: int = 256
CONST_DEFAULT_POINTS_PER_TILE: int = 80
CONST_DEFAULT_POINT_MIN_DISTANCE: int = 14
CONST_DEFAULT_POINT_QUALITY_LEVEL: float = 0.03
CONST_DEFAULT_POINT_BATCH_SIZE: int = 32
CONST_DEFAULT_DEDUP_IOU: float = 0.60
CONST_DEFAULT_BBOX_DEDUP_IOU: float = 0.85
CONST_DEFAULT_USE_POINT_PROMPTS: bool = True


class SecondaryParticleService(Sam2AspectRatioService):
    """2차 입자 전용 서비스 — width-only resize + bottom crop으로 load."""

    def load_image_bgr(self) -> np.ndarray:
        arr_image = cv2.imread(str(self.obj_config.path_input), cv2.IMREAD_COLOR)
        if arr_image is None:
            raise FileNotFoundError(
                f"이미지를 읽을 수 없습니다: {self.obj_config.path_input}")
        int_w = max(1, self.obj_config.int_preprocessWidth)
        int_crop = round(int_w * CONST_PREPROCESS_BOTTOM_CROP / CONST_PREPROCESS_WIDTH)
        int_img_h, int_img_w = arr_image.shape[:2]
        if int_img_w != int_w:
            int_new_h = max(1, round(int_img_h * int_w / int_img_w))
            arr_image = cv2.resize(arr_image, (int_w, int_new_h), interpolation=cv2.INTER_LINEAR)
        int_h_final = max(1, arr_image.shape[0] - int_crop)
        return arr_image[:int_h_final, :]


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

def _build_img_id_summary(
    str_imgId: str,
    path_outputRoot: Path,
    list_fileSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    int_totalObjects = int(sum((d.get("num_total_objects") or 0) for d in list_fileSummaries))
    int_particleCount = int(sum((d.get("num_particles") or 0) for d in list_fileSummaries))
    int_fragmentCount = int(sum((d.get("num_fragments") or 0) for d in list_fileSummaries))

    list_pooled_sphs: tp.List[float] = []
    list_pooled_sphs_prime: tp.List[float] = []
    list_pooled_sizes: tp.List[float] = []
    list_fine_ratios: tp.List[float] = []
    list_times: tp.List[float] = []
    for d in list_fileSummaries:
        list_pooled_sphs.extend(d.get("particle_sphericity_raw") or [])
        list_pooled_sphs_prime.extend(d.get("particle_sphericity_prime_raw") or [])
        list_pooled_sizes.extend(d.get("particle_size_um_raw") or [])
        if d.get("fine_particle_ratio_percent") is not None:
            try:
                list_fine_ratios.append(float(d["fine_particle_ratio_percent"]))
            except (TypeError, ValueError):
                pass
        if d.get("processing_time_sec") is not None:
            try:
                list_times.append(float(d["processing_time_sec"]))
            except (TypeError, ValueError):
                pass

    return {
        "img_id": str_imgId,
        "output_dir": str(path_outputRoot / str_imgId),
        "num_images": len(list_fileSummaries),
        "num_total_objects": int_totalObjects,
        "num_particles": int_particleCount,
        "num_fragments": int_fragmentCount,
        "fragment_count_total": int(sum((d.get("fragment_count") or 0) for d in list_fileSummaries)),
        "total_object_count": int_totalObjects,
        "normal_particle_count": int_particleCount,
        "fine_particle_count": int_fragmentCount,
        "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
        "fine_particle_ratio_percent_stats": pooled_stats(list_fine_ratios),
        "fragment_count_mean_per_image": calculate_mean_from_optional_values(
            float(d.get("fragment_count") or 0) for d in list_fileSummaries),
        "particle_sphericity_mean": calculate_mean_from_optional_values(
            d.get("particle_sphericity_mean") for d in list_fileSummaries),
        "particle_sphericity": pooled_stats(list_pooled_sphs),
        "particle_sphericity_prime_mean": calculate_mean_from_optional_values(
            d.get("particle_sphericity_prime_mean") for d in list_fileSummaries),
        "particle_sphericity_prime": pooled_stats(list_pooled_sphs_prime),
        "particle_mean_size_um": calculate_mean_from_optional_values(
            d.get("particle_mean_size_um") for d in list_fileSummaries),
        "particle_size_um": pooled_stats(list_pooled_sizes),
        "particle_sphericity_raw": list_pooled_sphs,
        "particle_sphericity_prime_raw": list_pooled_sphs_prime,
        "particle_size_um_raw": list_pooled_sizes,
        "processing_time_sec": pooled_stats(list_times),
        "files": list_fileSummaries,
    }


def _build_batch_summary(
    path_input: Path,
    path_outputDir: Path,
    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]],
) -> tp.Dict[str, tp.Any]:
    int_totalObjects = int(sum((d.get("num_total_objects") or 0) for d in list_groupSummaries))
    int_particleCount = int(sum((d.get("num_particles") or 0) for d in list_groupSummaries))
    int_fragmentCount = int(sum((d.get("num_fragments") or 0) for d in list_groupSummaries))

    list_all_sphs: tp.List[float] = []
    list_all_sphs_prime: tp.List[float] = []
    list_all_sizes: tp.List[float] = []
    list_all_fine_ratios: tp.List[float] = []
    list_all_times: tp.List[float] = []
    for g in list_groupSummaries:
        list_all_sphs.extend(g.get("particle_sphericity_raw") or [])
        list_all_sphs_prime.extend(g.get("particle_sphericity_prime_raw") or [])
        list_all_sizes.extend(g.get("particle_size_um_raw") or [])
        for f in g.get("files") or []:
            if f.get("fine_particle_ratio_percent") is not None:
                try:
                    list_all_fine_ratios.append(float(f["fine_particle_ratio_percent"]))
                except (TypeError, ValueError):
                    pass
            if f.get("processing_time_sec") is not None:
                try:
                    list_all_times.append(float(f["processing_time_sec"]))
                except (TypeError, ValueError):
                    pass

    return {
        "input_path": str(path_input),
        "output_dir": str(path_outputDir),
        "num_img_ids": len(list_groupSummaries),
        "num_images": int(sum((d.get("num_images") or 0) for d in list_groupSummaries)),
        "num_total_objects": int_totalObjects,
        "num_particles": int_particleCount,
        "num_fragments": int_fragmentCount,
        "fragment_count_total": int(sum((d.get("fragment_count_total") or 0) for d in list_groupSummaries)),
        "fine_particle_ratio_percent": calculate_percentage(int_fragmentCount, int_totalObjects),
        "fine_particle_ratio_percent_stats": pooled_stats(list_all_fine_ratios),
        "fragment_count_mean_per_img_id": calculate_mean_from_optional_values(
            d.get("fragment_count_mean_per_image") for d in list_groupSummaries),
        "particle_sphericity_mean": calculate_mean_from_optional_values(
            d.get("particle_sphericity_mean") for d in list_groupSummaries),
        "particle_sphericity": pooled_stats(list_all_sphs),
        "particle_sphericity_prime_mean": calculate_mean_from_optional_values(
            d.get("particle_sphericity_prime_mean") for d in list_groupSummaries),
        "particle_sphericity_prime": pooled_stats(list_all_sphs_prime),
        "particle_mean_size_um": calculate_mean_from_optional_values(
            d.get("particle_mean_size_um") for d in list_groupSummaries),
        "particle_size_um": pooled_stats(list_all_sizes),
        "processing_time_sec": pooled_stats(list_all_times),
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
    str_promptMode: str = "hct",
    float_scalePixels: tp.Optional[float] = None,
    float_scaleMicrometers: tp.Optional[float] = None,
    str_device: tp.Optional[str] = None,
    bool_retinaMasks: bool = CONST_DEFAULT_RETINA_MASKS,
    bool_saveIndividualMasks: bool = CONST_DEFAULT_SAVE_INDIVIDUAL_MASKS,
    bool_useEqDiameter: bool = True,
    int_preprocessWidth: int = 1024,
    int_numGpus: int = 1,
    bool_useOpenCV: bool = False,
    bool_debug: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Run secondary particle segmentation and measurement pipeline."""
    path_input = Path(str_input)
    path_outputRoot = Path(str_outputDir)
    list_inputGroups = collect_input_groups(path_input)
    bool_isBatch = path_input.is_dir()

    # Explicit scale overrides take priority; fall back to magnification presets
    if float_scalePixels is None or float_scaleMicrometers is None:
        float_scalePixels = CONST_SCALE_PIXELS
        float_scaleMicrometers = CONST_SCALE_MICROMETERS

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
            str_promptMode=str_promptMode,
            float_scalePixels=float_scalePixels,
            float_scaleMicrometers=float_scaleMicrometers,
            str_device=str_device,
            bool_retinaMasks=bool_retinaMasks,
            bool_saveIndividualMasks=bool_saveIndividualMasks,
            bool_useEqDiameter=bool_useEqDiameter,
            int_preprocessWidth=int_preprocessWidth,
            bool_useOpenCV=bool_useOpenCV,
            bool_debug=bool_debug,
        )

    if not bool_isBatch:
        str_groupId, list_imagePaths = list_inputGroups[0]
        print(f"[single] processing: {list_imagePaths[0].name}", flush=True)
        obj_result = SecondaryParticleService(_create_config(str_groupId, list_imagePaths[0])).process()
        print(f"[single] done: {list_imagePaths[0].name}", flush=True)
        return obj_result.dict_summary

    path_outputRoot.mkdir(parents=True, exist_ok=True)

    # GPU 디바이스 목록 결정
    list_devices: tp.List[tp.Optional[str]] = []
    if int_numGpus > 1:
        try:
            import torch
            int_avail = torch.cuda.device_count()
            int_base = 0
            if str_device and ":" in str_device:
                try:
                    int_base = int(str_device.split(":")[-1])
                except ValueError:
                    int_base = 0
            list_devices = [
                f"cuda:{int_base + i}"
                for i in range(min(int_numGpus, int_avail - int_base))
            ]
        except Exception:
            pass
    if not list_devices:
        list_devices = [str_device]  # 단일 디바이스 (None 포함)

    # 디바이스별 모델 초기화 (OpenCV 모드에서는 생략)
    str_firstGroupId, list_firstImages = list_inputGroups[0]
    list_gpu_services: tp.List[SecondaryParticleService] = []
    for str_dev in list_devices:
        cfg_dev = dataclasses.replace(
            _create_config(str_firstGroupId, list_firstImages[0]),
            str_device=str_dev,
        )
        obj_svc = SecondaryParticleService(cfg_dev)
        if not bool_useOpenCV:
            print(f"[batch] model initialized on {str_dev or 'auto'}", flush=True)
            obj_svc.initialize_model()
        list_gpu_services.append(obj_svc)

    # 모델을 워커에 라운드로빈으로 분배하는 큐
    obj_gpu_queue: Queue = Queue()
    for obj_svc in list_gpu_services:
        obj_gpu_queue.put(obj_svc)

    list_groupSummaries: tp.List[tp.Dict[str, tp.Any]] = []

    for str_groupId, list_imagePaths in tqdm(list_inputGroups, desc="groups", unit="group"):

        def _run_image(path_image: Path) -> tp.Optional[tp.Dict[str, tp.Any]]:
            path_imageOutDir = _build_image_output_dir(path_outputRoot, str_groupId, path_image, bool_isBatch)
            path_cachedSummary = path_imageOutDir / "summary.json"
            if path_cachedSummary.exists():
                try:
                    with path_cachedSummary.open(encoding="utf-8") as obj_f:
                        dict_fs = json.load(obj_f)
                    dict_fs.setdefault("img_id", str_groupId)
                    dict_fs.setdefault("image_name", path_image.name)
                    dict_fs.setdefault("image_path", str(path_image))
                    print(f"  [skip] {path_image.name} (이미 처리됨)", flush=True)
                    return dict_fs
                except Exception:
                    pass  # 읽기 실패 시 재처리

            if bool_useOpenCV:
                # OpenCV 모드: 모델 공유 불필요 → 큐 경합 없이 독립 실행
                try:
                    obj_service = SecondaryParticleService(_create_config(str_groupId, path_image))
                    float_t0 = time.perf_counter()
                    obj_result = obj_service.process()
                    dict_fs = dict(obj_result.dict_summary)
                    dict_fs["img_id"] = str_groupId
                    dict_fs["image_name"] = path_image.name
                    dict_fs["image_path"] = str(path_image)
                    dict_fs["processing_time_sec"] = round(time.perf_counter() - float_t0, 3)
                    return dict_fs
                except Exception as exc:
                    print(f"[WARN] {path_image.name} 처리 실패 (skip): {exc}", flush=True)
                    return None
            # SAM2 모드: GPU 큐에서 모델 인스턴스 빌림
            obj_gpu = obj_gpu_queue.get()
            try:
                obj_service = SecondaryParticleService(
                    dataclasses.replace(
                        _create_config(str_groupId, path_image),
                        str_device=obj_gpu.obj_config.str_device,
                    )
                )
                obj_service.obj_model = obj_gpu.obj_model
                obj_service.dict_modelConfig = dict(obj_gpu.dict_modelConfig)
                float_t0 = time.perf_counter()
                obj_result = obj_service.process()
                dict_fs = dict(obj_result.dict_summary)
                dict_fs["img_id"] = str_groupId
                dict_fs["image_name"] = path_image.name
                dict_fs["image_path"] = str(path_image)
                dict_fs["processing_time_sec"] = round(time.perf_counter() - float_t0, 3)
                return dict_fs
            except Exception as exc:
                print(f"[WARN] {path_image.name} 처리 실패 (skip): {exc}", flush=True)
                return None
            finally:
                obj_gpu_queue.put(obj_gpu)

        int_workers = (
            min(os.cpu_count() or 4, 8) if bool_useOpenCV else len(list_gpu_services)
        )
        list_fileSummaries: tp.List[tp.Dict[str, tp.Any]]
        if int_workers == 1:
            list_fileSummaries = [
                r for r in (
                    _run_image(p)
                    for p in tqdm(list_imagePaths, desc=str_groupId, unit="img", leave=False)
                ) if r is not None
            ]
        else:
            with ThreadPoolExecutor(max_workers=int_workers) as executor:
                list_fileSummaries = [
                    r for r in tqdm(
                        executor.map(_run_image, list_imagePaths),
                        total=len(list_imagePaths), desc=str_groupId, unit="img", leave=False,
                    ) if r is not None
                ]

        dict_groupSummary = _build_img_id_summary(str_groupId, path_outputRoot, list_fileSummaries)
        dict_groupSummary["num_images_attempted"] = len(list_imagePaths)
        int_failed = len(list_imagePaths) - len(list_fileSummaries)
        if int_failed > 0:
            print(f"[batch][group {str_groupId}] {int_failed}개 이미지 처리 실패", flush=True)
        path_groupDir = path_outputRoot / str_groupId
        path_groupDir.mkdir(parents=True, exist_ok=True)
        with (path_groupDir / "img_id_summary.json").open("w", encoding="utf-8") as obj_f:
            json_dump_safe(dict_groupSummary, obj_f)
        list_groupSummaries.append(dict_groupSummary)
        print(f"[batch][group done] {str_groupId}  "
              f"particles={dict_groupSummary['num_particles']}  "
              f"fragments={dict_groupSummary['num_fragments']}", flush=True)

    dict_batchSummary = _build_batch_summary(path_input, path_outputRoot, list_groupSummaries)
    with (path_outputRoot / "batch_summary.json").open("w", encoding="utf-8") as obj_f:
        json_dump_safe(dict_batchSummary, obj_f)
    try:
        save_secondary_batch_histograms(
            dict_batchSummary, path_outputRoot, str_lot=path_input.stem)
    except Exception as exc:
        print(f"[WARN] batch 히스토그램 저장 실패 (결과에는 영향 없음): {exc}", flush=True)
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
    obj_parser.add_argument("--input", default="img/synthetic_secondary.jpg",
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
    obj_parser.add_argument("--preprocess_width", type=lambda x: max(64, int(x)),
                            default=1024,
                            help="전처리 이미지 가로 크기 (px, 최솟값 64). 세로는 비율에 맞게 자동 계산. 기본값: 1024.")
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
    obj_parser.add_argument("--prompt_mode", default="hct", choices=["hct", "legacy"],
                            help="포인트 프롬프트 추출 방식. "
                                 "'hct': HCT+CC 기반 (기본값). "
                                 "'legacy': goodFeaturesToTrack 텍스처 기반 (negative 없음).")
    obj_parser.add_argument("--scale_pixels", type=float, default=None,
                            help="스케일 기준 pixel 수. 미지정 시 --magnification 또는 기본값(74) 사용. "
                                 "20k@1024=74, 50k@1024=185")
    obj_parser.add_argument("--scale_um", type=float, default=None,
                            help="스케일 기준 µm 값. 미지정 시 1.0 사용.")
    obj_parser.add_argument("--eq_diameter", action=argparse.BooleanOptionalAction, default=True,
                            help="크기 측정 방법: equivalent circle diameter(기본값) "
                                 "또는 --no-eq_diameter로 수평/수직 span 평균 사용")
    obj_parser.add_argument("--save_mask_imgs", "--save_individual_masks",
                            dest="save_mask_imgs",
                            action=argparse.BooleanOptionalAction, default=True)
    obj_parser.add_argument("--retina_masks", action=argparse.BooleanOptionalAction, default=True)
    obj_parser.add_argument("--device", default=None,
                            help="추론 device (예: cpu, cuda:0)")
    obj_parser.add_argument("--num_gpus", type=int, default=1,
                            help="멀티 GPU 병렬 처리 수. GPU가 여러 장이면 이미지를 분산 처리. 기본값: 1.")
    obj_parser.add_argument("--opencv", action=argparse.BooleanOptionalAction, default=False,
                            help="SAM2 대신 OpenCV CLAHE+Otsu 기반 세그멘테이션을 사용한다. 빠르지만 단순함.")
    obj_parser.add_argument("--debug", action="store_true", default=False,
                            help="디버그 이미지/파일 저장 (tiles, prompts, masks_raw, overlay, CSV, JSON 등).")
    return obj_parser
