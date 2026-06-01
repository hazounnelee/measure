#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1차 입자 두께 측정 CLI 진입점."""
import argparse
import json
import sys
import time

from services.primary_particle import run_primary_particle_analysis, build_primary_arg_parser
from configs import get_analysis_preset, load_paths_config, parse_magnification, mag_to_scale_pixels, mag_to_preset_key
from utils.metrics import _SafeJSONEncoder

_DEFAULT_PATHS_CONFIG = "configs/paths.yaml"


def main() -> None:
    # Windows 한글 출력 처리
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 1차 파싱: --config / --particle_type / --big/--small/--active / --magnification 만 먼저 읽는다.
    obj_preParser = argparse.ArgumentParser(add_help=False)
    obj_preParser.add_argument("--config", default=_DEFAULT_PATHS_CONFIG)
    obj_preParser.add_argument("--particle_type", default=None)
    obj_preParser.add_argument("--magnification", default=None)
    obj_preParser.add_argument("--big", action="store_true", default=False)
    obj_preParser.add_argument("--small", action="store_true", default=False)
    obj_preParser.add_argument("--active", action="store_true", default=False)
    obj_preArgs, _ = obj_preParser.parse_known_args()

    # 메인 파서 구성
    obj_parser = build_primary_arg_parser()
    obj_parser.add_argument(
        "--config",
        default=_DEFAULT_PATHS_CONFIG,
        metavar="FILE",
        help=(
            f"경로 설정 YAML 파일 (기본값: {_DEFAULT_PATHS_CONFIG}). "
            "input / output_dir / model / model_cfg / device 를 지정할 수 있다."
        ),
    )

    # 우선순위 (낮 → 높):
    #   parser default → paths config → preset → 배율 스케일 → CLI 인자

    # 1) paths config 적용
    dict_paths = load_paths_config(obj_preArgs.config)
    if dict_paths:
        obj_parser.set_defaults(**dict_paths)
        print(f"[config] {obj_preArgs.config} 에서 경로 설정 로드 ({', '.join(dict_paths.keys())})", flush=True)

    # 2) preset 적용
    try:
        float_mag = parse_magnification(obj_preArgs.magnification)
    except ValueError as exc:
        print(f"[ERROR] --magnification: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    # --big/--small/--active 플래그에서 particle_type 도출 (--particle_type 명시 시 우선)
    str_effective_type = obj_preArgs.particle_type
    if str_effective_type is None:
        if obj_preArgs.big:
            str_effective_type = "acicular"
        elif obj_preArgs.small:
            str_effective_type = "plate"
        elif obj_preArgs.active:
            str_effective_type = "active"

    if str_effective_type is not None:
        str_preset_key = mag_to_preset_key(float_mag) if float_mag else "20k"
        dict_preset = get_analysis_preset(str_effective_type, str_preset_key)
        if dict_preset:
            obj_parser.set_defaults(**dict_preset)
            print(f"[preset] {str_effective_type}/{str_preset_key} 프리셋 적용 ({len(dict_preset)}개 파라미터)", flush=True)
        else:
            print(f"[preset] 알 수 없는 조합: {str_effective_type}/{str_preset_key} (기본값 사용)", flush=True)
        if obj_preArgs.particle_type is None:
            obj_parser.set_defaults(particle_type=str_effective_type)

    # 3) 배율에서 scale_pixels 자동 계산 (preset보다 우선, CLI보다 후순위)
    if float_mag is not None:
        obj_parser.set_defaults(scale_pixels=mag_to_scale_pixels(float_mag), scale_um=1.0)
        print(f"[scale] {float_mag:.0f}x → scale_pixels={mag_to_scale_pixels(float_mag):.4f} px/µm", flush=True)

    # 4) 전체 파싱: CLI 인자가 최우선
    obj_args = obj_parser.parse_args()
    float_start = time.perf_counter()

    dict_summary = run_primary_particle_analysis(
        str_input=obj_args.input,
        str_outputDir=obj_args.output_dir,
        str_modelWeights=obj_args.model,
        str_modelConfig=obj_args.model_cfg,
        str_device=obj_args.device,
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
        bool_autoCenterCrop=obj_args.auto_center_crop,
        float_centerCropRatio=obj_args.center_crop_ratio,
        bool_saveIndividualMasks=obj_args.save_mask_imgs,
        bool_retinaMasks=obj_args.retina_masks,
        str_particleType=obj_args.particle_type or "unknown",
        str_magnification=obj_args.magnification or "unknown",
        str_particleMode=obj_args.particle_mode,
        bool_autoDetectSphere=obj_args.auto_detect_sphere,
        float_sphereCapFraction=obj_args.sphere_cap_fraction,
        str_measureMode=obj_args.measure_mode,
        bool_lsdAdaptiveThresh=obj_args.lsd_adaptive_thresh,
        bool_arScreen=obj_args.ar_screen,
        int_lsdMinLengthPx=obj_args.min_length,
        bool_fuseContours=obj_args.fuse,
        bool_advancedFuseContours=obj_args.advanced_fuse,
        int_preprocessWidth=obj_args.preprocess_width,
    )

    print("===== 1차 입자 분석 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2, cls=_SafeJSONEncoder))
    print(f"Elapsed time: {time.perf_counter() - float_start:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
