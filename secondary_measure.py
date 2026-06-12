#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2차 입자 분석 CLI 진입점."""
import argparse
import json
import sys
import time

from services.secondary_particle import run_secondary_particle_analysis, build_secondary_arg_parser
from configs import load_paths_config, parse_magnification, mag_to_scale_pixels
from utils.metrics import _SafeJSONEncoder

_DEFAULT_PATHS_CONFIG = "configs/paths.yaml"


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    obj_preParser = argparse.ArgumentParser(add_help=False)
    obj_preParser.add_argument("--config", default=_DEFAULT_PATHS_CONFIG)
    obj_preParser.add_argument("--magnification", default=None)
    obj_preArgs, _ = obj_preParser.parse_known_args()

    obj_parser = build_secondary_arg_parser()
    obj_parser.add_argument(
        "--config",
        default=_DEFAULT_PATHS_CONFIG,
        metavar="FILE",
        help=(
            f"경로 설정 YAML 파일 (기본값: {_DEFAULT_PATHS_CONFIG}). "
            "input / output_dir / model / model_cfg / device 를 지정할 수 있다."
        ),
    )
    obj_parser.add_argument(
        "--magnification", default=None,
        help="배율 (예: 1500, 3000, 20000, '20k'). scale_pixels를 자동 계산한다.",
    )

    dict_paths = load_paths_config(obj_preArgs.config)
    if dict_paths:
        obj_parser.set_defaults(**dict_paths)
        print(f"[config] {obj_preArgs.config} 에서 경로 설정 로드 ({', '.join(dict_paths.keys())})", flush=True)

    try:
        float_mag = parse_magnification(obj_preArgs.magnification)
    except ValueError as exc:
        print(f"[ERROR] --magnification: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    if float_mag is not None:
        obj_parser.set_defaults(scale_pixels=mag_to_scale_pixels(float_mag), scale_um=1.0)
        print(f"[scale] {float_mag:.0f}x → scale_pixels={mag_to_scale_pixels(float_mag):.4f} px/µm", flush=True)

    obj_args = obj_parser.parse_args()

    # area_threshold 자동 산출: (d0² × mag²) / 150000
    float_area_threshold = obj_args.area_threshold
    if float_area_threshold is None:
        if float_mag is not None:
            float_area_threshold = (obj_args.d0 ** 2 * float_mag ** 2) / 150000.0
            print(f"[threshold] d0={obj_args.d0}, mag={float_mag:.0f}x "
                  f"→ area_threshold={float_area_threshold:.1f}", flush=True)
        else:
            float_area_threshold = 1500.0

    float_start = time.perf_counter()

    dict_summary = run_secondary_particle_analysis(
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
        float_particleAreaThreshold=float_area_threshold,
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
        str_promptMode=obj_args.prompt_mode,
        float_scalePixels=obj_args.scale_pixels,
        float_scaleMicrometers=obj_args.scale_um,
        bool_retinaMasks=obj_args.retina_masks,
        bool_saveIndividualMasks=obj_args.save_mask_imgs,
        bool_useEqDiameter=obj_args.eq_diameter,
        int_preprocessWidth=obj_args.preprocess_width,
        int_numGpus=obj_args.num_gpus,
        bool_useOpenCV=obj_args.opencv,
        int_numNegativePoints=obj_args.num_negative_points,
        float_brightnessK=obj_args.brightness_k,
        float_punchOverlapThresh=obj_args.punch_overlap_thresh,
        bool_convexMasks=obj_args.convex_masks,
        bool_restoreMasks=obj_args.restore_masks,
        bool_debug=obj_args.debug,
    )

    print("===== 2차 입자 분석 결과 요약 =====")
    print(json.dumps(dict_summary, ensure_ascii=False, indent=2, cls=_SafeJSONEncoder))
    print(f"Elapsed time: {time.perf_counter() - float_start:.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
