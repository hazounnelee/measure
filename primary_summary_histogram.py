#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_summary.json을 읽어 1차 입자 히스토그램을 재생성하는 독립 스크립트."""
import argparse
import json
import sys
from pathlib import Path

from utils.histograms import save_primary_batch_histograms


def main() -> None:
    obj_parser = argparse.ArgumentParser(
        description="batch_summary.json에서 1차 입자 thickness/density 히스토그램을 생성합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obj_parser.add_argument(
        "summary",
        metavar="BATCH_SUMMARY_JSON",
        help="batch_summary.json 경로",
    )
    obj_parser.add_argument(
        "--output_dir", "-o", default=None,
        help="히스토그램 저장 디렉토리. 미지정 시 summary 파일과 같은 디렉토리.",
    )
    obj_parser.add_argument(
        "--lot", default=None,
        help="차트 제목에 표시할 lot 이름. 미지정 시 summary 파일의 부모 디렉토리명 사용.",
    )
    obj_args = obj_parser.parse_args()

    path_summary = Path(obj_args.summary)
    if not path_summary.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {path_summary}", file=sys.stderr)
        sys.exit(1)

    with path_summary.open(encoding="utf-8") as obj_f:
        dict_batchSummary = json.load(obj_f)

    path_outputDir = Path(obj_args.output_dir) if obj_args.output_dir else path_summary.parent
    path_outputDir.mkdir(parents=True, exist_ok=True)

    str_lot = obj_args.lot or path_summary.parent.name

    print(f"[info] summary: {path_summary}")
    print(f"[info] output : {path_outputDir}")
    print(f"[info] lot    : {str_lot}")

    save_primary_batch_histograms(dict_batchSummary, path_outputDir, str_lot=str_lot)

    print("[done] batch_hist_thickness.png, batch_hist_thickness_per_image.png, batch_hist_density.png 저장 완료")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
