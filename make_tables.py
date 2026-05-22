#!/usr/bin/env python3
"""batch_summary.json(대입경/소입경)으로부터 기초통계량 및 등급화 Excel 표 생성."""
import argparse
import json
import shutil
import sys
from pathlib import Path

# 지표별 (폴더명, per-image 값 키, batch 값 키, reverse)
_GRADE_METRICS = [
    ("입도_표준편차", "particle_size_std_um",         "particle_size_um.std",           False),
    ("타원도",        "particle_sphericity_prime_mean","particle_sphericity_prime.mean",  True),
    ("구형도",        "particle_sphericity_mean",      "particle_sphericity.mean",        True),
    ("미분_깨짐",     "fine_particle_ratio_percent",   "fine_particle_ratio_percent",     False),
]

# 출력 디렉토리에서 탐색할 파일 후보 (우선순위 순)
_ROI_CANDIDATES        = ["input_roi.png", "02_input_roi.png"]
_CLASSIFIED_CANDIDATES = ["classified.png", "06_pipeline_classified.png"]

_TEMPLATE = Path(__file__).parent / "tables.xlsx"


def _load(path: str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] 파일 없음: {p}", file=sys.stderr)
        sys.exit(1)
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _get(d: dict, key: str) -> float:
    """점(.) 구분 중첩 키 또는 단순 키로 float 값 반환."""
    v = d
    for part in key.split("."):
        v = v[part]
    return float(v)


def _quartiles(
    d1: dict | None, d2: dict | None, per_image_key: str
) -> tuple[float, float, float]:
    """두 배치의 per-image 값을 모아 Q1/Q2/Q3 반환."""
    import numpy as np
    vals = []
    for d in (d1, d2):
        if d is None:
            continue
        for img_id_entry in d.get("img_ids", []):
            for fe in img_id_entry.get("files", []):
                v = fe.get(per_image_key)
                if v is not None:
                    vals.append(float(v))
    if not vals:
        return 0.0, 0.0, 0.0
    return (
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 75)),
    )


def _grade_quartile(
    value: float, q1: float, q2: float, q3: float, reverse: bool = False
) -> int:
    """4분위수 기반 4등급 반환 (1=최우수)."""
    if not reverse:
        if value <= q1: return 1
        if value <= q2: return 2
        if value <= q3: return 3
        return 4
    else:
        if value >= q3: return 1
        if value >= q2: return 2
        if value >= q1: return 3
        return 4


def _grade_min(
    g: int, q1: float, q2: float, q3: float, reverse: bool
) -> float | None:
    """등급 g의 최솟값(하한). 하한이 없으면 None."""
    if not reverse:
        return [None, q1, q2, q3][g - 1]
    else:
        return [q3, q2, q1, None][g - 1]


def make_tables(
    d_large: dict | None,
    d_small: dict | None,
    path_template: Path,
    path_output: Path,
) -> None:
    import openpyxl

    shutil.copy2(path_template, path_output)
    wb = openpyxl.load_workbook(path_output)

    # ── 기초통계량 및 처리 시간 ──────────────────────────────────────────
    # 열: C=평균, D=중앙값, E=표준편차
    ws = wb["기초통계량 및 처리 시간"]
    stat_rows = [
        # (row_대입경, row_소입경, stats_dict_key)
        (3,  4,  "particle_size_um"),
        (5,  6,  "particle_sphericity_prime"),
        (7,  8,  "particle_sphericity"),
        (9,  10, "fine_particle_ratio_percent_stats"),
        (11, 12, "processing_time_sec"),
    ]
    for row_l, row_s, key in stat_rows:
        for row, d in ((row_l, d_large), (row_s, d_small)):
            if d is None:
                continue
            s = d[key]
            ws.cell(row=row, column=3).value = round(float(s["mean"]),   3)
            ws.cell(row=row, column=4).value = round(float(s["median"]), 3)
            ws.cell(row=row, column=5).value = round(float(s["std"]),    3)

    # ── 등급화 ────────────────────────────────────────────────────────────
    # 열: C=1등급, D=2등급, E=3등급, F=4등급  →  col = grade + 2
    ws2 = wb["등급화"]

    q1_size, q2_size, q3_size = _quartiles(d_large, d_small, "particle_size_std_um")
    q1_ell,  q2_ell,  q3_ell  = _quartiles(d_large, d_small, "particle_sphericity_prime_mean")
    q1_sph,  q2_sph,  q3_sph  = _quartiles(d_large, d_small, "particle_sphericity_mean")
    q1_frag, q2_frag, q3_frag = _quartiles(d_large, d_small, "fine_particle_ratio_percent")

    grade_rows = [
        # (row_대입경, row_소입경, batch_val_key, q1, q2, q3, reverse)
        (4,  5,  "particle_size_um.std",            q1_size, q2_size, q3_size, False),
        (7,  8,  "particle_sphericity_prime.mean",   q1_ell,  q2_ell,  q3_ell,  True),
        (10, 11, "particle_sphericity.mean",          q1_sph,  q2_sph,  q3_sph,  True),
        (13, 14, "fine_particle_ratio_percent",       q1_frag, q2_frag, q3_frag, False),
    ]
    for row_l, row_s, val_key, q1, q2, q3, reverse in grade_rows:
        for row, d in ((row_l, d_large), (row_s, d_small)):
            if d is None:
                continue
            value = _get(d, val_key)
            g = _grade_quartile(value, q1, q2, q3, reverse=reverse)
            min_val = _grade_min(g, q1, q2, q3, reverse)
            if min_val is not None:
                ws2.cell(row=row, column=g + 2).value = round(min_val, 4)

    wb.save(path_output)
    print(f"[done] {path_output}")


def _find(output_dir: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = output_dir / name
        if p.exists():
            return p
    return None


def export_grade_images(
    d: dict,
    label: str,
    path_outdir: Path,
    q1_size: float, q2_size: float, q3_size: float,
    q1_ell: float,  q2_ell: float,  q3_ell: float,
    q1_sph: float,  q2_sph: float,  q3_sph: float,
    q1_frag: float, q2_frag: float, q3_frag: float,
) -> None:
    """배치 내 각 이미지를 지표별 등급 폴더에 복사."""
    refs = {
        "입도_표준편차": (q1_size, q2_size, q3_size, False),
        "타원도":        (q1_ell,  q2_ell,  q3_ell,  True),
        "구형도":        (q1_sph,  q2_sph,  q3_sph,  True),
        "미분_깨짐":     (q1_frag, q2_frag, q3_frag, False),
    }

    copied = 0
    for img_id_entry in d.get("img_ids", []):
        for file_entry in img_id_entry.get("files", []):
            out_src = Path(file_entry.get("output_dir", ""))
            img_name = Path(file_entry.get("image_name", file_entry.get("input_path", "unknown"))).stem

            path_roi        = _find(out_src, _ROI_CANDIDATES)
            path_classified = _find(out_src, _CLASSIFIED_CANDIDATES)
            if path_roi is None or path_classified is None:
                print(f"  [skip] 이미지 없음: {out_src}")
                continue

            for metric_name, val_key, _, _ in _GRADE_METRICS:
                val = file_entry.get(val_key)
                if val is None:
                    continue
                q1, q2, q3, reverse = refs[metric_name]
                g = _grade_quartile(float(val), q1, q2, q3, reverse=reverse)

                dest = path_outdir / label / metric_name / f"grade_{g}" / img_name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path_roi,        dest / "input_roi.png")
                shutil.copy2(path_classified, dest / "classified.png")
                copied += 1

    print(f"  [{label}] {copied}개 이미지 복사 완료 → {path_outdir / label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="batch_summary.json → 기초통계량/등급화 Excel 표 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--large", metavar="JSON", default=None,
                        help="대입경 batch_summary.json 경로")
    parser.add_argument("--small", metavar="JSON", default=None,
                        help="소입경 batch_summary.json 경로")
    parser.add_argument("--template", metavar="XLSX", default=str(_TEMPLATE),
                        help="표 템플릿 xlsx 경로")
    parser.add_argument("-o", "--output", metavar="XLSX", default="batch_tables.xlsx",
                        help="출력 xlsx 경로")
    parser.add_argument("--grade-images", metavar="DIR", default=None,
                        help="등급별 이미지 쌍(input_roi+classified)을 내보낼 최상위 디렉토리")
    args = parser.parse_args()

    if args.large is None and args.small is None:
        parser.error("--large / --small 중 하나 이상 지정하세요.")

    d_large = _load(args.large)
    d_small = _load(args.small)

    path_template = Path(args.template)
    if not path_template.exists():
        print(f"[ERROR] 템플릿 없음: {path_template}", file=sys.stderr)
        sys.exit(1)

    make_tables(d_large, d_small, path_template, Path(args.output))

    if args.grade_images:
        path_gi = Path(args.grade_images)
        q1_size, q2_size, q3_size = _quartiles(d_large, d_small, "particle_size_std_um")
        q1_ell,  q2_ell,  q3_ell  = _quartiles(d_large, d_small, "particle_sphericity_prime_mean")
        q1_sph,  q2_sph,  q3_sph  = _quartiles(d_large, d_small, "particle_sphericity_mean")
        q1_frag, q2_frag, q3_frag = _quartiles(d_large, d_small, "fine_particle_ratio_percent")
        print("[grade-images]")
        if d_large is not None:
            export_grade_images(d_large, "large", path_gi,
                                q1_size, q2_size, q3_size,
                                q1_ell,  q2_ell,  q3_ell,
                                q1_sph,  q2_sph,  q3_sph,
                                q1_frag, q2_frag, q3_frag)
        if d_small is not None:
            export_grade_images(d_small, "small", path_gi,
                                q1_size, q2_size, q3_size,
                                q1_ell,  q2_ell,  q3_ell,
                                q1_sph,  q2_sph,  q3_sph,
                                q1_frag, q2_frag, q3_frag)


if __name__ == "__main__":
    main()
