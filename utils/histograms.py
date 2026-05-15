from __future__ import annotations
import csv
import math
import typing as tp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


def get_lot_number_from_input_path(path_input: Path) -> str:
    return path_input.resolve().parent.name or "UnknownLot"


def load_particle_mean_sizes_from_csv(path_csv: Path) -> tp.List[float]:
    if not path_csv.exists():
        return []
    list_vals: tp.List[float] = []
    with path_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["float_eqDiameterUm"])
                if not math.isnan(v):
                    list_vals.append(v)
            except (KeyError, ValueError):
                pass
    return list_vals


def load_particle_sphericities_from_csv(path_csv: Path) -> tp.List[float]:
    if not path_csv.exists():
        return []
    list_vals: tp.List[float] = []
    with path_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                s = row.get("float_sphericity", "")
                if s and s.lower() not in ("none", "nan", ""):
                    v = float(s)
                    if not math.isnan(v):
                        list_vals.append(v)
            except ValueError:
                pass
    return list_vals


def save_particle_distribution_histogram(
    path_particlesCsv: Path,
    path_outputImage: Path,
    path_inputImage: Path,
) -> None:
    list_sizes = load_particle_mean_sizes_from_csv(path_particlesCsv)
    str_lot = get_lot_number_from_input_path(path_inputImage)

    obj_fig = Figure(figsize=(10, 6), dpi=100)
    obj_ax = obj_fig.add_subplot(111)
    try:
        obj_ax.set_title(f"{str_lot} — Secondary Particle Size", fontsize=18)
        obj_ax.set_xlabel("Equivalent Diameter (µm)", fontsize=14)
        obj_ax.set_ylabel("Count", fontsize=14)
        obj_ax.tick_params(labelsize=12)

        if list_sizes:
            arr_v = np.array(list_sizes, dtype=np.float32)
            int_bins = int(np.clip(np.sqrt(len(arr_v)), 5, 20))
            float_mean = float(np.mean(arr_v))
            obj_ax.hist(arr_v, bins=int_bins, alpha=0.65, color="#5588ff",
                        edgecolor="#333333", linewidth=0.8, label="Particle")
            obj_ax.axvline(float_mean, linestyle="--", linewidth=1.5, color="#5588ff")
            obj_ax.text(float_mean, obj_ax.get_ylim()[1] * 0.95,
                        f"  mean: {float_mean:.3f} µm",
                        color="#5588ff", fontsize=11, va="top")
            obj_ax.legend(fontsize=12)
            obj_ax.grid(axis="y", linestyle="--", alpha=0.3)
        else:
            obj_ax.text(0.5, 0.5, "No particle data", ha="center", va="center",
                        transform=obj_ax.transAxes, fontsize=13, color="#666666")

        obj_fig.tight_layout()
        obj_fig.savefig(str(path_outputImage), bbox_inches="tight")
    finally:
        obj_fig.clf()


def save_sphericity_distribution_histogram(
    path_particlesCsv: Path,
    path_outputImage: Path,
    path_inputImage: Path,
) -> None:
    list_sphs = load_particle_sphericities_from_csv(path_particlesCsv)
    str_lot = get_lot_number_from_input_path(path_inputImage)

    obj_fig = Figure(figsize=(10, 6), dpi=100)
    obj_ax = obj_fig.add_subplot(111)
    try:
        obj_ax.set_title(f"{str_lot} — Secondary Particle Sphericity", fontsize=18)
        obj_ax.set_xlabel("Sphericity", fontsize=14)
        obj_ax.set_ylabel("Count", fontsize=14)
        obj_ax.tick_params(labelsize=12)

        if list_sphs:
            arr_v = np.array(list_sphs, dtype=np.float32)
            int_bins = int(np.clip(np.sqrt(len(arr_v)), 5, 20))
            float_mean = float(np.mean(arr_v))
            obj_ax.hist(arr_v, bins=int_bins, alpha=0.65, color="#44cc44",
                        edgecolor="#333333", linewidth=0.8, label="Particle")
            obj_ax.axvline(float_mean, linestyle="--", linewidth=1.5, color="#44cc44")
            obj_ax.text(float_mean, obj_ax.get_ylim()[1] * 0.95,
                        f"  mean: {float_mean:.3f}",
                        color="#44cc44", fontsize=11, va="top")
            obj_ax.set_xlim(0, 1)
            obj_ax.legend(fontsize=12)
            obj_ax.grid(axis="y", linestyle="--", alpha=0.3)
        else:
            obj_ax.text(0.5, 0.5, "No sphericity data", ha="center", va="center",
                        transform=obj_ax.transAxes, fontsize=13, color="#666666")

        obj_fig.tight_layout()
        obj_fig.savefig(str(path_outputImage), bbox_inches="tight")
    finally:
        obj_fig.clf()


# ── Batch histogram helpers ────────────────────────────────────────────────────

def _draw_quartile_hist(
    obj_ax: Axes,
    arr_v: np.ndarray,
    str_color: str,
    str_unit: str,
    float_xlim_min: tp.Optional[float],
    float_xlim_max: tp.Optional[float],
    int_bins_factor: int = 1,
) -> None:
    """Draw a histogram with Q1/Q2/Q3 markers and IQR shading on an existing Axes."""
    int_bins = int(np.clip(np.sqrt(len(arr_v)) * int_bins_factor, 5 * int_bins_factor, 150))
    obj_ax.hist(arr_v, bins=int_bins, alpha=0.65, color=str_color,
                edgecolor="#333333", linewidth=0.8)

    float_q1   = float(np.percentile(arr_v, 25))
    float_q2   = float(np.median(arr_v))
    float_q3   = float(np.percentile(arr_v, 75))
    float_mean = float(np.mean(arr_v))

    if float_xlim_min is not None and float_xlim_max is not None:
        obj_ax.set_xlim(float_xlim_min, float_xlim_max)
    elif float_xlim_min is not None:
        obj_ax.set_xlim(left=float_xlim_min)
    elif float_xlim_max is not None:
        obj_ax.set_xlim(right=float_xlim_max)

    float_ymax = obj_ax.get_ylim()[1]
    obj_ax.axvspan(float_q1, float_q3, alpha=0.12, color=str_color,
                   label=f"IQR  [{float_q1:.3f} – {float_q3:.3f}]{str_unit}")

    for float_val, str_lbl, str_lc, float_yf, str_ls in [
        (float_q1,   "Q1",     "#cc6600", 0.95, ":"),
        (float_q2,   "Q2",     "#cc0000", 0.82, ":"),
        (float_q3,   "Q3",     "#cc6600", 0.95, ":"),
        (float_mean, "mean",   str_color, 0.68, "--"),
    ]:
        obj_ax.axvline(float_val, linestyle=str_ls, linewidth=1.6, color=str_lc)
        obj_ax.text(float_val, float_ymax * float_yf,
                    f" {str_lbl}\n {float_val:.3f}{str_unit}",
                    color=str_lc, fontsize=9, va="top")

    obj_ax.legend(fontsize=10)
    obj_ax.grid(axis="y", linestyle="--", alpha=0.3)


def _save_batch_hist(
    list_vals: tp.List[float],
    path_output: Path,
    str_title: str,
    str_xlabel: str,
    str_color: str,
    str_unit: str = "",
    float_xlim_min: tp.Optional[float] = None,
    float_xlim_max: tp.Optional[float] = None,
    int_bins_factor: int = 1,
) -> None:
    obj_fig = Figure(figsize=(10, 6), dpi=100)
    obj_ax = obj_fig.add_subplot(111)
    try:
        obj_ax.set_title(str_title, fontsize=14)
        obj_ax.set_xlabel(str_xlabel, fontsize=12)
        obj_ax.set_ylabel("Count", fontsize=12)
        obj_ax.tick_params(labelsize=11)

        if list_vals:
            arr_v = np.array(list_vals, dtype=np.float64)
            _draw_quartile_hist(obj_ax, arr_v, str_color, str_unit,
                                float_xlim_min, float_xlim_max, int_bins_factor)
        else:
            obj_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=obj_ax.transAxes, fontsize=13, color="#666666")

        obj_fig.tight_layout()
        obj_fig.savefig(str(path_output), bbox_inches="tight")
    finally:
        obj_fig.clf()


def save_secondary_batch_histograms(
    dict_batchSummary: tp.Dict[str, tp.Any],
    path_outputDir: Path,
    str_lot: str = "",
) -> None:
    """Save batch histograms for secondary (1차 입자) analysis.

    Outputs: batch_hist_size.png, batch_hist_sphericity.png, batch_hist_fine_ratio.png
    """
    str_prefix = f"{str_lot}  " if str_lot else ""

    # Raw values live in img_id summaries (particle_size_um_raw / particle_sphericity_raw)
    # which are pooled at the group level, not stored at the batch level.
    list_sizes: tp.List[float] = []
    list_sphs: tp.List[float] = []
    list_fine: tp.List[float] = []
    list_size_stds: tp.List[float] = []      # 이미지(파일)별 입도 표준편차
    list_size_per_image: tp.List[float] = [] # 이미지(파일)별 평균 입도
    list_sph_per_image: tp.List[float] = []  # 이미지(파일)별 평균 구형도
    for dict_g in (dict_batchSummary.get("img_ids") or []):
        for v in (dict_g.get("particle_size_um_raw") or []):
            try:
                fv = float(v)
                if not math.isnan(fv):
                    list_sizes.append(fv)
            except (TypeError, ValueError):
                pass
        for v in (dict_g.get("particle_sphericity_raw") or []):
            try:
                fv = float(v)
                if not math.isnan(fv):
                    list_sphs.append(fv)
            except (TypeError, ValueError):
                pass
        for dict_f in (dict_g.get("files") or []):
            # per-particle raw (fallback if img_id level is absent)
            if not dict_g.get("particle_size_um_raw"):
                for v in (dict_f.get("particle_size_um_raw") or []):
                    try:
                        fv = float(v)
                        if not math.isnan(fv):
                            list_sizes.append(fv)
                    except (TypeError, ValueError):
                        pass

            # std: prefer pre-computed per-file value, fallback to raw
            v_std = dict_f.get("particle_size_std_um")
            if v_std is not None:
                try:
                    fv = float(v_std)
                    if not math.isnan(fv):
                        list_size_stds.append(fv)
                except (TypeError, ValueError):
                    pass
            else:
                list_raw = [float(r) for r in (dict_f.get("particle_size_um_raw") or [])
                            if not math.isnan(float(r))]
                if len(list_raw) >= 2:
                    list_size_stds.append(float(np.std(list_raw, ddof=1)))

            # per-image mean
            v = dict_f.get("particle_mean_size_um")
            if v is not None:
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        list_size_per_image.append(fv)
                except (TypeError, ValueError):
                    pass
            v = dict_f.get("particle_sphericity_mean")
            if v is not None:
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        list_sph_per_image.append(fv)
                except (TypeError, ValueError):
                    pass

            v = dict_f.get("fine_particle_ratio_percent")
            if v is not None:
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        list_fine.append(fv)
                except (TypeError, ValueError):
                    pass

    float_size_xmin,     float_size_xmax     = _std_xlim(list_sizes)
    float_sph_xmin,      float_sph_xmax      = _std_xlim(list_sphs)
    float_sph_xmax = min(float_sph_xmax, 1.0) if float_sph_xmax is not None else 1.0
    float_fine_xmin,     float_fine_xmax     = _std_xlim(list_fine)
    float_size_std_xmin, float_size_std_xmax = _std_xlim(list_size_stds)
    float_size_pi_xmin,  float_size_pi_xmax  = _std_xlim(list_size_per_image)
    float_sph_pi_xmin,   float_sph_pi_xmax   = _std_xlim(list_sph_per_image)
    float_sph_pi_xmax = min(float_sph_pi_xmax, 1.0) if float_sph_pi_xmax is not None else 1.0

    _save_batch_hist(
        list_vals=list_sizes,
        path_output=path_outputDir / "batch_hist_size.png",
        str_title=f"{str_prefix}Particle Size — Batch Distribution (per particle)",
        str_xlabel="Equivalent Diameter (µm)",
        str_color="#5588ff",
        str_unit=" µm",
        float_xlim_min=float_size_xmin, float_xlim_max=float_size_xmax,
        int_bins_factor=5,
    )
    _save_batch_hist(
        list_vals=list_size_per_image,
        path_output=path_outputDir / "batch_hist_size_per_image.png",
        str_title=f"{str_prefix}Particle Size — Batch Distribution (per-image mean)",
        str_xlabel="Mean Equivalent Diameter (µm)",
        str_color="#3366cc",
        str_unit=" µm",
        float_xlim_min=float_size_pi_xmin, float_xlim_max=float_size_pi_xmax,
        int_bins_factor=3,
    )
    _save_batch_hist(
        list_vals=list_size_stds,
        path_output=path_outputDir / "batch_hist_size_std.png",
        str_title=f"{str_prefix}Particle Size Std per Image — Batch Distribution",
        str_xlabel="Size Std (µm)",
        str_color="#2266cc",
        str_unit=" µm",
        float_xlim_min=float_size_std_xmin, float_xlim_max=float_size_std_xmax,
    )
    _save_batch_hist(
        list_vals=list_sphs,
        path_output=path_outputDir / "batch_hist_sphericity.png",
        str_title=f"{str_prefix}Sphericity — Batch Distribution (per particle)",
        str_xlabel="Sphericity",
        str_color="#44cc44",
        str_unit="",
        float_xlim_min=float_sph_xmin, float_xlim_max=float_sph_xmax,
    )
    _save_batch_hist(
        list_vals=list_sph_per_image,
        path_output=path_outputDir / "batch_hist_sphericity_per_image.png",
        str_title=f"{str_prefix}Sphericity — Batch Distribution (per-image mean)",
        str_xlabel="Mean Sphericity",
        str_color="#229922",
        str_unit="",
        float_xlim_min=float_sph_pi_xmin, float_xlim_max=float_sph_pi_xmax,
    )
    _save_batch_hist(
        list_vals=list_fine,
        path_output=path_outputDir / "batch_hist_fine_ratio.png",
        str_title=f"{str_prefix}Fine Particle Ratio per Image — Batch Distribution",
        str_xlabel="Fine Particle Ratio (%)",
        str_color="#ff6622",
        str_unit="%",
        float_xlim_min=float_fine_xmin, float_xlim_max=float_fine_xmax,
    )


def _std_xlim(
    list_vals: tp.List[float],
    float_z: float = 1.96,
) -> tp.Tuple[tp.Optional[float], tp.Optional[float]]:
    """Return (xmin, xmax) as mean ± z*std (default z=1.96 for 95%)."""
    if len(list_vals) < 4:
        return None, None
    arr = np.array(list_vals, dtype=np.float64)
    float_mean = float(np.mean(arr))
    float_std = float(np.std(arr))
    return float_mean - float_z * float_std, float_mean + float_z * float_std


def save_primary_batch_histograms(
    dict_batchSummary: tp.Dict[str, tp.Any],
    path_outputDir: Path,
    str_lot: str = "",
) -> None:
    """Save batch histograms for primary (2차 입자) analysis.

    Outputs: batch_hist_thickness.png, batch_hist_density.png
    """
    str_prefix = f"{str_lot}  " if str_lot else ""

    list_thickness: tp.List[float] = []
    for v in (dict_batchSummary.get("all_primary_thickness_um_raw") or []):
        try:
            fv = float(v)
            if not math.isnan(fv):
                list_thickness.append(fv)
        except (TypeError, ValueError):
            pass

    list_thickness_per_image: tp.List[float] = []
    for dict_g in (dict_batchSummary.get("img_ids") or []):
        for dict_f in (dict_g.get("files") or []):
            dict_th = dict_f.get("all_primary_thickness_um")
            if isinstance(dict_th, dict):
                v = dict_th.get("mean")
                if v is not None:
                    try:
                        fv = float(v)
                        if not math.isnan(fv):
                            list_thickness_per_image.append(fv)
                    except (TypeError, ValueError):
                        pass

    # density: prefer pre-pooled raw list, fall back to img_id file traversal
    list_densities: tp.List[float] = []
    for v in (dict_batchSummary.get("roi_density_raw") or []):
        try:
            fv = float(v)
            if not math.isnan(fv):
                list_densities.append(fv)
        except (TypeError, ValueError):
            pass
    if not list_densities:
        for dict_g in (dict_batchSummary.get("img_ids") or []):
            for dict_f in (dict_g.get("files") or []):
                v = dict_f.get("roi_density")
                if v is not None:
                    try:
                        fv = float(v)
                        if not math.isnan(fv):
                            list_densities.append(fv)
                    except (TypeError, ValueError):
                        pass

    _save_batch_hist(
        list_vals=list_thickness,
        path_output=path_outputDir / "batch_hist_thickness.png",
        str_title=f"{str_prefix}Primary Particle Thickness — Batch Distribution (per particle)",
        str_xlabel="Thickness (µm)",
        str_color="#9944ee",
        str_unit=" µm",
        **(dict(zip(("float_xlim_min", "float_xlim_max"), _std_xlim(list_thickness)))),
    )
    _save_batch_hist(
        list_vals=list_thickness_per_image,
        path_output=path_outputDir / "batch_hist_thickness_per_image.png",
        str_title=f"{str_prefix}Primary Particle Thickness — Batch Distribution (per-image mean)",
        str_xlabel="Mean Thickness (µm)",
        str_color="#7722bb",
        str_unit=" µm",
        **(dict(zip(("float_xlim_min", "float_xlim_max"), _std_xlim(list_thickness_per_image)))),
    )
    _save_batch_hist(
        list_vals=list_densities,
        path_output=path_outputDir / "batch_hist_density.png",
        str_title=f"{str_prefix}ROI Density — Batch Distribution",
        str_xlabel="ROI Density (foreground fraction)",
        str_color="#ff8844",
        str_unit="",
        **(dict(zip(("float_xlim_min", "float_xlim_max"), _std_xlim(list_densities)))),
    )


def save_lot_particle_scatter_histogram(
    path_lot_dir: Path,
    path_output: Path,
    str_lot: str = "",
) -> None:
    """LOT 하위 폴더의 모든 objects.csv를 읽어 히스토그램 + 1D 산점도를 저장한다.

    위 패널: 전구체/미분 분류별 입도 히스토그램 (mean ± 1.96σ x축)
    아래 패널: 동일 x축 공유, 각 입자를 점으로 찍고 x축까지 수직선 연결
    """
    import matplotlib.gridspec as gridspec

    list_particle_sizes: tp.List[float] = []
    list_fragment_sizes: tp.List[float] = []

    for path_csv in sorted(path_lot_dir.glob("**/objects.csv")):
        try:
            with path_csv.open(encoding="utf-8-sig") as obj_f:
                for dict_row in csv.DictReader(obj_f):
                    str_cat = dict_row.get("str_category", "")
                    try:
                        fv = float(dict_row.get("float_eqDiameterUm", ""))
                        if not math.isnan(fv) and fv > 0:
                            if str_cat == "particle":
                                list_particle_sizes.append(fv)
                            elif str_cat == "fragment":
                                list_fragment_sizes.append(fv)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

    list_all = list_particle_sizes + list_fragment_sizes
    if not list_all:
        import sys
        print(f"[WARN] {path_lot_dir} 에서 입자 데이터를 찾지 못했습니다. objects.csv 경로를 확인하세요.", file=sys.stderr)
        return

    float_xmin, float_xmax = _std_xlim(list_all)
    arr_all = np.array(list_all, dtype=np.float64)
    if float_xmin is None:
        float_xmin = float(arr_all.min())
    if float_xmax is None:
        float_xmax = float(arr_all.max())

    str_prefix = f"{str_lot}  " if str_lot else ""
    rng = np.random.default_rng(seed=0)

    obj_fig = Figure(figsize=(12, 7), dpi=100)
    gs = gridspec.GridSpec(2, 1, figure=obj_fig, height_ratios=[4, 1], hspace=0.08)
    obj_ax_hist = obj_fig.add_subplot(gs[0])
    obj_ax_scat = obj_fig.add_subplot(gs[1], sharex=obj_ax_hist)

    try:
        # ── 히스토그램 ──────────────────────────────────────────────────
        obj_ax_hist.set_title(
            f"{str_prefix}Particle Size — All Objects  "
            f"(particle={len(list_particle_sizes)}, fragment={len(list_fragment_sizes)})",
            fontsize=13)
        obj_ax_hist.set_ylabel("Count", fontsize=11)
        obj_ax_hist.tick_params(labelsize=10)
        obj_ax_hist.set_xlim(float_xmin, float_xmax)

        int_bins = int(np.clip(np.sqrt(len(list_all)), 8, 40))

        for list_vals, str_label, str_color in [
            (list_particle_sizes, "Particle", "#5588ff"),
            (list_fragment_sizes, "Fragment", "#ff6622"),
        ]:
            if list_vals:
                obj_ax_hist.hist(
                    list_vals, bins=int_bins, alpha=0.55, color=str_color,
                    edgecolor="#333333", linewidth=0.6, label=str_label,
                    range=(float_xmin, float_xmax),
                )

        float_mean_all = float(np.mean(arr_all))
        float_std_all  = float(np.std(arr_all))
        obj_ax_hist.axvline(float_mean_all, color="#222222", linewidth=1.5, linestyle="--",
                            label=f"mean={float_mean_all:.3f} µm")
        obj_ax_hist.axvline(float_mean_all - 1.96 * float_std_all,
                            color="#888888", linewidth=1.0, linestyle=":")
        obj_ax_hist.axvline(float_mean_all + 1.96 * float_std_all,
                            color="#888888", linewidth=1.0, linestyle=":")
        obj_ax_hist.legend(fontsize=10)
        obj_ax_hist.grid(axis="y", linestyle="--", alpha=0.3)
        obj_ax_hist.tick_params(labelbottom=False)

        # ── 1D 산점도 (rug) ─────────────────────────────────────────────
        obj_ax_scat.set_xlabel("Equivalent Diameter (µm)", fontsize=11)
        obj_ax_scat.set_ylabel("", fontsize=1)
        obj_ax_scat.set_ylim(0, 1)
        obj_ax_scat.tick_params(left=False, labelleft=False, labelsize=10)
        obj_ax_scat.set_xlim(float_xmin, float_xmax)

        for list_vals, str_color in [
            (list_particle_sizes, "#5588ff"),
            (list_fragment_sizes, "#ff6622"),
        ]:
            if list_vals:
                arr_x = np.array(list_vals, dtype=np.float64)
                # x축 범위 밖 점은 그리지 않음
                arr_x = arr_x[(arr_x >= float_xmin) & (arr_x <= float_xmax)]
                # y 방향 jitter로 겹침 완화
                arr_y = rng.uniform(0.15, 0.85, size=len(arr_x))
                obj_ax_scat.vlines(arr_x, 0, arr_y, color=str_color, alpha=0.25, linewidth=0.8)
                obj_ax_scat.scatter(arr_x, arr_y, color=str_color, alpha=0.5, s=8, linewidths=0)

        obj_fig.tight_layout()
        obj_fig.savefig(str(path_output), bbox_inches="tight")
    finally:
        obj_fig.clf()
