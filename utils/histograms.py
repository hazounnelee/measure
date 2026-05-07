from __future__ import annotations
import csv
import math
import typing as tp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
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
