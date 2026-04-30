#!/usr/bin/env python3
"""Generate pipeline PDF diagrams for primary_measure.py and secondary_measure.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ── Colour palette ─────────────────────────────────────────────────────────────
C_IO      = "#1D4ED8"   # blue     – input / output nodes
C_PREP    = "#15803D"   # green    – preprocessing
C_BRANCH  = "#B45309"   # amber    – decision / branch
C_LSD     = "#6D28D9"   # violet   – LSD-specific
C_SAM2    = "#0E7490"   # cyan     – SAM2-specific
C_MEAS    = "#B91C1C"   # red      – measurement / classification
C_OUT     = "#1E3A8A"   # indigo   – outputs
C_COMMON  = "#374151"   # slate    – shared steps (density, summary)
BG        = "#F8FAFC"   # page background


# ── Drawing helpers ────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, text, color=C_COMMON, fs=7.8, tc="white",
        bold=False, alpha=1.0, style="round,pad=0.15"):
    """Rounded rectangle with centred multiline text."""
    fb = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle=style, linewidth=1.0,
                        edgecolor="#FFFFFF", facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(fb)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            multialignment="center", zorder=3, linespacing=1.35)


def diamond(ax, cx, cy, w, h, text, color=C_BRANCH, fs=7.8):
    """Diamond shape for decision nodes."""
    pts = np.array([[cx, cy + h / 2], [cx + w / 2, cy],
                    [cx, cy - h / 2], [cx - w / 2, cy]])
    poly = plt.Polygon(pts, closed=True, facecolor=color,
                       edgecolor="#FFFFFF", linewidth=1.0, zorder=2)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", multialignment="center", zorder=3)


def arr(ax, x1, y1, x2, y2, label=None, lc="#444444", lw=1.4, ls="-"):
    """Arrow from (x1,y1) → (x2,y2) with optional side label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=lc, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=1)
    if label:
        mx, my = (x1 + x2) / 2 + 0.08, (y1 + y2) / 2
        ax.text(mx, my, label, fontsize=6.5, color="#555555", style="italic",
                va="center", zorder=3)


def hconn(ax, x1, y, x2, yend, lc="#444444", lw=1.4):
    """Horizontal then vertical connector (L-shape) ending with arrowhead."""
    ax.plot([x1, x2], [y, y], color=lc, lw=lw, zorder=1)
    ax.annotate("", xy=(x2, yend), xytext=(x2, y),
                arrowprops=dict(arrowstyle="->", color=lc, lw=lw), zorder=1)


def leg_patch(color, label):
    return mpatches.Patch(facecolor=color, edgecolor="white", label=label)


# ── PRIMARY PIPELINE ───────────────────────────────────────────────────────────

def draw_primary():
    fig = plt.figure(figsize=(20, 30))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

    W  = 20      # page width
    BW = 14      # common box width
    BH = 1.05   # standard box height
    XC = 10      # horizontal centre

    # thin separator lines between sections
    def sep(y): ax.plot([0.4, W - 0.4], [y, y], color="#CBD5E1", lw=0.6, ls="--", zorder=0)

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(XC, 29.2, "primary_measure.py — Full Pipeline",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1E293B")
    ax.text(XC, 28.6, "particle_type ∈ {acicular, plate}  ×  magnification ∈ {20k, 50k}",
            ha="center", va="center", fontsize=9, color="#475569")

    # ── Input ──────────────────────────────────────────────────────────────────
    box(ax, XC, 27.8, BW, BH,
        "INPUT  —  SEM image (.jpg / .png / .tiff / …)\n"
        "single file  or  batch directory",
        color=C_IO, fs=8.5, bold=True)
    arr(ax, XC, 27.27, XC, 26.7)
    sep(27.0)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    box(ax, XC, 26.2, BW, 0.95,
        "IMAGE NORMALISATION\n"
        "resize → 2048 × 1636  (bilinear)    ·    crop bottom 100 px  →  2048 × 1536",
        color=C_PREP, fs=8)
    arr(ax, XC, 25.72, XC, 25.15)
    sep(24.9)

    # ── ROI extraction ─────────────────────────────────────────────────────────
    box(ax, XC, 24.4, BW, 1.05,
        "ROI EXTRACTION  (auto_detect_sphere=False, default)\n"
        "center crop  (crop_ratio = 0.60)   —   applies to both 20k and 50k\n"
        "manual override:  --roi_x_min/y_min/x_max/y_max",
        color=C_PREP, fs=7.5)
    arr(ax, XC, 23.75, XC, 23.2)
    sep(22.95)

    # ── Branch decision ────────────────────────────────────────────────────────
    diamond(ax, XC, 22.55, 5.5, 0.9, "measure_mode?", fs=9)
    # LSD label
    ax.text(5.0, 22.55, "lsd", fontsize=8, color="#6D28D9",
            fontweight="bold", va="center", ha="right")
    # SAM2 label
    ax.text(15.0, 22.55, "sam2", fontsize=8, color="#0E7490",
            fontweight="bold", va="center", ha="left")

    # left branch line
    ax.plot([7.25, 4.5], [22.55, 22.55], color="#444", lw=1.4, zorder=1)
    ax.annotate("", xy=(4.5, 21.9), xytext=(4.5, 22.55),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)
    # right branch line
    ax.plot([12.75, 15.5], [22.55, 22.55], color="#444", lw=1.4, zorder=1)
    ax.annotate("", xy=(15.5, 21.9), xytext=(15.5, 22.55),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)

    sep(22.0)

    # ── Column headers ─────────────────────────────────────────────────────────
    ax.text(4.5, 21.7, "LSD  (acicular)", ha="center", fontsize=9.5,
            fontweight="bold", color=C_LSD)
    ax.text(15.5, 21.7, "SAM2  (plate / acicular hybrid)", ha="center",
            fontsize=9.5, fontweight="bold", color=C_SAM2)
    ax.plot([10, 10], [22.0, 3.4], color="#CBD5E1", lw=1.0, ls=":", zorder=0)

    LX, RX = 4.5, 15.5   # column centres
    CW = 8.4              # column box width
    CH = 0.95             # column box height

    def larr(y1, y2):  arr(ax, LX, y1, LX, y2)
    def rarr(y1, y2):  arr(ax, RX, y1, RX, y2)

    # ── LSD column ─────────────────────────────────────────────────────────────
    y = 21.2
    box(ax, LX, y, CW, CH,
        "CONTRAST ENHANCEMENT\n"
        "CLAHE  (clipLimit=2.0, 8×8 tile)   +   GaussianBlur (3×3)",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, LX, y, CW, 1.1,
        "THRESHOLDING   (--lsd_adaptive_thresh)\n"
        "Otsu  →  single global threshold  (float_otsu_thresh)\n"
        "Adaptive  →  local Gaussian threshold  (block_size = max(11, ⌊min(H,W)/12⌋))\n"
        "→  density  =  white_px / total_px  (computed here)",
        color=C_LSD, fs=7.2); larr(y - 0.55, y - 1.15)

    y -= 1.45
    box(ax, LX, y, CW, CH,
        "LSD DETECTION\n"
        "cv2.createLineSegmentDetector(0).detect(blur)\n"
        "output: N × (x1, y1, x2, y2) + per-segment width",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, LX, y, CW, CH,
        "SEGMENT FILTER\n"
        "length  ≥  20 px\n"
        "--ar_screen OFF (default)  →  AR filter skipped  (all segments pass)",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, LX, y, CW, CH,
        "DEDUPLICATION  (greedy, longest-first)\n"
        "reject if:  dist(centre_i, centre_j) < 12 px\n"
        "           AND  |angle_i − angle_j|  < 25°",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, LX, y, CW, 1.05,
        "SEGMENT FUSION  (--lsd_fuse_segments, ON by default)\n"
        "union-find over collinear neighbours:\n"
        "  Δangle < 10°  ·  perp-dist < 8 px  ·  axial gap < 15 px\n"
        "merge: length-weighted axis + extreme endpoint projection",
        color=C_LSD, fs=7.2, alpha=0.88); larr(y - 0.53, y - 1.18)

    y -= 1.48
    box(ax, LX, y, CW, 1.15,
        "PERPENDICULAR THICKNESS\n"
        "7 sample positions along segment  (t = 0.2 … 0.8)\n"
        "scan ⊥ axis  ±  0.5 × px_per_µm\n"
        "pick nearest bright run to scan centre  →  width\n"
        "thickness  =  median of 7 estimates   (discard < 2 px)",
        color=C_LSD, fs=7.2); larr(y - 0.58, y - 1.22)

    y -= 1.52
    box(ax, LX, y, CW, 1.05,
        "CLASSIFICATION  (LSD)\n"
        "AR  =  thickness / length\n"
        "AR < acicular_threshold (0.40)  →  acicular\n"
        "AR ≥ threshold  →  plate\n"
        "mask area < area_threshold  →  discard",
        color=C_MEAS, fs=7.5)

    LSD_END_Y = y - 0.53

    # ── SAM2 column ────────────────────────────────────────────────────────────
    y = 21.2
    box(ax, RX, y, CW, CH,
        "TILE GRID\n"
        "ROI → overlapping tiles  (tile_size × stride)\n"
        "edge tiles extended to ROI boundary  ·  duplicates removed",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "TEXTURE ENHANCEMENT  (per tile)\n"
        "CLAHE  +  Sobel gradient  +  Laplacian  +  morphological blackhat\n"
        "→  single enhanced grayscale channel",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "INTEREST-POINT SAMPLING  (Shi-Tomasi)\n"
        "cv2.goodFeaturesToTrack on enhanced tile\n"
        "fallback: Otsu → contour centroids  (if sparse)\n"
        "up to points_per_tile,  spacing ≥ point_min_distance",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "SAM2 BATCH INFERENCE  (Ultralytics)\n"
        "foreground point prompts  (label = 1),  batch_size points per call\n"
        "raw logit masks  →  binarise at mask_binarize_threshold",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "BBOX IoU DEDUP  (tile-level fast filter)\n"
        "reject mask if bbox IoU ≥ 0.85 with any accepted mask\n"
        "→  promote tile-coord mask → ROI-coord mask",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "PIXEL IoU DEDUP  (ROI-level)\n"
        "binary mask IoU ≥ 0.60  →  reject as duplicate",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, CH,
        "MASK REFINEMENT  (morphology)\n"
        "open  (int_maskMorphOpenIterations)\n"
        "close  (int_maskMorphCloseIterations)",
        color=C_SAM2, fs=7.5); rarr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, RX, y, CW, 1.05,
        "minAreaRect  +  CLASSIFICATION  (SAM2)\n"
        "thickness = min(rect_W, rect_H)   ·   long_axis = max\n"
        "AR  =  thickness / long_axis\n"
        "area < area_threshold → fragment\n"
        "AR < 0.40 → acicular    ·    AR ≥ 0.40 → plate",
        color=C_MEAS, fs=7.5)

    SAM2_END_Y = y - 0.53

    sep(3.5)

    # ── Merge + density ────────────────────────────────────────────────────────
    MERGE_Y = 3.15
    # lines from branches down to merge
    ax.plot([LX, LX], [LSD_END_Y, MERGE_Y + 0.4], color="#444", lw=1.4, zorder=1)
    ax.plot([RX, RX], [SAM2_END_Y, MERGE_Y + 0.4], color="#444", lw=1.4, zorder=1)
    ax.plot([LX, RX], [MERGE_Y + 0.4, MERGE_Y + 0.4], color="#444", lw=1.4, zorder=1)
    ax.annotate("", xy=(XC, MERGE_Y + 0.05), xytext=(XC, MERGE_Y + 0.4),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)

    box(ax, XC, MERGE_Y - 0.32, BW, 0.95,
        "ROI DENSITY\n"
        "LSD mode: density = white_px / total_px  (from binary already computed)\n"
        "SAM2 mode: Otsu-binarise ROI grey  →  density = white_px / total_px",
        color=C_COMMON, fs=7.8)
    arr(ax, XC, MERGE_Y - 0.80, XC, MERGE_Y - 1.35)
    sep(1.55)

    # ── Outputs ────────────────────────────────────────────────────────────────
    box(ax, XC, 1.0, BW, 0.9,
        "OUTPUTS  (per image)\n"
        "01_input.png  02_input_roi.png  03_overlay_roi.png  04_overlay_full.png\n"
        "lsd_01…06_*.png  |  05_opencv_candidates.png  |  06_sphere_detection.png\n"
        "objects.csv  acicular.csv / plate.csv  |  thickness_dist.png\n"
        "summary.json  (roi_density)   objects.json   debug.json\n"
        "Batch:  img_id_summary.json  ·  batch_summary.json  (roi_density_mean)",
        color=C_OUT, fs=7.5, bold=True)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_patches = [
        leg_patch(C_IO,     "Input / Output"),
        leg_patch(C_PREP,   "Preprocessing / ROI"),
        leg_patch(C_BRANCH, "Decision / Branch"),
        leg_patch(C_LSD,    "LSD-specific"),
        leg_patch(C_SAM2,   "SAM2-specific"),
        leg_patch(C_MEAS,   "Measurement / Classification"),
        leg_patch(C_COMMON, "Shared (both modes)"),
        leg_patch(C_OUT,    "Output"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              bbox_to_anchor=(0.01, 0.00), fontsize=7.5,
              framealpha=0.9, edgecolor="#CBD5E1", ncol=4,
              handlelength=1.2, handleheight=0.8, columnspacing=1.0)

    return fig


# ── SECONDARY PIPELINE ─────────────────────────────────────────────────────────

def draw_secondary():
    fig = plt.figure(figsize=(14, 22))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

    XC = 7
    BW = 11
    BH = 1.0

    def sep(y): ax.plot([0.4, 13.6], [y, y], color="#CBD5E1", lw=0.6, ls="--", zorder=0)
    def dn(y1, y2): arr(ax, XC, y1, XC, y2)

    # Title
    ax.text(XC, 21.3, "secondary_measure.py — Full Pipeline",
            ha="center", fontsize=14, fontweight="bold", color="#1E293B")
    ax.text(XC, 20.75, "Segment-Anything SAM2  →  particle / fragment classification\n"
            "equivalent circle diameter  ·  sphericity  (µm)",
            ha="center", fontsize=8.5, color="#475569", multialignment="center")

    # Input
    box(ax, XC, 19.9, BW, BH,
        "INPUT  —  SEM image (.jpg / .png / .tiff / …)\n"
        "single file  or  batch directory",
        color=C_IO, fs=8.5, bold=True)
    dn(19.4, 18.85); sep(18.65)

    # Preprocessing
    box(ax, XC, 18.3, BW, BH,
        "IMAGE NORMALISATION\n"
        "resize → 2048 × 1636  (bilinear)   ·   crop bottom 100 px  →  2048 × 1536",
        color=C_PREP, fs=8)
    dn(17.8, 17.25); sep(17.05)

    # ROI
    box(ax, XC, 16.55, BW, 0.95,
        "ROI EXTRACTION\n"
        "--roi_x_min/y_min/x_max/y_max  (default: full normalised image)\n"
        "or auto_center_crop if enabled",
        color=C_PREP, fs=8)
    dn(16.07, 15.5); sep(15.3)

    # Tiling
    box(ax, XC, 14.85, BW, BH,
        "TILE GRID\n"
        "ROI → overlapping tiles  (tile_size=512 × stride=256)\n"
        "edge tiles extended  ·  duplicates deduplicated",
        color=C_SAM2, fs=8)
    dn(14.35, 13.8)

    # Texture
    box(ax, XC, 13.3, BW, BH,
        "TEXTURE ENHANCEMENT  (per tile)\n"
        "CLAHE  +  Sobel gradient  +  Laplacian  +  morphological blackhat\n"
        "→  single enhanced grayscale channel",
        color=C_SAM2, fs=8)
    dn(12.8, 12.25)

    # Point sampling
    box(ax, XC, 11.75, BW, BH,
        "INTEREST-POINT SAMPLING  (Shi-Tomasi)\n"
        "cv2.goodFeaturesToTrack on enhanced tile\n"
        "fallback: Otsu → contour centroids  (if sparse)\n"
        "up to points_per_tile=80,  spacing ≥ point_min_distance=14 px",
        color=C_SAM2, fs=8)
    dn(11.25, 10.7)

    # SAM2
    box(ax, XC, 10.2, BW, BH,
        "SAM2 BATCH INFERENCE  (Ultralytics)\n"
        "foreground point prompts  (label = 1),  batch_size = 32 points per call\n"
        "raw logit masks  →  binarise at mask_binarize_threshold = 0.0",
        color=C_SAM2, fs=8)
    dn(9.7, 9.15)

    # BBox dedup
    box(ax, XC, 8.65, BW, BH,
        "BOUNDING-BOX IoU DEDUP  (tile-level, fast)\n"
        "reject mask if bbox IoU ≥ 0.85 with any accepted mask\n"
        "promote tile-coord mask → ROI-coord mask",
        color=C_SAM2, fs=8)
    dn(8.15, 7.6)

    # Pixel dedup
    box(ax, XC, 7.1, BW, BH,
        "PIXEL IoU DEDUP  (ROI-level)\n"
        "binary mask IoU ≥ 0.60  →  reject as duplicate",
        color=C_SAM2, fs=8)
    dn(6.6, 6.05)

    # Morphology
    box(ax, XC, 5.55, BW, BH,
        "MASK REFINEMENT  (morphology, optional)\n"
        "morphological open  (remove small spurs)\n"
        "morphological close  (fill interior holes)",
        color=C_SAM2, fs=8)
    dn(5.05, 4.5); sep(4.3)

    # Measurement
    box(ax, XC, 3.85, BW, 1.05,
        "MEASUREMENT\n"
        "eq_diameter_um  =  2 × √(mask_area / π)   →   px → µm  (Equivalent Circle Diameter)\n"
        "sphericity      =  4π × mask_area / perimeter²   (Wadell 2D isoperimetric ratio)\n"
        "--eq_diameter ON (default): size = eq_diameter   |   OFF: size = (spanH + spanV) / 2",
        color=C_MEAS, fs=8)
    dn(3.25, 2.75)

    # Classification
    box(ax, XC, 2.25, BW, 0.9,
        "CLASSIFICATION\n"
        "mask_area  <  area_threshold (1500 px²)  →  fragment\n"
        "otherwise  →  particle",
        color=C_MEAS, fs=8)
    dn(1.8, 1.3); sep(1.1)

    # Outputs
    box(ax, XC, 0.65, BW, 0.88,
        "OUTPUTS  (per image)\n"
        "01…04_*.png  overlay  |  objects.csv  particles.csv  fragment_masks/  particle_masks/\n"
        "particle_dist.png  sphericity_dist.png  |  summary.json  objects.json  debug.json\n"
        "Batch:  img_id_summary.json  ·  batch_summary.json",
        color=C_OUT, fs=7.8, bold=True)

    # Legend
    legend_patches = [
        leg_patch(C_IO,   "Input / Output"),
        leg_patch(C_PREP, "Preprocessing / ROI"),
        leg_patch(C_SAM2, "SAM2 inference"),
        leg_patch(C_MEAS, "Measurement / Classification"),
        leg_patch(C_OUT,  "Output"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              bbox_to_anchor=(0.01, 0.00), fontsize=8,
              framealpha=0.9, edgecolor="#CBD5E1", ncol=5,
              handlelength=1.2, handleheight=0.8)

    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating pipeline_primary.pdf …")
    with PdfPages("pipeline_primary.pdf") as pdf:
        fig = draw_primary()
        pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    print("  → pipeline_primary.pdf")

    print("Generating pipeline_secondary.pdf …")
    with PdfPages("pipeline_secondary.pdf") as pdf:
        fig = draw_secondary()
        pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    print("  → pipeline_secondary.pdf")

    print("Done.")
