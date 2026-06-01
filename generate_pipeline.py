#!/usr/bin/env python3
"""Generate pipeline PDF diagrams for primary_measure.py and secondary_measure.py."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
_NANUM_PATH = (
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8"
    "/7a0b5c0f3c1d41c4c52a33343496c9c65ad52c50.asset/AssetData/NanumGothic.ttc"
)
try:
    font_manager.fontManager.addfont(_NANUM_PATH)
    matplotlib.rcParams["font.family"] = "Nanum Gothic"
except Exception:
    pass  # 폰트 없으면 경고만 뜨고 계속 진행

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
        "IMAGE NORMALISATION  (--preprocess_width W,  default 1024)\n"
        "resize → W × round(W×1636/2048)  (bilinear)    ·    crop bottom round(W×100/2048) px\n"
        "default:  1024 × 818  →  crop 50 px  →  1024 × 768",
        color=C_PREP, fs=7.5)
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
        "Otsu  →  single global threshold  |  Adaptive  →  local Gaussian\n"
        "auto-invert if >55% white (ensures particles = foreground)\n"
        "→  density  =  foreground(particle)_px / total_px  (computed here)",
        color=C_LSD, fs=7.2); larr(y - 0.55, y - 1.15)

    y -= 1.45
    box(ax, LX, y, CW, CH,
        "LSD SEGMENT DETECTION\n"
        "cv2.createLineSegmentDetector()  →  line segments from gradient\n"
        "→  (x1,y1)-(x2,y2), length, angle per segment",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.3
    box(ax, LX, y, CW, CH,
        "DEDUP  (sort longest-first)\n"
        "sort candidates by length descending  →  greedy deduplication\n"
        "reject if centre_dist < 12 px  AND  |Δangle| < 25°",
        color=C_LSD, fs=7.5); larr(y - 0.48, y - 1.0)

    y -= 1.48
    box(ax, LX, y, CW, 1.15,
        "PERPENDICULAR THICKNESS\n"
        "7 sample positions along segment  (t = 0.2 … 0.8)\n"
        "scan ⊥ axis  ±  max(40, 2×px/µm)  pixels\n"
        "binary lookup (polarity-normalised)  →  nearest foreground run\n"
        "thickness  =  median of 7 estimates   (discard < 2 px)",
        color=C_LSD, fs=7.2); larr(y - 0.58, y - 1.22)

    y -= 1.52
    box(ax, LX, y, CW, 1.05,
        "CLASSIFICATION  (LSD)\n"
        "AR  =  thickness / length\n"
        "AR < acicular_threshold (0.40)  →  acicular\n"
        "AR ≥ threshold  →  plate\n"
        "mask area < area_threshold  →  discard",
        color=C_MEAS, fs=7.5); larr(y - 0.53, y - 1.05)

    y -= 1.35
    box(ax, LX, y, CW, 1.0,
        "POST-PROCESSING  (optional)\n"
        "--fuse: Union-Find merge of overlapping parallel segments\n"
        "        Δangle < 15°,  overlap ≥ 70%  →  fuse into one\n"
        "--min_length (default 10 px × scale): discard if long_axis < threshold",
        color=C_LSD, fs=7.2)

    LSD_END_Y = y - 0.50

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
    fig = plt.figure(figsize=(16, 34))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 34)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

    XC = 8
    BW = 13.5
    LC = "#444444"

    def sep(y): ax.plot([0.3, 15.7], [y, y], color="#CBD5E1", lw=0.6, ls="--", zorder=0)
    def dn(y1, y2): arr(ax, XC, y1, XC, y2)
    def side_arr(x, y1, y2):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="->", color=LC, lw=1.4), zorder=1)
    def hline(x1, x2, y):
        ax.plot([x1, x2], [y, y], color=LC, lw=1.4, zorder=1)
    def vline(x, y1, y2):
        ax.plot([x, x], [y1, y2], color=LC, lw=1.4, zorder=1)

    # Title
    ax.text(XC, 33.55, "secondary_measure.py — 파이프라인",
            ha="center", fontsize=14, fontweight="bold", color="#1E293B")
    ax.text(XC, 33.05, "HCT 포인트 프롬프트  →  SAM2  →  밝기 필터  →  Kasa 복원  →  hull 병합  →  S / S' / eq_d 측정",
            ha="center", fontsize=8.5, color="#475569")
    sep(32.75)

    # ── INPUT ─────────────────────────────────────────────────────────────────
    box(ax, XC, 32.25, BW, 0.7,
        "INPUT  —  SEM image  (single file  or  batch directory)",
        color=C_IO, fs=8.5, bold=True)
    dn(31.9, 31.4); sep(31.2)

    # ── PREPROCESSING ─────────────────────────────────────────────────────────
    box(ax, XC, 30.75, BW, 1.0,
        "전처리  (IMAGE NORMALISATION)\n"
        "resize → preprocess_width W (default 1024) × round(W×1636/2048)  [bilinear]\n"
        "bottom crop: round(W×100/2048) px  스케일바 제거  →  default 1024 × 768",
        color=C_PREP, fs=7.8)
    dn(30.25, 29.75); sep(29.55)

    # ── ROI ───────────────────────────────────────────────────────────────────
    box(ax, XC, 29.1, BW, 0.85,
        "ROI 추출\n"
        "--roi_x_min / y_min / x_max / y_max  직접 지정  (default: 전체 이미지)\n"
        "또는  auto_center_crop 활성 시 center crop",
        color=C_PREP, fs=7.8)
    dn(28.675, 28.175); sep(27.975)

    # ── TILE GRID ─────────────────────────────────────────────────────────────
    box(ax, XC, 27.6, BW, 0.75,
        "타일 분할  (SLIDING WINDOW TILE GRID)\n"
        "ROI → 겹치는 tile grid  (tile_size=512, stride=256)  ·  경계 타일 확장  ·  중복 제거",
        color=C_SAM2, fs=7.8)
    dn(27.225, 26.825)

    # ── detect_hct_prompts header ──────────────────────────────────────────────
    box(ax, XC, 26.55, BW, 0.5,
        "detect_hct_prompts()  [per tile]",
        color=C_SAM2, fs=8.5, bold=True)
    dn(26.3, 25.95)

    # ── DIAMOND: HCT 검출? ─────────────────────────────────────────────────────
    diamond(ax, XC, 25.6, 5.5, 0.75, "HCT 검출?", fs=8.5)
    ax.text(11.1, 25.6, "YES", fontsize=7.5, color=C_SAM2, fontweight="bold", va="center")
    ax.text(4.85, 25.6, "NO", fontsize=7.5, color=C_BRANCH, fontweight="bold", va="center", ha="right")

    # YES → right → 밝기 검증 box
    hline(10.75, 13.0, 25.6); side_arr(13.0, 25.6, 25.1)
    box(ax, 13.0, 24.75, 3.5, 0.7,
        "밝기 검증\npatch 평균 < Otsu×0.85\n→ reject (배경 갭)",
        color=C_SAM2, fs=7.2)

    # NO → left → Canny fallback box
    hline(5.25, 3.0, 25.6); side_arr(3.0, 25.6, 25.1)
    box(ax, 3.0, 24.75, 3.5, 0.7,
        "Canny fallback\nMORPH_CLOSE(7×7,×3)\nRERT_CCOMP fill\n→ dist peak",
        color=C_BRANCH, fs=7.2)

    # Merge both branches back to center
    vline(13.0, 24.4, 24.15); vline(3.0, 24.4, 24.15)
    hline(3.0, 13.0, 24.15)
    ax.annotate("", xy=(XC, 23.95), xytext=(XC, 24.15),
                arrowprops=dict(arrowstyle="->", color=LC, lw=1.4), zorder=1)

    # ── 미커버 블롭 보완 ────────────────────────────────────────────────────────
    box(ax, XC, 23.6, BW, 0.7,
        "미커버 전경 보완\n"
        "fg AND NOT circles_mask → connectedComponents → 블롭별 거리변환 peak  (fragment · 비원형 입자)",
        color=C_SAM2, fs=7.8)
    dn(23.25, 22.8)

    # ── 네거티브 샘플링 ─────────────────────────────────────────────────────────
    box(ax, XC, 22.5, BW, 0.6,
        "네거티브 샘플링  ~fg → erode(×4) → linspace N개 균등 샘플링",
        color=C_SAM2, fs=7.8)
    dn(22.2, 21.7); sep(21.5)

    # ── SAM2 ──────────────────────────────────────────────────────────────────
    box(ax, XC, 21.1, BW, 0.8,
        "SAM2 배치 추론  (Ultralytics)\n"
        "포지티브 (label=1) + 네거티브 (label=0)  ·  batch_size=32\n"
        "raw logit mask → binarize  (mask_binarize_threshold=0.0)",
        color=C_SAM2, fs=7.8)
    dn(20.7, 20.2); sep(20.0)

    # ── DEDUP ─────────────────────────────────────────────────────────────────
    box(ax, XC, 19.55, BW, 1.0,
        "마스크 중복 제거  (DEDUP)\n"
        "① bbox IoU ≥ 0.85 → reject  [tile-level fast filter]\n"
        "② binary mask IoU ≥ 0.60 → reject  [ROI-level]\n"
        "③ 포함 관계: inter / small_area ≥ 0.75 → reject  [ROI-level]",
        color=C_SAM2, fs=7.8)
    dn(19.05, 18.55)

    # ── POSTPROCESS ───────────────────────────────────────────────────────────
    box(ax, XC, 18.1, BW, 0.85,
        "마스크 후처리  (SMOOTH + PEANUT SPLIT)\n"
        "Smooth: MORPH_CLOSE(5×5,×2) + OPEN(5×5,×1) → 최대 연결 컴포넌트 유지\n"
        "Peanut split: minAreaRect AR < 0.6 → 거리변환 2-peak → Watershed 분리",
        color=C_SAM2, fs=7.8)
    dn(17.675, 17.175); sep(16.975)

    # ── PUNCH-OUT ─────────────────────────────────────────────────────────────
    box(ax, XC, 16.6, BW, 0.75,
        "포함 마스크 펀치아웃  (PUNCH-OUT)\n"
        "작은 마스크 J가 큰 마스크 I에 97%+ 포함 → I에서 J 픽셀 제거\n"
        "→ 입자 테두리만 남겨 밝기 필터의 판별 정확도 향상",
        color=C_SAM2, fs=7.8)
    dn(16.225, 15.775)

    # ── DIAMOND: 밝기 필터 ─────────────────────────────────────────────────────
    diamond(ax, XC, 15.4, 7.0, 0.75, "마스크 평균 밝기\n< Otsu × 0.5?", fs=7.8)
    ax.text(12.5, 15.4, "NO", fontsize=7.5, color="#555", fontweight="bold", va="center")
    ax.text(3.5, 15.4, "YES", fontsize=7.5, color=C_BRANCH, fontweight="bold", va="center", ha="right")

    # YES → left → 마스크 제거 (dead-end)
    hline(4.5, 1.8, 15.4)
    box(ax, 1.2, 15.4, 1.5, 0.6, "마스크\n제거", color=C_BRANCH, fs=7.5)

    # NO → straight down
    dn(15.025, 14.575)

    # ── DIAMOND: Kasa 복원 조건 ────────────────────────────────────────────────
    diamond(ax, XC, 14.2, 7.5, 0.75, "solidity < 0.97\nor 직선≥15px\nor ROI 경계?", fs=7.5)
    ax.text(12.6, 14.2, "YES", fontsize=7.5, color=C_MEAS, fontweight="bold", va="center")
    ax.text(3.25, 14.2, "NO", fontsize=7.5, color="#555", fontweight="bold", va="center", ha="right")

    # YES → right → Kasa box
    hline(11.75, 13.8, 14.2); side_arr(13.8, 14.2, 13.65)
    box(ax, 13.8, 13.1, 3.8, 1.1,
        "Kasa 원 피팅\nhull pts → 최소제곱\n(cx, cy, r)\n검증: solidity·CV·면적비\n밝은 픽셀(≥Otsu×0.75) 복원",
        color=C_MEAS, fs=7.0)

    # Kasa box bottom → merge
    vline(13.8, 12.55, 12.3); hline(8.0, 13.8, 12.3)

    # NO → straight down from diamond bottom, meets merge
    vline(XC, 13.825, 12.3)
    ax.annotate("", xy=(XC, 12.1), xytext=(XC, 12.3),
                arrowprops=dict(arrowstyle="->", color=LC, lw=1.4), zorder=1)

    # ── HULL MASK ─────────────────────────────────────────────────────────────
    box(ax, XC, 11.75, BW, 0.7,
        "Hull 마스크 적용\n"
        "convex hull fill 적용  →  면적·크기·S'·분류 모두 hull 기준    (S 원형도는 원본 컨투어 유지)",
        color=C_MEAS, fs=7.8)
    dn(11.4, 10.9)

    # ── HULL MERGE ────────────────────────────────────────────────────────────
    box(ax, XC, 10.55, BW, 0.7,
        "Hull 마스크 97%+ 겹침 병합  (UNION-FIND)\n"
        "inter / min_area ≥ 0.97 → Union-Find 그룹화 → 합집합(OR) 마스크로 병합 후 재측정",
        color=C_MEAS, fs=7.8)
    dn(10.2, 9.7); sep(9.5)

    # ── MEASUREMENT ───────────────────────────────────────────────────────────
    box(ax, XC, 9.05, BW, 1.0,
        "측정  (MEASUREMENT)\n"
        "S   = 4π × hull_area / hull_perimeter²             [원형도, Wadell 2D]  ←  원본 컨투어 hull\n"
        "S'  = b / a  (cv2.fitEllipse 단축 / 장축비)         [타원도]  ←  hull 마스크 컨투어\n"
        "eq_diameter = 2 × √(hull_area / π)  →  × (µm/px)  [등가원 지름]",
        color=C_MEAS, fs=7.8)
    dn(8.55, 8.05)

    # ── DIAMOND: 분류 ──────────────────────────────────────────────────────────
    diamond(ax, XC, 7.7, 6.0, 0.75, "hull_area ≥\n1500 px²?", fs=8.0)
    ax.text(11.2, 7.7, "YES", fontsize=7.5, color=C_MEAS, fontweight="bold", va="center")
    ax.text(4.8, 7.7, "NO", fontsize=7.5, color=C_COMMON, fontweight="bold", va="center", ha="right")

    # YES → right → particle
    hline(11.0, 12.8, 7.7); side_arr(12.8, 7.7, 7.25)
    box(ax, 12.8, 6.95, 2.5, 0.6, "particle", color=C_MEAS, fs=8.5, bold=True)

    # NO → left → fragment
    hline(5.0, 3.2, 7.7); side_arr(3.2, 7.7, 7.25)
    box(ax, 3.2, 6.95, 2.5, 0.6, "fragment", color=C_COMMON, fs=8.5, bold=True)

    # Merge both at bottom
    vline(12.8, 6.65, 6.4); vline(3.2, 6.65, 6.4)
    hline(3.2, 12.8, 6.4)
    ax.annotate("", xy=(XC, 6.2), xytext=(XC, 6.4),
                arrowprops=dict(arrowstyle="->", color=LC, lw=1.4), zorder=1)
    sep(6.05)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    box(ax, XC, 5.55, BW, 0.9,
        "출력  (OUTPUT)\n"
        "input_roi.png  ·  classified.png  ·  overlay_roi.png  ·  summary.json\n"
        "[--debug]  tiles.png  prompts.png  objects.csv  size/sph_dist.png  overlay_S/Sp.png\n"
        "Batch:  img_id_summary.json  ·  batch_summary.json  ·  히스토그램 PNG",
        color=C_OUT, fs=7.8, bold=True)

    # Legend
    legend_patches = [
        leg_patch(C_IO,     "Input / Output"),
        leg_patch(C_PREP,   "전처리 / ROI"),
        leg_patch(C_SAM2,   "HCT / SAM2 / Dedup"),
        leg_patch(C_MEAS,   "Kasa / Hull / 측정 / 분류"),
        leg_patch(C_BRANCH, "분기 (YES 경로)"),
        leg_patch(C_COMMON, "fragment"),
        leg_patch(C_OUT,    "Output"),
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              bbox_to_anchor=(0.01, 0.005), fontsize=8,
              framealpha=0.9, edgecolor="#CBD5E1", ncol=4,
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
