#!/usr/bin/env python3
"""
make_fig2_clip_structure.py
===========================
Generate Fig 2 for the paper: clip structure diagram with three example frames
from a real takeover clip aligned to ON / TakeOver / OFF phases.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
OUT  = ROOT / "Code" / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

frame_on       = OUT / "frame_on.jpg"
frame_takeover = OUT / "frame_takeover.jpg"
frame_off      = OUT / "frame_off.jpg"

# ── Color palette (consistent with paper_figures.py) ───────────────────
C = dict(
    blue   = "#4C78A8",
    orange = "#F58518",
    red    = "#E45756",
    teal   = "#72B7B2",
    green  = "#54A24B",
    gray   = "#BAB0AC",
)

# ── Global style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

# ── Load images ────────────────────────────────────────────────────────
img_on  = mpimg.imread(str(frame_on))
img_to  = mpimg.imread(str(frame_takeover))
img_off = mpimg.imread(str(frame_off))

# ── Figure layout: top = timeline schematic, bottom = 3 frames ────────
fig = plt.figure(figsize=(7.16, 3.8))

# Top: clip structure timeline (spans full width)
ax_top = fig.add_axes([0.05, 0.58, 0.90, 0.38])

# Bottom: three image panels
gap = 0.03
img_w = 0.28
img_h = 0.42
y_img = 0.06
x_start = 0.05 + (0.90 - 3 * img_w - 2 * gap) / 2

ax_on  = fig.add_axes([x_start,                    y_img, img_w, img_h])
ax_to  = fig.add_axes([x_start + img_w + gap,      y_img, img_w, img_h])
ax_off = fig.add_axes([x_start + 2*(img_w + gap),  y_img, img_w, img_h])

# ════════════════════════════════════════════════════════════════════════
#  Top panel: clip structure timeline
# ════════════════════════════════════════════════════════════════════════
ax = ax_top
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-12, 12)
ax.set_ylim(-0.8, 2.0)

# Main timeline bar
bar_y = 1.0
bar_h = 0.45

# ADAS ON bar
ax.barh(bar_y, 10, left=-10, height=bar_h, color=C["blue"], alpha=0.85,
        edgecolor="none")
ax.text(-5, bar_y, "ADAS Engaged (ON)", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold")

# ADAS OFF bar
ax.barh(bar_y, 10, left=0, height=bar_h, color=C["gray"], alpha=0.70,
        edgecolor="none")
ax.text(5, bar_y, "Manual Driving (OFF)", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold")

# Event marker (red triangle)
ax.plot(0, bar_y, marker="v", color=C["red"], markersize=11, zorder=5,
        clip_on=False)
ax.plot([0, 0], [bar_y - bar_h/2 - 0.05, bar_y + bar_h/2 + 0.05],
        color=C["red"], linewidth=1.5, zorder=4)

# "Takeover Event" label
ax.text(0, bar_y - bar_h/2 - 0.22, "Takeover\nEvent ($t=0$)",
        ha="center", va="top", fontsize=8, color=C["red"], fontweight="bold")

# Window bracket at top
bw_y = bar_y + bar_h/2 + 0.15
ax.annotate("", xy=(-10, bw_y), xytext=(10, bw_y),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=0.8))
ax.text(0, bw_y + 0.08, "20 s clip window", ha="center", va="bottom",
        fontsize=8, color="#333")

# Time labels at top
for x, label in [(-10, "$-10$ s"), (0, "$0$ s"), (10, "$+10$ s")]:
    ax.text(x, bw_y + 0.35, label, ha="center", va="bottom",
            fontsize=7.5, color="#555")

# Dashed lines connecting timeline to images
for x_pos, target_ax in [(-7, ax_on), (0, ax_to), (7, ax_off)]:
    # Get the target axes position in figure coords
    bbox = target_ax.get_position()
    target_x_fig = bbox.x0 + bbox.width / 2
    target_y_fig = bbox.y0 + bbox.height

    source_pos = ax.get_position()
    source_x_fig = source_pos.x0 + (x_pos - (-12)) / 24 * source_pos.width
    source_y_fig = source_pos.y0

    fig.add_artist(matplotlib.lines.Line2D(
        [source_x_fig, target_x_fig],
        [source_y_fig, target_y_fig + 0.005],
        color="#999", linewidth=0.6, linestyle="--",
        transform=fig.transFigure, clip_on=False
    ))

# Small triangles on timeline pointing to frame times
for x_pos, col in [(-7, C["blue"]), (0, C["red"]), (7, C["gray"])]:
    ax.plot(x_pos, bar_y - bar_h/2 - 0.02, marker="^", color=col,
            markersize=6, zorder=5, clip_on=False)

# ════════════════════════════════════════════════════════════════════════
#  Bottom panels: three frames
# ════════════════════════════════════════════════════════════════════════
for ax_img, img, title, border_col in [
    (ax_on,  img_on,  "(a) ADAS ON ($t = -7$ s)",     C["blue"]),
    (ax_to,  img_to,  "(b) Takeover ($t = 0$ s)",     C["red"]),
    (ax_off, img_off, "(c) Manual ($t = +7$ s)",       C["gray"]),
]:
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    # Colored border
    for sp in ax_img.spines.values():
        sp.set_color(border_col)
        sp.set_linewidth(2.0)
        sp.set_visible(True)
    ax_img.set_title(title, fontsize=8, pad=4, fontweight="bold",
                     color=border_col)

# ── Save ───────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_clip_structure.{ext}")
plt.close()
print(f"Saved fig_clip_structure.pdf/png → {OUT}")
