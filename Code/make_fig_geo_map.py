#!/usr/bin/env python3
"""
make_fig_geo_map.py
====================
Publication-quality geographic distribution figure for the TakeOver paper.
Matches the style of paper_figures.py (IEEE / serif / minimal spines).
Uses geopandas for the basemap and KDE for density visualization.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.stats import gaussian_kde

ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
GPS_CSV = ROOT / "Code" / "stats_output" / "route_gps.csv"
OUT = ROOT / "Code" / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Paper color palette (same as paper_figures.py) ──
C = dict(
    blue="#4C78A8", orange="#F58518", red="#E45756",
    teal="#72B7B2", green="#54A24B", purple="#B279A2",
    gray="#BAB0AC", brown="#9D755D",
)
C_LAND   = "#EDEAE3"
C_OCEAN  = "#F7F7F7"
C_BORDER = "#D0D0D0"
C_COAST  = "#AAAAAA"

# ── Global style (matches paper_figures.py exactly) ──
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          8,
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "axes.grid":          False,
})

# ── Load data ──
df = pd.read_csv(GPS_CSV)
print(f"Loaded {len(df)} GPS points, {df.dongle_id.nunique()} devices")

# Load Natural Earth basemap
ne_path = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical/ne_110m_land.shp"
if not ne_path.exists():
    # Try geopandas built-in
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
else:
    world = gpd.read_file(ne_path)

# ── Region classification ──
region_defs = {
    "North America": ((24, 50), (-130, -60)),
    "Europe":        ((35, 72), (-12, 40)),
    "Asia":          ((0, 55),  (60, 145)),
    "Oceania":       ((-50, -10), (110, 180)),
    "South America": ((-56, 15), (-82, -34)),
}

region_masks = {}
classified = pd.Series(False, index=df.index)
for name, ((lat_lo, lat_hi), (lng_lo, lng_hi)) in region_defs.items():
    m = (df.lat > lat_lo) & (df.lat < lat_hi) & (df.lng > lng_lo) & (df.lng < lng_hi) & ~classified
    region_masks[name] = m
    classified |= m
region_masks["Other"] = ~classified

for name, mask in region_masks.items():
    print(f"  {name}: {mask.sum()} ({mask.sum()/len(df)*100:.1f}%)")


def draw_basemap(ax, xlim, ylim):
    """Draw land/ocean basemap."""
    ax.set_facecolor(C_OCEAN)
    world.plot(ax=ax, color=C_LAND, edgecolor=C_COAST, linewidth=0.25)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def scatter_with_density(ax, lats, lngs, s=5, alpha_base=0.5, cmap_name="blue"):
    """Scatter points colored by local density."""
    if len(lats) < 3:
        ax.scatter(lngs, lats, s=s, c=C["red"], alpha=0.6, edgecolors="none", zorder=3)
        return

    xy = np.vstack([lngs, lats])
    try:
        kde = gaussian_kde(xy, bw_method=0.15)
        density = kde(xy)
        density = (density - density.min()) / (density.max() - density.min() + 1e-10)
    except Exception:
        density = np.ones(len(lats)) * 0.5

    # Custom colormap using paper palette: blue -> red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "paper", [C["blue"], C["teal"], C["orange"], C["red"]], N=256
    )
    order = np.argsort(density)
    ax.scatter(
        np.array(lngs)[order], np.array(lats)[order],
        s=s, c=density[order], cmap=cmap, alpha=alpha_base,
        edgecolors="none", zorder=3, vmin=0, vmax=1,
    )
    return cmap


# ── Figure layout: world map (top) + US inset (bottom-left) ──
fig = plt.figure(figsize=(7.16, 3.8))

# Main world map axes
ax_main = fig.add_axes([0.0, 0.22, 1.0, 0.78])
draw_basemap(ax_main, (-170, 180), (-55, 75))

# Scatter all points with density coloring
cmap = scatter_with_density(ax_main, df.lat.values, df.lng.values, s=5, alpha_base=0.55)

# ── Region annotations on world map ──
text_style = dict(
    fontsize=7, fontweight="bold", color="#444",
    ha="center", va="top",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              alpha=0.88, edgecolor=C["gray"], linewidth=0.4),
    zorder=6,
)

region_labels = {
    "Europe":        (18, 38),
    "Asia":          (100, 8),
}
for name, (lx, ly) in region_labels.items():
    n = region_masks[name].sum()
    pct = n / len(df) * 100
    ax_main.text(lx, ly, f"{name}\n{n:,} routes ({pct:.0f}%)", **text_style)

# Other regions combined
n_other = sum(region_masks[r].sum() for r in ["Oceania", "South America", "Other"])
pct_other = n_other / len(df) * 100
ax_main.text(148, -38, f"Other\n{n_other:,} ({pct_other:.0f}%)",
             **{**text_style, "fontsize": 6})

# ── US inset (bottom-left, inside world map) ──
ax_us = fig.add_axes([0.005, 0.23, 0.40, 0.45])
draw_basemap(ax_us, (-127, -64), (23, 51))

us_df = df[region_masks["North America"]]
scatter_with_density(ax_us, us_df.lat.values, us_df.lng.values, s=3, alpha_base=0.45)

# Inset border
for sp in ax_us.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor("#888")
    sp.set_linewidth(0.6)
na_n = region_masks["North America"].sum()
na_pct = na_n / len(df) * 100
ax_us.set_title(f"North America  ({na_n:,} routes, {na_pct:.0f}%)",
                fontsize=6.5, pad=2, color="#444", fontweight="bold")

# Dashed rectangle on world map showing US inset extent
import matplotlib.patches as mpatches
rect = mpatches.Rectangle((-127, 23), 63, 28,
                           linewidth=0.6, edgecolor="#888",
                           facecolor="none", linestyle="--", zorder=4)
ax_main.add_patch(rect)

# ── Bottom strip: stacked proportion bar + legend ──
ax_bar = fig.add_axes([0.0, 0.0, 1.0, 0.20])
ax_bar.set_xlim(0, 1)
ax_bar.set_ylim(0, 1)
ax_bar.axis("off")

bar_colors = [C["blue"], C["teal"], C["green"], C["orange"], C["purple"], C["gray"]]
bar_regions = ["North America", "Europe", "Asia", "Oceania", "South America", "Other"]
bar_counts = [region_masks[r].sum() for r in bar_regions]
bar_total = sum(bar_counts)
bar_fracs = [c / bar_total for c in bar_counts]

# Draw stacked horizontal bar
x0 = 0.03
bar_y, bar_h = 0.62, 0.22
for i, (name, frac, count) in enumerate(zip(bar_regions, bar_fracs, bar_counts)):
    w = frac * 0.94
    ax_bar.barh(bar_y, w, height=bar_h, left=x0,
                color=bar_colors[i], edgecolor="white", linewidth=0.5, zorder=2)
    # Only label if segment is wide enough
    if frac > 0.08:
        pct_str = f"{frac*100:.0f}%"
        ax_bar.text(x0 + w / 2, bar_y, pct_str,
                    ha="center", va="center", fontsize=6.5,
                    color="white", fontweight="bold", zorder=3)
    x0 += w

# Legend row: colored dot + region name + count
legend_y = 0.18
x_positions = [0.03, 0.20, 0.34, 0.47, 0.60, 0.76]
for i, (name, count) in enumerate(zip(bar_regions, bar_counts)):
    xl = x_positions[i]
    ax_bar.plot(xl, legend_y, 'o', color=bar_colors[i], markersize=3.5, zorder=3)
    ax_bar.text(xl + 0.012, legend_y, f"{name} ({count:,})",
                fontsize=5.8, va="center", color="#555")

# Summary line (right-aligned at bar level)
n_total_drivers = 327  # from dataset_statistics.py (per_clip.csv)
n_gps_drivers = df.dongle_id.nunique()  # 323 with GPS
n_unknown = n_total_drivers - n_gps_drivers
summary_txt = (f"{len(df):,} routes  |  "
               f"{n_total_drivers} drivers ({n_unknown} without GPS)")
ax_bar.text(0.97, 0.42, summary_txt,
            ha="right", va="top", fontsize=5.8, color="#666",
            fontstyle="italic")

# ── Density colorbar (top-right of world map) ──
from matplotlib.colorbar import ColorbarBase
ax_cb = fig.add_axes([0.82, 0.88, 0.12, 0.013])
cmap_obj = mcolors.LinearSegmentedColormap.from_list(
    "paper", [C["blue"], C["teal"], C["orange"], C["red"]], N=256
)
norm = mcolors.Normalize(vmin=0, vmax=1)
cb = ColorbarBase(ax_cb, cmap=cmap_obj, norm=norm, orientation="horizontal")
cb.set_ticks([0, 1])
cb.set_ticklabels(["Low", "High"])
cb.ax.tick_params(labelsize=5, length=1.5, width=0.3, pad=1)
cb.outline.set_linewidth(0.3)
ax_cb.set_title("Density", fontsize=5.5, pad=1.5, color="#666")

# ── Save ──
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_geo_distribution.{ext}")
plt.close()
print(f"\nSaved fig_geo_distribution.pdf/png → {OUT}")
