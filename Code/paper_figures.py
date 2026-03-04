#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_figures.py
================
Publication-quality figures for the OpenLKA TakeOver dataset paper.
Style: AI top-conference (NeurIPS / CVPR / IV), no grids, minimal spines.

Color Palette (used across ALL figures in the paper):
    Blue    #4C78A8      Orange  #F58518
    Red     #E45756      Teal    #72B7B2
    Green   #54A24B      Purple  #B279A2
    Gray    #BAB0AC      Brown   #9D755D

Semantic Color Bindings (fixed across ALL figures):
    Powertrain:  ICE = Blue,  BEV = Green,  HEV/PHEV = Teal
    Trigger:     Steering = Blue,  Brake = Teal,  Gas = Green,  System = Gray
    Statistics:  Mean line = Red,  Median line = Orange
    Histograms:  Primary = Blue,  Secondary = Teal
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
DATA = ROOT / "Code" / "stats_output" / "per_clip.csv"
OUT  = ROOT / "Code" / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Color Palette
# ═══════════════════════════════════════════════════════════════════════
C = dict(
    blue   = "#4C78A8",
    orange = "#F58518",
    red    = "#E45756",
    teal   = "#72B7B2",
    green  = "#54A24B",
    purple = "#B279A2",
    gray   = "#BAB0AC",
    brown  = "#9D755D",
)

# ═══════════════════════════════════════════════════════════════════════
#  Global Style (IEEE / serif / minimal)
# ═══════════════════════════════════════════════════════════════════════
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
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ═══════════════════════════════════════════════════════════════════════
#  Load
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA)
N = len(df)
print(f"Loaded {N:,} clips")

# ═══════════════════════════════════════════════════════════════════════
#  Fig 2: Dataset Overview  (1 × 5)  — double-column IEEE figure
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2),
                         gridspec_kw={"wspace": 0.42})
LABEL_KW = dict(fontsize=10, fontweight="bold", va="top")

# ── (a) Top-10 vehicle brands ────────────────────────────────────────
ax = axes[0]
n_brands_shown = 10
n_brands_total = df["brand"].nunique()
brand_cnts = df.groupby("brand").size().sort_values(ascending=False).head(n_brands_shown)
y = brand_cnts.sort_values()
bars = ax.barh(y.index, y.values, color=C["blue"], height=0.68,
               edgecolor="white", linewidth=0.4)
for bar, v in zip(bars, y.values):
    ax.text(v + 10, bar.get_y() + bar.get_height() / 2,
            f"{v:,}", va="center", fontsize=6.5, color="#333")
ax.set_xlabel("Number of Clips")
ax.set_xlim(0, y.max() * 1.15)
ax.text(-0.14, 1.08, "(a)", transform=ax.transAxes, **LABEL_KW)

# ── (b) Speed distribution ───────────────────────────────────────────
ax = axes[1]
spd = df["speed_kmh"].dropna()
spd = spd[spd >= 0]
bins = np.arange(0, int(spd.max()) + 5, 3)
ax.hist(spd, bins=bins, color=C["blue"], edgecolor="white", linewidth=0.2,
        alpha=0.85, zorder=2)
mu, med = spd.mean(), spd.median()
ax.axvline(mu,  color=C["red"],    lw=1.2, ls="--", zorder=3,
           label=f"Mean = {mu:.1f}")
ax.axvline(med, color=C["orange"], lw=1.2, ls="-.", zorder=3,
           label=f"Median = {med:.1f}")
ax.set_xlabel("Speed at Takeover (km/h)")
ax.set_ylabel("Count")
ax.legend(frameon=False, loc="upper right", handlelength=1.4, fontsize=6.5)
ax.text(-0.14, 1.08, "(b)", transform=ax.transAxes, **LABEL_KW)

# ── (c) Lead vehicle distance distribution ───────────────────────────
ax = axes[2]
lead = df[df["has_lead"]]
drel = lead["lead_drel_m"].dropna()
drel = drel[(drel > 0) & (drel < 200)]
bins_d = np.arange(0, 155, 3)
ax.hist(drel, bins=bins_d, color=C["green"], edgecolor="white", linewidth=0.2,
        alpha=0.85, zorder=2)
mu_d, med_d = drel.mean(), drel.median()
ax.axvline(mu_d,  color=C["red"],    lw=1.0, ls="--", zorder=3,
           label=f"Mean = {mu_d:.1f}")
ax.axvline(med_d, color=C["orange"], lw=1.0, ls="-.", zorder=3,
           label=f"Median = {med_d:.1f}")
ax.set_xlabel("Distance to Lead Vehicle (m)")
ax.set_ylabel("Count")
ax.legend(frameon=False, loc="upper right", handlelength=1.4, fontsize=6.5)
ax.text(-0.14, 1.08, "(c)", transform=ax.transAxes, **LABEL_KW)

# ── (d) Primary takeover action ──────────────────────────────────────
ax = axes[3]
pt_df = df["primary_trigger"].value_counts()
_trig_order = ["Steering", "Brake", "Gas", "Mixed", "System / Unknown"]
_trig_short = ["Steer", "Brake", "Gas", "Mixed", "Sys."]
_trig_colors = [C["blue"], C["teal"], C["green"], C["purple"], C["gray"]]
_pairs = [(s, pt_df.get(o, 0), c)
          for o, s, c in zip(_trig_order, _trig_short, _trig_colors)
          if pt_df.get(o, 0) > 0]
short  = [p[0] for p in _pairs]
vals   = [p[1] for p in _pairs]
colors = [p[2] for p in _pairs]
bars = ax.bar(short, vals, color=colors, width=0.60,
              edgecolor="white", linewidth=0.4)

raw_pcts = [v / N * 100 for v in vals]
floored = [int(p * 10) / 10 for p in raw_pcts]
remainders = [round(r - f, 4) for r, f in zip(raw_pcts, floored)]
deficit = round(100.0 - sum(floored), 1)
n_bumps = int(round(deficit * 10))
bump_idx = sorted(range(len(remainders)), key=lambda i: -remainders[i])
display_pcts = list(floored)
for i in range(min(n_bumps, len(display_pcts))):
    display_pcts[bump_idx[i]] = round(display_pcts[bump_idx[i]] + 0.1, 1)

for bar, pct in zip(bars, display_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 25,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=6.5, color="#333")
ax.set_ylabel("Number of Clips")
ax.set_xlabel("Primary Takeover Action")
ax.tick_params(axis="x", labelsize=7)
ax.text(-0.14, 1.08, "(d)", transform=ax.transAxes, **LABEL_KW)

plt.tight_layout(w_pad=1.2)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_dataset_overview.{ext}")
plt.close()
print("  ✓ fig_dataset_overview")

# ═══════════════════════════════════════════════════════════════════════
#  Fig 3: Clip structure timeline (conceptual schematic)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 1.3))
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-12, 12)
ax.set_ylim(-0.5, 1.6)

# ON bar
ax.barh(0.8, 10, left=-10, height=0.5, color=C["blue"], alpha=0.85,
        edgecolor="none")
ax.text(-5, 0.8, "ADAS ON", ha="center", va="center",
        fontsize=8, color="white", fontweight="bold")
# OFF bar
ax.barh(0.8, 10, left=0, height=0.5, color=C["gray"], alpha=0.65,
        edgecolor="none")
ax.text(5, 0.8, "ADAS OFF", ha="center", va="center",
        fontsize=8, color="white", fontweight="bold")

# Event marker
ax.plot(0, 0.8, marker="v", color=C["red"], markersize=9, zorder=5)
ax.text(0, 0.15, "Takeover\nEvent", ha="center", va="top",
        fontsize=7, color=C["red"], fontweight="bold")

# Window bracket
bw = 0.55
ax.annotate("", xy=(-10, bw), xytext=(10, bw),
            arrowprops=dict(arrowstyle="<->", color="#333", lw=0.8))
ax.text(0, bw - 0.12, "20 s clip window", ha="center", va="top",
        fontsize=7, color="#333")

# Time labels
for x, label in [(-10, "−10 s"), (0, "0 s"), (10, "+10 s")]:
    ax.text(x, 1.35, label, ha="center", va="bottom", fontsize=7, color="#555")

plt.tight_layout(pad=0.1)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_clip_structure.{ext}")
plt.close()
print("  ✓ fig_clip_structure")

# ═══════════════════════════════════════════════════════════════════════
#  Fig (optional): Lead vehicle distance histogram
# ═══════════════════════════════════════════════════════════════════════
lead = df[df["has_lead"]]
drel = lead["lead_drel_m"].dropna()
drel = drel[(drel > 0) & (drel < 200)]

fig, ax = plt.subplots(figsize=(3.5, 2.2))
bins_d = np.arange(0, 155, 3)
ax.hist(drel, bins=bins_d, color=C["teal"], edgecolor="white", linewidth=0.2,
        alpha=0.85)
mu_d, med_d = drel.mean(), drel.median()
ax.axvline(mu_d,  color=C["red"],    lw=1.0, ls="--",
           label=f"Mean = {mu_d:.1f} m")
ax.axvline(med_d, color=C["orange"], lw=1.0, ls="-.",
           label=f"Median = {med_d:.1f} m")
ax.set_xlabel("Distance to Lead Vehicle (m)")
ax.set_ylabel("Count")
ax.legend(frameon=False, loc="upper right", handlelength=1.6)
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_lead_distance.{ext}")
plt.close()
print("  ✓ fig_lead_distance")

# ═══════════════════════════════════════════════════════════════════════
#  Fig (optional): Speed CDF
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.2))
s_sorted = np.sort(spd.values)
cdf = np.arange(1, len(s_sorted) + 1) / len(s_sorted) * 100
ax.plot(s_sorted, cdf, color=C["blue"], lw=1.5)
for pct, col, ls in [(50, C["gray"], ":"), (90, C["orange"], "--"),
                      (95, C["red"], "--")]:
    v = float(np.percentile(s_sorted, pct))
    ax.axhline(pct, color=col, lw=0.7, ls=ls, alpha=0.6)
    ax.text(s_sorted.max() * 0.97, pct + 1.5,
            f"P{pct} = {v:.0f} km/h", ha="right", fontsize=6.5, color=col)
ax.set_xlabel("Speed (km/h)")
ax.set_ylabel("Cumulative (%)")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_speed_cdf.{ext}")
plt.close()
print("  ✓ fig_speed_cdf")

# ═══════════════════════════════════════════════════════════════════════
#  Fig: Clips per driver (dongle_id) — long-tail distribution
# ═══════════════════════════════════════════════════════════════════════
drv = df.groupby("dongle_id").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(3.5, 2.2))
ax.bar(range(len(drv)), drv.values, color=C["blue"], width=1.0,
       edgecolor="none", alpha=0.85)
# Highlight top-10
ax.bar(range(10), drv.values[:10], color=C["orange"], width=1.0,
       edgecolor="none", alpha=0.85)
top10_pct = drv.values[:10].sum() / drv.sum() * 100
ax.text(10, drv.values[9], f"  Top-10: {top10_pct:.1f}%",
        fontsize=7, color=C["orange"], va="center")
ax.set_xlabel("Driver Index (sorted by clip count)")
ax.set_ylabel("Number of Clips")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_clips_per_driver.{ext}")
plt.close()
print("  ✓ fig_clips_per_driver")

print(f"\nAll figures → {OUT}/")
