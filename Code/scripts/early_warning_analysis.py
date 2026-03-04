#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
early_warning_analysis.py
=========================
Parse VLM early_warning_hypothesis to quantify when visual cues first
appeared, then visualize the temporal advantage by archetype.

Output: fig7_early_warning_advantage.png/.pdf
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
OUT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver"
           "/Code/long_lat/longtail")
CSV = OUT / "longtail_analysis_summary.csv"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          12,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

# Temporal-window palette — green→amber→red traffic-light metaphor
WW_PAL = {
    "5+ s":  "#2D8E47",   # rich green
    "3–5 s": "#F0A030",   # warm amber
    "< 3 s": "#CC3333",   # clear red
}

# Archetype accent colors (for dot plot)
ARCH_PAL = {
    "Traffic Dyn.": "#4A9A5B",
    "Infra. Deg.":  "#3D7EAA",
    "Adverse Env.": "#D48A2C",
    "All Clips":    "#555555",
}

# ══════════════════════════════════════════════════════════════════════════════
#  ARCHETYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
P1 = {"Faded Lane Lines", "Construction", "Sharp Curve", "Obstruction"}
P2 = {"Wet Road", "Glare/Weather", "Night Driving"}
P3 = {"Slow Vehicle", "Hard Braking", "Cut-in", "Tailgating",
      "Congestion", "Merging", "Pedestrian"}


def classify_archetype(rf: str) -> str:
    if pd.isna(rf) or not rf.strip():
        return "Other"
    tags = {t.strip() for t in rf.split(";")}
    if tags & P1:
        return "Infra. Deg."
    if tags & P2:
        return "Adverse Env."
    if tags & P3:
        return "Traffic Dyn."
    return "Other"


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL CUE PARSING
# ══════════════════════════════════════════════════════════════════════════════
RE_T5 = re.compile(
    r"T[-−–]\s*5"
    r"|(?:as early as|at least)\s+5\s*s"
    r"|5\s*s(?:econds?)?\s*(?:before|prior|earlier)"
    r"|(?:at|from|by)\s+5\s*s",
    re.IGNORECASE
)

RE_T3 = re.compile(
    r"T[-−–]\s*3"
    r"|(?:as early as|at least)\s+3\s*s"
    r"|3\s*s(?:econds?)?\s*(?:before|prior|earlier)"
    r"|(?:at|from|by)\s+3\s*s",
    re.IGNORECASE
)


def classify_warning_window(text: str) -> str:
    """Classify earliest actionable cue into one of three temporal bins."""
    if pd.isna(text) or not text.strip():
        return "< 3 s"
    if RE_T5.search(text):
        return "5+ s"
    if RE_T3.search(text):
        return "3–5 s"
    return "< 3 s"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    df = pd.read_csv(CSV)
    total = len(df)
    print(f"  Loaded {total} clips")

    df["warning_window"] = df["early_warning_hypothesis"].apply(
        classify_warning_window)
    df["archetype"] = df["risk_factors"].apply(classify_archetype)

    # ── Stats ──
    ww_counts = df["warning_window"].value_counts()
    print("\n  Warning Window Distribution:")
    for cat in ["5+ s", "3–5 s", "< 3 s"]:
        n = ww_counts.get(cat, 0)
        print(f"    {cat:10s}  {n:4d}  ({100*n/total:.1f}%)")

    early_pct = 100 * (ww_counts.get("5+ s", 0) + ww_counts.get("3–5 s", 0)) / total
    print(f"\n  Cues visible at >= 3s before takeover: {early_pct:.1f}%")

    # ── Compute per-group breakdowns ──
    cat_order = ["5+ s", "3–5 s", "< 3 s"]
    row_labels = ["All Clips", "Traffic Dyn.", "Infra. Deg.", "Adverse Env."]

    group_data = {}   # label -> {cat: count}
    group_n = {}      # label -> total N

    # All clips
    group_n["All Clips"] = total
    group_data["All Clips"] = {c: ww_counts.get(c, 0) for c in cat_order}

    # Per archetype
    for arch in ["Traffic Dyn.", "Infra. Deg.", "Adverse Env."]:
        sub = df[df["archetype"] == arch]
        n = len(sub)
        group_n[arch] = n
        vc = sub["warning_window"].value_counts()
        group_data[arch] = {c: vc.get(c, 0) for c in cat_order}

    # ── Figure: 2 panels ──
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 3.8),
        gridspec_kw={"wspace": 0.55, "width_ratios": [1.3, 1],
                     "left": 0.08, "right": 0.96, "top": 0.88, "bottom": 0.12})

    # ═══════════════════════════════════════════════════════
    #  (a) 100% stacked horizontal bars by group
    # ═══════════════════════════════════════════════════════
    bar_h = 0.55
    y_pos = np.arange(len(row_labels))[::-1]  # top-to-bottom

    for i, label in enumerate(row_labels):
        y = y_pos[i]
        n = group_n[label]
        left = 0.0
        for cat in cat_order:
            cnt = group_data[label][cat]
            pct = 100.0 * cnt / n if n > 0 else 0
            bar = ax_a.barh(y, pct, left=left, height=bar_h,
                            color=WW_PAL[cat], edgecolor="white", lw=0.8)
            # Label inside bar if wide enough
            if pct > 8:
                ax_a.text(left + pct / 2, y, f"{pct:.0f}%",
                          ha="center", va="center", fontsize=10,
                          fontweight="bold", color="white",
                          path_effects=[pe.withStroke(linewidth=1.5,
                                                     foreground="black",
                                                     alpha=0.3)])
            left += pct

        # N label at end of bar
        ax_a.text(101.5, y, f"n={n}", ha="left", va="center",
                  fontsize=10, color="#666666")

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(row_labels, fontsize=12, fontweight="bold")
    ax_a.set_xlim(0, 115)
    ax_a.set_xlabel("Proportion (%)", fontsize=13)
    ax_a.set_title("(a)  Warning Window Composition", fontsize=14,
                    fontweight="bold", loc="left", pad=10)

    # Subtle grid
    for xv in [25, 50, 75]:
        ax_a.axvline(xv, color="#e0e0e0", lw=0.5, zorder=0)

    # Legend — horizontal below x-label
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=WW_PAL[c], edgecolor="white", label=c)
                      for c in cat_order]
    ax_a.legend(handles=legend_patches, loc="upper center",
                bbox_to_anchor=(0.42, -0.18), ncol=3, frameon=False,
                fontsize=11, handlelength=1.2, handletextpad=0.5,
                columnspacing=1.5)

    # ═══════════════════════════════════════════════════════
    #  (b) Lollipop / Cleveland dot plot — % predictable
    # ═══════════════════════════════════════════════════════
    lollipop_labels = ["All Clips", "Traffic Dyn.", "Infra. Deg.", "Adverse Env."]
    lollipop_y = np.arange(len(lollipop_labels))[::-1]
    pct_vals = []
    for label in lollipop_labels:
        n = group_n[label]
        n5 = group_data[label].get("5+ s", 0)
        n3 = group_data[label].get("3–5 s", 0)
        pct_vals.append(100.0 * (n5 + n3) / n if n > 0 else 0)

    for y, label, pct in zip(lollipop_y, lollipop_labels, pct_vals):
        c = ARCH_PAL[label]
        # Stem line
        ax_b.plot([0, pct], [y, y], color=c, lw=2.0, solid_capstyle="round",
                  zorder=2)
        # Dot
        ax_b.scatter(pct, y, s=140, color=c, edgecolors="white", linewidths=1.2,
                     zorder=3)
        # Value label
        ax_b.text(pct + 1.8, y, f"{pct:.0f}%",
                  ha="left", va="center", fontsize=13,
                  fontweight="bold", color=c)

    ax_b.set_yticks(lollipop_y)
    ax_b.set_yticklabels(lollipop_labels, fontsize=12, fontweight="bold")
    ax_b.set_xlim(0, 85)
    ax_b.set_xlabel("Clips with cue $\\geq$ 3 s before takeover (%)", fontsize=12)
    ax_b.set_title("(b)  Predictability by Archetype", fontsize=14,
                    fontweight="bold", loc="left", pad=10)

    # Subtle vertical grid
    for xv in [20, 40, 60, 80]:
        ax_b.axvline(xv, color="#e0e0e0", lw=0.5, zorder=0)

    # Remove left spine for cleaner look on panel b
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)

    fig.savefig(OUT / "fig7_early_warning_advantage.png")
    fig.savefig(OUT / "fig7_early_warning_advantage.pdf")
    plt.close(fig)
    print("\n  Saved fig7_early_warning_advantage.png / .pdf")

    # Per-archetype detail
    print("\n  Predictability by Archetype:")
    for label, pct in zip(lollipop_labels, pct_vals):
        print(f"    {label:16s}  {pct:5.1f}%  (N={group_n[label]})")

    print("\nDone.")


if __name__ == "__main__":
    main()
