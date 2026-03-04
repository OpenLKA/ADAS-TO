#!/usr/bin/env python3
"""Regenerate fig2_ttc_thw_scatter with improved layout."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver/Code/long_lat")
CSV = OUT / "per_clip_all_metrics.csv"

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

C = {
    "ego": "#4C78A8", "lead": "#E45756", "steer": "#F58518",
    "brake": "#E45756", "gas": "#54A24B",
    "ttc": "#FF9DA7", "thw": "#9D755D", "grey": "#888888",
}


def main():
    df = pd.read_csv(CSV)
    valid = df.dropna(subset=["min_ttc_pre", "thw_at_min_ttc"])
    valid = valid[(valid["min_ttc_pre"] > 0) & (valid["thw_at_min_ttc"] > 0)]
    ttc_col, thw_col = "min_ttc_pre", "thw_at_min_ttc"
    print(f"  Fig 2: {len(valid)} clips with verified lead + TTC/THW")

    ttc = valid[ttc_col].clip(upper=30)
    thw = valid[thw_col].clip(upper=10)

    n_ttc15 = (valid[ttc_col] < 1.5).sum()
    n_ttc30 = (valid[ttc_col] < 3.0).sum()
    n_thw08 = (valid[thw_col] < 0.8).sum()

    # ── Layout ──
    fig = plt.figure(figsize=(8, 6.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.06, wspace=0.06)

    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # ── Scatter (main) ──
    colors_map = {"Brake": C["brake"], "Gas": C["gas"],
                  "Steering": C["steer"]}
    clrs = valid["primary_trigger"].map(colors_map).fillna(C["grey"])

    ax_main.scatter(ttc, thw, c=clrs, s=16, alpha=0.45,
                    edgecolors="none", rasterized=True)

    # Threshold lines
    ax_main.axvline(1.5, color=C["ttc"], ls="--", lw=1.0, alpha=0.8)
    ax_main.axvline(3.0, color=C["ttc"], ls=":",  lw=0.8, alpha=0.5)
    ax_main.axhline(0.8, color=C["thw"], ls="--", lw=1.0, alpha=0.8)
    ax_main.axhline(1.5, color=C["thw"], ls=":",  lw=0.8, alpha=0.5)

    ax_main.set_xlabel("Min TTC in pre-window (s)")
    ax_main.set_ylabel("THW at min-TTC (s)")

    # Stats annotation — bottom-left to avoid scatter overlap

    # Legend — upper-left (scatter is sparse there, away from marginals)
    handles = [mpatches.Patch(color=c, label=l) for l, c in colors_map.items()
               if l in valid["primary_trigger"].values]
    if handles:
        ax_main.legend(handles=handles, loc="upper left", frameon=True,
                       fancybox=True, framealpha=0.9, edgecolor="#cccccc")

    # ── Marginal top (TTC histogram) ──
    ax_top.hist(ttc, bins=60, color=C["ego"], alpha=0.6,
                edgecolor="white", lw=0.3)
    ax_top.axvline(1.5, color=C["ttc"], ls="--", lw=1.0, alpha=0.8)
    ax_top.axvline(3.0, color=C["ttc"], ls=":",  lw=0.8, alpha=0.5)
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["bottom"].set_visible(False)

    # ── Marginal right (THW histogram) ──
    ax_right.hist(thw, bins=60, orientation="horizontal",
                  color=C["ego"], alpha=0.6, edgecolor="white", lw=0.3)
    ax_right.axhline(0.8, color=C["thw"], ls="--", lw=1.0, alpha=0.8)
    ax_right.set_xlabel("Count")
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.spines["left"].set_visible(False)

    # No title
    fig.savefig(OUT / "fig2_ttc_thw_scatter.pdf")
    fig.savefig(OUT / "fig2_ttc_thw_scatter.png")
    plt.close(fig)
    print("  Saved fig2_ttc_thw_scatter.pdf / .png")


if __name__ == "__main__":
    main()
