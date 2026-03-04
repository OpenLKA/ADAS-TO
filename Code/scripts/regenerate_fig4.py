#!/usr/bin/env python3
"""
Regenerate fig4_action_sequence with corrected [-0.2, +0.5]s trigger window.

Panel (a): Stacked histogram of action onsets relative to event_t (t=0),
           within the causal window [-0.2, +0.5]s.
Panel (b): Horizontal bar of primary trigger modality (from per_clip.csv
           computed with the tight [-0.2, +0.5]s window).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver/Code")
OUT  = ROOT / "long_lat"
CSV_METRICS = OUT / "per_clip_all_metrics.csv"
CSV_PERCLIP = ROOT / "stats_output" / "per_clip.csv"   # new trigger labels

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
    "brake":  "#E45756",
    "gas":    "#54A24B",
    "steer":  "#F58518",
    "mixed":  "#B279A2",
    "system": "#BAB0AC",
    "grey":   "#888888",
}

# Trigger detection window (must match dataset_statistics.py)
WIN_LO = -0.2
WIN_HI =  0.5


def main():
    df = pd.read_csv(CSV_METRICS)

    # ── Compute onset relative to event_t, filter to [-0.2, +0.5]s ──
    actions = ["brakePressed", "gasPressed", "steeringPressed"]
    labels  = ["Brake", "Gas", "Steering"]

    records = []
    for _, row in df.iterrows():
        evt = row["event_t"]
        if np.isnan(evt):
            continue
        for act, label in zip(actions, labels):
            onset = row.get(f"{act}_onset_s", np.nan)
            if np.isnan(onset):
                continue
            rel = onset - evt
            if WIN_LO <= rel <= WIN_HI:
                records.append({"action": label, "rel_onset": rel})

    act_df = pd.DataFrame(records)
    print(f"  Action records in [{WIN_LO}, {WIN_HI}]s window: {len(act_df)}")

    # ── Load new trigger labels from per_clip.csv ──
    pc = pd.read_csv(CSV_PERCLIP)
    print(f"  per_clip.csv: {len(pc)} clips")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4),
                             gridspec_kw={"wspace": 0.35})

    # ═══════════════════════════════════════════════════════
    #  (a) Onset distribution within [-0.2, +0.5]s
    # ═══════════════════════════════════════════════════════
    ax = axes[0]
    bin_edges = np.linspace(WIN_LO, WIN_HI, 36)  # 35 bins, each 0.02s wide

    stack_order  = ["Brake", "Gas", "Steering"]
    stack_colors = [C["brake"], C["gas"], C["steer"]]
    stack_data = []
    stack_labels = []
    for label, color in zip(stack_order, stack_colors):
        subset = act_df[act_df["action"] == label]["rel_onset"]
        n_total = len(subset)
        stack_data.append(subset.values)
        stack_labels.append(f"{label} ({n_total:,})")

    ax.hist(stack_data, bins=bin_edges, stacked=True,
            color=stack_colors, label=stack_labels,
            edgecolor="white", lw=0.3)

    # t=0 reference line
    ax.axvline(0, color="k", lw=1.2, ls="--", alpha=0.6, zorder=5)
    ax.text(0.01, 0.95, "$t = 0$", transform=ax.transAxes,
            fontsize=10, color="#333", va="top", ha="left")

    ax.set_xlim(WIN_LO, WIN_HI)
    ax.set_xlabel("Time relative to ADAS-OFF event (s)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", frameon=True, fancybox=True,
              framealpha=0.9, edgecolor="#cccccc")

    # ═══════════════════════════════════════════════════════
    #  (b) Primary trigger — horizontal bar (from per_clip.csv)
    # ═══════════════════════════════════════════════════════
    ax = axes[1]
    pt_counts = pc["primary_trigger"].value_counts()
    trigger_order  = ["Brake", "Steering", "Gas", "Mixed", "System / Unknown"]
    trigger_short  = ["Brake", "Steering", "Gas", "Mixed", "System"]
    trigger_colors = [C["brake"], C["steer"], C["gas"], C["mixed"], C["system"]]

    # Keep only non-zero
    pairs = [(s, pt_counts.get(o, 0), c)
             for o, s, c in zip(trigger_order, trigger_short, trigger_colors)
             if pt_counts.get(o, 0) > 0]
    bar_labels = [p[0] for p in pairs]
    counts     = [p[1] for p in pairs]
    bar_colors = [p[2] for p in pairs]
    total = sum(counts)

    bars = ax.barh(bar_labels, counts, color=bar_colors,
                   edgecolor="white", lw=0.5, height=0.6)
    for bar, cnt in zip(bars, counts):
        if cnt > 0 and total > 0:
            ax.text(bar.get_width() + total * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,} ({100*cnt/total:.1f}%)",
                    va="center", fontsize=11, color=C["grey"])
    ax.set_xlabel("Number of Clips")
    ax.invert_yaxis()

    fig.savefig(OUT / "fig4_action_sequence.pdf")
    fig.savefig(OUT / "fig4_action_sequence.png")
    plt.close(fig)
    print("  Saved fig4_action_sequence.pdf / .png")


if __name__ == "__main__":
    main()
