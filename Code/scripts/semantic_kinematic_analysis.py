#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_kinematic_analysis.py
==============================
Semantic clustering of VLM-annotated risk factors into Macro-Archetypes,
then cross-modal causal analysis linking archetypes to kinematic severity.

Outputs:
  fig5_semantic_clustering.png   — archetype donut + risk factor bar
  fig6_kinematic_signatures.png  — violin+box of decel/steer_rate/jerk by archetype
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
OUT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver/Code/long_lat/longtail")
CSV = OUT / "longtail_analysis_summary.csv"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          11,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

# Academic muted palette
PAL = {
    "Infrastructure Degradation": "#5B8DB8",   # muted blue
    "Adverse Environment":        "#E8943A",   # muted orange
    "Traffic Dynamics":           "#6BAF6B",   # muted green
    "Other":                      "#AAAAAA",   # grey
}

# Priority-based archetype mapping
P1_TAGS = {"Faded Lane Lines", "Construction", "Sharp Curve", "Obstruction"}
P2_TAGS = {"Wet Road", "Glare/Weather", "Night Driving"}
P3_TAGS = {"Slow Vehicle", "Hard Braking", "Cut-in", "Tailgating",
           "Congestion", "Merging", "Pedestrian"}


def classify_archetype(risk_str: str) -> str:
    if pd.isna(risk_str) or not risk_str.strip():
        return "Other"
    tags = {t.strip() for t in risk_str.split(";")}
    if tags & P1_TAGS:
        return "Infrastructure Degradation"
    if tags & P2_TAGS:
        return "Adverse Environment"
    if tags & P3_TAGS:
        return "Traffic Dynamics"
    return "Other"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    df = pd.read_csv(CSV)
    df["archetype"] = df["risk_factors"].apply(classify_archetype)

    # Explode individual tags for bar chart
    tags_all = (df["risk_factors"].dropna()
                .str.split(r"\s*;\s*")
                .explode()
                .str.strip())
    tags_all = tags_all[tags_all != ""]

    return df, tags_all


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5: Semantic Clustering & Archetypes
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig5(df, tags_all):
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(10, 4),
                                          gridspec_kw={"wspace": 0.35})

    # ── Left: Donut chart ──
    arch_order = ["Traffic Dynamics", "Infrastructure Degradation",
                  "Adverse Environment", "Other"]
    counts = df["archetype"].value_counts()
    sizes  = [counts.get(a, 0) for a in arch_order]
    colors = [PAL[a] for a in arch_order]
    # Drop archetypes with 0 count
    non_zero = [(a, s, c) for a, s, c in zip(arch_order, sizes, colors) if s > 0]
    if non_zero:
        a_nz, s_nz, c_nz = zip(*non_zero)
    else:
        a_nz, s_nz, c_nz = arch_order, sizes, colors

    wedges, texts, autotexts = ax_pie.pie(
        s_nz, labels=None, colors=c_nz, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5))

    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight("bold")

    # Center text
    total = sum(s_nz)
    ax_pie.text(0, 0, f"N={total}", ha="center", va="center",
                fontsize=13, fontweight="bold", color="#333333")

    # Legend below
    ax_pie.legend(
        wedges, [f"{a} ({s})" for a, s in zip(a_nz, s_nz)],
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, frameon=False, fontsize=8.5)

    ax_pie.set_aspect("equal")

    # ── Right: Top 10 risk factors bar ──
    top10 = tags_all.value_counts().head(10)
    y_pos = np.arange(len(top10))

    bars = ax_bar.barh(y_pos, top10.values, color="#5B8DB8", alpha=0.85,
                       edgecolor="white", lw=0.5, height=0.65)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(top10.index, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Frequency")

    # Value labels
    for bar, val in zip(bars, top10.values):
        ax_bar.text(bar.get_width() + max(top10.values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9, color="#555555")

    fig.savefig(OUT / "fig5_semantic_clustering.png")
    fig.savefig(OUT / "fig5_semantic_clustering.pdf")
    plt.close(fig)
    print("  Saved fig5_semantic_clustering.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 6: Cross-Modal Kinematic Signatures
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig6(df):
    # Exclude "Other"
    plot_df = df[df["archetype"] != "Other"].copy()

    # Map column names
    metrics = [
        ("peak_decel_post",     "Peak Deceleration\n(m/s²)",   -10, 0),
        ("steer_rate_max_post", "Max Steering Rate\n(°/s)",     0, 100),
        ("jerk_max_post",       "Max Jerk\n(m/s³)",             0, 30),
    ]

    arch_order = ["Traffic\nDyn.", "Infra.\nDeg.", "Adverse\nEnv."]
    rename_map = {
        "Traffic Dynamics":           "Traffic\nDyn.",
        "Infrastructure Degradation": "Infra.\nDeg.",
        "Adverse Environment":        "Adverse\nEnv.",
    }
    plot_df["archetype_short"] = plot_df["archetype"].map(rename_map)

    palette = {rename_map[k]: v for k, v in PAL.items() if k in rename_map}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2),
                              gridspec_kw={"wspace": 0.32})

    for ax, (col, ylabel, ylo, yhi) in zip(axes, metrics):
        data = plot_df.dropna(subset=[col]).copy()
        data[col] = data[col].clip(lower=ylo, upper=yhi)

        sns.violinplot(x="archetype_short", y=col, data=data,
                       hue="archetype_short", order=arch_order,
                       palette=palette, legend=False,
                       inner=None, linewidth=0.8, cut=0,
                       saturation=0.7, ax=ax)

        sns.boxplot(x="archetype_short", y=col, data=data,
                    order=arch_order,
                    width=0.15, boxprops=dict(facecolor="white", zorder=3),
                    medianprops=dict(color="#333333", lw=1.5),
                    whiskerprops=dict(color="#555555"),
                    capprops=dict(color="#555555"),
                    fliersize=0, ax=ax)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_ylim(ylo, yhi)
        ax.tick_params(axis="x", labelsize=9)

    fig.savefig(OUT / "fig6_kinematic_signatures.png")
    fig.savefig(OUT / "fig6_kinematic_signatures.pdf")
    plt.close(fig)
    print("  Saved fig6_kinematic_signatures.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(df):
    print("\n" + "═" * 60)
    print("  Archetype Distribution")
    print("═" * 60)
    for a in ["Traffic Dynamics", "Infrastructure Degradation",
              "Adverse Environment", "Other"]:
        n = (df["archetype"] == a).sum()
        print(f"  {a:35s}  {n:4d}  ({100*n/len(df):.1f}%)")

    print("\n" + "═" * 60)
    print("  Median Kinematics by Archetype (excl. Other)")
    print("═" * 60)
    cols = ["peak_decel_post", "steer_rate_max_post", "jerk_max_post"]
    sub = df[df["archetype"] != "Other"]
    table = sub.groupby("archetype")[cols].median()
    table.columns = ["Peak Decel (m/s²)", "Steer Rate (°/s)", "Jerk (m/s³)"]
    print(table.to_string())
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("═" * 60)
    print("  Semantic Clustering & Cross-Modal Kinematic Analysis")
    print("═" * 60)

    df, tags_all = load_data()
    print(f"  Loaded {len(df)} clips, {len(tags_all)} individual tags")

    plot_fig5(df, tags_all)
    plot_fig6(df)
    print_summary(df)
    print("Done.")


if __name__ == "__main__":
    main()
