#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_figures.py
===================
Publication-quality figures for Section IV (Take Over Analysis).
Style matches paper_figures.py (IEEE / serif / minimal spines).

All figures include N= annotations per panel/group as required.

Run:
    python3 analysis_figures.py

Outputs (paper_figs/):
    fig_safety_metrics.pdf        — 2×2: TTC, THW, accel violin, jerk violin
    fig_scenario_distribution.pdf — horizontal bar of 9 scenarios
    fig_scenario_by_trigger.pdf   — 100% stacked bar: scenario × trigger
    fig_thw_ttc_scatter.pdf       — TTC vs THW scatter by scenario
    fig_safety_by_scenario.pdf    — box plots: safety metrics grouped by scenario
    fig_risk_maneuver_quadrant.pdf— risk score vs maneuver score scatter
    fig_oem_vs_op_comparison.pdf  — IPW effect sizes with 95% CI
    fig_smd_balance.pdf           — Love plot: SMD before/after propensity weighting
    fig_qlog_rlog_sensitivity.pdf — qlog vs rlog derivative metric comparison
    fig_mixed_model_forest.pdf    — forest plot of model coefficients
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
DATA = CODE / "stats_output" / "analysis_master.csv"
MODEL_RESULTS = CODE / "stats_output" / "model_results.json"
OUT  = CODE / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

with open(CODE / "configs" / "analysis_thresholds.yaml") as f:
    CFG = yaml.safe_load(f)

# ═══════════════════════════════════════════════════════════════════════
#  Color Palette (identical to paper_figures.py)
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

SCENARIO_COLORS = {
    "longitudinal_conflict": C["red"],
    "lateral_conflict":      C["orange"],
    "planned_lane_change":   C["blue"],
    "planned_acceleration":  C["green"],
    "intersection_odd":      C["brown"],
    "ride_discomfort":       C["purple"],
    "system_boundary":       C["gray"],
    "discretionary":         C["teal"],
    "uncertain_mixed":       "#D4D4D4",
}

SCENARIO_ORDER = CFG["scenario_categories"]

SCENARIO_SHORT = {
    "longitudinal_conflict": "Long. Conflict",
    "lateral_conflict":      "Lat. Conflict",
    "planned_lane_change":   "Planned LC",
    "planned_acceleration":  "Planned Accel.",
    "intersection_odd":      "Intersect./ODD",
    "ride_discomfort":       "Ride Discomf.",
    "system_boundary":       "System Bound.",
    "discretionary":         "Discretionary",
    "uncertain_mixed":       "Uncertain",
}

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

COL_W = 3.5   # IEEE single column width (inches)
DBL_W = 7.16  # IEEE double column width


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}")


def annot_n(ax, x, y, n, fontsize=5.5):
    """Add N=... annotation at position."""
    ax.annotate(f"N={n:,}", (x, y), fontsize=fontsize, ha="center",
                va="bottom", color="gray")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 1: Safety Metrics Overview (2×2)
# ═══════════════════════════════════════════════════════════════════════
def fig_safety_metrics(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(DBL_W, 4.5))

    # (a) TTC histogram
    ax = axes[0, 0]
    pre = df["pre_ttc_min_s"].dropna().clip(0, 15)
    post = df["post_ttc_min_s"].dropna().clip(0, 15)
    bins = np.linspace(0, 15, 40)
    ax.hist(pre, bins=bins, alpha=0.7, color=C["blue"],
            label=f"Pre (N={len(pre):,})", density=True)
    ax.hist(post, bins=bins, alpha=0.5, color=C["teal"],
            label=f"Post (N={len(post):,})", density=True)
    ax.axvline(CFG["ttc_critical_s"], color=C["red"], ls="--", lw=0.8,
               label=f'Critical ({CFG["ttc_critical_s"]}s)')
    ax.set_xlabel("Min TTC (s)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Time-to-Collision")
    ax.legend(frameon=False, fontsize=6)

    # (b) THW histogram
    ax = axes[0, 1]
    pre = df["pre_thw_min_s"].dropna().clip(0, 6)
    post = df["post_thw_min_s"].dropna().clip(0, 6)
    bins = np.linspace(0, 6, 40)
    ax.hist(pre, bins=bins, alpha=0.7, color=C["blue"],
            label=f"Pre (N={len(pre):,})", density=True)
    ax.hist(post, bins=bins, alpha=0.5, color=C["teal"],
            label=f"Post (N={len(post):,})", density=True)
    ax.axvline(CFG["thw_critical_s"], color=C["red"], ls="--", lw=0.8,
               label=f'Critical ({CFG["thw_critical_s"]}s)')
    ax.set_xlabel("Min THW (s)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Time Headway")
    ax.legend(frameon=False, fontsize=6)

    # (c) Pre vs Post acceleration (violin)
    ax = axes[1, 0]
    accel_data = []
    accel_labels = []
    for col_name, lab in [("pre_min_accel_mps2", "Pre min"),
                           ("post_min_accel_mps2", "Post min"),
                           ("pre_max_accel_mps2", "Pre max"),
                           ("post_max_accel_mps2", "Post max")]:
        vals = df[col_name].dropna().clip(-10, 10)
        accel_data.append(vals.values if len(vals) > 0 else np.array([0]))
        accel_labels.append(f"{lab}\n(N={len(vals):,})")
    parts = ax.violinplot(accel_data, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(accel_labels) + 1))
    ax.set_xticklabels(accel_labels, fontsize=6)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("(c) Acceleration Distribution")

    # (d) Pre vs Post jerk (violin)
    ax = axes[1, 1]
    jerk_data = []
    jerk_labels = []
    for col_name, lab in [("pre_max_abs_jerk_mps3", "Pre"),
                           ("post_max_abs_jerk_mps3", "Post")]:
        vals = df[col_name].dropna().clip(0, 30)
        jerk_data.append(vals.values if len(vals) > 0 else np.array([0]))
        jerk_labels.append(f"{lab}\n(N={len(vals):,})")
    parts = ax.violinplot(jerk_data, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(jerk_labels) + 1))
    ax.set_xticklabels(jerk_labels, fontsize=7)
    ax.set_ylabel("Max |Jerk| (m/s³)")
    ax.set_title("(d) Jerk Distribution")

    fig.tight_layout()
    save(fig, "fig_safety_metrics")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 2: Scenario Distribution
# ═══════════════════════════════════════════════════════════════════════
def fig_scenario_distribution(df: pd.DataFrame):
    counts = df["scenario"].value_counts()
    N = len(df)
    cats = [c for c in SCENARIO_ORDER if c in counts.index]
    vals = [counts.get(c, 0) for c in cats]
    colors = [SCENARIO_COLORS.get(c, C["gray"]) for c in cats]
    labels = [SCENARIO_SHORT.get(c, c) for c in cats]

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    y_pos = range(len(cats))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Number of Clips")
    ax.invert_yaxis()

    for bar, v in zip(bars, vals):
        pct = v / N * 100
        ax.text(bar.get_width() + max(vals) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({pct:.1f}%)", va="center", fontsize=6)

    ax.set_title(f"Scenario Distribution (N={N:,})", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_scenario_distribution")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 3: Scenario by Trigger (100% stacked bar)
# ═══════════════════════════════════════════════════════════════════════
def fig_scenario_by_trigger(df: pd.DataFrame):
    if "primary_trigger" not in df.columns:
        print("  [SKIP] fig_scenario_by_trigger — no primary_trigger column")
        return

    df = df.copy()
    def _trig_group(pt):
        pt = str(pt)
        if "Steer" in pt: return "Steering"
        if "Brake" in pt: return "Brake"
        if "Gas" in pt:   return "Gas"
        return "System/Other"

    df["trigger_group"] = df["primary_trigger"].apply(_trig_group)
    trig_order = ["Steering", "Brake", "Gas", "System/Other"]

    ct = pd.crosstab(df["trigger_group"], df["scenario"])
    ct = ct.reindex(index=trig_order, columns=SCENARIO_ORDER, fill_value=0)
    totals = ct.sum(axis=1)
    ct_pct = ct.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(DBL_W, 2.5))
    x = np.arange(len(trig_order))
    bottom = np.zeros(len(trig_order))

    for cat in SCENARIO_ORDER:
        if cat in ct_pct.columns:
            vals = ct_pct[cat].values
            ax.bar(x, vals, bottom=bottom,
                   color=SCENARIO_COLORS.get(cat, C["gray"]),
                   label=SCENARIO_SHORT.get(cat, cat),
                   edgecolor="white", linewidth=0.3)
            bottom += vals

    # N annotations at top of each bar
    for i, trig in enumerate(trig_order):
        ax.text(i, 102, f"N={totals[trig]:,}", ha="center", fontsize=6, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(trig_order)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 108)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=5.5,
              frameon=False, ncol=1)

    fig.tight_layout()
    save(fig, "fig_scenario_by_trigger")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 4: TTC vs THW Scatter
# ═══════════════════════════════════════════════════════════════════════
def fig_thw_ttc_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(COL_W, 3.2))

    mask = df["pre_ttc_min_s"].notna() & df["pre_thw_min_s"].notna()
    sub = df[mask].copy()
    sub["pre_ttc_min_s"] = sub["pre_ttc_min_s"].clip(0, 15)
    sub["pre_thw_min_s"] = sub["pre_thw_min_s"].clip(0, 6)

    for cat in SCENARIO_ORDER:
        m = sub["scenario"] == cat
        if m.sum() == 0:
            continue
        ax.scatter(sub.loc[m, "pre_thw_min_s"], sub.loc[m, "pre_ttc_min_s"],
                   s=3, alpha=0.35, color=SCENARIO_COLORS.get(cat, C["gray"]),
                   label=f"{SCENARIO_SHORT.get(cat, cat)} (N={m.sum():,})",
                   rasterized=True)

    ax.axhline(CFG["ttc_critical_s"], color=C["red"], ls="--", lw=0.7)
    ax.axvline(CFG["thw_critical_s"], color=C["red"], ls=":", lw=0.7)
    ax.set_xlabel("Min THW (s)")
    ax.set_ylabel("Min TTC (s)")
    ax.set_title(f"TTC vs THW (N={len(sub):,})", fontsize=9)
    ax.legend(fontsize=4.5, frameon=False, markerscale=2, ncol=2,
              loc="upper right")

    fig.tight_layout()
    save(fig, "fig_thw_ttc_scatter")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 5: Safety by Scenario (box plots)
# ═══════════════════════════════════════════════════════════════════════
def fig_safety_by_scenario(df: pd.DataFrame):
    metrics = [
        ("pre_ttc_min_s",           "Min TTC (s)",       0, 15),
        ("pre_thw_min_s",           "Min THW (s)",       0, 6),
        ("pre_drac_max_mps2",       "Max DRAC (m/s²)",   0, 10),
        ("post_max_abs_jerk_mps3",  "Max |Jerk| (m/s³)", 0, 25),
    ]

    cats_present = [c for c in SCENARIO_ORDER if c in df["scenario"].values]
    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 3.0))

    for ax, (col_name, ylabel, lo, hi) in zip(axes, metrics):
        data = []
        colors = []
        ns = []
        for cat in cats_present:
            vals = df.loc[df["scenario"] == cat, col_name].dropna().clip(lo, hi)
            data.append(vals.values if len(vals) > 0 else np.array([]))
            colors.append(SCENARIO_COLORS.get(cat, C["gray"]))
            ns.append(len(vals))

        if any(len(d) > 0 for d in data):
            bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color=C["red"], lw=0.8))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xticklabels([SCENARIO_SHORT.get(c, c)[:6] for c in cats_present],
                           rotation=60, ha="right", fontsize=5)
        ax.set_ylabel(ylabel)
        # N annotations below x-axis
        for i, n in enumerate(ns):
            ax.text(i + 1, lo - (hi - lo) * 0.08, f"{n:,}",
                    ha="center", fontsize=4.5, color="gray")

    fig.tight_layout()
    save(fig, "fig_safety_by_scenario")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 6: Risk vs Maneuver Quadrant Scatter
# ═══════════════════════════════════════════════════════════════════════
def fig_risk_maneuver_quadrant(df: pd.DataFrame):
    if "risk_score" not in df.columns or "maneuver_score" not in df.columns:
        print("  [SKIP] fig_risk_maneuver_quadrant — missing scores")
        return

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))

    for cat in SCENARIO_ORDER:
        m = df["scenario"] == cat
        if m.sum() == 0:
            continue
        ax.scatter(df.loc[m, "maneuver_score"], df.loc[m, "risk_score"],
                   s=2, alpha=0.3, color=SCENARIO_COLORS.get(cat, C["gray"]),
                   label=SCENARIO_SHORT.get(cat, cat), rasterized=True)

    # Quadrant lines at 0.25
    ax.axhline(0.25, color="gray", ls=":", lw=0.5)
    ax.axvline(0.25, color="gray", ls=":", lw=0.5)

    # Quadrant counts
    hi_r = df["risk_score"] >= 0.25
    hi_m = df["maneuver_score"] >= 0.25
    for mask, pos, ha, va in [
        (hi_r & hi_m,   (0.85, 0.85), "right", "top"),
        (hi_r & ~hi_m,  (0.05, 0.85), "left",  "top"),
        (~hi_r & hi_m,  (0.85, 0.05), "right", "bottom"),
        (~hi_r & ~hi_m, (0.05, 0.05), "left",  "bottom"),
    ]:
        n = mask.sum()
        ax.text(pos[0], pos[1], f"N={n:,}", transform=ax.transAxes,
                fontsize=6, ha=ha, va=va, color="gray",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                          alpha=0.7, lw=0.3))

    ax.set_xlabel("Maneuver Score")
    ax.set_ylabel("Risk Score")
    ax.set_title(f"Risk vs Maneuver (N={len(df):,})", fontsize=9)
    ax.legend(fontsize=4.5, frameon=False, markerscale=3, ncol=2,
              loc="center right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    save(fig, "fig_risk_maneuver_quadrant")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 7: OEM vs OP Effect Sizes
# ═══════════════════════════════════════════════════════════════════════
def fig_oem_vs_op_comparison(results: dict):
    prop = results.get("propensity_oem_vs_op", {})
    effects = prop.get("effects", {})
    if not effects:
        print("  [SKIP] fig_oem_vs_op_comparison — no propensity effects")
        return

    names = list(effects.keys())
    labels = [effects[n]["label"] for n in names]
    diffs = [effects[n]["difference"] for n in names]
    ci_lo = [effects[n]["ci_lower"] for n in names]
    ci_hi = [effects[n]["ci_upper"] for n in names]
    ns_op = [effects[n]["n_op"] for n in names]
    ns_oem = [effects[n]["n_oem"] for n in names]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    y = np.arange(len(names))
    xerr = [np.array(diffs) - np.array(ci_lo),
            np.array(ci_hi) - np.array(diffs)]
    ax.errorbar(diffs, y, xerr=xerr, fmt="o", color=C["blue"],
                ecolor=C["gray"], capsize=3, markersize=4)
    ax.axvline(0, color="black", ls="-", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\n(N={no}+{ne})" for lab, no, ne
                        in zip(labels, ns_op, ns_oem)], fontsize=6)
    ax.set_xlabel("OP - OEM (IPW-weighted difference)")
    ax.set_title("OEM vs openpilot Comparison", fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    save(fig, "fig_oem_vs_op_comparison")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 8: SMD Balance (Love Plot)
# ═══════════════════════════════════════════════════════════════════════
def fig_smd_balance(results: dict):
    prop = results.get("propensity_oem_vs_op", {})
    smd_before = prop.get("smd_before", {})
    smd_after = prop.get("smd_after", {})
    if not smd_before:
        print("  [SKIP] fig_smd_balance — no SMD data")
        return

    covs = list(smd_before.keys())
    before = [abs(smd_before[c]) for c in covs]
    after = [abs(smd_after.get(c, 0)) for c in covs]
    threshold = prop.get("smd_threshold", 0.1)

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    y = np.arange(len(covs))
    ax.scatter(before, y, marker="x", color=C["red"], s=30, zorder=3,
               label="Before weighting")
    ax.scatter(after, y, marker="o", color=C["blue"], s=30, zorder=3,
               label="After weighting")
    # Connect pairs
    for i in range(len(covs)):
        ax.plot([before[i], after[i]], [y[i], y[i]],
                color=C["gray"], lw=0.5, zorder=1)

    ax.axvline(threshold, color=C["red"], ls="--", lw=0.7,
               label=f"Threshold ({threshold})")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("pre_", "").replace("_", " ") for c in covs],
                       fontsize=7)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title("Covariate Balance", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    ax.invert_yaxis()

    fig.tight_layout()
    save(fig, "fig_smd_balance")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 9: qlog vs rlog Sensitivity
# ═══════════════════════════════════════════════════════════════════════
def fig_qlog_rlog_sensitivity(df: pd.DataFrame):
    if "log_kind" not in df.columns:
        # Try log_kind from merged data
        if "log_kind_ds" in df.columns:
            df["log_kind"] = df["log_kind_ds"]
        else:
            print("  [SKIP] fig_qlog_rlog_sensitivity — no log_kind column")
            return

    metrics = [
        ("pre_max_abs_jerk_mps3",  "Pre max |jerk| (m/s³)", 0, 30),
        ("post_max_abs_jerk_mps3", "Post max |jerk| (m/s³)", 0, 30),
        ("pre_steer_rate_max_deg_per_s", "Pre steer rate (deg/s)", 0, 100),
        ("post_steer_rate_max_deg_per_s", "Post steer rate (deg/s)", 0, 100),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 2.5))

    for ax, (col_name, ylabel, lo, hi) in zip(axes, metrics):
        if col_name not in df.columns:
            ax.set_visible(False)
            continue

        data_q = df[df["log_kind"] == "qlog"][col_name].dropna().clip(lo, hi)
        data_r = df[df["log_kind"] == "rlog"][col_name].dropna().clip(lo, hi)

        data = []
        labels = []
        for d, lk, hz in [(data_q, "qlog", "10Hz"), (data_r, "rlog", "100Hz")]:
            data.append(d.values if len(d) > 0 else np.array([0]))
            labels.append(f"{lk}\n({hz})\nN={len(d):,}")

        if any(len(d) > 0 for d in data):
            bp = ax.boxplot(data, widths=0.5, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color=C["red"], lw=0.8))
            bp["boxes"][0].set_facecolor(C["blue"])
            bp["boxes"][0].set_alpha(0.6)
            if len(bp["boxes"]) > 1:
                bp["boxes"][1].set_facecolor(C["teal"])
                bp["boxes"][1].set_alpha(0.6)

        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel(ylabel, fontsize=7)

    fig.suptitle("Sampling Rate Sensitivity: qlog (10 Hz) vs rlog (100 Hz)",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    save(fig, "fig_qlog_rlog_sensitivity")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 10: Mixed Model Forest Plot
# ═══════════════════════════════════════════════════════════════════════
def fig_mixed_model_forest(results: dict):
    # Select LMM models for forest plot
    lmm_models = {k: v for k, v in results.items()
                  if k.startswith("lmm_") and "coefficients" in v}
    if not lmm_models:
        print("  [SKIP] fig_mixed_model_forest — no LMM results")
        return

    n_models = len(lmm_models)
    fig, axes = plt.subplots(1, n_models, figsize=(DBL_W, 3.5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, mdata) in zip(axes, lmm_models.items()):
        coefs = mdata["coefficients"]
        # Filter out intercept for clarity
        names = [k for k in coefs if k != "Intercept"]
        if not names:
            continue

        vals = [coefs[n]["estimate"] for n in names]
        ci_lo = [coefs[n].get("ci_lower", vals[i]) for i, n in enumerate(names)]
        ci_hi = [coefs[n].get("ci_upper", vals[i]) for i, n in enumerate(names)]

        y_pos = np.arange(len(names))
        xerr = [np.array(vals) - np.array(ci_lo),
                np.array(ci_hi) - np.array(vals)]
        ax.errorbar(vals, y_pos, xerr=xerr, fmt="o", color=C["blue"],
                    ecolor=C["gray"], capsize=2, markersize=3)
        ax.axvline(0, color="black", ls="-", lw=0.5)

        # Clean up coefficient names
        clean_names = []
        for n in names:
            cn = (n.replace("C(scenario)[T.", "").replace("C(trigger)[T.", "")
                  .rstrip("]").replace("_", " "))
            clean_names.append(cn)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(clean_names, fontsize=5.5)
        ax.set_xlabel("Coefficient", fontsize=7)
        icc = mdata.get("icc", 0)
        ax.set_title(f"{mdata.get('label', model_name)}\n(ICC={icc:.3f}, "
                     f"N={mdata.get('n_obs', '?'):,})", fontsize=7)
        ax.invert_yaxis()

    fig.tight_layout()
    save(fig, "fig_mixed_model_forest")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("Loading analysis_master.csv …")
    df = pd.read_csv(DATA, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Load model results if available
    model_results = {}
    if MODEL_RESULTS.exists():
        with open(MODEL_RESULTS) as f:
            model_results = json.load(f)
        print(f"  Loaded model results ({len(model_results)} models)")

    print("Generating figures …")
    fig_safety_metrics(df)
    fig_scenario_distribution(df)
    fig_scenario_by_trigger(df)
    fig_thw_ttc_scatter(df)
    fig_safety_by_scenario(df)
    fig_risk_maneuver_quadrant(df)
    fig_oem_vs_op_comparison(model_results)
    fig_smd_balance(model_results)
    fig_qlog_rlog_sensitivity(df)
    fig_mixed_model_forest(model_results)

    print(f"\nAll figures saved to {OUT}")


if __name__ == "__main__":
    main()
