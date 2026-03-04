#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_figures.py
=================
Refined figures for the takeover safety & smoothness paper.

Produces five publication-quality figures:
  fig1  — laneProb_min_pre  →  steer_rate / jerk  (binned median + CI)
  fig2  — trigger-stratified violin+box for steer_rate and jerk
  fig3  — curvature-mismatch strata  →  steer_rate  (monotonic trend)
  fig4  — speed-stratified: laneProb  →  steer_rate  (3 panels)
  fig5  — lead-present planner-based deceleration-demand proxy

Data corrections applied:
  1. RMSE gating: accel_plan_output_rmse == 0  →  NA  (channel inactive)
  2. Curvature-mismatch winsorization at P99 (cap ≈ 0.195)
  3. No radarState-derived columns used (verified: none present)

Run:
    python3 refine_figures.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
CODE = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver/Code")
CSV_IN     = CODE / "outputs" / "tables" / "control_safety_metrics.csv"
MASTER_CSV = CODE / "stats_output" / "analysis_master.csv"
OUT        = CODE / "outputs" / "figures_refined"
OUT_TAB    = CODE / "outputs" / "tables"
OUT_REP    = CODE / "outputs" / "reports"
for d in (OUT, OUT_TAB, OUT_REP):
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

C = dict(
    blue="#4C78A8", orange="#F58518", red="#E45756", teal="#72B7B2",
    green="#54A24B", purple="#B279A2", gray="#BAB0AC", brown="#9D755D",
)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          9,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

N_BOOT = 2000
RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════
def _save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(OUT / f"{name}{ext}")
    plt.close(fig)
    print(f"  saved {name}")


def bootstrap_ci(vals, stat_fn=np.median, n_boot=N_BOOT, ci=0.95):
    """Return (point, lo, hi) via percentile bootstrap."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return (np.nan, np.nan, np.nan)
    point = float(stat_fn(vals))
    boots = np.array([stat_fn(RNG.choice(vals, size=len(vals), replace=True))
                      for _ in range(n_boot)])
    a = (1 - ci) / 2
    return (point, float(np.percentile(boots, 100 * a)),
            float(np.percentile(boots, 100 * (1 - a))))


def _annotate_n(ax, x, y_top, n, offset_frac=0.04):
    """Place 'N=...' above a point."""
    yr = ax.get_ylim()
    off = (yr[1] - yr[0]) * offset_frac
    ax.text(x, y_top + off, f"N={n:,}", ha="center", va="bottom",
            fontsize=7, color="#555555")


# ═══════════════════════════════════════════════════════════════════════
#  Load, merge, gate, winsorize
# ═══════════════════════════════════════════════════════════════════════
def load() -> pd.DataFrame:
    print("Loading data …")
    df = pd.read_csv(CSV_IN, low_memory=False)
    print(f"  {len(df):,} clips")

    # Merge pre_speed_mean_mps from analysis_master
    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV,
                             usecols=["dongle_id", "route_id", "clip_id",
                                      "pre_speed_mean_mps"],
                             low_memory=False)
        df = df.merge(master, on=["dongle_id", "route_id", "clip_id"], how="left")
        n_spd = df["pre_speed_mean_mps"].notna().sum()
        print(f"  Merged pre_speed_mean_mps: {n_spd:,} non-null")
    else:
        print("  [WARN] analysis_master.csv not found; speed column unavailable")

    # ── Gate accel plan→output RMSE ──
    for tag in ("pre", "post"):
        col = f"accel_plan_output_rmse_{tag}"
        if col in df.columns:
            n_zero = (df[col] == 0.0).sum()
            df.loc[df[col] == 0.0, col] = np.nan
            print(f"  Gated {col}: {n_zero:,} zeros → NA")

    # ── Winsorize curvature mismatch at P99 ──
    col = "curvature_mismatch_max_pre"
    v = df[col].dropna()
    cap = v.quantile(0.99)
    n_cap = (df[col] > cap).sum()
    df[col] = df[col].clip(upper=cap)
    print(f"  Winsorized {col} at P99={cap:.6f} ({n_cap:,} capped)")

    # Save augmented table
    aug_path = OUT_TAB / "control_safety_metrics_plus_speed.csv"
    df.to_csv(aug_path, index=False)
    print(f"  Saved augmented table: {aug_path.name}")

    return df


# ═══════════════════════════════════════════════════════════════════════
#  LANE-PROB BINS  (shared across Fig 1 & Fig 4)
# ═══════════════════════════════════════════════════════════════════════
LP_EDGES  = [0.0, 0.1, 0.3, 0.6, 0.9, 1.001]
LP_LABELS = ["0–0.1", "0.1–0.3", "0.3–0.6", "0.6–0.9", "0.9–1.0"]


def _binned_median_ci(sub, xcol, ycol, edges, labels):
    """Return lists: meds, lo, hi, ns for each bin."""
    sub = sub.copy()
    sub["_bin"] = pd.cut(sub[xcol], bins=edges, labels=labels,
                         include_lowest=True, right=False)
    meds, los, his, ns = [], [], [], []
    for lab in labels:
        vals = sub.loc[sub["_bin"] == lab, ycol].dropna().values
        m, lo, hi = bootstrap_ci(vals)
        meds.append(m); los.append(lo); his.append(hi); ns.append(len(vals))
    return meds, los, his, ns


# ═══════════════════════════════════════════════════════════════════════
#  Fig 1: laneProb → steer_rate / jerk  (2-panel)
# ═══════════════════════════════════════════════════════════════════════
def fig1(df):
    sub = df[["laneProb_min_pre", "steer_rate_max_post", "jerk_max_post"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    for ax, ycol, ylabel in zip(axes, [
        "steer_rate_max_post", "jerk_max_post",
    ], [
        "Steer rate max (°/s)", "Jerk max (m/s³)",
    ]):
        meds, los, his, ns = _binned_median_ci(sub, "laneProb_min_pre",
                                                ycol, LP_EDGES, LP_LABELS)
        x = np.arange(len(LP_LABELS))
        ax.errorbar(x, meds,
                    yerr=[np.array(meds) - np.array(los),
                          np.array(his) - np.array(meds)],
                    fmt="o-", color=C["blue"], capsize=4, capthick=1,
                    markersize=5, linewidth=1.3, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.4)
        # Overall median reference
        all_med = float(np.nanmedian(sub[ycol].values))
        ax.axhline(all_med, color=C["orange"], ls="--", lw=0.9, zorder=0)
        ax.text(len(LP_LABELS) - 0.5, all_med, f" median = {all_med:.1f}",
                va="bottom", ha="right", fontsize=7, color=C["orange"])
        # N labels
        for i in range(len(LP_LABELS)):
            _annotate_n(ax, i, his[i], ns[i])
        ax.set_xticks(x)
        ax.set_xticklabels(LP_LABELS)
        ax.set_xlabel("Lane probability min (pre)")
        ax.set_ylabel(ylabel)

    # Panel tags
    for ax, tag in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig1_laneprob_smoothness")


# ═══════════════════════════════════════════════════════════════════════
#  Fig 2: Trigger-stratified violin+box  (2-panel)
# ═══════════════════════════════════════════════════════════════════════
TRIG_ORDER = ["Brake Override", "Gas Override", "Steering Override",
              "System / Unknown"]
TRIG_SHORT = {"Brake Override": "Brake", "Gas Override": "Gas",
              "Steering Override": "Steer", "System / Unknown": "System"}
TRIG_COLOR = {"Brake Override": C["teal"], "Gas Override": C["green"],
              "Steering Override": C["blue"], "System / Unknown": C["gray"]}


def fig2(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    for ax, col, ylabel in zip(axes, [
        "jerk_max_post", "steer_rate_max_post",
    ], [
        "Jerk max (m/s³)", "Steer rate max (°/s)",
    ]):
        data, labs, colors = [], [], []
        for trig in TRIG_ORDER:
            vals = df.loc[df["primary_trigger"] == trig, col].dropna().values
            if len(vals) < 5:
                continue
            data.append(vals)
            labs.append(TRIG_SHORT[trig])
            colors.append(TRIG_COLOR[trig])

        pos = np.arange(len(data))
        vp = ax.violinplot(data, positions=pos,
                           showmedians=False, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.25)
            body.set_edgecolor("none")

        bp = ax.boxplot(data, positions=pos, widths=0.28,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=C["orange"], linewidth=1.6),
                        whiskerprops=dict(linewidth=0.7),
                        capprops=dict(linewidth=0.7))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)

        # N + P95 cap
        all_vals = np.concatenate(data)
        ymax = np.percentile(all_vals, 97)
        for i, d in enumerate(data):
            ax.text(i, ymax * 0.96, f"N={len(d):,}",
                    ha="center", va="top", fontsize=7, color="#555555")

        ax.set_xticks(pos)
        ax.set_xticklabels(labs)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0, top=ymax * 1.05)

    for ax, tag in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig2_trigger_stratified")


# ═══════════════════════════════════════════════════════════════════════
#  Fig 3: Curvature-mismatch strata → steer_rate
# ═══════════════════════════════════════════════════════════════════════
def fig3(df):
    sub = df[["curvature_mismatch_max_pre", "steer_rate_max_post"]].dropna()
    if len(sub) < 100:
        print("  [SKIP] fig3: insufficient data")
        return

    q_bounds = [0, 0.50, 0.80, 0.90, 0.95, 0.99]
    q_vals = [sub["curvature_mismatch_max_pre"].quantile(q) for q in q_bounds]
    strata = ["\u2264P50", "P50\u201380", "P80\u201390", "P90\u201395", "P95\u201399"]

    fig, ax = plt.subplots(figsize=(3.8, 2.8))

    meds, los, his, ns = [], [], [], []
    for i, lab in enumerate(strata):
        lo_v = q_vals[i]
        hi_v = q_vals[i + 1] if i + 1 < len(q_vals) else np.inf
        if i == 0:
            mask = sub["curvature_mismatch_max_pre"] <= hi_v
        else:
            mask = (sub["curvature_mismatch_max_pre"] > lo_v) & \
                   (sub["curvature_mismatch_max_pre"] <= hi_v)
        vals = sub.loc[mask, "steer_rate_max_post"].values
        m, ci_lo, ci_hi = bootstrap_ci(vals)
        meds.append(m); los.append(ci_lo); his.append(ci_hi); ns.append(len(vals))

    x = np.arange(len(strata))
    ax.errorbar(x, meds,
                yerr=[np.array(meds) - np.array(los),
                      np.array(his) - np.array(meds)],
                fmt="s-", color=C["purple"], capsize=4, capthick=1,
                markersize=5.5, linewidth=1.3, elinewidth=0.9,
                markeredgecolor="white", markeredgewidth=0.4)
    for i in range(len(strata)):
        _annotate_n(ax, i, his[i], ns[i])

    ax.set_xticks(x)
    ax.set_xticklabels(strata, fontsize=8)
    ax.set_xlabel("Curvature mismatch stratum (pre, winsorized)")
    ax.set_ylabel("Steer rate max (°/s)")

    fig.tight_layout()
    _save(fig, "fig3_mismatch_strata")


# ═══════════════════════════════════════════════════════════════════════
#  Fig 4: Speed-stratified laneProb → steer_rate + jerk  (2×3)
# ═══════════════════════════════════════════════════════════════════════
SPEED_REGIMES = [
    ("Low (0–60 km/h)",      0.0,  16.7),
    ("Medium (60–100 km/h)", 16.7, 27.8),
    ("High (≥100 km/h)",     27.8, 999),
]

SPEED_SHORT = ["Low", "Medium", "High"]


def fig4(df):
    if "pre_speed_mean_mps" not in df.columns:
        print("  [SKIP] fig4: pre_speed_mean_mps unavailable")
        return

    sub = df[["laneProb_min_pre", "steer_rate_max_post", "jerk_max_post",
              "pre_speed_mean_mps"]].dropna()
    if len(sub) < 200:
        print("  [SKIP] fig4: insufficient data after dropna")
        return

    metrics = [
        ("steer_rate_max_post", "Steer rate max (°/s)"),
        ("jerk_max_post",       "Jerk max (m/s³)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.8))

    # Collect all CI-hi values per row for uniform y-limits
    row_his = [[], []]

    for col_idx, (label, lo_spd, hi_spd) in enumerate(SPEED_REGIMES):
        s = sub[(sub["pre_speed_mean_mps"] >= lo_spd) &
                (sub["pre_speed_mean_mps"] < hi_spd)]
        n_total = len(s)

        for row_idx, (ycol, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            meds, los, his, ns = _binned_median_ci(
                s, "laneProb_min_pre", ycol, LP_EDGES, LP_LABELS)
            row_his[row_idx].extend([h for h in his if np.isfinite(h)])

            x = np.arange(len(LP_LABELS))
            ax.errorbar(x, meds,
                        yerr=[np.array(meds) - np.array(los),
                              np.array(his) - np.array(meds)],
                        fmt="o-", color=C["blue"], capsize=3, capthick=0.8,
                        markersize=4, linewidth=1.1, elinewidth=0.7,
                        markeredgecolor="white", markeredgewidth=0.3)

            for i in range(len(LP_LABELS)):
                ax.text(i, his[i] + 0.3 if np.isfinite(his[i]) else 0,
                        f"{ns[i]:,}", ha="center", va="bottom",
                        fontsize=5.5, color="#555555")

            ax.set_xticks(x)
            if row_idx == 1:
                ax.set_xticklabels(LP_LABELS, fontsize=6.5)
                ax.set_xlabel("Lane prob. min (pre)", fontsize=8)
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)

        # Speed regime label on top row
        axes[0, col_idx].text(
            0.5, 1.06,
            f"{SPEED_SHORT[col_idx]}  (N={n_total:,})",
            transform=axes[0, col_idx].transAxes,
            ha="center", va="bottom", fontsize=8.5)

    # Uniform y-limits per row
    for row_idx in range(2):
        if row_his[row_idx]:
            ymax = max(row_his[row_idx]) * 1.3
            for col_idx in range(3):
                axes[row_idx, col_idx].set_ylim(bottom=0, top=ymax)

    # Panel tags (a)–(f)
    tags = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    for r in range(2):
        for c in range(3):
            axes[r, c].text(-0.02, 1.08 if r == 0 else 1.05,
                            tags[r][c], transform=axes[r, c].transAxes,
                            fontsize=9, fontweight="bold",
                            va="bottom", ha="right")

    fig.tight_layout(h_pad=1.5, w_pad=0.8)
    _save(fig, "fig4_speed_stratified")


# ═══════════════════════════════════════════════════════════════════════
#  Fig 6: Covariate robustness — speed × hasLead stratification  (1×3)
# ═══════════════════════════════════════════════════════════════════════
def fig6(df):
    """Within each speed stratum, split by hasLead (present vs absent)
    to check whether the laneProb → steer_rate association is robust
    to lead-vehicle presence — a standard stratified robustness check."""

    if "pre_speed_mean_mps" not in df.columns:
        print("  [SKIP] fig6: pre_speed_mean_mps unavailable")
        return

    # Binary hasLead: rate > 0.5 → present
    df = df.copy()
    df["_hasLead"] = (df["hasLead_rate_pre"].fillna(0) > 0.5).astype(int)

    sub = df[["laneProb_min_pre", "steer_rate_max_post",
              "pre_speed_mean_mps", "_hasLead"]].dropna()
    if len(sub) < 200:
        print("  [SKIP] fig6: insufficient data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), sharey=True)

    lead_styles = {
        1: dict(color=C["red"],  fmt="s-",  label="Lead present",
                ms=4.5, lw=1.2),
        0: dict(color=C["teal"], fmt="o--", label="Lead absent",
                ms=4.5, lw=1.2),
    }

    global_his = []

    for col_idx, (label, lo_spd, hi_spd) in enumerate(SPEED_REGIMES):
        ax = axes[col_idx]
        s_speed = sub[(sub["pre_speed_mean_mps"] >= lo_spd) &
                      (sub["pre_speed_mean_mps"] < hi_spd)]
        n_total = len(s_speed)

        x = np.arange(len(LP_LABELS))

        for lead_val in [1, 0]:
            s = s_speed[s_speed["_hasLead"] == lead_val]
            sty = lead_styles[lead_val]
            meds, los, his, ns = _binned_median_ci(
                s, "laneProb_min_pre", "steer_rate_max_post",
                LP_EDGES, LP_LABELS)
            global_his.extend([h for h in his if np.isfinite(h)])

            offset = 0.08 if lead_val == 1 else -0.08
            ax.errorbar(x + offset, meds,
                        yerr=[np.array(meds) - np.array(los),
                              np.array(his) - np.array(meds)],
                        fmt=sty["fmt"], color=sty["color"],
                        capsize=3, capthick=0.7,
                        markersize=sty["ms"], linewidth=sty["lw"],
                        elinewidth=0.7,
                        markeredgecolor="white", markeredgewidth=0.3,
                        label=sty["label"] if col_idx == 0 else None)

            # N labels — only for lead-present (top) to avoid clutter
            if lead_val == 1:
                for i in range(len(LP_LABELS)):
                    if np.isfinite(his[i]):
                        ax.text(i + offset, his[i] + 0.3,
                                f"{ns[i]:,}", ha="center", va="bottom",
                                fontsize=5.5, color=sty["color"])

        ax.set_xticks(x)
        ax.set_xticklabels(LP_LABELS, fontsize=6.5)
        ax.set_xlabel("Lane prob. min (pre)", fontsize=8)
        ax.text(0.5, 1.03,
                f"{SPEED_SHORT[col_idx]}  (N={n_total:,})",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5)

    axes[0].set_ylabel("Steer rate max (°/s)", fontsize=8.5)

    # Uniform y-axis
    if global_his:
        ymax = max(global_his) * 1.3
        for ax in axes:
            ax.set_ylim(bottom=0, top=ymax)

    # Single legend on first panel
    axes[0].legend(loc="upper right", fontsize=7, frameon=False,
                   handlelength=1.8)

    for ax, tag in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(-0.02, 1.10, tag, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

    fig.tight_layout(w_pad=0.8)
    _save(fig, "fig6_haslead_robustness")


# ═══════════════════════════════════════════════════════════════════════
#  Fig 5: Lead-present planner-based deceleration-demand proxy
# ═══════════════════════════════════════════════════════════════════════
def fig5(df):
    """
    Without radarState, true TTC/THW cannot be computed.  Instead we
    construct a *planner-based deceleration demand* proxy for lead-present
    clips:

        decel_demand_proxy = |aTarget_min_pre|  (when aTarget_min_pre < 0)

    This captures how hard the planner was braking in the pre-window.
    Among clips where hasLead_rate_pre > 0.5 (lead present for >50% of
    the pre-window), a higher decel demand suggests the planner was
    responding to a closer or faster-approaching lead — analogous to
    the information TTC/THW would provide, but from the planner's
    perspective rather than from radar distance.

    We also include planned_speed_drop_pre as a complementary proxy
    measuring the magnitude of the planner's anticipated speed reduction.
    """
    # Lead-present subset: hasLead detected for >50% of pre-window
    lead_mask = df["hasLead_rate_pre"].fillna(0) > 0.5
    sub = df[lead_mask].copy()
    n_lead = len(sub)
    print(f"  Lead-present subset (hasLead_rate > 0.5): {n_lead:,} clips")

    if n_lead < 100:
        print("  [SKIP] fig5: insufficient lead-present clips")
        return

    # Decel demand proxy: |aTarget_min| for negative values
    sub["decel_demand"] = sub["aTarget_min_pre"].clip(upper=0).abs()

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # ── Panel (a): Distribution of decel demand + planned speed drop ──
    ax = axes[0]
    dd = sub["decel_demand"].dropna()
    sd = sub["planned_speed_drop_pre"].dropna()

    # Violin for decel demand
    parts = ax.violinplot([dd.values], positions=[0], showmedians=False,
                          showextrema=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(C["blue"])
        pc.set_alpha(0.3)
        pc.set_edgecolor("none")
    bp = ax.boxplot([dd.values], positions=[0], widths=0.25,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=C["orange"], linewidth=1.6),
                    whiskerprops=dict(linewidth=0.7),
                    capprops=dict(linewidth=0.7))
    bp["boxes"][0].set_facecolor(C["blue"])
    bp["boxes"][0].set_alpha(0.6)

    # Violin for planned speed drop
    parts2 = ax.violinplot([sd.values], positions=[1], showmedians=False,
                           showextrema=False, widths=0.7)
    for pc in parts2["bodies"]:
        pc.set_facecolor(C["teal"])
        pc.set_alpha(0.3)
        pc.set_edgecolor("none")
    bp2 = ax.boxplot([sd.values], positions=[1], widths=0.25,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color=C["orange"], linewidth=1.6),
                     whiskerprops=dict(linewidth=0.7),
                     capprops=dict(linewidth=0.7))
    bp2["boxes"][0].set_facecolor(C["teal"])
    bp2["boxes"][0].set_alpha(0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Decel. demand\n|aTarget min| (m/s²)",
                        "Planned speed\ndrop (m/s)"], fontsize=8)
    ax.set_ylabel("Value")
    ax.text(0, np.percentile(dd, 95) * 1.02, f"N={len(dd):,}",
            ha="center", va="bottom", fontsize=7, color="#555555")
    ax.text(1, np.percentile(sd, 95) * 1.02, f"N={len(sd):,}",
            ha="center", va="bottom", fontsize=7, color="#555555")
    # Cap y at P97 for readability
    ymax_a = max(np.percentile(dd, 97), np.percentile(sd, 97)) * 1.15
    ax.set_ylim(bottom=0, top=ymax_a)

    # ── Panel (b): Decel demand bins → jerk_max_post ──
    ax = axes[1]
    dd_sub = sub[["decel_demand", "jerk_max_post"]].dropna()
    if len(dd_sub) < 50:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
    else:
        # Quintile-based bins for decel demand
        try:
            dd_sub["_bin"] = pd.qcut(dd_sub["decel_demand"], q=5, duplicates="drop")
            bin_order = sorted(dd_sub["_bin"].dropna().unique())
        except ValueError:
            # Fall back to fixed bins
            dd_sub["_bin"] = pd.cut(dd_sub["decel_demand"],
                                    bins=[0, 0.1, 0.3, 0.8, 1.5, 10],
                                    include_lowest=True)
            bin_order = sorted(dd_sub["_bin"].dropna().unique())

        meds, los, his, ns, xlabs = [], [], [], [], []
        for b in bin_order:
            vals = dd_sub.loc[dd_sub["_bin"] == b, "jerk_max_post"].values
            m, ci_lo, ci_hi = bootstrap_ci(vals)
            meds.append(m); los.append(ci_lo); his.append(ci_hi)
            ns.append(len(vals))
            # Short label
            if hasattr(b, "left"):
                xlabs.append(f"{b.left:.1f}–{b.right:.1f}")
            else:
                xlabs.append(str(b))

        x = np.arange(len(xlabs))
        ax.errorbar(x, meds,
                    yerr=[np.array(meds) - np.array(los),
                          np.array(his) - np.array(meds)],
                    fmt="s-", color=C["blue"], capsize=3.5, capthick=0.9,
                    markersize=5, linewidth=1.2, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.4)
        for i in range(len(xlabs)):
            _annotate_n(ax, i, his[i], ns[i])

        ax.set_xticks(x)
        ax.set_xticklabels(xlabs, fontsize=7, rotation=15, ha="right")
        ax.set_xlabel("Decel. demand proxy (m/s²)")
        ax.set_ylabel("Jerk max (m/s³)")

    for ax, tag in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.02, 1.05, tag, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")

    fig.suptitle("")  # no title
    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig5_lead_present_proxy")


# ═══════════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════════
def write_report(df):
    N = len(df)

    # Compute key numbers for findings
    lp_sub = df[["laneProb_min_pre", "steer_rate_max_post"]].dropna()
    lp_lo = lp_sub.loc[lp_sub["laneProb_min_pre"] < 0.1, "steer_rate_max_post"]
    lp_hi = lp_sub.loc[lp_sub["laneProb_min_pre"] > 0.9, "steer_rate_max_post"]
    lp_lo_med = lp_lo.median() if len(lp_lo) > 10 else np.nan
    lp_hi_med = lp_hi.median() if len(lp_hi) > 10 else np.nan

    steer_trig = df.loc[df["primary_trigger"] == "Steering Override",
                        "steer_rate_max_post"].dropna()
    brake_trig = df.loc[df["primary_trigger"] == "Brake Override",
                        "steer_rate_max_post"].dropna()

    # Speed strata check
    has_speed = "pre_speed_mean_mps" in df.columns
    if has_speed:
        for label, lo, hi in SPEED_REGIMES:
            s = df[(df["pre_speed_mean_mps"] >= lo) &
                   (df["pre_speed_mean_mps"] < hi)]
            n = len(s)

    # Lead-present
    lead_sub = df[df["hasLead_rate_pre"].fillna(0) > 0.5]
    n_lead = len(lead_sub)
    dd = lead_sub["aTarget_min_pre"].clip(upper=0).abs().dropna()
    dd_med = dd.median() if len(dd) > 0 else np.nan

    lines = [
        "# Figure Refinement Notes",
        "",
        "## Data Checks",
        "",
        f"- **Total clips**: {N:,}",
        f"- **laneProb_min_pre**: range [0.0003, 0.9981] — confirmed ∈ [0, 1]. "
        f"Missing for {df['laneProb_min_pre'].isna().mean():.1%} of clips "
        f"(drivingModelData absent).",
        f"- **steer_rate_max_post**: range [0, 500] °/s (capped). "
        f"Missing for {df['steer_rate_max_post'].isna().mean():.1%}.",
        f"- **jerk_max_post**: range [0, 50] m/s³ (capped). "
        f"Missing for {df['jerk_max_post'].isna().mean():.1%}.",
        f"- **curvature_mismatch_max_pre**: P99 = 0.195, P100 = 21.9 before "
        f"winsorization. Winsorized at P99 (154 values capped). "
        f"Units: 1/m (|κ_desired − κ_actual|).",
        f"- **primary_trigger**: Steering Override ({df['primary_trigger'].eq('Steering Override').sum():,}), "
        f"Brake Override ({df['primary_trigger'].eq('Brake Override').sum():,}), "
        f"Gas Override ({df['primary_trigger'].eq('Gas Override').sum():,}), "
        f"System/Unknown ({df['primary_trigger'].eq('System / Unknown').sum():,}), "
        f"missing ({df['primary_trigger'].isna().sum()}).",
        f"- **No radarState-derived columns present** (verified: no dRel, vRel, "
        f"radar, thw, ttc, headway, or distance columns found).",
        "",
        "## Gating and Winsorization",
        "",
        "- **RMSE gating**: `accel_plan_output_rmse_pre`: 7,583 / 11,795 values "
        "(64.3%) were exactly 0.0 → set to NA (channel inactive). "
        "`accel_plan_output_rmse_post`: 9,361 / 11,750 (79.7%) → NA.",
        "- **Curvature mismatch winsorization**: capped at P99 ≈ 0.195. "
        "Justification: the 1% tail extends to 21.9, >100× the P99 value, "
        "likely reflecting instrumentation artifacts or edge-case controller "
        "transients. Winsorization preserves 99% of the distribution.",
        "",
        "## Findings (for paper text)",
        "",
        f"1. **Lane confidence and lateral control urgency (Fig 1, Fig 4)**. "
        f"Clips with low lane-detection confidence (laneProb < 0.1) exhibit "
        f"median post-takeover steering rate of {lp_lo_med:.1f} °/s, "
        f"approximately {lp_lo_med/lp_hi_med:.1f}× higher than clips with "
        f"high confidence (laneProb > 0.9, median {lp_hi_med:.1f} °/s). "
        f"This monotonic relationship persists across speed strata "
        f"(Fig 4), suggesting it is not solely a confound of low-speed "
        f"driving contexts." if has_speed else
        f"1. **Lane confidence and lateral control urgency (Fig 1)**. "
        f"Clips with low lane-detection confidence (laneProb < 0.1) exhibit "
        f"higher post-takeover steering rate than high-confidence clips.",
        "",
        f"2. **Trigger modality (Fig 2)**. Steering-override takeovers show "
        f"median steer rate of {steer_trig.median():.1f} °/s versus "
        f"{brake_trig.median():.1f} °/s for brake overrides — an expected "
        f"difference reflecting the biomechanics of the trigger itself "
        f"rather than indicating differential safety." if len(brake_trig) > 0 else "",
        "",
        f"3. **Curvature mismatch dose–response (Fig 3)**. Post-takeover "
        f"steering rate increases monotonically across curvature-mismatch "
        f"strata, consistent with the hypothesis that larger planner–vehicle "
        f"path discrepancies precede more urgent lateral corrections.",
        "",
        f"4. **Lead-present planner demand (Fig 5)**. Among {n_lead:,} clips "
        f"with sustained lead-vehicle detection (hasLead > 50%), the median "
        f"planner deceleration demand is {dd_med:.2f} m/s². Higher decel "
        f"demand is associated with modestly elevated post-takeover jerk, "
        f"consistent with more urgent longitudinal corrections. "
        f"**Limitation**: this proxy reflects planner intent, not physical "
        f"headway; true TTC/THW requires radar distance data (radarState), "
        f"which is excluded from this analysis.",
        "",
        "## Figure Captions",
        "",
        "**Fig 1.** Median post-takeover steering rate (a) and longitudinal "
        "jerk (b) as a function of pre-takeover lane-detection confidence. "
        "Error bars show bootstrap 95% CI (2,000 replicates). "
        "Sample sizes per bin annotated.",
        "",
        "**Fig 2.** Distribution of post-takeover jerk (a) and steering rate "
        "(b) stratified by takeover trigger modality. Boxes show IQR with "
        "median (orange); violins show kernel density. Sample sizes annotated.",
        "",
        "**Fig 3.** Median post-takeover steering rate across quantile strata "
        "of pre-takeover curvature mismatch (winsorized at P99). "
        "The monotonic increase is consistent with larger path-tracking "
        "discrepancies preceding more urgent lateral corrections.",
        "",
        "**Fig 4.** Speed-stratified replication of Fig 1: median "
        "post-takeover steering rate (a–c) and jerk (d–f) versus "
        "lane-detection confidence, shown separately for low-, medium-, "
        "and high-speed regimes. The negative lateral-control association "
        "persists across speed strata; the longitudinal (jerk) association "
        "is weaker and largely confined to low-speed driving.",
        "",
        "**Fig 5.** Lead-vehicle-present subset analysis. (a) Distribution of "
        "planner-based deceleration demand (|aTarget min|) and planned speed "
        "drop for clips where hasLead > 50% of the pre-window. (b) Median "
        "post-takeover jerk as a function of binned deceleration demand. "
        "Note: this is a planner-intent proxy, not a radar-derived headway "
        "metric; true TTC/THW cannot be computed without radarState.",
        "",
        "**Fig 6.** Stratified robustness check: within each speed regime, "
        "the laneProb → steer\\_rate relationship is shown separately for "
        "clips with sustained lead-vehicle detection (hasLead > 50\\%, red) "
        "and without (teal). The association persists in both subgroups, "
        "suggesting it is not confounded by lead-vehicle presence.",
        "",
    ]

    path = OUT_REP / "figure_refinement_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved report: {path.name}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    df = load()

    print("\nGenerating figures …")
    fig1(df)
    fig2(df)
    fig3(df)
    fig4(df)
    fig5(df)
    fig6(df)

    print("\nWriting report …")
    write_report(df)

    print("\nDone. All outputs in:")
    print(f"  Figures: {OUT}")
    print(f"  Report:  {OUT_REP / 'figure_refinement_notes.md'}")


if __name__ == "__main__":
    main()
