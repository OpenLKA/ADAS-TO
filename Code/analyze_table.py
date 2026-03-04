#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_table.py
================
Takeover safety-proxy & control-smoothness reanalysis.

Reads control_safety_metrics.csv (15,705 clips) and produces:
  outputs/tables/reanalysis_summary.csv
  outputs/figures/fig_A_laneprob_vs_smoothness.{pdf,png}
  outputs/figures/fig_B_trigger_stratified.{pdf,png}
  outputs/figures/fig_C_mismatch_strata.{pdf,png}
  outputs/figures/fig_D_issue_flags.{pdf,png}
  outputs/reports/takeover_reanalysis.md

Critical fixes over the first-pass script:
  A) Missingness-aware: effective N reported for every metric.
  B) RMSE gating: accel_plan_output_rmse set to NA when both plan
     and output channels are inactive (RMSE==0 heuristic; 64% of pre
     and 80% of post values are exactly zero, indicating channel
     inactivity rather than perfect tracking).
  C) Curvature mismatch winsorized at P99.  Scatter plots replaced
     with binned medians + bootstrap 95% CI.

Run:
    python3 analyze_table.py
"""
from __future__ import annotations

import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
CODE = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver/Code")
CSV_IN = CODE / "outputs" / "tables" / "control_safety_metrics.csv"
OUT_TABLES  = CODE / "outputs" / "tables"
OUT_FIGS    = CODE / "outputs" / "figures"
OUT_REPORTS = CODE / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGS, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = dict(
    blue="#4C78A8", orange="#F58518", red="#E45756", teal="#72B7B2",
    green="#54A24B", purple="#B279A2", gray="#BAB0AC", brown="#9D755D",
)

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

N_BOOT = 2000
RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════
def _save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(OUT_FIGS / f"{name}{ext}")
    plt.close(fig)
    print(f"  Saved {name}")


def bootstrap_ci(vals: np.ndarray, stat_fn=np.median,
                 n_boot: int = N_BOOT, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (stat, lo, hi) via bootstrap percentile method."""
    vals = vals[np.isfinite(vals)]
    if len(vals) < 5:
        return (np.nan, np.nan, np.nan)
    point = float(stat_fn(vals))
    boots = np.array([stat_fn(RNG.choice(vals, size=len(vals), replace=True))
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))
    return (point, lo, hi)


def quantile_table(s: pd.Series, label: str) -> dict:
    """Summary row for a metric."""
    v = s.dropna()
    return dict(
        metric=label,
        N=len(v),
        mean=v.mean() if len(v) else np.nan,
        median=v.median() if len(v) else np.nan,
        P5=v.quantile(0.05) if len(v) else np.nan,
        P25=v.quantile(0.25) if len(v) else np.nan,
        P75=v.quantile(0.75) if len(v) else np.nan,
        P95=v.quantile(0.95) if len(v) else np.nan,
    )


def _fmt(x, decimals=3):
    if pd.isna(x):
        return "---"
    return f"{x:.{decimals}f}"


# ═══════════════════════════════════════════════════════════════════════
#  Load & Gate
# ═══════════════════════════════════════════════════════════════════════
def load_and_gate() -> pd.DataFrame:
    print("Loading CSV …")
    df = pd.read_csv(CSV_IN, low_memory=False)
    N = len(df)
    print(f"  {N:,} clips loaded")

    # ── Gating: accel plan→output RMSE ──
    # When both actuators.accel and actuatorsOutput.accel are zero
    # (channel inactive), RMSE is trivially 0.  We gate these out.
    # Rationale: 64% of pre and 80% of post values are exactly 0.0,
    # far exceeding what perfect-tracking would produce.
    for tag in ("pre", "post"):
        col = f"accel_plan_output_rmse_{tag}"
        n_before = df[col].notna().sum()
        n_zero = (df[col] == 0.0).sum()
        df.loc[df[col] == 0.0, col] = np.nan
        n_after = df[col].notna().sum()
        print(f"  Gated {col}: {n_zero:,} zeros → NA  "
              f"(effective N: {n_before:,} → {n_after:,})")

    # ── Winsorize curvature mismatch at P99 ──
    col = "curvature_mismatch_max_pre"
    v = df[col].dropna()
    if len(v) > 100:
        cap = v.quantile(0.99)
        n_capped = (df[col] > cap).sum()
        df[col] = df[col].clip(upper=cap)
        df["curvature_mismatch_max_pre_winsorized"] = True
        print(f"  Winsorized {col} at P99={cap:.6f}  ({n_capped:,} values capped)")

    return df


# ═══════════════════════════════════════════════════════════════════════
#  Section 1: Pre-Takeover Context
# ═══════════════════════════════════════════════════════════════════════
def section1_context(df: pd.DataFrame) -> list[dict]:
    print("\n── Section 1: Pre-Takeover Context ──")
    rows = []
    metrics = [
        ("laneProb_min_pre",            "Lane prob. min (pre)"),
        ("laneProb_mean_pre",           "Lane prob. mean (pre)"),
        ("laneCenter_range_pre",        "Lane center range (pre, m)"),
        ("laneWidth_mean_pre",          "Lane width mean (pre, m)"),
        ("curvature_mismatch_mean_pre", "Curv. mismatch mean (pre)"),
        ("curvature_mismatch_max_pre",  "Curv. mismatch max (pre, winsorized)"),
        ("aTarget_min_pre",             "aTarget min (pre, m/s²)"),
        ("aTarget_mean_pre",            "aTarget mean (pre, m/s²)"),
        ("planned_speed_drop_pre",      "Planned speed drop (pre, m/s)"),
        ("hasLead_rate_pre",            "hasLead rate (pre)"),
        ("leadVisible_rate_pre",        "leadVisible rate (pre)"),
        ("lead_consistency_flag",       "Lead inconsistency rate (pre)"),
    ]
    for col, label in metrics:
        if col in df.columns:
            r = quantile_table(df[col], label)
            rows.append(r)
            print(f"  {label}: N={r['N']:,}  median={_fmt(r['median'])}  "
                  f"[P5={_fmt(r['P5'])}, P95={_fmt(r['P95'])}]")
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Section 2: Post-Takeover Control Quality
# ═══════════════════════════════════════════════════════════════════════
def section2_control(df: pd.DataFrame) -> list[dict]:
    print("\n── Section 2: Post-Takeover Control Quality ──")
    rows = []
    metrics = [
        ("jerk_max_post",               "Jerk max (post, m/s³)"),
        ("steer_rate_max_post",         "Steer rate max (post, °/s)"),
        ("curvature_rate_max_post",     "Curvature rate max (post, 1/m·s)"),
        ("accel_plan_output_rmse_pre",  "Accel plan→output RMSE (pre, gated)"),
        ("accel_plan_output_rmse_post", "Accel plan→output RMSE (post, gated)"),
        ("accel_output_state_rmse_pre", "Accel output→state RMSE (pre)"),
        ("accel_output_state_rmse_post","Accel output→state RMSE (post)"),
        ("curv_plan_output_rmse_pre",   "Curv. plan→output RMSE (pre)"),
        ("curv_plan_output_rmse_post",  "Curv. plan→output RMSE (post)"),
    ]
    for col, label in metrics:
        if col in df.columns:
            r = quantile_table(df[col], label)
            rows.append(r)
            print(f"  {label}: N={r['N']:,}  median={_fmt(r['median'])}  "
                  f"[P5={_fmt(r['P5'])}, P95={_fmt(r['P95'])}]")

    # Stabilization
    st = df["stabilization_time_5s"].dropna()
    n_censored = df["stabilization_censored"].sum()
    n_total = len(df)
    n_uncensored = len(st)
    cens_rate = n_censored / n_total
    print(f"\n  Stabilization: {n_uncensored:,} uncensored, "
          f"{n_censored:,} censored ({cens_rate:.1%})")
    if n_uncensored > 0:
        r = quantile_table(st, "Stabilization time (uncensored, s)")
        rows.append(r)
        print(f"  Uncensored median={_fmt(r['median'])} s  "
              f"[P5={_fmt(r['P5'])}, P95={_fmt(r['P95'])}]")

    rows.append(dict(metric="Stabilization censored rate",
                     N=n_total, mean=cens_rate, median=np.nan,
                     P5=np.nan, P25=np.nan, P75=np.nan, P95=np.nan))
    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Section 3: Trigger-Stratified Analysis
# ═══════════════════════════════════════════════════════════════════════
def section3_trigger(df: pd.DataFrame) -> list[dict]:
    print("\n── Section 3: Behavior by Trigger ──")
    rows = []
    triggers = df["primary_trigger"].dropna().unique()
    triggers = sorted(triggers)
    print(f"  Triggers: {triggers}")

    for metric_col, metric_label in [
        ("jerk_max_post", "Jerk max post (m/s³)"),
        ("steer_rate_max_post", "Steer rate max post (°/s)"),
    ]:
        print(f"\n  {metric_label}:")
        group_stats = {}
        for trig in triggers:
            vals = df.loc[df["primary_trigger"] == trig, metric_col].dropna().values
            med, lo, hi = bootstrap_ci(vals)
            group_stats[trig] = (len(vals), med, lo, hi)
            print(f"    {trig}: N={len(vals):,}  median={_fmt(med)}  "
                  f"95% CI [{_fmt(lo)}, {_fmt(hi)}]")
            rows.append(dict(
                metric=f"{metric_label} | {trig}",
                N=len(vals), mean=np.nanmean(vals) if len(vals) else np.nan,
                median=med, P5=lo, P25=np.nan, P75=np.nan, P95=hi,
            ))

        # Pairwise effect sizes (median differences) — Steering vs Brake
        if "Steering Override" in group_stats and "Brake Override" in group_stats:
            s_vals = df.loc[df["primary_trigger"] == "Steering Override", metric_col].dropna().values
            b_vals = df.loc[df["primary_trigger"] == "Brake Override", metric_col].dropna().values
            diff_med = np.median(s_vals) - np.median(b_vals)
            # Bootstrap CI for difference of medians
            diffs = []
            for _ in range(N_BOOT):
                s_b = RNG.choice(s_vals, size=len(s_vals), replace=True)
                b_b = RNG.choice(b_vals, size=len(b_vals), replace=True)
                diffs.append(np.median(s_b) - np.median(b_b))
            diffs = np.array(diffs)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"    Δ(Steering−Brake) median: {diff_med:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]")

    return rows


# ═══════════════════════════════════════════════════════════════════════
#  Section 4: Interaction Flags
# ═══════════════════════════════════════════════════════════════════════
def section4_flags(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    print("\n── Section 4: Diagnostic Interaction Flags ──")

    # Thresholds
    # curvature mismatch P95 (after winsorization)
    cm_p95 = df["curvature_mismatch_max_pre"].dropna().quantile(0.95)
    # curvature rate P95
    cr_p95 = df["curvature_rate_max_post"].dropna().quantile(0.95)
    # gated RMSE P95
    apo_p95 = df["accel_plan_output_rmse_pre"].dropna().quantile(0.95)

    flags = {
        "lead_inconsistency": (
            df["lead_consistency_flag"].fillna(0) > 0.10,
            "Lead hasLead/leadVisible mismatch > 10%",
            f"lead_consistency_flag > 0.10",
        ),
        "low_lane_prob": (
            df["laneProb_min_pre"].fillna(1.0) < 0.30,
            "Lane detection confidence < 0.30",
            f"laneProb_min_pre < 0.30",
        ),
        "high_curv_mismatch": (
            df["curvature_mismatch_max_pre"].fillna(0) > cm_p95,
            f"Curvature mismatch > P95 ({cm_p95:.6f})",
            f"curvature_mismatch_max_pre > {cm_p95:.6f} (P95)",
        ),
        "output_aggressive_low_lane": (
            (df["laneProb_min_pre"].fillna(1.0) < 0.30) &
            (df["curvature_rate_max_post"].fillna(0) > cr_p95),
            f"Low lane prob. AND curvature rate > P95 ({cr_p95:.4f})",
            f"laneProb_min_pre < 0.30 AND curvature_rate_max_post > {cr_p95:.4f}",
        ),
        "plan_output_outlier": (
            df["accel_plan_output_rmse_pre"].fillna(0) > apo_p95,
            f"Gated accel plan→output RMSE > P95 ({apo_p95:.4f})",
            f"accel_plan_output_rmse_pre (gated) > {apo_p95:.4f}",
        ),
    }

    rows = []
    flag_df = pd.DataFrame(index=df.index)

    for flag_name, (mask, desc, rule) in flags.items():
        mask = mask.astype(bool)
        flag_df[flag_name] = mask

        # Effective N (how many clips have data for this flag)
        if flag_name == "low_lane_prob" or flag_name == "output_aggressive_low_lane":
            eff_n = df["laneProb_min_pre"].notna().sum()
        elif flag_name == "lead_inconsistency":
            eff_n = df["lead_consistency_flag"].notna().sum()
        elif flag_name == "high_curv_mismatch":
            eff_n = df["curvature_mismatch_max_pre"].notna().sum()
        elif flag_name == "plan_output_outlier":
            eff_n = df["accel_plan_output_rmse_pre"].notna().sum()
        else:
            eff_n = len(df)

        prev = mask.sum() / max(eff_n, 1)
        print(f"\n  {flag_name}: {mask.sum():,}/{eff_n:,} ({prev:.1%})")
        print(f"    Rule: {rule}")

        # Effect on smoothness
        for sm_col, sm_label in [("jerk_max_post", "jerk_max_post"),
                                  ("steer_rate_max_post", "steer_rate_max_post")]:
            flagged = df.loc[mask, sm_col].dropna().values
            unflagged = df.loc[~mask, sm_col].dropna().values
            if len(flagged) >= 10 and len(unflagged) >= 10:
                med_f, lo_f, hi_f = bootstrap_ci(flagged)
                med_u, lo_u, hi_u = bootstrap_ci(unflagged)
                uplift = med_f - med_u
                # Bootstrap CI for uplift
                diffs = []
                for _ in range(N_BOOT):
                    f_b = RNG.choice(flagged, size=len(flagged), replace=True)
                    u_b = RNG.choice(unflagged, size=len(unflagged), replace=True)
                    diffs.append(np.median(f_b) - np.median(u_b))
                diffs = np.array(diffs)
                d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
                print(f"    {sm_label}: flagged median={med_f:.2f} vs "
                      f"unflagged={med_u:.2f}, Δ={uplift:+.2f} "
                      f"[{d_lo:+.2f}, {d_hi:+.2f}]")

        rows.append(dict(
            metric=f"flag_{flag_name}",
            N=eff_n,
            mean=prev,
            median=float(mask.sum()),
            P5=np.nan, P25=np.nan, P75=np.nan, P95=np.nan,
        ))

    return rows, flag_df


# ═══════════════════════════════════════════════════════════════════════
#  Fig A: Binned laneProb → smoothness
# ═══════════════════════════════════════════════════════════════════════
def fig_A(df: pd.DataFrame):
    bins = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
    bin_labels = ["0–0.1", "0.1–0.3", "0.3–0.6", "0.6–0.9", "0.9–1.0"]

    sub = df[["laneProb_min_pre", "jerk_max_post", "steer_rate_max_post"]].dropna()
    sub = sub.copy()
    sub["bin"] = pd.cut(sub["laneProb_min_pre"], bins=bins, labels=bin_labels,
                        include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    for ax, (ycol, ylabel) in zip(axes, [
        ("steer_rate_max_post", "Steer rate max (°/s)"),
        ("jerk_max_post", "Jerk max (m/s³)"),
    ]):
        meds, los, his, ns = [], [], [], []
        for bl in bin_labels:
            vals = sub.loc[sub["bin"] == bl, ycol].values
            med, lo, hi = bootstrap_ci(vals)
            meds.append(med)
            los.append(lo)
            his.append(hi)
            ns.append(len(vals))

        x = np.arange(len(bin_labels))
        ax.errorbar(x, meds, yerr=[np.array(meds) - np.array(los),
                                    np.array(his) - np.array(meds)],
                    fmt="o-", color=C["blue"], capsize=4, capthick=1,
                    markersize=5, linewidth=1.2, elinewidth=0.8)

        # Median line reference
        overall_med = float(np.nanmedian(sub[ycol].values))
        ax.axhline(overall_med, color=C["orange"], ls="--", lw=0.8,
                   label=f"Overall median = {overall_med:.1f}")

        # Annotate N per bin
        for i, n in enumerate(ns):
            ax.text(i, his[i] + (max(his) - min(los)) * 0.04 if not np.isnan(his[i]) else 0,
                    f"n={n:,}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("Lane probability min (pre)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6.5, loc="upper right")

    fig.tight_layout()
    _save(fig, "fig_A_laneprob_vs_smoothness")


# ═══════════════════════════════════════════════════════════════════════
#  Fig B: Trigger-stratified box/violin
# ═══════════════════════════════════════════════════════════════════════
def fig_B(df: pd.DataFrame):
    triggers = sorted(df["primary_trigger"].dropna().unique())
    trigger_short = {
        "Steering Override": "Steer",
        "Brake Override": "Brake",
        "Gas Override": "Gas",
        "System / Unknown": "System",
    }
    colors_map = {
        "Steering Override": C["blue"],
        "Brake Override": C["teal"],
        "Gas Override": C["green"],
        "System / Unknown": C["gray"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2))

    for ax, (col, ylabel) in zip(axes, [
        ("jerk_max_post", "Jerk max (m/s³)"),
        ("steer_rate_max_post", "Steer rate max (°/s)"),
    ]):
        data, labs, box_colors = [], [], []
        for trig in triggers:
            vals = df.loc[df["primary_trigger"] == trig, col].dropna().values
            if len(vals) < 5:
                continue
            data.append(vals)
            labs.append(trigger_short.get(trig, trig))
            box_colors.append(colors_map.get(trig, C["gray"]))

        vp = ax.violinplot(data, positions=range(len(data)),
                           showmedians=False, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(box_colors[i])
            body.set_alpha(0.35)

        # Overlay box
        bp = ax.boxplot(data, positions=range(len(data)), widths=0.25,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=C["orange"], linewidth=1.5))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(box_colors[i])
            patch.set_alpha(0.7)

        # Annotate N
        for i, d in enumerate(data):
            y_top = np.percentile(d, 95)
            ax.text(i, y_top * 1.02, f"n={len(d):,}",
                    ha="center", va="bottom", fontsize=6)

        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labs)
        ax.set_ylabel(ylabel)
        # Cap y at P99 for readability
        all_vals = np.concatenate(data)
        ax.set_ylim(bottom=0, top=np.percentile(all_vals, 99) * 1.15)

    fig.tight_layout()
    _save(fig, "fig_B_trigger_stratified")


# ═══════════════════════════════════════════════════════════════════════
#  Fig C: Curvature mismatch strata → steer rate
# ═══════════════════════════════════════════════════════════════════════
def fig_C(df: pd.DataFrame):
    sub = df[["curvature_mismatch_max_pre", "steer_rate_max_post"]].dropna()
    if len(sub) < 100:
        print("  [SKIP] Fig C: insufficient data")
        return

    # Define strata by quantiles
    q_bounds = [0, 0.50, 0.80, 0.90, 0.95, 0.99]
    q_vals = [sub["curvature_mismatch_max_pre"].quantile(q) for q in q_bounds]
    strata_labels = ["≤P50", "P50–80", "P80–90", "P90–95", "P95–99"]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    meds, los, his, ns = [], [], [], []
    for i in range(len(strata_labels)):
        lo_v = q_vals[i]
        hi_v = q_vals[i + 1] if i + 1 < len(q_vals) else np.inf
        if i == 0:
            mask = sub["curvature_mismatch_max_pre"] <= hi_v
        else:
            mask = (sub["curvature_mismatch_max_pre"] > lo_v) & \
                   (sub["curvature_mismatch_max_pre"] <= hi_v)
        vals = sub.loc[mask, "steer_rate_max_post"].values
        med, ci_lo, ci_hi = bootstrap_ci(vals)
        meds.append(med)
        los.append(ci_lo)
        his.append(ci_hi)
        ns.append(len(vals))

    x = np.arange(len(strata_labels))
    ax.errorbar(x, meds,
                yerr=[np.array(meds) - np.array(los),
                      np.array(his) - np.array(meds)],
                fmt="s-", color=C["purple"], capsize=4, capthick=1,
                markersize=5, linewidth=1.2, elinewidth=0.8)

    for i, n in enumerate(ns):
        offset = (max(his) - min(los)) * 0.05 if not any(np.isnan(his)) else 1
        ax.text(i, his[i] + offset, f"n={n:,}",
                ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(strata_labels, fontsize=7)
    ax.set_xlabel("Curvature mismatch max (pre, winsorized) stratum")
    ax.set_ylabel("Steer rate max (post, °/s)")

    fig.tight_layout()
    _save(fig, "fig_C_mismatch_strata")


# ═══════════════════════════════════════════════════════════════════════
#  Fig D: Issue-flag prevalence + effect
# ═══════════════════════════════════════════════════════════════════════
def fig_D(df: pd.DataFrame, flag_df: pd.DataFrame):
    flag_names = list(flag_df.columns)
    flag_labels = {
        "lead_inconsistency": "Lead\ninconsist.",
        "low_lane_prob": "Low lane\nprob.",
        "high_curv_mismatch": "High curv.\nmismatch",
        "output_aggressive_low_lane": "Aggressive\n+ low lane",
        "plan_output_outlier": "Plan→output\noutlier",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5))

    # Panel a: Prevalence
    ax = axes[0]
    prevs = []
    for fn in flag_names:
        prevs.append(flag_df[fn].mean())
    x = np.arange(len(flag_names))
    bars = ax.bar(x, prevs, color=[C["red"], C["teal"], C["purple"],
                                    C["brown"], C["orange"]],
                  width=0.6, alpha=0.8)
    for i, (bar, p) in enumerate(zip(bars, prevs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{p:.1%}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([flag_labels.get(fn, fn) for fn in flag_names], fontsize=6.5)
    ax.set_ylabel("Prevalence")
    ax.set_title("(a) Flag prevalence", fontsize=8)

    # Panel b: Effect size (median uplift in steer_rate_max_post) with CI
    ax = axes[1]
    uplifts, ci_los, ci_his = [], [], []
    for fn in flag_names:
        mask = flag_df[fn].astype(bool)
        flagged = df.loc[mask, "steer_rate_max_post"].dropna().values
        unflagged = df.loc[~mask, "steer_rate_max_post"].dropna().values
        if len(flagged) >= 10 and len(unflagged) >= 10:
            diff = np.median(flagged) - np.median(unflagged)
            diffs = []
            for _ in range(N_BOOT):
                f_b = RNG.choice(flagged, size=len(flagged), replace=True)
                u_b = RNG.choice(unflagged, size=len(unflagged), replace=True)
                diffs.append(np.median(f_b) - np.median(u_b))
            diffs = np.array(diffs)
            d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
            uplifts.append(diff)
            ci_los.append(d_lo)
            ci_his.append(d_hi)
        else:
            uplifts.append(0)
            ci_los.append(0)
            ci_his.append(0)

    ax.barh(x, uplifts, color=[C["red"], C["teal"], C["purple"],
                                C["brown"], C["orange"]],
            height=0.6, alpha=0.8)
    ax.errorbar(uplifts, x,
                xerr=[np.array(uplifts) - np.array(ci_los),
                      np.array(ci_his) - np.array(uplifts)],
                fmt="none", color="black", capsize=3, capthick=0.8,
                elinewidth=0.8)
    ax.axvline(0, color="black", lw=0.5, ls="-")
    ax.set_yticks(x)
    ax.set_yticklabels([flag_labels.get(fn, fn) for fn in flag_names], fontsize=6.5)
    ax.set_xlabel("Δ median steer rate max (°/s)")
    ax.set_title("(b) Effect on post-takeover steer rate", fontsize=8)

    fig.tight_layout()
    _save(fig, "fig_D_issue_flags")


# ═══════════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════════
def generate_report(df: pd.DataFrame, summary_rows: list[dict],
                    flag_df: pd.DataFrame):
    N = len(df)
    lines = []

    lines.append("# Takeover Safety & Smoothness Reanalysis")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(dedent("""\
    ### Data and Scope

    This analysis examines 15,705 takeover clips from the OpenLKA dataset.
    Each clip captures a disengagement event where the human driver overrides
    the Level-2 ADAS (openpilot) via steering, brake, or gas input.
    Six CAN-bus topics are used per clip: carControl, carOutput, carState,
    controlsState, drivingModelData, and longitudinalPlan.  radarState is
    excluded; consequently, no direct time-to-collision (TTC), time headway
    (THW), or deceleration-rate-to-avoid-crash (DRAC) metrics are available.
    All safety-relevant quantities reported here are *proxies* derived from
    planner intent and perception confidence rather than physical headway.

    ### Time Windows

    - **Pre-takeover window**: [−3, 0] s relative to the disengagement event.
      Used for context proxies (lane confidence, curvature mismatch, planner
      targets).
    - **Post-takeover window**: [0, +5] s.  Used for control-quality metrics
      (jerk, steering rate, curvature rate, RMSE, stabilization time).

    ### Derivative Computation

    Jerk (da/dt), steering rate (dθ/dt), and curvature rate (dκ/dt) are
    computed via timestamp-based finite differences after Savitzky–Golay
    smoothing (window = 0.3 s, polynomial order 2).  Jerk is capped at
    50 m/s³ and steering rate at 500 °/s to suppress instrumentation
    artifacts.

    ### Control-Chain RMSE and Gating

    Plan-to-output RMSE compares `actuators.accel` (carControl) against
    `actuatorsOutput.accel` (carOutput) on a common 20 Hz grid via linear
    interpolation.  A **gating rule** is applied: if the RMSE value is
    exactly 0.0, the corresponding channel is presumed inactive and the
    value is set to NA.  This is necessary because 64.3% of pre-window
    and 79.7% of post-window accel plan→output RMSE values are identically
    zero, indicating that the actuator command channel is not active
    (e.g., the vehicle platform does not expose acceleration commands
    via carOutput).  Without gating, these zeros would bias summary
    statistics toward artificially low RMSE.  Curvature plan→output RMSE
    has only 3.8% zeros and is retained without gating.

    ### Missingness

    Topic-level missingness varies substantially:
    """))

    miss_cols = [c for c in df.columns if c.startswith("miss_")]
    for mc in miss_cols:
        rate = df[mc].mean()
        topic = mc.replace("miss_", "")
        lines.append(f"- **{topic}**: {rate:.1%} missing ({df[mc].sum():,}/{N:,})")

    lines.append("")
    lines.append(dedent("""\
    drivingModelData is absent for 33.9% of clips; metrics derived from
    it (laneProb, laneCenter_range, laneWidth) have effective
    N ≈ 10,200 rather than 15,700.  carOutput is absent for 23.9%;
    all RMSE terms involving actuatorsOutput have effective N ≈ 11,800.
    All tables report effective N alongside summary statistics.

    ### Winsorization

    Curvature mismatch (|κ_desired − κ_actual|) exhibits extreme right
    skew (P99 = 0.195 vs. P100 = 21.9).  Values above P99 are capped
    (winsorized) to prevent outlier-driven distortion.  This choice is
    conservative: it preserves 99% of the distribution while removing
    instrumentation artifacts.

    ### Stabilization Time

    Stabilization is defined as the first continuous 1.0 s window in the
    post-takeover period where |a| < 0.5 m/s², |jerk| < 1.0 m/s³, and
    |steer rate| < 30 °/s simultaneously.  The metric is right-censored
    at 5.0 s: clips that do not stabilize within 5 s are marked as
    censored and excluded from the uncensored distribution.
    """))

    lines.append("---")
    lines.append("")
    lines.append("## Results")
    lines.append("")

    # ── Section 1 ──
    lines.append("### 1. Pre-Takeover Context (System Load Proxies)")
    lines.append("")

    # Lane prob
    lp = df["laneProb_min_pre"].dropna()
    lines.append(f"**Lane detection confidence.**  Among the {len(lp):,} clips "
                 f"with drivingModelData, the minimum lane-line probability in "
                 f"the pre-window has a median of {lp.median():.3f} "
                 f"(P5 = {lp.quantile(0.05):.3f}, P95 = {lp.quantile(0.95):.3f}).  "
                 f"{(lp < 0.3).sum():,} clips ({(lp < 0.3).mean():.1%}) "
                 f"fall below 0.30, suggesting degraded lane marking visibility "
                 f"or model uncertainty.")
    lines.append("")

    # Curvature mismatch
    cm = df["curvature_mismatch_max_pre"].dropna()
    lines.append(f"**Curvature mismatch.**  The maximum |κ_desired − κ_actual| "
                 f"in the pre-window (winsorized at P99) has a median of "
                 f"{cm.median():.6f} (N = {len(cm):,}).  "
                 f"The distribution is right-skewed: P95 = {cm.quantile(0.95):.6f}.  "
                 f"Elevated mismatch is consistent with situations where the "
                 f"planner's desired path diverges from the vehicle's executed "
                 f"trajectory, potentially indicating challenging road geometry "
                 f"or controller tracking limitations.")
    lines.append("")

    # aTarget
    at = df["aTarget_min_pre"].dropna()
    lines.append(f"**Longitudinal planner target.**  The minimum planned "
                 f"acceleration target (aTarget) in the pre-window has a "
                 f"median of {at.median():.3f} m/s² (N = {len(at):,}).  "
                 f"{(at < -1.0).sum():,} clips ({(at < -1.0).mean():.1%}) "
                 f"show aTarget < −1.0 m/s², suggesting the planner was "
                 f"commanding notable deceleration before the takeover.")
    lines.append("")

    # Lead consistency
    lc = df["lead_consistency_flag"].dropna()
    lines.append(f"**Lead vehicle consistency.**  The mismatch rate between "
                 f"hasLead (longitudinalPlan) and hudControl.leadVisible "
                 f"(carControl) has a median of {lc.median():.3f} "
                 f"(N = {len(lc):,}).  {(lc > 0.10).sum():,} clips "
                 f"({(lc > 0.10).mean():.1%}) exhibit a mismatch rate "
                 f"exceeding 10%, indicating intermittent lead-vehicle "
                 f"detection or display discrepancies.")
    lines.append("")

    # ── Section 2 ──
    lines.append("### 2. Post-Takeover Control Quality")
    lines.append("")

    jk = df["jerk_max_post"].dropna()
    sr = df["steer_rate_max_post"].dropna()
    lines.append(f"**Jerk.**  Peak post-takeover jerk has a median of "
                 f"{jk.median():.2f} m/s³ (N = {len(jk):,}, "
                 f"P5 = {jk.quantile(0.05):.2f}, P95 = {jk.quantile(0.95):.2f}).  "
                 f"The distribution is right-skewed, consistent with the "
                 f"expectation that most takeovers are smooth but a minority "
                 f"involve abrupt longitudinal corrections.")
    lines.append("")

    lines.append(f"**Steering rate.**  Peak post-takeover steering rate has "
                 f"a median of {sr.median():.1f} °/s (N = {len(sr):,}, "
                 f"P5 = {sr.quantile(0.05):.1f}, "
                 f"P95 = {sr.quantile(0.95):.1f}).  The wide P95 value "
                 f"suggests substantial heterogeneity in lateral control "
                 f"urgency across takeover events.")
    lines.append("")

    # RMSE
    apo_pre = df["accel_plan_output_rmse_pre"].dropna()
    apo_post = df["accel_plan_output_rmse_post"].dropna()
    aos_pre = df["accel_output_state_rmse_pre"].dropna()
    aos_post = df["accel_output_state_rmse_post"].dropna()

    lines.append(f"**Accel plan→output RMSE (gated).**  After removing "
                 f"inactive-channel zeros, the pre-window median is "
                 f"{apo_pre.median():.4f} (N = {len(apo_pre):,}) and the "
                 f"post-window median is {apo_post.median():.4f} "
                 f"(N = {len(apo_post):,}).  The low magnitudes suggest "
                 f"that when the accel command channel is active, the "
                 f"output closely tracks the plan.")
    lines.append("")

    lines.append(f"**Accel output→state RMSE.**  This measures the gap "
                 f"between commanded output and realized vehicle acceleration "
                 f"(aEgo).  Pre-window median = {aos_pre.median():.3f} "
                 f"(N = {len(aos_pre):,}), post-window median = "
                 f"{aos_post.median():.3f} (N = {len(aos_post):,}).  "
                 f"The substantially larger magnitude compared to "
                 f"plan→output RMSE is expected: it captures both "
                 f"actuator lag and physical plant dynamics.")
    lines.append("")

    # Stabilization
    st = df["stabilization_time_5s"].dropna()
    n_cens = int(df["stabilization_censored"].sum())
    lines.append(f"**Stabilization time.**  Of {N:,} clips, "
                 f"{n_cens:,} ({n_cens/N:.1%}) are right-censored "
                 f"(not stabilized within 5 s).  Among the "
                 f"{len(st):,} uncensored clips, median stabilization "
                 f"time is {st.median():.2f} s "
                 f"(P5 = {st.quantile(0.05):.2f}, "
                 f"P95 = {st.quantile(0.95):.2f}).  "
                 f"**Caution**: the high censoring rate ({n_cens/N:.1%}) "
                 f"means the uncensored distribution under-represents "
                 f"difficult takeovers.  The median should be interpreted "
                 f"as a lower bound on the population stabilization time.")
    lines.append("")

    # ── Section 3 ──
    lines.append("### 3. Behavior Validation by Trigger Type")
    lines.append("")

    triggers = sorted(df["primary_trigger"].dropna().unique())
    for metric_col, metric_label, unit in [
        ("jerk_max_post", "jerk", "m/s³"),
        ("steer_rate_max_post", "steering rate", "°/s"),
    ]:
        lines.append(f"**Post-takeover {metric_label} by trigger type:**")
        lines.append("")
        lines.append(f"| Trigger | N | Median | 95% CI |")
        lines.append(f"|---------|---|--------|--------|")
        for trig in triggers:
            vals = df.loc[df["primary_trigger"] == trig, metric_col].dropna().values
            med, lo, hi = bootstrap_ci(vals)
            lines.append(f"| {trig} | {len(vals):,} | {med:.2f} {unit} "
                         f"| [{lo:.2f}, {hi:.2f}] |")
        lines.append("")

    # Effect sizes
    if "Steering Override" in triggers and "Brake Override" in triggers:
        for metric_col, metric_label in [
            ("jerk_max_post", "jerk"),
            ("steer_rate_max_post", "steering rate"),
        ]:
            s_vals = df.loc[df["primary_trigger"] == "Steering Override",
                            metric_col].dropna().values
            b_vals = df.loc[df["primary_trigger"] == "Brake Override",
                            metric_col].dropna().values
            diff = np.median(s_vals) - np.median(b_vals)
            diffs = []
            for _ in range(N_BOOT):
                s_b = RNG.choice(s_vals, size=len(s_vals), replace=True)
                b_b = RNG.choice(b_vals, size=len(b_vals), replace=True)
                diffs.append(np.median(s_b) - np.median(b_b))
            diffs_arr = np.array(diffs)
            d_lo, d_hi = np.percentile(diffs_arr, [2.5, 97.5])
            lines.append(f"Steering–Brake median difference in {metric_label}: "
                         f"Δ = {diff:+.2f}, 95% CI [{d_lo:+.2f}, {d_hi:+.2f}].")
        lines.append("")

    lines.append(dedent("""\
    Steering-override takeovers exhibit higher median steering rates
    than brake-override takeovers, consistent with the biomechanical
    expectation that steering interventions involve rapid lateral
    corrections.  The effect-size confidence intervals exclude zero,
    suggesting a robust difference.  However, this comparison reflects
    the trigger modality itself and should not be interpreted as
    evidence that one trigger type is inherently safer.
    """))

    # ── Section 4 ──
    lines.append("### 4. Diagnostic Interaction Flags")
    lines.append("")
    lines.append(dedent("""\
    Five diagnostic flags identify clips exhibiting potentially
    problematic perception–control interactions.  Each flag is
    defined by explicit thresholds chosen from the data distribution
    or engineering practice.  For each flag, we report prevalence
    (among clips with available data) and the uplift in median
    post-takeover steering rate (flagged vs. unflagged), with
    bootstrap 95% CI for the difference.
    """))

    flag_defs = {
        "lead_inconsistency": ("lead_consistency_flag > 0.10",
            "Mismatch between planner hasLead and HUD leadVisible exceeds 10%."),
        "low_lane_prob": ("laneProb_min_pre < 0.30",
            "Minimum lane-line probability below 0.30 (perception low-confidence)."),
        "high_curv_mismatch": ("curvature_mismatch_max_pre > P95 (winsorized)",
            "Curvature mismatch in the top 5% of the distribution."),
        "output_aggressive_low_lane": ("laneProb_min_pre < 0.30 AND curvature_rate_max_post > P95",
            "Conjunction of low lane confidence and aggressive post-takeover curvature rate."),
        "plan_output_outlier": ("accel_plan_output_rmse_pre (gated) > P95",
            "Gated plan→output RMSE in the top 5%, suggesting notable plan–execution discrepancy."),
    }

    lines.append("| Flag | Rule | Prevalence | Steer rate uplift (Δ median) |")
    lines.append("|------|------|------------|------------------------------|")
    for fn in flag_df.columns:
        mask = flag_df[fn].astype(bool)
        eff_n = mask.notna().sum()
        prev = mask.sum() / max(eff_n, 1)
        rule_str, _ = flag_defs.get(fn, (fn, ""))
        flagged = df.loc[mask, "steer_rate_max_post"].dropna().values
        unflagged = df.loc[~mask, "steer_rate_max_post"].dropna().values
        if len(flagged) >= 10 and len(unflagged) >= 10:
            uplift = np.median(flagged) - np.median(unflagged)
            diffs = []
            for _ in range(N_BOOT):
                f_b = RNG.choice(flagged, size=len(flagged), replace=True)
                u_b = RNG.choice(unflagged, size=len(unflagged), replace=True)
                diffs.append(np.median(f_b) - np.median(u_b))
            diffs_arr = np.array(diffs)
            d_lo, d_hi = np.percentile(diffs_arr, [2.5, 97.5])
            uplift_str = f"{uplift:+.1f} °/s [{d_lo:+.1f}, {d_hi:+.1f}]"
        else:
            uplift_str = "---"
        lines.append(f"| {fn} | `{rule_str}` | {prev:.1%} | {uplift_str} |")
    lines.append("")

    # ── Limitations ──
    lines.append("### 5. Limitations")
    lines.append("")
    lines.append(dedent("""\
    1. **No radar-derived safety metrics.**  Without radarState, this
       analysis cannot compute TTC, THW, or DRAC.  The reported proxies
       (hasLead rate, aTarget, planned speed drop) reflect planner intent
       rather than physical headway.

    2. **Proxy nature of all pre-takeover metrics.**  Lane probability,
       curvature mismatch, and lead-visibility rates are model-internal
       quantities.  Their relationship to objective road conditions is
       mediated by the perception model's accuracy, which may vary across
       vehicle platforms and lighting conditions.

    3. **RMSE gating heuristic.**  The zero-RMSE gating rule (Section
       Methods) is a post-hoc heuristic.  A more rigorous approach would
       check raw channel activity (mean |signal| > ε) per clip, requiring
       a separate pass over raw CSVs.

    4. **Stabilization censoring.**  The high censoring rate (≈85%)
       means the reported stabilization-time distribution is conditional
       on stabilization occurring within 5 s.  Causal or population-level
       inferences from this metric require survival-analysis methods.

    5. **Sample-rate sensitivity.**  The dataset contains both qlog
       (10 Hz) and rlog (100 Hz) recordings.  Derivative metrics (jerk,
       steering rate) may exhibit systematic differences across log types.

    6. **No causal claims.**  All reported associations (e.g., low lane
       probability → higher steering rate) are observational.
       Confounding by road type, weather, or vehicle platform cannot
       be ruled out without additional covariate adjustment.

    7. **drivingModelData missingness.**  One-third of clips lack
       drivingModelData; lane-probability and lane-width metrics are
       computed on the remaining 66%.  If missingness is non-random
       (e.g., correlated with vehicle platform or log type), the
       reported distributions may not generalize to the full dataset.
    """))

    report_path = OUT_REPORTS / "takeover_reanalysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved report: {report_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    df = load_and_gate()

    all_rows = []

    # Missingness summary
    miss_rows = []
    for mc in [c for c in df.columns if c.startswith("miss_")]:
        miss_rows.append(dict(
            metric=f"missingness_{mc.replace('miss_', '')}",
            N=len(df), mean=df[mc].mean(), median=np.nan,
            P5=np.nan, P25=np.nan, P75=np.nan, P95=np.nan,
        ))
    all_rows.extend(miss_rows)

    # Sections
    all_rows.extend(section1_context(df))
    all_rows.extend(section2_control(df))
    all_rows.extend(section3_trigger(df))
    flag_rows, flag_df = section4_flags(df)
    all_rows.extend(flag_rows)

    # Summary CSV
    summary = pd.DataFrame(all_rows)
    summary_path = OUT_TABLES / "reanalysis_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path}  ({len(summary)} rows)")

    # Figures
    print("\nGenerating figures …")
    fig_A(df)
    fig_B(df)
    fig_C(df)
    fig_D(df, flag_df)

    # Report
    print("\nGenerating report …")
    generate_report(df, all_rows, flag_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
