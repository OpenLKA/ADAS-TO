#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_nonego_radar_ttc_thw.py
================================
Radar-based TTC / THW longitudinal analysis for Non-ego takeover clips.
Reads radarState.csv (leadOne) + carState.csv per clip.

Definitions:
  THW = dRel / vEgo          (time headway; vEgo > 0.5 m/s guard)
  TTC = dRel / (-vRel)       (time-to-collision; only when closing: vRel < -0.1)

Pre-window: [-3, 0] s around takeover.
Only samples where leadOne.status == True are used.

Run:
    python3 scripts/analyze_nonego_radar_ttc_thw.py
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════
CODE   = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver/Code")
ROOT   = CODE.parent
LABELS = CODE / "DatasetClassification" / "ego_nonego_labels.csv"
OUTDIR = CODE / "outputs" / "metric"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════
PRE_S       = 3.0
POST_S      = 5.0
EPS         = 1e-6
V_MIN       = 0.5       # m/s — guard for THW denominator
VREL_CLOSE  = -0.1      # m/s — closing threshold for TTC
TTC_CAP     = 15.0      # s — cap TTC at reasonable max
THW_CAP     = 15.0      # s
N_WORKERS   = 12
N_BOOT      = 2000
SPEED_LOW   = 16.7      # m/s (~60 km/h)
SPEED_HIGH  = 27.8      # m/s (~100 km/h)

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

RNG = np.random.default_rng(42)


def _save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(OUTDIR / f"{name}{ext}")
    plt.close(fig)
    print(f"  saved {name}")


def bootstrap_median_ci(arr, n_boot=N_BOOT, alpha=0.05):
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return np.nan, np.nan, np.nan
    med = np.nanmedian(arr)
    boots = np.array([np.nanmedian(RNG.choice(arr, len(arr), replace=True))
                      for _ in range(n_boot)])
    return med, np.percentile(boots, 100*alpha/2), np.percentile(boots, 100*(1-alpha/2))


# ═══════════════════════════════════════════════════════════════════════
#  Clip discovery (same as main script)
# ═══════════════════════════════════════════════════════════════════════

def find_all_clips():
    lookup = {}
    for mj in ROOT.rglob("meta.json"):
        d = mj.parent
        try:
            with open(mj) as f:
                meta = json.load(f)
            key = (meta["dongle_id"], meta["route_id"], meta["clip_id"])
            lookup[key] = d
        except Exception:
            continue
    return lookup


# ═══════════════════════════════════════════════════════════════════════
#  Per-clip worker
# ═══════════════════════════════════════════════════════════════════════

def process_clip(clip_dir: Path) -> dict | None:
    """Extract radar-based longitudinal metrics for one clip."""
    meta_path = clip_dir / "meta.json"
    radar_path = clip_dir / "radarState.csv"
    state_path = clip_dir / "carState.csv"

    if not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None

    event_t = meta.get("video_time_s", 10.0)

    row = dict(
        dongle_id=meta.get("dongle_id", ""),
        route_id=meta.get("route_id", ""),
        clip_id=meta.get("clip_id", -1),
        car_model=meta.get("car_model", ""),
        event_t=event_t,
        has_radar=False,
        n_lead_samples_pre=0,
        n_lead_samples_post=0,
    )

    # ── Read carState for vEgo ───────────────────────────────────────
    vego_at_to = np.nan
    try:
        cs = pd.read_csv(state_path, usecols=["time_s", "vEgo", "aEgo"])
        # vEgo at takeover
        mask_at = (cs["time_s"] >= event_t - 0.3) & (cs["time_s"] <= event_t + 0.3)
        if mask_at.sum() > 0:
            idx = np.argmin(np.abs(cs.loc[mask_at, "time_s"].values - event_t))
            vego_at_to = float(cs.loc[mask_at, "vEgo"].values[idx])
    except Exception:
        cs = None
    row["vEgo_at_to"] = vego_at_to

    # ── Read radarState ──────────────────────────────────────────────
    if not radar_path.exists():
        row["has_radar"] = False
        # Fill all radar metrics with NaN
        for k in _radar_nan_keys():
            row[k] = np.nan
        return row

    try:
        rd = pd.read_csv(radar_path)
    except Exception:
        for k in _radar_nan_keys():
            row[k] = np.nan
        return row

    row["has_radar"] = True

    # ── Pre-window: [-3, 0] ──────────────────────────────────────────
    pre_mask = (rd["time_s"] >= event_t - PRE_S) & (rd["time_s"] <= event_t)
    rd_pre = rd[pre_mask].copy()

    # Filter lead active
    if "leadOne.status" in rd_pre.columns:
        # Handle both bool and numeric
        status = rd_pre["leadOne.status"]
        if status.dtype == object:
            status = status.map({"True": True, "true": True,
                                 "False": False, "false": False})
        lead_pre = rd_pre[status.astype(bool)].copy()
    else:
        lead_pre = pd.DataFrame()

    row["n_lead_samples_pre"] = len(lead_pre)

    if len(lead_pre) >= 2:
        dRel = lead_pre["leadOne.dRel"].values
        vRel = lead_pre["leadOne.vRel"].values
        vLead = lead_pre["leadOne.vLead"].values
        t_radar = lead_pre["time_s"].values

        # Get vEgo by interpolation from carState
        if cs is not None:
            cs_pre = cs[(cs["time_s"] >= event_t - PRE_S) & (cs["time_s"] <= event_t)]
            if len(cs_pre) >= 2:
                vEgo_interp = np.interp(t_radar, cs_pre["time_s"].values,
                                        cs_pre["vEgo"].values)
            else:
                vEgo_interp = vLead - vRel  # vEgo ≈ vLead - vRel
        else:
            vEgo_interp = vLead - vRel

        # ── THW = dRel / vEgo ────────────────────────────────────────
        safe_vego = np.where(vEgo_interp > V_MIN, vEgo_interp, np.nan)
        thw = dRel / safe_vego
        thw = np.clip(thw, 0, THW_CAP)

        row["thw_min_pre"]  = float(np.nanmin(thw)) if np.any(np.isfinite(thw)) else np.nan
        row["thw_p5_pre"]   = float(np.nanpercentile(thw[np.isfinite(thw)], 5)) if np.sum(np.isfinite(thw)) > 2 else np.nan
        row["thw_median_pre"] = float(np.nanmedian(thw)) if np.any(np.isfinite(thw)) else np.nan

        # ── TTC = dRel / (-vRel), only when closing ─────────────────
        closing = vRel < VREL_CLOSE
        if closing.sum() > 0:
            ttc_raw = dRel[closing] / (-vRel[closing])
            ttc_raw = np.clip(ttc_raw, 0, TTC_CAP)
            row["ttc_min_pre"]    = float(np.nanmin(ttc_raw))
            row["ttc_p5_pre"]     = float(np.nanpercentile(ttc_raw, 5)) if len(ttc_raw) > 2 else np.nan
            row["ttc_median_pre"] = float(np.nanmedian(ttc_raw))
            row["n_closing_pre"]  = int(closing.sum())
        else:
            row["ttc_min_pre"]    = np.nan
            row["ttc_p5_pre"]     = np.nan
            row["ttc_median_pre"] = np.nan
            row["n_closing_pre"]  = 0

        # ── dRel at takeover ─────────────────────────────────────────
        idx_to = np.argmin(np.abs(t_radar - event_t))
        row["dRel_at_to"]  = float(dRel[idx_to])
        row["vRel_at_to"]  = float(vRel[idx_to])
        row["vLead_at_to"] = float(vLead[idx_to])

        # ── Summary stats ────────────────────────────────────────────
        row["dRel_min_pre"]    = float(np.nanmin(dRel))
        row["dRel_mean_pre"]   = float(np.nanmean(dRel))
        row["vRel_min_pre"]    = float(np.nanmin(vRel))   # most negative = fastest closing
        row["lead_rate_pre"]   = float(len(lead_pre) / max(len(rd_pre), 1))
    else:
        for k in _radar_nan_keys():
            row[k] = np.nan
        row["n_closing_pre"] = 0
        row["lead_rate_pre"] = 0.0

    # ── Post-window: [0, +5] ─────────────────────────────────────────
    post_mask = (rd["time_s"] >= event_t) & (rd["time_s"] <= event_t + POST_S)
    rd_post = rd[post_mask].copy()
    if "leadOne.status" in rd_post.columns:
        status_p = rd_post["leadOne.status"]
        if status_p.dtype == object:
            status_p = status_p.map({"True": True, "true": True,
                                     "False": False, "false": False})
        lead_post = rd_post[status_p.astype(bool)]
    else:
        lead_post = pd.DataFrame()
    row["n_lead_samples_post"] = len(lead_post)

    if len(lead_post) >= 2:
        dRel_p = lead_post["leadOne.dRel"].values
        vRel_p = lead_post["leadOne.vRel"].values
        row["dRel_min_post"]  = float(np.nanmin(dRel_p))
        row["vRel_min_post"]  = float(np.nanmin(vRel_p))
    else:
        row["dRel_min_post"]  = np.nan
        row["vRel_min_post"]  = np.nan

    return row


def _radar_nan_keys():
    """Keys to set NaN when radar is missing."""
    return [
        "thw_min_pre", "thw_p5_pre", "thw_median_pre",
        "ttc_min_pre", "ttc_p5_pre", "ttc_median_pre",
        "n_closing_pre", "dRel_at_to", "vRel_at_to", "vLead_at_to",
        "dRel_min_pre", "dRel_mean_pre", "vRel_min_pre", "lead_rate_pre",
        "dRel_min_post", "vRel_min_post",
    ]


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_nonego():
    print("Loading Non-ego labels …")
    labels = pd.read_csv(LABELS)
    ne = labels[labels["label"] == "Non-ego"].copy()
    print(f"  Non-ego: {len(ne)}")

    print("Building clip lookup …")
    lookup = find_all_clips()
    print(f"  Total clip dirs: {len(lookup)}")

    dirs = []
    for _, r in ne.iterrows():
        key = (r["dongle_id"], r["route_id"], int(r["clip_id"]))
        dirs.append(lookup.get(key))
    ne["clip_dir"] = dirs
    ne = ne.dropna(subset=["clip_dir"])
    print(f"  Non-ego with dirs: {len(ne)}")
    return ne


def extract_all(ne_df):
    clip_dirs = ne_df["clip_dir"].tolist()
    results = []
    total = len(clip_dirs)
    done = errors = 0

    print(f"Extracting radar metrics from {total} clips ({N_WORKERS} workers) …")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(process_clip, Path(d)): d for d in clip_dirs}
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
                else:
                    errors += 1
            except Exception:
                errors += 1
            if done % 2000 == 0 or done == total:
                print(f"  {done}/{total}  ({errors} errors)")

    df = pd.DataFrame(results)
    print(f"  Rows: {len(df)}, errors: {errors}")
    return df


def postprocess(df):
    v = df["vEgo_at_to"]
    df["speed_regime"] = pd.cut(
        v, bins=[0, SPEED_LOW, SPEED_HIGH, 999],
        labels=["Low (<60 km/h)", "Medium", "High (>100 km/h)"],
        right=False,
    )
    df["lead_present_pre"] = df["n_lead_samples_pre"] >= 3
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════

def fig_ttc_dist(df):
    """TTC distribution for lead-present, closing clips."""
    vals = df["ttc_min_pre"].dropna().values
    if len(vals) < 30:
        print("  SKIP fig_ttc_dist: insufficient data")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    # Clip for display
    v = vals[vals <= 10]
    ax.hist(v, bins=60, color=C["red"], alpha=0.7,
            edgecolor="white", linewidth=0.3)
    # Threshold lines
    for thr, ls, lab in [(1.5, "--", "1.5 s"), (2.0, "-.", "2.0 s"),
                          (3.0, ":", "3.0 s")]:
        cnt = (vals < thr).sum()
        pct = 100 * cnt / len(vals)
        ax.axvline(thr, color=C["blue"], ls=ls, lw=1.0,
                   label=f"TTC < {lab}: {cnt} ({pct:.1f}%)")
    med = np.nanmedian(vals)
    ax.axvline(med, color=C["orange"], ls="--", lw=1.2,
               label=f"Median = {med:.2f} s")
    ax.set_xlabel("TTC min (s)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 10)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax.text(0.98, 0.65, f"n = {len(vals)}", transform=ax.transAxes,
            ha="right", fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, "fig_radar_ttc_dist")


def fig_thw_dist(df):
    """THW distribution for lead-present clips."""
    vals = df["thw_min_pre"].dropna().values
    if len(vals) < 30:
        print("  SKIP fig_thw_dist")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    v = vals[vals <= 8]
    ax.hist(v, bins=60, color=C["teal"], alpha=0.7,
            edgecolor="white", linewidth=0.3)
    for thr, ls, lab in [(0.8, "--", "0.8 s"), (1.0, "-.", "1.0 s"),
                          (1.5, ":", "1.5 s")]:
        cnt = (vals < thr).sum()
        pct = 100 * cnt / len(vals)
        ax.axvline(thr, color=C["red"], ls=ls, lw=1.0,
                   label=f"THW < {lab}: {cnt} ({pct:.1f}%)")
    med = np.nanmedian(vals)
    ax.axvline(med, color=C["orange"], ls="--", lw=1.2,
               label=f"Median = {med:.2f} s")
    ax.set_xlabel("THW min (s)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 8)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    ax.text(0.98, 0.65, f"n = {len(vals)}", transform=ax.transAxes,
            ha="right", fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, "fig_radar_thw_dist")


def fig_ttc_thw_joint(df):
    """Joint scatter: TTC min vs THW min."""
    sub = df.dropna(subset=["ttc_min_pre", "thw_min_pre"])
    if len(sub) < 30:
        print("  SKIP fig_ttc_thw_joint")
        return
    # Subsample
    if len(sub) > 4000:
        sub = sub.sample(4000, random_state=42)
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.scatter(sub["thw_min_pre"], sub["ttc_min_pre"],
               s=3, alpha=0.2, color=C["blue"], rasterized=True)
    # Threshold box
    ax.axvline(0.8, color=C["red"], ls="--", lw=0.8, alpha=0.7)
    ax.axhline(1.5, color=C["red"], ls="--", lw=0.8, alpha=0.7)
    ax.fill_between([0, 0.8], 0, 1.5, color=C["red"], alpha=0.08)
    ax.text(0.4, 0.75, "Critical\nzone", ha="center", va="center",
            fontsize=7, color=C["red"], fontstyle="italic")
    ax.set_xlabel("THW min (s)")
    ax.set_ylabel("TTC min (s)")
    ax.set_xlim(0, min(8, sub["thw_min_pre"].quantile(0.98)))
    ax.set_ylim(0, min(10, sub["ttc_min_pre"].quantile(0.98)))
    ax.text(0.98, 0.95, f"n = {len(sub)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, "fig_radar_ttc_thw_joint")


def fig_drel_dist(df):
    """Lead distance distribution at takeover."""
    vals = df["dRel_at_to"].dropna().values
    vals = vals[(vals > 0) & (vals < 150)]
    if len(vals) < 30:
        print("  SKIP fig_drel_dist")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.hist(vals, bins=60, color=C["blue"], alpha=0.7,
            edgecolor="white", linewidth=0.3)
    med = np.nanmedian(vals)
    ax.axvline(med, color=C["orange"], ls="--", lw=1.2,
               label=f"Median = {med:.1f} m")
    ax.axvline(10, color=C["red"], ls=":", lw=1.0,
               label=f"dRel < 10 m: {(vals<10).sum()} ({100*(vals<10).mean():.1f}%)")
    ax.set_xlabel("Lead distance at takeover (m)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6, frameon=False)
    ax.text(0.98, 0.88, f"n = {len(vals)}", transform=ax.transAxes,
            ha="right", fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, "fig_radar_drel_dist")


def fig_ttc_by_speed(df):
    """TTC and THW distributions by speed regime."""
    regimes = ["Low (<60 km/h)", "Medium", "High (>100 km/h)"]
    colors_r = [C["teal"], C["blue"], C["red"]]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    metrics = [
        ("ttc_min_pre", "TTC min (s)"),
        ("thw_min_pre", "THW min (s)"),
        ("dRel_min_pre", "Min lead distance (m)"),
    ]
    for ax, (col, lab) in zip(axes, metrics):
        data = []
        for r in regimes:
            v = df.loc[df["speed_regime"] == r, col].dropna().values
            if col in ("ttc_min_pre",):
                v = v[v <= TTC_CAP]
            if col in ("thw_min_pre",):
                v = v[v <= THW_CAP]
            if col == "dRel_min_pre":
                v = v[(v > 0) & (v < 150)]
            data.append(v)
        bp = ax.boxplot(data, positions=range(3), widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        for patch, clr in zip(bp["boxes"], colors_r):
            patch.set_facecolor(clr)
            patch.set_alpha(0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(regimes, fontsize=7)
        ax.set_ylabel(lab)
        for i, d in enumerate(data):
            ax.text(i, ax.get_ylim()[1] * 0.95, f"n={len(d)}",
                    ha="center", fontsize=5.5, color="#555")
    fig.tight_layout()
    _save(fig, "fig_radar_longit_by_speed")


def fig_vrel_dist(df):
    """Relative velocity distribution at takeover (lead-present)."""
    vals = df["vRel_at_to"].dropna().values
    if len(vals) < 30:
        print("  SKIP fig_vrel_dist")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    # Clip to reasonable range
    v = vals[(vals > -20) & (vals < 15)]
    ax.hist(v, bins=60, color=C["purple"], alpha=0.7,
            edgecolor="white", linewidth=0.3)
    med = np.nanmedian(v)
    ax.axvline(med, color=C["orange"], ls="--", lw=1.2,
               label=f"Median = {med:.2f} m/s")
    ax.axvline(0, color=C["gray"], ls="-", lw=0.8)
    ax.set_xlabel("Relative velocity at takeover (m/s)\n← closing    separating →")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6, frameon=False)
    closing_pct = 100 * (vals < -0.1).sum() / len(vals)
    ax.text(0.98, 0.88, f"n = {len(v)}\nClosing: {closing_pct:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=6.5, color="#555")
    fig.tight_layout()
    _save(fig, "fig_radar_vrel_dist")


def fig_ttc_thw_binned_by_speed(df):
    """Binned median TTC and THW as function of ego speed at takeover."""
    sub = df.dropna(subset=["ttc_min_pre", "vEgo_at_to"])
    if len(sub) < 100:
        print("  SKIP fig_ttc_thw_binned")
        return

    speed_bins = np.arange(0, 40, 5)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    for ax, col, lab, clr in [
        (axes[0], "ttc_min_pre", "TTC min (s)", C["red"]),
        (axes[1], "thw_min_pre", "THW min (s)", C["teal"]),
    ]:
        s = df.dropna(subset=[col, "vEgo_at_to"])
        xv = s["vEgo_at_to"].values
        yv = s[col].values
        # Bin
        meds, los, his, xs, ns = [], [], [], [], []
        for i in range(len(speed_bins) - 1):
            mask = (xv >= speed_bins[i]) & (xv < speed_bins[i+1])
            yb = yv[mask]
            yb = yb[np.isfinite(yb) & (yb <= (TTC_CAP if "ttc" in col else THW_CAP))]
            if len(yb) < 10:
                continue
            m, lo, hi = bootstrap_median_ci(yb)
            meds.append(m); los.append(lo); his.append(hi)
            xs.append((speed_bins[i] + speed_bins[i+1]) / 2)
            ns.append(len(yb))
        if not xs:
            continue
        xs, meds, los, his = map(np.array, [xs, meds, los, his])
        ax.plot(xs, meds, "o-", color=clr, lw=1.2, ms=4)
        ax.fill_between(xs, los, his, color=clr, alpha=0.15)
        ax.set_xlabel("Ego speed at takeover (m/s)")
        ax.set_ylabel(lab)
        for xi, ni in zip(xs, ns):
            ax.text(xi, ax.get_ylim()[0], f"{ni}", ha="center",
                    fontsize=5, color="#999", va="bottom")
        # Threshold line
        if "ttc" in col:
            ax.axhline(1.5, color=C["gray"], ls="--", lw=0.7, alpha=0.6)
            ax.text(xs[-1], 1.5, " 1.5 s", va="bottom", fontsize=6, color=C["gray"])
        else:
            ax.axhline(0.8, color=C["gray"], ls="--", lw=0.7, alpha=0.6)
            ax.text(xs[-1], 0.8, " 0.8 s", va="bottom", fontsize=6, color=C["gray"])

    fig.tight_layout()
    _save(fig, "fig_radar_ttc_thw_vs_speed")


def fig_critical_rates(df):
    """Bar chart: rates of critical TTC/THW/dRel events by speed regime."""
    sub = df[df["lead_present_pre"]].copy()
    if len(sub) < 50:
        print("  SKIP fig_critical_rates")
        return
    regimes = ["Low (<60 km/h)", "Medium", "High (>100 km/h)"]
    thresholds = [
        ("ttc_min_pre", 1.5, "TTC < 1.5 s"),
        ("ttc_min_pre", 2.0, "TTC < 2.0 s"),
        ("thw_min_pre", 0.8, "THW < 0.8 s"),
        ("thw_min_pre", 1.0, "THW < 1.0 s"),
        ("dRel_min_pre", 10, "dRel < 10 m"),
    ]
    colors = [C["red"], C["orange"], C["teal"], C["blue"], C["purple"]]

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    x = np.arange(len(regimes))
    width = 0.15
    for j, ((col, thr, lab), clr) in enumerate(zip(thresholds, colors)):
        rates = []
        for r in regimes:
            s = sub.loc[sub["speed_regime"] == r, col].dropna()
            if len(s) > 0:
                rates.append(100 * (s < thr).mean())
            else:
                rates.append(0)
        ax.bar(x + j * width, rates, width, color=clr, alpha=0.75, label=lab)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(regimes, fontsize=8)
    ax.set_ylabel("Event rate (%)")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig_radar_critical_rates")


def fig_drel_pre_post(df):
    """Lead distance pre→post distribution (min pre vs min post)."""
    sub = df.dropna(subset=["dRel_min_pre", "dRel_min_post"])
    sub = sub[(sub["dRel_min_pre"] > 0) & (sub["dRel_min_pre"] < 150) &
              (sub["dRel_min_post"] > 0) & (sub["dRel_min_post"] < 150)]
    if len(sub) < 30:
        print("  SKIP fig_drel_pre_post")
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    if len(sub) > 3000:
        sub = sub.sample(3000, random_state=42)
    ax.scatter(sub["dRel_min_pre"], sub["dRel_min_post"],
               s=3, alpha=0.2, color=C["blue"], rasterized=True)
    lim = max(sub["dRel_min_pre"].quantile(0.98),
              sub["dRel_min_post"].quantile(0.98))
    ax.plot([0, lim], [0, lim], color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("Min lead distance pre (m)")
    ax.set_ylabel("Min lead distance post (m)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.text(0.98, 0.05, f"n = {len(sub)}", transform=ax.transAxes,
            ha="right", fontsize=7, color="#555")
    fig.tight_layout()
    _save(fig, "fig_radar_drel_pre_post")


# ═══════════════════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════════════════

def write_report(df):
    n = len(df)
    has_radar = df["has_radar"].sum()
    lead_pre  = df["lead_present_pre"].sum()

    lines = [
        "# Non-Ego Radar-Based Longitudinal Analysis (TTC / THW)\n",
        "## 1. Dataset Overview\n",
        f"- **Total Non-ego clips**: {n}",
        f"- **With radarState.csv**: {has_radar} ({100*has_radar/n:.1f}%)",
        f"- **Lead-present in pre-window** (≥3 radar samples with leadOne.status=True): "
        f"{lead_pre} ({100*lead_pre/n:.1f}%)",
    ]

    # Definitions
    lines.append("\n## 2. Metric Definitions\n")
    lines.extend([
        "| Metric | Formula | Guard / Notes |",
        "|--------|---------|---------------|",
        "| **THW** (time headway) | dRel / vEgo | vEgo > 0.5 m/s; capped at 15 s |",
        "| **TTC** (time-to-collision) | dRel / (−vRel) | Only closing (vRel < −0.1 m/s); capped at 15 s |",
        "| **dRel** | leadOne.dRel | Direct radar measurement (m) |",
        "| **vRel** | leadOne.vRel | Relative velocity (m/s); negative = closing |",
    ])

    # Summary table for lead-present clips
    lp = df[df["lead_present_pre"]]
    lines.append(f"\n## 3. Key Metrics (lead-present, n = {len(lp)})\n")
    lines.append("| Metric | Median | IQR | P5 | P95 |")
    lines.append("|--------|-------:|:---:|---:|----:|")
    for col, lab in [
        ("ttc_min_pre",    "TTC min (s)"),
        ("thw_min_pre",    "THW min (s)"),
        ("dRel_min_pre",   "dRel min (m)"),
        ("dRel_at_to",     "dRel at takeover (m)"),
        ("vRel_at_to",     "vRel at takeover (m/s)"),
        ("vLead_at_to",    "vLead at takeover (m/s)"),
    ]:
        v = lp[col].dropna()
        if len(v) > 5:
            lines.append(
                f"| {lab} | {v.median():.2f} | "
                f"{v.quantile(0.25):.2f}–{v.quantile(0.75):.2f} | "
                f"{v.quantile(0.05):.2f} | {v.quantile(0.95):.2f} |")

    # Critical event rates
    lines.append(f"\n## 4. Critical Event Rates (lead-present)\n")
    lines.append("| Threshold | Count | Rate |")
    lines.append("|-----------|------:|-----:|")
    for col, thr, lab in [
        ("ttc_min_pre", 1.5, "TTC < 1.5 s"),
        ("ttc_min_pre", 2.0, "TTC < 2.0 s"),
        ("ttc_min_pre", 3.0, "TTC < 3.0 s"),
        ("thw_min_pre", 0.8, "THW < 0.8 s"),
        ("thw_min_pre", 1.0, "THW < 1.0 s"),
        ("thw_min_pre", 1.5, "THW < 1.5 s"),
        ("dRel_min_pre", 10,  "dRel < 10 m"),
    ]:
        v = lp[col].dropna()
        if len(v) > 0:
            cnt = (v < thr).sum()
            pct = 100 * cnt / len(v)
            lines.append(f"| {lab} | {cnt} | {pct:.1f}% |")

    # Speed-stratified summary
    regimes = ["Low (<60 km/h)", "Medium", "High (>100 km/h)"]
    lines.append(f"\n## 5. Speed-Stratified Summary (lead-present)\n")
    lines.append("| Speed | n | TTC med (s) | THW med (s) | dRel med (m) | TTC<1.5s rate |")
    lines.append("|-------|--:|:----------:|:----------:|:-----------:|:------------:|")
    for r in regimes:
        s = lp[lp["speed_regime"] == r]
        ttc = s["ttc_min_pre"].dropna()
        thw = s["thw_min_pre"].dropna()
        dr  = s["dRel_min_pre"].dropna()
        ttc_crit = f"{100*(ttc<1.5).mean():.1f}%" if len(ttc) > 0 else "—"
        lines.append(
            f"| {r} | {len(s)} | "
            f"{ttc.median():.2f} | {thw.median():.2f} | {dr.median():.1f} | "
            f"{ttc_crit} |")

    # Findings
    lines.append(f"\n## 6. Key Findings\n")
    findings = []

    ttc_all = lp["ttc_min_pre"].dropna()
    if len(ttc_all) > 50:
        findings.append(
            f"**TTC distribution**: Among {len(ttc_all)} lead-present Non-ego clips "
            f"with closing dynamics, median TTC_min = {ttc_all.median():.2f} s. "
            f"{(ttc_all<1.5).sum()} clips ({100*(ttc_all<1.5).mean():.1f}%) "
            f"breach the 1.5 s critical threshold.")

    thw_all = lp["thw_min_pre"].dropna()
    if len(thw_all) > 50:
        findings.append(
            f"**THW distribution**: Median THW_min = {thw_all.median():.2f} s. "
            f"{(thw_all<0.8).sum()} clips ({100*(thw_all<0.8).mean():.1f}%) "
            f"below the 0.8 s safety threshold.")

    dr = lp["dRel_at_to"].dropna()
    if len(dr) > 50:
        findings.append(
            f"**Lead distance at takeover**: Median = {dr.median():.1f} m. "
            f"{(dr<10).sum()} clips ({100*(dr<10).mean():.1f}%) with dRel < 10 m, "
            f"indicating very close following at the moment of takeover.")

    vr = lp["vRel_at_to"].dropna()
    if len(vr) > 50:
        closing_pct = 100 * (vr < -0.1).mean()
        findings.append(
            f"**Closing dynamics**: {closing_pct:.1f}% of lead-present clips "
            f"have negative vRel (closing) at takeover. "
            f"Median vRel = {vr.median():.2f} m/s.")

    # Speed effect
    for r in regimes:
        s = lp[lp["speed_regime"] == r]
        ttc_r = s["ttc_min_pre"].dropna()
        if len(ttc_r) > 30:
            crit = 100 * (ttc_r < 1.5).mean()
            if crit > 10:
                findings.append(
                    f"**{r} speed**: TTC < 1.5 s rate = {crit:.1f}% "
                    f"(n = {len(ttc_r)}), suggesting elevated longitudinal "
                    f"conflict risk in this regime.")

    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}\n")

    # Limitations
    lines.append("\n## 7. Limitations\n")
    lines.extend([
        "1. **Radar availability**: radarState.csv is present for "
        f"{100*has_radar/n:.1f}% of Non-ego clips. "
        "Missing radar clips cannot contribute to TTC/THW.\n",
        "2. **Lead detection gaps**: leadOne.status may be False even when a "
        "lead vehicle exists (e.g., radar occlusion, lateral offset). "
        "The lead_present_pre flag requires ≥3 active samples in [-3, 0] s.\n",
        "3. **TTC only for closing**: TTC is undefined when ego is not closing "
        "on the lead vehicle. Non-closing lead-present clips are excluded "
        "from TTC statistics.\n",
        "4. **THW denominator guard**: THW = dRel/vEgo is set to NaN when "
        "vEgo < 0.5 m/s (near-stationary).\n",
        "5. **Single lead vehicle**: Only leadOne is used; leadTwo is ignored.\n",
    ])

    rpt_path = OUTDIR / "non_ego_radar_ttc_thw_report.md"
    rpt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {rpt_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ne = load_nonego()
    df = extract_all(ne)
    df = postprocess(df)

    # Save CSV
    csv_path = OUTDIR / "non_ego_radar_ttc_thw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}  ({len(df)} rows)")

    # Summary
    print("\n" + "=" * 60)
    print("RADAR DATA AVAILABILITY")
    print("=" * 60)
    print(f"  has_radar:        {df['has_radar'].sum()} / {len(df)}")
    print(f"  lead_present_pre: {df['lead_present_pre'].sum()}")
    lp = df[df["lead_present_pre"]]
    for col in ["ttc_min_pre", "thw_min_pre", "dRel_at_to", "vRel_at_to"]:
        nn = lp[col].notna().sum()
        print(f"  {col:25s} non-missing: {nn} ({100*nn/max(len(lp),1):.1f}%)")

    # Figures
    print("\nGenerating figures …")
    fig_ttc_dist(df[df["lead_present_pre"]])
    fig_thw_dist(df[df["lead_present_pre"]])
    fig_ttc_thw_joint(df[df["lead_present_pre"]])
    fig_drel_dist(df[df["lead_present_pre"]])
    fig_vrel_dist(df[df["lead_present_pre"]])
    fig_ttc_by_speed(df[df["lead_present_pre"]])
    fig_ttc_thw_binned_by_speed(df[df["lead_present_pre"]])
    fig_critical_rates(df)
    fig_drel_pre_post(df[df["lead_present_pre"]])

    # Report
    print("\nWriting report …")
    write_report(df)

    # List figures
    print("\n" + "=" * 60)
    print("GENERATED FIGURES")
    print("=" * 60)
    for f in sorted(OUTDIR.glob("fig_radar*")):
        print(f"  {f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
