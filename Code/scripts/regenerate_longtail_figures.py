#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regenerate_longtail_figures.py
==============================
Regenerate fig1_longitudinal_context.png and fig3_lateral_steering_jerk.png
for all 285 longtail clips with improved layout:
  - No suptitle (no overlap)
  - Key info (car model, clip id, TTC, THW) shown as legend-style text box
    in upper-right of the top subplot
  - Clean spacing between subplots (no overlap)
  - Consistent serif style at 300 dpi

Usage:
    python3 scripts/regenerate_longtail_figures.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
LONGTAIL = CODE / "long_lat" / "longtail"
CSV_PATH = CODE / "long_lat" / "per_clip_all_metrics.csv"

PRE_S  = 5.0
POST_S = 5.0
SMOOTH_WINDOW_S = 0.3
SMOOTH_POLY     = 2
EPS = 1e-6
N_WORKERS = 8

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          10,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

CLR = {
    "ego":   "#4C78A8",
    "lead":  "#E45756",
    "steer": "#F58518",
    "brake": "#E45756",
    "gas":   "#54A24B",
    "lane":  "#72B7B2",
    "jerk":  "#B279A2",
    "grey":  "#888888",
}

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def safe_read_csv(path: Path, usecols=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, low_memory=False)
            if usecols:
                existing = [c for c in usecols if c in df.columns]
                return df[existing] if existing else pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()


def parse_bool_col(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(False))


def smooth_signal(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    n = len(values)
    if n < 5:
        return values
    win = max(3, int(SMOOTH_WINDOW_S * sample_rate_hz))
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    poly = min(SMOOTH_POLY, win - 1)
    if win >= 3 and poly >= 1:
        try:
            return savgol_filter(values, win, poly)
        except Exception:
            pass
    return pd.Series(values).rolling(3, center=True, min_periods=1).mean().values


def _info_text(row: dict) -> str:
    """Build compact info string for legend-style annotation."""
    car = row.get("car_model", "")
    cid = row.get("clip_id", "")
    parts = [f"{car}  clip {cid}"]

    ttc_val = _best_ttc(row)
    thw_val = _best_thw(row)
    metrics = []
    if not np.isnan(ttc_val):
        metrics.append(f"TTC={ttc_val:.1f}s")
    if not np.isnan(thw_val):
        metrics.append(f"THW={thw_val:.2f}s")
    if metrics:
        parts.append("  ".join(metrics))
    return "\n".join(parts)


def _best_ttc(row):
    vals = [row.get("min_ttc_pre", np.nan), row.get("ttc_at_takeover", np.nan)]
    valid = [v for v in vals if not np.isnan(v)]
    return min(valid) if valid else np.nan

def _best_thw(row):
    vals = [row.get("thw_at_min_ttc", np.nan), row.get("thw_at_takeover", np.nan)]
    valid = [v for v in vals if not np.isnan(v)]
    return min(valid) if valid else np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_figures(clip_dir: Path, row: dict) -> dict:
    """Generate fig1 + fig3 for one clip. Returns status dict."""
    result = {"clip_dir": str(clip_dir), "fig1_ok": False, "fig3_ok": False}

    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return result
    with open(meta_path) as f:
        meta = json.load(f)

    log_hz = meta.get("log_hz", 20)
    first_to = row.get("first_takeover_s", np.nan)
    if np.isnan(first_to):
        first_to = meta.get("video_time_s", np.nan)
    if np.isnan(first_to):
        return result

    cs    = safe_read_csv(clip_dir / "carState.csv")
    radar = safe_read_csv(clip_dir / "radarState.csv")
    model = safe_read_csv(clip_dir / "drivingModelData.csv")

    if cs.empty or "time_s" not in cs.columns:
        return result

    for col in ["steeringPressed", "gasPressed", "brakePressed"]:
        if col in cs.columns:
            cs[col] = parse_bool_col(cs[col])
    if not radar.empty and "leadOne.status" in radar.columns:
        radar["leadOne.status"] = parse_bool_col(radar["leadOne.status"])

    cs["t_rel"] = cs["time_s"] - first_to
    if not radar.empty and "time_s" in radar.columns:
        radar["t_rel"] = radar["time_s"] - first_to
    if not model.empty and "time_s" in model.columns:
        model["t_rel"] = model["time_s"] - first_to

    cs_w = cs[(cs["t_rel"] >= -PRE_S) & (cs["t_rel"] <= POST_S)]
    radar_w = pd.DataFrame()
    if not radar.empty and "t_rel" in radar.columns:
        radar_w = radar[(radar["t_rel"] >= -PRE_S) & (radar["t_rel"] <= POST_S)]
    model_w = pd.DataFrame()
    if not model.empty and "t_rel" in model.columns:
        model_w = model[(model["t_rel"] >= -PRE_S) & (model["t_rel"] <= POST_S)]

    # Compute jerk on full window for fig3 panel (d)
    jerk_ts = None
    cs_post = cs[(cs["t_rel"] >= 0) & (cs["t_rel"] <= POST_S)]
    extreme_jerk = False

    if len(cs_post) > 5 and "aEgo" in cs_post.columns:
        cs_full = cs[(cs["t_rel"] >= -PRE_S) & (cs["t_rel"] <= POST_S)]
        if len(cs_full) > 5:
            t_f = cs_full["time_s"].values
            a_f = smooth_signal(cs_full["aEgo"].values, log_hz)
            dt_f = np.diff(t_f)
            dt_f[dt_f < EPS] = EPS
            jerk_vals = np.diff(a_f) / dt_f
            jerk_ts = (cs_full["t_rel"].values[1:], jerk_vals)

        t_post = cs_post["time_s"].values
        a_post = smooth_signal(cs_post["aEgo"].values, log_hz)
        dt_post = np.diff(t_post)
        dt_post[dt_post < EPS] = EPS
        jerk_post = np.abs(np.diff(a_post) / dt_post)
        jerk_max = min(float(np.nanmax(jerk_post)), 50.0)
        extreme_jerk = jerk_max > 5.0

    info = _info_text(row)

    # ── Fig 1: Longitudinal Context ────────────────────────────────────
    try:
        fig, axes = plt.subplots(3, 1, figsize=(5.5, 4.8), sharex=True,
                                 gridspec_kw={"hspace": 0.25})

        # (a) Speed
        ax = axes[0]
        ax.plot(cs_w["t_rel"], cs_w["vEgo"] * 3.6,
                color=CLR["ego"], lw=1.0, label="Ego")
        if not radar_w.empty and "leadOne.vLead" in radar_w.columns:
            rv = radar_w
            if "leadOne.status" in radar_w.columns:
                rv = radar_w[radar_w["leadOne.status"] == True]
            if not rv.empty:
                ax.plot(rv["t_rel"], rv["leadOne.vLead"] * 3.6,
                        color=CLR["lead"], lw=1.0, ls="--", label="Lead")
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_ylabel("Speed (km/h)")
        # Info box in upper-right as legend-style text
        ax.text(0.98, 0.95, info, transform=ax.transAxes,
                fontsize=6, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                          alpha=0.9, lw=0.5))
        ax.legend(loc="upper left", frameon=False)

        # (b) Distance to Lead
        ax = axes[1]
        if not radar_w.empty and "leadOne.dRel" in radar_w.columns:
            rv = radar_w
            if "leadOne.status" in radar_w.columns:
                rv = radar_w[radar_w["leadOne.status"] == True]
            if not rv.empty:
                ax.plot(rv["t_rel"], rv["leadOne.dRel"],
                        color=CLR["lead"], lw=1.0, label="dRel")
                ax.fill_between(rv["t_rel"], 0, rv["leadOne.dRel"],
                                color=CLR["lead"], alpha=0.06)
                ax.legend(loc="upper right", frameon=False)
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_ylabel("Rel. Distance (m)")

        # (c) Pedal
        ax = axes[2]
        if "brakePressed" in cs_w.columns:
            ax.fill_between(cs_w["t_rel"], 0, cs_w["brakePressed"].astype(float),
                            color=CLR["brake"], alpha=0.3, label="Brake", step="post")
        if "gasPressed" in cs_w.columns:
            ax.fill_between(cs_w["t_rel"], 0, cs_w["gasPressed"].astype(float) * 0.7,
                            color=CLR["gas"], alpha=0.3, label="Gas", step="post")
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_ylabel("Pedal")
        ax.set_xlabel("Time relative to takeover (s)")
        ax.legend(loc="upper right", frameon=False)
        ax.set_ylim(-0.05, 1.15)

        for a in axes:
            a.set_xlim(-PRE_S, POST_S)

        fig.savefig(clip_dir / "fig1_longitudinal_context.png")
        plt.close(fig)
        result["fig1_ok"] = True
    except Exception as e:
        print(f"  [ERR fig1] {clip_dir.name}: {e}")

    # ── Fig 3: Lateral + Jerk ──────────────────────────────────────────
    try:
        n_panels = 4 if jerk_ts is not None else 3
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(5.5, 1.5 * n_panels + 0.8),
                                 sharex=True,
                                 gridspec_kw={"hspace": 0.30})

        # (a) Lane Center Offset
        ax = axes[0]
        if not model_w.empty and "laneLineMeta.leftY" in model_w.columns:
            offset = (model_w["laneLineMeta.leftY"] + model_w["laneLineMeta.rightY"]) / 2.0
            ax.plot(model_w["t_rel"], offset, color=CLR["lane"], lw=1.0, label="Offset")
            ax.fill_between(model_w["t_rel"], 0, offset, color=CLR["lane"], alpha=0.10)
            ax.legend(loc="upper left", frameon=False)
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(0, color=CLR["grey"], lw=0.4, alpha=0.5)
        ax.set_ylabel("Lane Offset (m)")
        # Info box
        jerk_tag = "  [EXTREME JERK]" if extreme_jerk else ""
        ax.text(0.98, 0.95, info + jerk_tag, transform=ax.transAxes,
                fontsize=6, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc",
                          alpha=0.9, lw=0.5))

        # (b) Steering Angle
        ax = axes[1]
        if "steeringAngleDeg" in cs_w.columns:
            ax.plot(cs_w["t_rel"], cs_w["steeringAngleDeg"],
                    color=CLR["steer"], lw=0.9, label="Steer Angle")
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(0, color=CLR["grey"], lw=0.4, alpha=0.5)
        ax.set_ylabel("Steer Angle (\u00b0)")
        ax.legend(loc="upper right", frameon=False)

        # (c) Steering Torque
        ax = axes[2]
        if "steeringTorque" in cs_w.columns:
            ax.plot(cs_w["t_rel"], cs_w["steeringTorque"],
                    color=CLR["steer"], lw=0.9, alpha=0.7, label="Steer Torque")
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(0, color=CLR["grey"], lw=0.4, alpha=0.5)
        ax.set_ylabel("Steer Torque (Nm)")
        ax.legend(loc="upper right", frameon=False)

        # (d) Longitudinal Jerk (if available)
        if jerk_ts is not None:
            ax = axes[3]
            t_j, j_v = jerk_ts
            ax.plot(t_j, j_v, color=CLR["jerk"], lw=0.8, alpha=0.85, label="Jerk")
            ax.axhline(5.0, color=CLR["brake"], lw=0.6, ls="--", alpha=0.7)
            ax.axhline(-5.0, color=CLR["brake"], lw=0.6, ls="--", alpha=0.7,
                       label="\u00b15 m/s\u00b3")
            extreme_mask = np.abs(j_v) > 5.0
            if extreme_mask.any():
                ax.fill_between(t_j, -50, 50, where=extreme_mask,
                                color=CLR["brake"], alpha=0.08)
            ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
            ax.set_ylabel("Jerk (m/s\u00b3)")
            j_lim = min(30, max(10, np.nanpercentile(np.abs(j_v), 99) * 1.3))
            ax.set_ylim(-j_lim, j_lim)
            ax.legend(loc="upper right", frameon=False)

        axes[-1].set_xlabel("Time relative to takeover (s)")
        for a in axes:
            a.set_xlim(-PRE_S, POST_S)

        fig.savefig(clip_dir / "fig3_lateral_steering_jerk.png")
        plt.close(fig)
        result["fig3_ok"] = True
    except Exception as e:
        print(f"  [ERR fig3] {clip_dir.name}: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("═" * 70)
    print("Regenerating fig1 + fig3 for all longtail clips")
    print("═" * 70)

    # Load metrics
    df = pd.read_csv(CSV_PATH)
    print(f"  Metrics CSV: {len(df)} rows")

    # Find all longtail symlinks
    clips = sorted([p for p in LONGTAIL.iterdir() if p.is_symlink()])
    print(f"  Longtail clips: {len(clips)}")

    # Build lookup from clip_dir → row
    metrics_lookup = {}
    for _, row in df.iterrows():
        metrics_lookup[row["clip_dir"]] = row.to_dict()

    # Build work items
    work = []
    for link in clips:
        real = link.resolve()
        key = str(real)
        if key in metrics_lookup:
            work.append((real, metrics_lookup[key]))
        else:
            # Try without trailing slash differences
            for k, v in metrics_lookup.items():
                if Path(k).resolve() == real:
                    work.append((real, v))
                    break

    print(f"  Matched to metrics: {len(work)}")
    print()

    # Process in parallel
    results = []
    if len(work) <= 4:
        for clip_dir, row in work:
            results.append(generate_figures(clip_dir, row))
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(generate_figures, cd, r): cd for cd, r in work}
            done = 0
            for f in as_completed(futures):
                done += 1
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"  [FAIL] {futures[f].name}: {e}")
                if done % 50 == 0:
                    print(f"  ... {done}/{len(work)} done")

    fig1_ok = sum(1 for r in results if r["fig1_ok"])
    fig3_ok = sum(1 for r in results if r["fig3_ok"])
    print(f"\n  Done: fig1 {fig1_ok}/{len(work)}, fig3 {fig3_ok}/{len(work)}")


if __name__ == "__main__":
    main()
