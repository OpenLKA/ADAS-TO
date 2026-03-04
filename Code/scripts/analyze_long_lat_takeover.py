#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_long_lat_takeover.py
============================
Longitudinal & Lateral takeover analysis with TTC, THW, lane offset,
and multi-panel time-series visualizations.

Processes all 15,700+ clips in parallel. For each clip, computes TTC/THW
from radarState, lane-center offset from drivingModelData, and identifies
takeover onset from carState boolean columns.

Outputs:
    long_lat/
        per_clip_ttc_thw.csv               — per-clip TTC/THW at takeover
        fig1_longitudinal_context.pdf/png   — multi-panel time-series
        fig2_ttc_thw_scatter.pdf/png        — TTC vs THW at takeover
        fig3_lateral_trajectory.pdf/png     — lane offset + steering
        fig4_action_sequence.pdf/png        — gantt-style timeline
        summary_report.md                   — text report

Run:
    python3 scripts/analyze_long_lat_takeover.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  Paths & Constants
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
OUT  = CODE / "long_lat"
os.makedirs(OUT, exist_ok=True)

N_WORKERS       = 12
EPS             = 1e-6
PRE_S           = 5.0    # seconds before takeover
POST_S          = 5.0    # seconds after takeover
SMOOTH_WINDOW_S = 0.3
SMOOTH_POLY     = 2
MIN_V_THW       = 0.5   # m/s  — avoid huge THW at near-zero speed
TTC_CAP         = 100.0  # cap TTC at 100 s

# ──────────────────────────────────────────────────────────────────────────────
#  Utilities (adapted from compute_derived_signals.py)
# ──────────────────────────────────────────────────────────────────────────────
def safe_read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read CSV robustly, returning empty DataFrame on failure."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except (ValueError, Exception):
        try:
            df = pd.read_csv(path, low_memory=False)
            if usecols:
                existing = [c for c in usecols if c in df.columns]
                return df[existing] if existing else pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()


def parse_bool_col(series: pd.Series) -> pd.Series:
    """Convert string-encoded booleans to proper bool."""
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def time_window(df: pd.DataFrame, event_t: float,
                before_s: float, after_s: float) -> pd.DataFrame:
    """Slice rows within [event_t - before_s, event_t + after_s]."""
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame()
    lo = event_t - before_s
    hi = event_t + after_s
    mask = (df["time_s"] >= lo) & (df["time_s"] <= hi)
    return df[mask].copy()


def smooth_signal(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Savitzky-Golay smoothing with fallback to rolling mean."""
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


def find_all_clips() -> list[Path]:
    """Find all clip directories containing meta.json."""
    return sorted(p.parent for p in ROOT.rglob("meta.json")
                  if "Code" not in str(p))


_NAN = float("nan")


# ──────────────────────────────────────────────────────────────────────────────
#  Per-clip worker
# ──────────────────────────────────────────────────────────────────────────────
def process_clip(clip_dir: Path) -> dict | None:
    """Extract TTC, THW, lane offset, and takeover timing for one clip."""
    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None

    event_t = meta.get("video_time_s", _NAN)
    if np.isnan(event_t):
        return None

    log_hz = meta.get("log_hz", 20)

    # ── Read CSVs ──────────────────────────────────────────────────────
    cs = safe_read_csv(clip_dir / "carState.csv")
    radar = safe_read_csv(clip_dir / "radarState.csv")
    model = safe_read_csv(clip_dir / "drivingModelData.csv")
    lplan = safe_read_csv(clip_dir / "longitudinalPlan.csv")

    if cs.empty or "time_s" not in cs.columns:
        return None

    # Parse booleans
    for col in ["steeringPressed", "gasPressed", "brakePressed"]:
        if col in cs.columns:
            cs[col] = parse_bool_col(cs[col])

    # ── Detect takeover onsets ─────────────────────────────────────────
    result = {
        "clip_dir":   str(clip_dir),
        "car_model":  meta.get("car_model", ""),
        "dongle_id":  meta.get("dongle_id", ""),
        "route_id":   meta.get("route_id", ""),
        "clip_id":    meta.get("clip_id", ""),
        "event_t":    event_t,
        "log_hz":     log_hz,
    }

    # Find first transition False→True for each action
    for action in ["steeringPressed", "gasPressed", "brakePressed"]:
        if action not in cs.columns:
            result[f"{action}_onset_s"] = _NAN
            continue
        pressed = cs[action].values
        times   = cs["time_s"].values
        onset = _NAN
        for i in range(1, len(pressed)):
            if pressed[i] and not pressed[i - 1]:
                onset = times[i]
                break
        result[f"{action}_onset_s"] = onset

    # Determine first longitudinal and lateral takeover
    long_onsets = []
    for a in ["gasPressed", "brakePressed"]:
        v = result.get(f"{a}_onset_s", _NAN)
        if not np.isnan(v):
            long_onsets.append(v)
    result["long_takeover_s"] = min(long_onsets) if long_onsets else _NAN

    steer_onset = result.get("steeringPressed_onset_s", _NAN)
    result["lat_takeover_s"] = steer_onset

    # First takeover of any kind
    all_onsets = long_onsets[:]
    if not np.isnan(steer_onset):
        all_onsets.append(steer_onset)
    result["first_takeover_s"] = min(all_onsets) if all_onsets else _NAN

    # Primary trigger modality
    if all_onsets:
        ft = min(all_onsets)
        if not np.isnan(steer_onset) and steer_onset == ft:
            result["primary_trigger"] = "Steering"
        elif long_onsets and min(long_onsets) == ft:
            # distinguish brake vs gas
            brake_t = result.get("brakePressed_onset_s", _NAN)
            gas_t   = result.get("gasPressed_onset_s", _NAN)
            if not np.isnan(brake_t) and brake_t == ft:
                result["primary_trigger"] = "Brake"
            else:
                result["primary_trigger"] = "Gas"
        else:
            result["primary_trigger"] = "Unknown"
    else:
        result["primary_trigger"] = "None"

    # ── TTC and THW at takeover ────────────────────────────────────────
    result["ttc_at_takeover"] = _NAN
    result["thw_at_takeover"] = _NAN
    result["dRel_at_takeover"] = _NAN
    result["vEgo_at_takeover"] = _NAN
    result["vLead_at_takeover"] = _NAN
    result["hasLead_at_takeover"] = False

    ft = result["first_takeover_s"]
    if not np.isnan(ft) and not radar.empty and "time_s" in radar.columns:
        # Find radar row closest to takeover
        if "leadOne.dRel" in radar.columns and "leadOne.vLead" in radar.columns:
            idx = (radar["time_s"] - ft).abs().idxmin()
            dRel  = radar.loc[idx, "leadOne.dRel"]
            vLead = radar.loc[idx, "leadOne.vLead"]

            # Get ego speed from carState at same time
            cs_idx = (cs["time_s"] - ft).abs().idxmin()
            vEgo = cs.loc[cs_idx, "vEgo"] if "vEgo" in cs.columns else _NAN

            result["dRel_at_takeover"]  = dRel
            result["vEgo_at_takeover"]  = vEgo
            result["vLead_at_takeover"] = vLead

            # Check hasLead
            has_lead = False
            if not lplan.empty and "hasLead" in lplan.columns and "time_s" in lplan.columns:
                lp_idx = (lplan["time_s"] - ft).abs().idxmin()
                has_lead = parse_bool_col(pd.Series([lplan.loc[lp_idx, "hasLead"]])).iloc[0]
            elif not np.isnan(dRel) and dRel > 0 and dRel < 200:
                has_lead = True
            result["hasLead_at_takeover"] = has_lead

            if has_lead and not np.isnan(dRel) and not np.isnan(vEgo) and not np.isnan(vLead):
                # THW = dRel / vEgo
                if vEgo > MIN_V_THW:
                    result["thw_at_takeover"] = dRel / vEgo
                # TTC = dRel / (vEgo - vLead), only when closing
                closing_speed = vEgo - vLead
                if closing_speed > 0.5:
                    ttc = dRel / closing_speed
                    result["ttc_at_takeover"] = min(ttc, TTC_CAP)

    # ── Lane center offset at takeover ─────────────────────────────────
    result["lane_offset_at_takeover"] = _NAN
    result["lane_width_at_takeover"]  = _NAN
    if not np.isnan(ft) and not model.empty and "time_s" in model.columns:
        if "laneLineMeta.leftY" in model.columns and "laneLineMeta.rightY" in model.columns:
            m_idx = (model["time_s"] - ft).abs().idxmin()
            leftY  = model.loc[m_idx, "laneLineMeta.leftY"]
            rightY = model.loc[m_idx, "laneLineMeta.rightY"]
            if not np.isnan(leftY) and not np.isnan(rightY):
                result["lane_offset_at_takeover"] = (leftY + rightY) / 2.0
                result["lane_width_at_takeover"]  = abs(rightY - leftY)

    # ── Smoothed post-takeover metrics ─────────────────────────────────
    result["jerk_max_post"]       = _NAN
    result["steer_rate_max_post"] = _NAN
    result["peak_decel_post"]     = _NAN

    if not np.isnan(ft):
        cs_post = time_window(cs, ft, 0, POST_S)
        if len(cs_post) > 5:
            t_arr = cs_post["time_s"].values
            sr = log_hz

            if "aEgo" in cs_post.columns:
                a_smooth = smooth_signal(cs_post["aEgo"].values, sr)
                dt = np.diff(t_arr)
                dt[dt < EPS] = EPS
                jerk = np.abs(np.diff(a_smooth) / dt)
                result["jerk_max_post"] = min(float(np.nanmax(jerk)), 50.0)
                result["peak_decel_post"] = float(np.nanmin(a_smooth))

            if "steeringAngleDeg" in cs_post.columns:
                s_smooth = smooth_signal(cs_post["steeringAngleDeg"].values, sr)
                dt = np.diff(t_arr)
                dt[dt < EPS] = EPS
                sr_vals = np.abs(np.diff(s_smooth) / dt)
                result["steer_rate_max_post"] = min(float(np.nanmax(sr_vals)), 500.0)

    # ── Flags ──────────────────────────────────────────────────────────
    result["has_radar"]    = not radar.empty and "leadOne.dRel" in radar.columns
    result["has_model"]    = not model.empty and "laneLineMeta.leftY" in model.columns
    result["has_carState"] = not cs.empty

    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Time-series extraction for individual-clip plots
# ──────────────────────────────────────────────────────────────────────────────
def get_clip_timeseries(clip_dir: Path) -> dict | None:
    """Load full time-series data for a single clip (for plotting)."""
    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    event_t = meta.get("video_time_s", _NAN)
    if np.isnan(event_t):
        return None

    cs    = safe_read_csv(clip_dir / "carState.csv")
    radar = safe_read_csv(clip_dir / "radarState.csv")
    model = safe_read_csv(clip_dir / "drivingModelData.csv")

    if cs.empty:
        return None

    for col in ["steeringPressed", "gasPressed", "brakePressed"]:
        if col in cs.columns:
            cs[col] = parse_bool_col(cs[col])

    # Relative time
    cs["t_rel"] = cs["time_s"] - event_t
    if not radar.empty and "time_s" in radar.columns:
        radar["t_rel"] = radar["time_s"] - event_t
    if not model.empty and "time_s" in model.columns:
        model["t_rel"] = model["time_s"] - event_t

    return {
        "event_t": event_t, "meta": meta,
        "carState": cs, "radarState": radar, "drivingModelData": model,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting Setup (publication style)
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          9,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
})

C = {
    "ego":   "#4C78A8",
    "lead":  "#E45756",
    "steer": "#F58518",
    "brake": "#E45756",
    "gas":   "#54A24B",
    "lane":  "#72B7B2",
    "ttc":   "#FF9DA7",
    "thw":   "#9D755D",
    "grey":  "#888888",
}


def _save(fig, stem: str):
    """Save figure as PDF and PNG."""
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png")


# ──────────────────────────────────────────────────────────────────────────────
#  Figure 1: Longitudinal Takeover Context (exemplar clip)
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig1_longitudinal_context(df: pd.DataFrame, clips: list[Path]):
    """
    Multi-panel time-series [-5s, +5s] around first longitudinal takeover.
    Panels: (a) vEgo vs vLead, (b) dRel, (c) Brake/Gas inputs.
    Uses median-TTC clip as exemplar for time-series, plus aggregate subplot.
    """
    # Find a good exemplar: clip with longitudinal takeover and radar data
    # Pick one near the median TTC
    valid = df.dropna(subset=["long_takeover_s", "ttc_at_takeover"])
    if valid.empty:
        valid = df.dropna(subset=["long_takeover_s"])
    if valid.empty:
        print("  [SKIP] Fig 1: no clips with longitudinal takeover")
        return

    # Pick exemplar near median TTC (or median jerk if no TTC)
    if "ttc_at_takeover" in valid.columns and valid["ttc_at_takeover"].notna().sum() > 0:
        med = valid["ttc_at_takeover"].median()
        exemplar_idx = (valid["ttc_at_takeover"] - med).abs().idxmin()
    else:
        exemplar_idx = valid.index[len(valid) // 2]

    exemplar_dir = Path(valid.loc[exemplar_idx, "clip_dir"])
    ts = get_clip_timeseries(exemplar_dir)
    if ts is None:
        print("  [SKIP] Fig 1: cannot load exemplar time-series")
        return

    cs    = ts["carState"]
    radar = ts["radarState"]
    et    = ts["event_t"]

    # Window
    cs_w    = cs[(cs["t_rel"] >= -PRE_S) & (cs["t_rel"] <= POST_S)].copy()
    radar_w = pd.DataFrame()
    if not radar.empty and "t_rel" in radar.columns:
        radar_w = radar[(radar["t_rel"] >= -PRE_S) & (radar["t_rel"] <= POST_S)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.5), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    # ── Panel (a): Speed ──
    ax = axes[0]
    ax.plot(cs_w["t_rel"], cs_w["vEgo"] * 3.6, color=C["ego"],
            lw=1.2, label="Ego speed")
    if not radar_w.empty and "leadOne.vLead" in radar_w.columns:
        ax.plot(radar_w["t_rel"], radar_w["leadOne.vLead"] * 3.6,
                color=C["lead"], lw=1.2, ls="--", label="Lead speed")
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("(a) Ego vs Lead Speed", fontsize=9, fontweight="bold", loc="left")

    # ── Panel (b): Distance ──
    ax = axes[1]
    if not radar_w.empty and "leadOne.dRel" in radar_w.columns:
        ax.plot(radar_w["t_rel"], radar_w["leadOne.dRel"],
                color=C["lead"], lw=1.2)
        ax.fill_between(radar_w["t_rel"], 0, radar_w["leadOne.dRel"],
                        color=C["lead"], alpha=0.08)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Rel. Distance (m)")
    ax.set_title("(b) Distance to Lead Vehicle", fontsize=9, fontweight="bold", loc="left")

    # ── Panel (c): Pedal inputs ──
    ax = axes[2]
    if "brakePressed" in cs_w.columns:
        ax.fill_between(cs_w["t_rel"], 0, cs_w["brakePressed"].astype(float),
                        color=C["brake"], alpha=0.35, label="Brake", step="post")
    if "gasPressed" in cs_w.columns:
        ax.fill_between(cs_w["t_rel"], 0, cs_w["gasPressed"].astype(float) * 0.7,
                        color=C["gas"], alpha=0.35, label="Gas", step="post")
    if "brake" in cs_w.columns:
        ax.plot(cs_w["t_rel"], cs_w["brake"], color=C["brake"], lw=0.8,
                alpha=0.6, ls="--")
    if "gas" in cs_w.columns:
        ax.plot(cs_w["t_rel"], cs_w["gas"], color=C["gas"], lw=0.8,
                alpha=0.6, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Pedal Input")
    ax.set_xlabel("Time relative to takeover (s)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("(c) Brake / Gas Pedal Inputs", fontsize=9, fontweight="bold", loc="left")
    ax.set_ylim(-0.05, 1.15)

    for a in axes:
        a.set_xlim(-PRE_S, POST_S)

    clip_label = f"{ts['meta'].get('car_model','')}, clip {ts['meta'].get('clip_id','')}"
    fig.suptitle(f"Longitudinal Takeover Context — Exemplar ({clip_label})",
                 fontsize=10, fontweight="bold", y=1.01)
    _save(fig, "fig1_longitudinal_context")


# ──────────────────────────────────────────────────────────────────────────────
#  Figure 2: TTC and THW Distribution at Takeover
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig2_ttc_thw(df: pd.DataFrame):
    """Scatter plot of TTC vs THW at takeover initiation, with marginals."""
    valid = df.dropna(subset=["ttc_at_takeover", "thw_at_takeover"])
    valid = valid[(valid["ttc_at_takeover"] > 0) & (valid["thw_at_takeover"] > 0)]

    if len(valid) < 5:
        print(f"  [SKIP] Fig 2: only {len(valid)} valid TTC-THW pairs")
        return

    # Cap for readability
    ttc = valid["ttc_at_takeover"].clip(upper=30)
    thw = valid["thw_at_takeover"].clip(upper=10)

    fig = plt.figure(figsize=(7.16, 5.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Color by primary trigger
    colors_map = {"Brake": C["brake"], "Gas": C["gas"],
                  "Steering": C["steer"], "Unknown": C["grey"], "None": C["grey"]}
    clrs = valid["primary_trigger"].map(colors_map).fillna(C["grey"])

    ax_main.scatter(ttc, thw, c=clrs, s=12, alpha=0.4, edgecolors="none", rasterized=True)

    # Critical thresholds
    ax_main.axvline(1.5, color=C["ttc"], ls="--", lw=0.8, alpha=0.8, label="TTC = 1.5 s")
    ax_main.axhline(0.8, color=C["thw"], ls="--", lw=0.8, alpha=0.8, label="THW = 0.8 s")
    ax_main.set_xlabel("TTC at Takeover (s)")
    ax_main.set_ylabel("THW at Takeover (s)")
    ax_main.legend(loc="upper right", frameon=False, fontsize=7)

    # Marginal histograms
    ax_top.hist(ttc, bins=60, color=C["ego"], alpha=0.6, edgecolor="white", lw=0.3)
    ax_top.axvline(1.5, color=C["ttc"], ls="--", lw=0.8, alpha=0.8)
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["bottom"].set_visible(False)

    ax_right.hist(thw, bins=60, orientation="horizontal",
                  color=C["ego"], alpha=0.6, edgecolor="white", lw=0.3)
    ax_right.axhline(0.8, color=C["thw"], ls="--", lw=0.8, alpha=0.8)
    ax_right.set_xlabel("Count")
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.spines["left"].set_visible(False)

    # Legend for trigger types
    handles = [mpatches.Patch(color=c, label=l) for l, c in colors_map.items()
               if l in valid["primary_trigger"].values]
    if handles:
        ax_main.legend(handles=handles, loc="upper right", frameon=False,
                       fontsize=7, title="Trigger", title_fontsize=7)

    n_crit = ((valid["ttc_at_takeover"] < 1.5) | (valid["thw_at_takeover"] < 0.8)).sum()
    ax_main.text(0.02, 0.98,
                 f"N = {len(valid):,}  |  TTC<1.5s or THW<0.8s: {n_crit:,} ({100*n_crit/len(valid):.1f}%)",
                 transform=ax_main.transAxes, fontsize=7, va="top", ha="left",
                 color=C["grey"])

    fig.suptitle("TTC and THW at Takeover Initiation", fontsize=10, fontweight="bold")
    _save(fig, "fig2_ttc_thw_scatter")


# ──────────────────────────────────────────────────────────────────────────────
#  Figure 3: Lateral Trajectory & Steering Response
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig3_lateral_trajectory(df: pd.DataFrame, clips: list[Path]):
    """Time-series [-5s,+5s] around lateral takeover: lane offset, steering."""
    valid = df.dropna(subset=["lat_takeover_s"])
    valid = valid[valid["has_model"]]

    if valid.empty:
        print("  [SKIP] Fig 3: no clips with lateral takeover + lane data")
        return

    # Pick exemplar near median lane offset magnitude
    if valid["lane_offset_at_takeover"].notna().sum() > 0:
        med = valid["lane_offset_at_takeover"].abs().median()
        exemplar_idx = (valid["lane_offset_at_takeover"].abs() - med).abs().idxmin()
    else:
        exemplar_idx = valid.index[len(valid) // 2]

    exemplar_dir = Path(valid.loc[exemplar_idx, "clip_dir"])
    ts = get_clip_timeseries(exemplar_dir)
    if ts is None:
        print("  [SKIP] Fig 3: cannot load exemplar time-series")
        return

    cs    = ts["carState"]
    model = ts["drivingModelData"]
    et    = ts["event_t"]

    cs_w = cs[(cs["t_rel"] >= -PRE_S) & (cs["t_rel"] <= POST_S)].copy()
    model_w = pd.DataFrame()
    if not model.empty and "t_rel" in model.columns:
        model_w = model[(model["t_rel"] >= -PRE_S) & (model["t_rel"] <= POST_S)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.5), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    # ── Panel (a): Lane center offset ──
    ax = axes[0]
    if not model_w.empty and "laneLineMeta.leftY" in model_w.columns:
        leftY  = model_w["laneLineMeta.leftY"]
        rightY = model_w["laneLineMeta.rightY"]
        offset = (leftY + rightY) / 2.0
        ax.plot(model_w["t_rel"], offset, color=C["lane"], lw=1.2)
        ax.fill_between(model_w["t_rel"], 0, offset, color=C["lane"], alpha=0.12)
        ax.axhline(0, color=C["grey"], lw=0.5, ls="-", alpha=0.5)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Lane Offset (m)")
    ax.set_title("(a) Lane Center Offset", fontsize=9, fontweight="bold", loc="left")

    # ── Panel (b): Steering angle ──
    ax = axes[1]
    if "steeringAngleDeg" in cs_w.columns:
        ax.plot(cs_w["t_rel"], cs_w["steeringAngleDeg"],
                color=C["steer"], lw=1.0)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0, color=C["grey"], lw=0.5, ls="-", alpha=0.5)
    ax.set_ylabel("Steering Angle (°)")
    ax.set_title("(b) Steering Angle", fontsize=9, fontweight="bold", loc="left")

    # ── Panel (c): Steering torque ──
    ax = axes[2]
    if "steeringTorque" in cs_w.columns:
        ax.plot(cs_w["t_rel"], cs_w["steeringTorque"],
                color=C["steer"], lw=1.0, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0, color=C["grey"], lw=0.5, ls="-", alpha=0.5)
    ax.set_ylabel("Steering Torque (Nm)")
    ax.set_xlabel("Time relative to takeover (s)")
    ax.set_title("(c) Steering Torque", fontsize=9, fontweight="bold", loc="left")

    for a in axes:
        a.set_xlim(-PRE_S, POST_S)

    clip_label = f"{ts['meta'].get('car_model','')}, clip {ts['meta'].get('clip_id','')}"
    fig.suptitle(f"Lateral Takeover Response — Exemplar ({clip_label})",
                 fontsize=10, fontweight="bold", y=1.01)
    _save(fig, "fig3_lateral_trajectory")


# ──────────────────────────────────────────────────────────────────────────────
#  Figure 4: Takeover Action Sequence (Gantt-style)
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig4_action_sequence(df: pd.DataFrame):
    """
    Gantt-style timeline showing activation order of brake, gas, steering
    across all clips (relative to first takeover = 0).
    """
    # Compute relative onsets
    actions = ["brakePressed", "gasPressed", "steeringPressed"]
    labels  = ["Brake", "Gas", "Steering"]
    colors  = [C["brake"], C["gas"], C["steer"]]

    records = []
    for _, row in df.iterrows():
        ft = row["first_takeover_s"]
        if np.isnan(ft):
            continue
        for act, label in zip(actions, labels):
            onset = row.get(f"{act}_onset_s", _NAN)
            if not np.isnan(onset):
                records.append({"action": label, "rel_onset": onset - ft})

    if not records:
        print("  [SKIP] Fig 4: no action sequence data")
        return

    act_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             gridspec_kw={"wspace": 0.35})

    # ── Panel (a): Distribution of relative onset times ──
    ax = axes[0]
    for i, (label, color) in enumerate(zip(labels, colors)):
        subset = act_df[act_df["action"] == label]["rel_onset"]
        subset = subset.clip(-5, 10)  # clip outliers
        if len(subset) > 0:
            ax.hist(subset, bins=80, alpha=0.5, color=color, label=label,
                    edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Time relative to first action (s)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.set_title("(a) Action Onset Distribution", fontsize=9, fontweight="bold", loc="left")

    # ── Panel (b): First action priority (bar chart) ──
    ax = axes[1]
    first_counts = df["primary_trigger"].value_counts()
    trigger_order = ["Steering", "Brake", "Gas"]
    trigger_colors = [C["steer"], C["brake"], C["gas"]]
    counts = [first_counts.get(t, 0) for t in trigger_order]
    total = sum(counts)

    bars = ax.barh(trigger_order, counts, color=trigger_colors, edgecolor="white", lw=0.5)
    for bar, cnt in zip(bars, counts):
        if cnt > 0 and total > 0:
            ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,} ({100*cnt/total:.1f}%)",
                    va="center", fontsize=7, color=C["grey"])
    ax.set_xlabel("Number of Clips")
    ax.set_title("(b) First Action Modality", fontsize=9, fontweight="bold", loc="left")

    fig.suptitle("Takeover Action Sequence", fontsize=10, fontweight="bold", y=1.02)
    _save(fig, "fig4_action_sequence")


# ──────────────────────────────────────────────────────────────────────────────
#  Summary Report
# ──────────────────────────────────────────────────────────────────────────────
def write_report(df: pd.DataFrame):
    """Generate markdown summary report."""
    lines = ["# Longitudinal & Lateral Takeover Analysis Report\n"]

    # Dataset overview
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- **Total clips processed**: {len(df):,}")
    n_long = df["long_takeover_s"].notna().sum()
    n_lat  = df["lat_takeover_s"].notna().sum()
    n_any  = df["first_takeover_s"].notna().sum()
    lines.append(f"- **Clips with longitudinal takeover** (brake/gas): {n_long:,} ({100*n_long/len(df):.1f}%)")
    lines.append(f"- **Clips with lateral takeover** (steering): {n_lat:,} ({100*n_lat/len(df):.1f}%)")
    lines.append(f"- **Clips with any takeover**: {n_any:,} ({100*n_any/len(df):.1f}%)")
    lines.append(f"- **Clips with radar data**: {df['has_radar'].sum():,}")
    lines.append(f"- **Clips with lane model**: {df['has_model'].sum():,}")
    lines.append("")

    # Primary trigger
    lines.append("### Primary Trigger Distribution\n")
    lines.append("| Trigger | Count | % |")
    lines.append("|---------|------:|--:|")
    for trig, cnt in df["primary_trigger"].value_counts().items():
        lines.append(f"| {trig} | {cnt:,} | {100*cnt/len(df):.1f}% |")
    lines.append("")

    # TTC / THW
    lines.append("## 2. TTC and THW at Takeover\n")
    ttc_valid = df["ttc_at_takeover"].dropna()
    thw_valid = df["thw_at_takeover"].dropna()
    lines.append(f"- **TTC available**: {len(ttc_valid):,} clips (requires lead vehicle with closing speed)")
    if len(ttc_valid) > 0:
        lines.append(f"- **TTC median**: {ttc_valid.median():.2f} s (IQR {ttc_valid.quantile(0.25):.2f}–{ttc_valid.quantile(0.75):.2f})")
        lines.append(f"- **TTC < 1.5 s (critical)**: {(ttc_valid < 1.5).sum():,} ({100*(ttc_valid < 1.5).mean():.1f}%)")
        lines.append(f"- **TTC < 3.0 s**: {(ttc_valid < 3.0).sum():,} ({100*(ttc_valid < 3.0).mean():.1f}%)")
    lines.append(f"- **THW available**: {len(thw_valid):,} clips")
    if len(thw_valid) > 0:
        lines.append(f"- **THW median**: {thw_valid.median():.2f} s (IQR {thw_valid.quantile(0.25):.2f}–{thw_valid.quantile(0.75):.2f})")
        lines.append(f"- **THW < 0.8 s (critical)**: {(thw_valid < 0.8).sum():,} ({100*(thw_valid < 0.8).mean():.1f}%)")
    lines.append("")

    # Lane offset
    lines.append("## 3. Lane Center Offset at Takeover\n")
    lo_valid = df["lane_offset_at_takeover"].dropna()
    lw_valid = df["lane_width_at_takeover"].dropna()
    lines.append(f"- **Lane offset available**: {len(lo_valid):,} clips")
    if len(lo_valid) > 0:
        lines.append(f"- **Median |offset|**: {lo_valid.abs().median():.3f} m")
        lines.append(f"- **IQR**: {lo_valid.abs().quantile(0.25):.3f}–{lo_valid.abs().quantile(0.75):.3f} m")
        lines.append(f"- **P95 |offset|**: {lo_valid.abs().quantile(0.95):.3f} m")
    if len(lw_valid) > 0:
        lines.append(f"- **Median lane width**: {lw_valid.median():.2f} m")
    lines.append("")

    # Post-takeover metrics
    lines.append("## 4. Post-Takeover Stability\n")
    lines.append("| Metric | Median | IQR (Q1–Q3) | P5 | P95 |")
    lines.append("|--------|-------:|:-----------:|---:|----:|")
    for col, label in [("jerk_max_post", "Jerk max (m/s³)"),
                        ("steer_rate_max_post", "Steer rate max (°/s)"),
                        ("peak_decel_post", "Peak decel (m/s²)")]:
        v = df[col].dropna()
        if len(v) > 0:
            lines.append(f"| {label} | {v.median():.3f} | "
                         f"{v.quantile(0.25):.3f}–{v.quantile(0.75):.3f} | "
                         f"{v.quantile(0.05):.3f} | {v.quantile(0.95):.3f} |")
    lines.append("")

    # Limitations
    lines.append("## 5. Limitations\n")
    lines.append("1. **TTC/THW availability**: TTC requires a closing scenario (vEgo > vLead) with "
                 "radar-detected lead. Many clips have no lead vehicle or are not closing.")
    lines.append("2. **Lane model coverage**: drivingModelData is missing for ~35% of clips.")
    lines.append("3. **Exemplar figures**: Time-series figures (Figs 1, 3) show single exemplar clips "
                 "selected near the population median; they illustrate typical behavior but are not "
                 "representative of all clips.")
    lines.append("4. **vLead interpretation**: leadOne.vLead is assumed to be absolute speed. "
                 "If it is relative speed, TTC computation would need adjustment.")
    lines.append("")

    report_path = OUT / "summary_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved summary_report.md")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Longitudinal & Lateral Takeover Analysis")
    print("=" * 70)

    # ── Discover clips ─────────────────────────────────────────────────
    clips = find_all_clips()
    print(f"\nFound {len(clips):,} clips")

    # ── Parallel processing ────────────────────────────────────────────
    print(f"\nProcessing clips with {N_WORKERS} workers...")
    results = []
    failed  = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_clip, c): c for c in clips}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(clips):,} clips processed...")
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
                else:
                    failed += 1
            except Exception:
                failed += 1

    print(f"\nSuccessful: {len(results):,}  |  Failed: {failed:,}")

    df = pd.DataFrame(results)

    # Save per-clip CSV
    csv_path = OUT / "per_clip_ttc_thw.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ── Generate figures ───────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_fig1_longitudinal_context(df, clips)
    plot_fig2_ttc_thw(df)
    plot_fig3_lateral_trajectory(df, clips)
    plot_fig4_action_sequence(df)

    # ── Summary report ─────────────────────────────────────────────────
    print("\nWriting report...")
    write_report(df)

    print(f"\nAll outputs saved to: {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
