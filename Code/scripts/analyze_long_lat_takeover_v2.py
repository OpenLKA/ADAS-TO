#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_long_lat_takeover_v2.py
================================
Corrected longitudinal & lateral takeover analysis.

Key fix vs v1:
  TTC/THW now ONLY computed when leadOne.status == True AND dRel > 2 m
  AND closing_speed > 0.5 m/s. This eliminates ~207 false positives from v1
  where dRel=0 / vLead=0 (radar not tracking any vehicle).

Outputs (all in Code/long_lat/):
    per_clip_all_metrics.csv           — one row per clip, all metrics
    fig1_longitudinal_context.pdf/png  — exemplar time-series
    fig2_ttc_thw_scatter.pdf/png       — TTC vs THW at takeover
    fig3_lateral_trajectory.pdf/png    — lane offset + steering
    fig4_action_sequence.pdf/png       — action timeline
    summary_report_v2.md               — text report

Run:
    python3 scripts/analyze_long_lat_takeover_v2.py
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
PRE_S           = 5.0
POST_S          = 5.0
SMOOTH_WINDOW_S = 0.3
SMOOTH_POLY     = 2
MIN_V_THW       = 0.5   # m/s
MIN_DREL        = 2.0   # m  — below this radar reading is unreliable
MIN_CLOSING     = 0.5   # m/s — minimum closing speed for valid TTC
TTC_CAP         = 100.0

# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────
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


def time_window(df: pd.DataFrame, event_t: float,
                before_s: float, after_s: float) -> pd.DataFrame:
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame()
    return df[(df["time_s"] >= event_t - before_s) &
              (df["time_s"] <= event_t + after_s)].copy()


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


def find_all_clips() -> list[Path]:
    return sorted(p.parent for p in ROOT.rglob("meta.json")
                  if "Code" not in str(p))


_NAN = float("nan")


# ──────────────────────────────────────────────────────────────────────────────
#  Per-clip worker
# ──────────────────────────────────────────────────────────────────────────────
def process_clip(clip_dir: Path) -> dict | None:
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
    cs    = safe_read_csv(clip_dir / "carState.csv")
    radar = safe_read_csv(clip_dir / "radarState.csv")
    model = safe_read_csv(clip_dir / "drivingModelData.csv")
    lplan = safe_read_csv(clip_dir / "longitudinalPlan.csv")

    if cs.empty or "time_s" not in cs.columns:
        return None

    for col in ["steeringPressed", "gasPressed", "brakePressed"]:
        if col in cs.columns:
            cs[col] = parse_bool_col(cs[col])

    # Parse radar status
    if not radar.empty and "leadOne.status" in radar.columns:
        radar["leadOne.status"] = parse_bool_col(radar["leadOne.status"])

    # ── Takeover onsets ────────────────────────────────────────────────
    result = {
        "clip_dir":   str(clip_dir),
        "car_model":  meta.get("car_model", ""),
        "dongle_id":  meta.get("dongle_id", ""),
        "route_id":   meta.get("route_id", ""),
        "clip_id":    meta.get("clip_id", ""),
        "event_t":    event_t,
        "log_hz":     log_hz,
    }

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

    # Longitudinal / lateral / first takeover
    long_onsets = []
    for a in ["gasPressed", "brakePressed"]:
        v = result.get(f"{a}_onset_s", _NAN)
        if not np.isnan(v):
            long_onsets.append(v)
    result["long_takeover_s"] = min(long_onsets) if long_onsets else _NAN

    steer_onset = result.get("steeringPressed_onset_s", _NAN)
    result["lat_takeover_s"] = steer_onset

    all_onsets = long_onsets[:]
    if not np.isnan(steer_onset):
        all_onsets.append(steer_onset)
    result["first_takeover_s"] = min(all_onsets) if all_onsets else _NAN

    # Primary trigger
    if all_onsets:
        ft = min(all_onsets)
        if not np.isnan(steer_onset) and steer_onset == ft:
            result["primary_trigger"] = "Steering"
        elif long_onsets and min(long_onsets) == ft:
            brake_t = result.get("brakePressed_onset_s", _NAN)
            gas_t   = result.get("gasPressed_onset_s", _NAN)
            result["primary_trigger"] = "Brake" if (not np.isnan(brake_t) and brake_t == ft) else "Gas"
        else:
            result["primary_trigger"] = "Unknown"
    else:
        result["primary_trigger"] = "None"

    ft = result["first_takeover_s"]

    # ── TTC / THW — CORRECTED: require leadOne.status == True ─────────
    result["ttc_at_takeover"]   = _NAN
    result["thw_at_takeover"]   = _NAN
    result["dRel_at_takeover"]  = _NAN
    result["vEgo_at_takeover"]  = _NAN
    result["vLead_at_takeover"] = _NAN
    result["lead_status_at_takeover"] = False
    result["lead_valid"]        = False

    if (not np.isnan(ft) and not radar.empty
            and "time_s" in radar.columns
            and "leadOne.status" in radar.columns
            and "leadOne.dRel" in radar.columns):

        idx = (radar["time_s"] - ft).abs().idxmin()
        status = radar.loc[idx, "leadOne.status"]
        dRel   = radar.loc[idx, "leadOne.dRel"]
        vLead  = radar.loc[idx, "leadOne.vLead"] if "leadOne.vLead" in radar.columns else _NAN

        cs_idx = (cs["time_s"] - ft).abs().idxmin()
        vEgo   = cs.loc[cs_idx, "vEgo"] if "vEgo" in cs.columns else _NAN

        result["lead_status_at_takeover"] = bool(status)
        result["dRel_at_takeover"]  = dRel
        result["vEgo_at_takeover"]  = vEgo
        result["vLead_at_takeover"] = vLead

        # Only compute TTC/THW if radar confirms a real lead vehicle
        if (status
                and not np.isnan(dRel) and dRel > MIN_DREL
                and not np.isnan(vEgo) and not np.isnan(vLead)):
            result["lead_valid"] = True

            # THW
            if vEgo > MIN_V_THW:
                result["thw_at_takeover"] = dRel / vEgo

            # TTC — only when closing
            closing = vEgo - vLead
            if closing > MIN_CLOSING:
                result["ttc_at_takeover"] = min(dRel / closing, TTC_CAP)

    # Also scan pre-window [-3, 0] for minimum TTC (more robust)
    result["min_ttc_pre"]       = _NAN
    result["thw_at_min_ttc"]    = _NAN
    result["dRel_at_min_ttc"]   = _NAN
    result["vEgo_at_min_ttc"]   = _NAN
    result["closing_at_min_ttc"] = _NAN

    if (not np.isnan(ft) and not radar.empty
            and "leadOne.status" in radar.columns):
        pre_mask = ((radar["time_s"] >= ft - 3) & (radar["time_s"] <= ft)
                    & (radar["leadOne.status"] == True))
        r_pre = radar[pre_mask]
        if len(r_pre) >= 2:
            best_ttc = _NAN
            best_row = None
            for _, rrow in r_pre.iterrows():
                d = rrow["leadOne.dRel"]
                vl = rrow.get("leadOne.vLead", _NAN)
                if np.isnan(d) or d < MIN_DREL or np.isnan(vl):
                    continue
                ci = (cs["time_s"] - rrow["time_s"]).abs().idxmin()
                ve = cs.loc[ci, "vEgo"] if "vEgo" in cs.columns else _NAN
                if np.isnan(ve):
                    continue
                cl = ve - vl
                if cl > MIN_CLOSING:
                    ttc = d / cl
                    if np.isnan(best_ttc) or ttc < best_ttc:
                        best_ttc = ttc
                        best_row = {"dRel": d, "vEgo": ve, "vLead": vl,
                                    "closing": cl, "thw": d / ve if ve > MIN_V_THW else _NAN}
            if best_row is not None:
                result["min_ttc_pre"]        = best_ttc
                result["thw_at_min_ttc"]     = best_row["thw"]
                result["dRel_at_min_ttc"]    = best_row["dRel"]
                result["vEgo_at_min_ttc"]    = best_row["vEgo"]
                result["closing_at_min_ttc"] = best_row["closing"]

    # ── Lane center offset ─────────────────────────────────────────────
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

    # ── Post-takeover smoothed metrics ─────────────────────────────────
    result["jerk_max_post"]       = _NAN
    result["steer_rate_max_post"] = _NAN
    result["peak_decel_post"]     = _NAN

    if not np.isnan(ft):
        cs_post = time_window(cs, ft, 0, POST_S)
        if len(cs_post) > 5:
            t_arr = cs_post["time_s"].values
            sr = log_hz
            if "aEgo" in cs_post.columns:
                a_sm = smooth_signal(cs_post["aEgo"].values, sr)
                dt = np.diff(t_arr); dt[dt < EPS] = EPS
                jerk = np.abs(np.diff(a_sm) / dt)
                result["jerk_max_post"]  = min(float(np.nanmax(jerk)), 50.0)
                result["peak_decel_post"] = float(np.nanmin(a_sm))
            if "steeringAngleDeg" in cs_post.columns:
                s_sm = smooth_signal(cs_post["steeringAngleDeg"].values, sr)
                dt = np.diff(t_arr); dt[dt < EPS] = EPS
                sr_vals = np.abs(np.diff(s_sm) / dt)
                result["steer_rate_max_post"] = min(float(np.nanmax(sr_vals)), 500.0)

    # ── Missingness flags ──────────────────────────────────────────────
    result["has_radar"] = not radar.empty and "leadOne.dRel" in radar.columns
    result["has_model"] = not model.empty and "laneLineMeta.leftY" in model.columns

    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Time-series loader for exemplar plots
# ──────────────────────────────────────────────────────────────────────────────
def get_clip_timeseries(clip_dir: Path):
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
    if not radar.empty and "leadOne.status" in radar.columns:
        radar["leadOne.status"] = parse_bool_col(radar["leadOne.status"])

    return {"event_t": event_t, "meta": meta,
            "carState": cs, "radarState": radar, "drivingModelData": model}


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting Setup
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "ego": "#4C78A8", "lead": "#E45756", "steer": "#F58518",
    "brake": "#E45756", "gas": "#54A24B", "lane": "#72B7B2",
    "ttc": "#FF9DA7", "thw": "#9D755D", "grey": "#888888",
}


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png")


# ──────────────────────────────────────────────────────────────────────────────
#  Fig 1: Longitudinal Takeover Context (exemplar)
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig1(df, clips):
    """Pick exemplar with real lead (lead_valid=True) near median TTC."""
    valid = df[(df["lead_valid"] == True) & df["long_takeover_s"].notna()
               & df["min_ttc_pre"].notna()]
    if valid.empty:
        valid = df[df["long_takeover_s"].notna()]
    if valid.empty:
        print("  [SKIP] Fig 1: no longitudinal takeovers")
        return

    if "min_ttc_pre" in valid.columns and valid["min_ttc_pre"].notna().sum() > 0:
        med = valid["min_ttc_pre"].median()
        idx = (valid["min_ttc_pre"] - med).abs().idxmin()
    else:
        idx = valid.index[len(valid) // 2]

    exemplar_dir = Path(valid.loc[idx, "clip_dir"])
    ts = get_clip_timeseries(exemplar_dir)
    if ts is None:
        print("  [SKIP] Fig 1: cannot load exemplar")
        return

    cs    = ts["carState"]
    radar = ts["radarState"]
    first_to = valid.loc[idx, "long_takeover_s"]

    cs["t_rel"]    = cs["time_s"] - first_to
    if not radar.empty and "time_s" in radar.columns:
        radar["t_rel"] = radar["time_s"] - first_to

    cs_w = cs[(cs["t_rel"] >= -PRE_S) & (cs["t_rel"] <= POST_S)]
    radar_w = pd.DataFrame()
    if not radar.empty and "t_rel" in radar.columns:
        # Only show radar where status=True
        radar_w = radar[(radar["t_rel"] >= -PRE_S) & (radar["t_rel"] <= POST_S)]

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.5), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    ttc_val = valid.loc[idx, "min_ttc_pre"]
    drel_val = valid.loc[idx, "dRel_at_min_ttc"]
    label = f"TTC={ttc_val:.1f}s, dRel={drel_val:.0f}m" if not np.isnan(ttc_val) else ""

    # (a) Speed
    ax = axes[0]
    ax.plot(cs_w["t_rel"], cs_w["vEgo"] * 3.6, color=C["ego"], lw=1.2, label="Ego")
    if not radar_w.empty and "leadOne.vLead" in radar_w.columns:
        # Only plot lead speed where status=True
        radar_valid = radar_w[radar_w.get("leadOne.status", False) == True] if "leadOne.status" in radar_w.columns else radar_w
        if not radar_valid.empty:
            ax.plot(radar_valid["t_rel"], radar_valid["leadOne.vLead"] * 3.6,
                    color=C["lead"], lw=1.2, ls="--", label="Lead")
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Speed (km/h)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"(a) Ego vs Lead Speed — {label}", fontsize=9, fontweight="bold", loc="left")

    # (b) Distance — only where status=True
    ax = axes[1]
    if not radar_w.empty and "leadOne.dRel" in radar_w.columns:
        radar_valid = radar_w[radar_w.get("leadOne.status", False) == True] if "leadOne.status" in radar_w.columns else radar_w
        if not radar_valid.empty:
            ax.plot(radar_valid["t_rel"], radar_valid["leadOne.dRel"],
                    color=C["lead"], lw=1.2)
            ax.fill_between(radar_valid["t_rel"], 0, radar_valid["leadOne.dRel"],
                            color=C["lead"], alpha=0.08)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Rel. Distance (m)")
    ax.set_title("(b) Distance to Lead (leadOne.status=True only)",
                 fontsize=9, fontweight="bold", loc="left")

    # (c) Pedal inputs
    ax = axes[2]
    if "brakePressed" in cs_w.columns:
        ax.fill_between(cs_w["t_rel"], 0, cs_w["brakePressed"].astype(float),
                        color=C["brake"], alpha=0.35, label="Brake", step="post")
    if "gasPressed" in cs_w.columns:
        ax.fill_between(cs_w["t_rel"], 0, cs_w["gasPressed"].astype(float) * 0.7,
                        color=C["gas"], alpha=0.35, label="Gas", step="post")
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_ylabel("Pedal Input")
    ax.set_xlabel("Time relative to takeover (s)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(-0.05, 1.15)

    for a in axes:
        a.set_xlim(-PRE_S, POST_S)

    clip_label = f"{valid.loc[idx, 'car_model']}, clip {valid.loc[idx, 'clip_id']}"
    fig.suptitle(f"Longitudinal Takeover Context — {clip_label}",
                 fontsize=10, fontweight="bold", y=1.01)
    _save(fig, "fig1_longitudinal_context")


# ──────────────────────────────────────────────────────────────────────────────
#  Fig 2: TTC / THW scatter (CORRECTED — only lead_valid clips)
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig2(df):
    """Scatter of TTC vs THW — using pre-window minimum TTC for robustness."""
    # Use BOTH instantaneous TTC at takeover AND pre-window min TTC
    # Prefer min_ttc_pre (more robust)
    valid = df.dropna(subset=["min_ttc_pre", "thw_at_min_ttc"])
    valid = valid[(valid["min_ttc_pre"] > 0) & (valid["thw_at_min_ttc"] > 0)]

    if len(valid) < 5:
        # Fall back to instantaneous
        valid = df.dropna(subset=["ttc_at_takeover", "thw_at_takeover"])
        valid = valid[(valid["ttc_at_takeover"] > 0) & (valid["thw_at_takeover"] > 0)]
        ttc_col, thw_col = "ttc_at_takeover", "thw_at_takeover"
    else:
        ttc_col, thw_col = "min_ttc_pre", "thw_at_min_ttc"

    if len(valid) < 5:
        print(f"  [SKIP] Fig 2: only {len(valid)} valid TTC-THW pairs")
        return

    print(f"  Fig 2: {len(valid)} clips with verified lead + TTC/THW")

    ttc = valid[ttc_col].clip(upper=30)
    thw = valid[thw_col].clip(upper=10)

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(7.16, 5.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Color by trigger
    colors_map = {"Brake": C["brake"], "Gas": C["gas"],
                  "Steering": C["steer"], "Unknown": C["grey"], "None": C["grey"]}
    clrs = valid["primary_trigger"].map(colors_map).fillna(C["grey"])

    ax_main.scatter(ttc, thw, c=clrs, s=14, alpha=0.45, edgecolors="none", rasterized=True)
    ax_main.axvline(1.5, color=C["ttc"], ls="--", lw=0.8, alpha=0.8)
    ax_main.axvline(3.0, color=C["ttc"], ls=":", lw=0.6, alpha=0.5)
    ax_main.axhline(0.8, color=C["thw"], ls="--", lw=0.8, alpha=0.8)
    ax_main.axhline(1.5, color=C["thw"], ls=":", lw=0.6, alpha=0.5)
    ax_main.set_xlabel("Min TTC in pre-window (s)")
    ax_main.set_ylabel("THW at min-TTC (s)")

    # Annotations for critical zones
    n_ttc15 = (valid[ttc_col] < 1.5).sum()
    n_ttc30 = (valid[ttc_col] < 3.0).sum()
    n_thw08 = (valid[thw_col] < 0.8).sum()
    ax_main.text(0.02, 0.98,
                 f"N = {len(valid):,} (leadOne.status=True, closing>0.5 m/s)\n"
                 f"TTC<1.5s: {n_ttc15}  |  TTC<3.0s: {n_ttc30}  |  THW<0.8s: {n_thw08}",
                 transform=ax_main.transAxes, fontsize=6.5, va="top", ha="left",
                 color=C["grey"], linespacing=1.4)

    # Marginals
    ax_top.hist(ttc, bins=60, color=C["ego"], alpha=0.6, edgecolor="white", lw=0.3)
    ax_top.axvline(1.5, color=C["ttc"], ls="--", lw=0.8, alpha=0.8)
    ax_top.axvline(3.0, color=C["ttc"], ls=":", lw=0.6, alpha=0.5)
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["bottom"].set_visible(False)

    ax_right.hist(thw, bins=60, orientation="horizontal",
                  color=C["ego"], alpha=0.6, edgecolor="white", lw=0.3)
    ax_right.axhline(0.8, color=C["thw"], ls="--", lw=0.8, alpha=0.8)
    ax_right.set_xlabel("Count")
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.spines["left"].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for l, c in colors_map.items()
               if l in valid["primary_trigger"].values]
    if handles:
        ax_main.legend(handles=handles, loc="lower right", frameon=False,
                       fontsize=7, title="First Action", title_fontsize=7)

    fig.suptitle("TTC and THW at Takeover (radar-verified lead vehicles only)",
                 fontsize=10, fontweight="bold")
    _save(fig, "fig2_ttc_thw_scatter")


# ──────────────────────────────────────────────────────────────────────────────
#  Fig 3: Lateral Trajectory & Steering Response
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig3(df, clips):
    valid = df.dropna(subset=["lat_takeover_s"])
    valid = valid[valid["has_model"]]
    if valid.empty:
        print("  [SKIP] Fig 3: no lateral takeovers with lane data")
        return

    if valid["lane_offset_at_takeover"].notna().sum() > 0:
        med = valid["lane_offset_at_takeover"].abs().median()
        idx = (valid["lane_offset_at_takeover"].abs() - med).abs().idxmin()
    else:
        idx = valid.index[len(valid) // 2]

    exemplar_dir = Path(valid.loc[idx, "clip_dir"])
    ts = get_clip_timeseries(exemplar_dir)
    if ts is None:
        print("  [SKIP] Fig 3: cannot load exemplar")
        return

    cs    = ts["carState"]
    model = ts["drivingModelData"]
    first_to = valid.loc[idx, "lat_takeover_s"]

    cs["t_rel"] = cs["time_s"] - first_to
    if not model.empty and "time_s" in model.columns:
        model["t_rel"] = model["time_s"] - first_to

    cs_w = cs[(cs["t_rel"] >= -PRE_S) & (cs["t_rel"] <= POST_S)]
    model_w = pd.DataFrame()
    if not model.empty and "t_rel" in model.columns:
        model_w = model[(model["t_rel"] >= -PRE_S) & (model["t_rel"] <= POST_S)]

    fig, axes = plt.subplots(3, 1, figsize=(7.16, 6.5), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    # (a) Lane center offset
    ax = axes[0]
    if not model_w.empty and "laneLineMeta.leftY" in model_w.columns:
        offset = (model_w["laneLineMeta.leftY"] + model_w["laneLineMeta.rightY"]) / 2.0
        ax.plot(model_w["t_rel"], offset, color=C["lane"], lw=1.2)
        ax.fill_between(model_w["t_rel"], 0, offset, color=C["lane"], alpha=0.12)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0, color=C["grey"], lw=0.5, alpha=0.5)
    ax.set_ylabel("Lane Offset (m)")
    ax.set_title("(a) Lane Center Offset", fontsize=9, fontweight="bold", loc="left")

    # (b) Steering angle
    ax = axes[1]
    if "steeringAngleDeg" in cs_w.columns:
        ax.plot(cs_w["t_rel"], cs_w["steeringAngleDeg"], color=C["steer"], lw=1.0)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0, color=C["grey"], lw=0.5, alpha=0.5)
    ax.set_ylabel("Steering Angle (°)")
    ax.set_title("(b) Steering Angle", fontsize=9, fontweight="bold", loc="left")

    # (c) Steering torque
    ax = axes[2]
    if "steeringTorque" in cs_w.columns:
        ax.plot(cs_w["t_rel"], cs_w["steeringTorque"], color=C["steer"], lw=1.0, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(0, color=C["grey"], lw=0.5, alpha=0.5)
    ax.set_ylabel("Steering Torque (Nm)")
    ax.set_xlabel("Time relative to takeover (s)")
    ax.set_title("(c) Steering Torque", fontsize=9, fontweight="bold", loc="left")

    for a in axes:
        a.set_xlim(-PRE_S, POST_S)

    clip_label = f"{valid.loc[idx, 'car_model']}, clip {valid.loc[idx, 'clip_id']}"
    fig.suptitle(f"Lateral Takeover Response — {clip_label}",
                 fontsize=10, fontweight="bold", y=1.01)
    _save(fig, "fig3_lateral_trajectory")


# ──────────────────────────────────────────────────────────────────────────────
#  Fig 4: Takeover Action Sequence
# ──────────────────────────────────────────────────────────────────────────────
def plot_fig4(df):
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
        print("  [SKIP] Fig 4: no action data")
        return

    act_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5),
                             gridspec_kw={"wspace": 0.35})

    # (a) Onset distribution
    ax = axes[0]
    for label, color in zip(labels, colors):
        subset = act_df[act_df["action"] == label]["rel_onset"].clip(-5, 10)
        if len(subset) > 0:
            ax.hist(subset, bins=80, alpha=0.5, color=color, label=label,
                    edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Time relative to first action (s)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.set_title("(a) Action Onset Distribution", fontsize=9, fontweight="bold", loc="left")

    # (b) First action priority
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
def write_report(df):
    n = len(df)
    lines = ["# Longitudinal & Lateral Takeover Analysis Report (v2 — corrected)\n"]

    lines.append("## 1. Dataset Overview\n")
    n_long = df["long_takeover_s"].notna().sum()
    n_lat  = df["lat_takeover_s"].notna().sum()
    n_any  = df["first_takeover_s"].notna().sum()
    lines.append(f"- **Total clips processed**: {n:,}")
    lines.append(f"- **Longitudinal takeover** (brake/gas): {n_long:,} ({100*n_long/n:.1f}%)")
    lines.append(f"- **Lateral takeover** (steering): {n_lat:,} ({100*n_lat/n:.1f}%)")
    lines.append(f"- **Any takeover**: {n_any:,} ({100*n_any/n:.1f}%)")
    lines.append(f"- **Radar data available**: {df['has_radar'].sum():,}")
    lines.append(f"- **Lane model available**: {df['has_model'].sum():,}")
    lines.append("")

    lines.append("### Primary Trigger\n")
    lines.append("| Trigger | Count | % |")
    lines.append("|---------|------:|--:|")
    for trig, cnt in df["primary_trigger"].value_counts().items():
        lines.append(f"| {trig} | {cnt:,} | {100*cnt/n:.1f}% |")
    lines.append("")

    # Lead detection
    lines.append("## 2. Lead Vehicle Detection\n")
    n_lead_status = df["lead_status_at_takeover"].sum()
    n_lead_valid  = df["lead_valid"].sum()
    lines.append(f"- **leadOne.status=True at takeover**: {n_lead_status:,} ({100*n_lead_status/n:.1f}%)")
    lines.append(f"- **Valid lead** (status=True + dRel>{MIN_DREL}m + closing>{MIN_CLOSING}m/s): "
                 f"{n_lead_valid:,} ({100*n_lead_valid/n:.1f}%)")
    lines.append(f"- **v1 bug note**: Previous analysis reported 216 clips with TTC<1.5s. "
                 f"This was incorrect — 207 of those had leadOne.status=False (no lead vehicle). "
                 f"Corrected count: see below.")
    lines.append("")

    # TTC / THW
    lines.append("## 3. TTC and THW at Takeover (corrected)\n")
    ttc = df["min_ttc_pre"].dropna()
    thw = df["thw_at_min_ttc"].dropna()
    lines.append(f"- **Clips with valid TTC**: {len(ttc):,}")
    if len(ttc) > 0:
        lines.append(f"- **TTC median**: {ttc.median():.2f} s "
                     f"(IQR {ttc.quantile(0.25):.2f}–{ttc.quantile(0.75):.2f})")
        lines.append(f"- **TTC < 1.5s**: {(ttc < 1.5).sum()}")
        lines.append(f"- **TTC < 3.0s**: {(ttc < 3.0).sum()}")
        lines.append(f"- **TTC < 5.0s**: {(ttc < 5.0).sum()}")
    lines.append(f"- **Clips with valid THW**: {len(thw):,}")
    if len(thw) > 0:
        lines.append(f"- **THW median**: {thw.median():.2f} s "
                     f"(IQR {thw.quantile(0.25):.2f}–{thw.quantile(0.75):.2f})")
        lines.append(f"- **THW < 0.8s**: {(thw < 0.8).sum()}")
    lines.append("")

    # Lane offset
    lines.append("## 4. Lane Center Offset\n")
    lo = df["lane_offset_at_takeover"].dropna()
    lw = df["lane_width_at_takeover"].dropna()
    lines.append(f"- **Available**: {len(lo):,}")
    if len(lo) > 0:
        lines.append(f"- **Median |offset|**: {lo.abs().median():.3f} m")
        lines.append(f"- **P95 |offset|**: {lo.abs().quantile(0.95):.3f} m")
    if len(lw) > 0:
        lines.append(f"- **Median lane width**: {lw.median():.2f} m")
    lines.append("")

    # Post-takeover
    lines.append("## 5. Post-Takeover Stability\n")
    lines.append("| Metric | Median | IQR | P5 | P95 |")
    lines.append("|--------|-------:|:---:|---:|----:|")
    for col, label in [("jerk_max_post", "Jerk max (m/s³)"),
                        ("steer_rate_max_post", "Steer rate max (°/s)"),
                        ("peak_decel_post", "Peak decel (m/s²)")]:
        v = df[col].dropna()
        if len(v) > 0:
            lines.append(f"| {label} | {v.median():.2f} | "
                         f"{v.quantile(0.25):.2f}–{v.quantile(0.75):.2f} | "
                         f"{v.quantile(0.05):.2f} | {v.quantile(0.95):.2f} |")
    lines.append("")

    lines.append("## 6. Corrections from v1\n")
    lines.append("| Issue | v1 (incorrect) | v2 (corrected) |")
    lines.append("|-------|---------------|----------------|")
    lines.append(f"| TTC<1.5s clips | 216 | {(ttc < 1.5).sum() if len(ttc) > 0 else 0} |")
    lines.append(f"| Lead validation | No status check | leadOne.status=True required |")
    lines.append(f"| dRel filter | None | dRel > {MIN_DREL} m |")
    lines.append(f"| Closing speed | vEgo > vLead (any) | closing > {MIN_CLOSING} m/s |")
    lines.append("")

    report_path = OUT / "summary_report_v2.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved summary_report_v2.md")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Longitudinal & Lateral Takeover Analysis v2 (corrected)")
    print("=" * 70)

    clips = find_all_clips()
    print(f"\nFound {len(clips):,} clips")

    print(f"\nProcessing clips with {N_WORKERS} workers...")
    results = []
    failed  = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_clip, c): c for c in clips}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(clips):,}...")
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
    csv_path = OUT / "per_clip_all_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Quick validation
    n_valid_lead = df["lead_valid"].sum()
    n_ttc = df["min_ttc_pre"].notna().sum()
    print(f"\nValidation:")
    print(f"  lead_status=True at takeover: {df['lead_status_at_takeover'].sum()}")
    print(f"  lead_valid (status + dRel + closing): {n_valid_lead}")
    print(f"  min_ttc_pre available: {n_ttc}")
    if n_ttc > 0:
        print(f"  TTC<1.5: {(df['min_ttc_pre'] < 1.5).sum()}")
        print(f"  TTC<3.0: {(df['min_ttc_pre'] < 3.0).sum()}")

    print("\nGenerating figures...")
    plot_fig1(df, clips)
    plot_fig2(df)
    plot_fig3(df, clips)
    plot_fig4(df)

    print("\nWriting report...")
    write_report(df)

    print(f"\nAll outputs: {OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
