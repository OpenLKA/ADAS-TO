#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
longtail_deep_dive_v2.py
========================
Four-phase pipeline for deep-dive analysis of critical takeover clips
defined by the early-warning window: TTC < 3.0 s OR THW < 0.8 s.

Uses per_clip_all_metrics.csv (v2, radar-verified with leadOne.status check).

Phase 1 — Curation:   Filter + symlink into longtail/
Phase 2 — Kinematic:  Per-clip fig1 (longitudinal) + fig3 (lateral + jerk)
Phase 3 — VLM:        3-frame extraction → GPT-4o annotation with sensor context
Phase 4 — Aggregate:  Compile longtail_analysis_summary.csv + markdown report

Run:
    python3 scripts/longtail_deep_dive_v2.py
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
TEST_MODE = False          # ← Set False to process all clips

ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
LONG_LAT  = CODE / "long_lat"
LONGTAIL  = LONG_LAT / "longtail"
CSV_PATH  = LONG_LAT / "per_clip_all_metrics.csv"

# Critical thresholds (union)
TTC_THRESHOLD = 3.0   # seconds
THW_THRESHOLD = 0.8   # seconds

PRE_S           = 5.0
POST_S          = 5.0
SMOOTH_WINDOW_S = 0.3
SMOOTH_POLY     = 2
EPS             = 1e-6

# Frame extraction: seconds before first_takeover_s
FRAME_OFFSETS_S = [5.0, 3.0, 1.0]

# VLM config — Self-Consistency with GPT-4o
GPT_MODEL       = "gpt-4o"           # switched to gpt-4o (cheaper + faster)
N_VLM_ROUNDS    = 3                   # independent annotation rounds
VLM_TEMPERATURE = 0.7                 # diversity for independent rounds
CONSENSUS_TEMPERATURE = 0.0           # deterministic for final consensus
GPT_MAX_TOKENS  = 800
GPT_TIMEOUT     = 60                  # gpt-4o is faster
GPT_RETRY_WAIT  = 5
GPT_MAX_RETRIES = 3

N_WORKERS = 1

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


_NAN = float("nan")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Curation & Linking
# ══════════════════════════════════════════════════════════════════════════════
def phase1_curate() -> pd.DataFrame:
    """Filter clips by TTC<3.0 OR THW<0.8 (both radar-verified) and symlink."""
    print("═" * 70)
    print("PHASE 1 — Curation & Linking")
    print("═" * 70)

    df = pd.read_csv(CSV_PATH)
    print(f"  Total clips in CSV: {len(df)}")

    # Use BOTH pre-window min AND instantaneous at-takeover metrics
    # A clip qualifies if EITHER measure is critical
    ttc_pre  = df["min_ttc_pre"] < TTC_THRESHOLD
    thw_pre  = df["thw_at_min_ttc"] < THW_THRESHOLD
    ttc_inst = df["ttc_at_takeover"] < TTC_THRESHOLD
    thw_inst = df["thw_at_takeover"] < THW_THRESHOLD

    mask = ttc_pre | thw_pre | ttc_inst | thw_inst
    crit = df[mask].copy()

    # Sort by most critical first (lowest TTC)
    # Combine both TTC columns, take minimum
    crit["_ttc_min"] = crit[["min_ttc_pre", "ttc_at_takeover"]].min(axis=1)
    crit = crit.sort_values("_ttc_min").reset_index(drop=True)

    # Classify criticality reason
    reasons = []
    for _, r in crit.iterrows():
        parts = []
        ttc_val = min(r.get("min_ttc_pre", 999), r.get("ttc_at_takeover", 999))
        thw_val = min(r.get("thw_at_min_ttc", 999), r.get("thw_at_takeover", 999))
        if ttc_val < 1.5:
            parts.append(f"TTC={ttc_val:.1f}s [EXTREME]")
        elif ttc_val < TTC_THRESHOLD:
            parts.append(f"TTC={ttc_val:.1f}s")
        if thw_val < THW_THRESHOLD:
            parts.append(f"THW={thw_val:.2f}s")
        reasons.append("; ".join(parts) if parts else "")
    crit["criticality_reason"] = reasons

    n_ttc = (ttc_pre | ttc_inst).sum()
    n_thw = (thw_pre | thw_inst).sum()
    n_both = ((ttc_pre | ttc_inst) & (thw_pre | thw_inst)).sum()
    print(f"  Critical clips: {len(crit)}")
    print(f"    TTC < {TTC_THRESHOLD}s: {n_ttc}")
    print(f"    THW < {THW_THRESHOLD}s: {n_thw}")
    print(f"    Both: {n_both}")
    print(f"    TTC < 1.5s (extreme): {(crit['_ttc_min'] < 1.5).sum()}")

    # Clean old symlinks and create new ones
    LONGTAIL.mkdir(parents=True, exist_ok=True)
    for old in LONGTAIL.iterdir():
        if old.is_symlink():
            old.unlink()

    linked = 0
    for _, row in crit.iterrows():
        src = Path(row["clip_dir"])
        if not src.exists():
            continue
        cid = row.get("clip_id", 0)
        cid_str = str(int(cid)) if not np.isnan(cid) else "0"
        name = f"{row['car_model']}__{row['dongle_id']}__{row['route_id']}__{cid_str}"
        dst = LONGTAIL / name
        if not dst.exists():
            try:
                dst.symlink_to(src)
                linked += 1
            except OSError:
                pass
        else:
            linked += 1

    print(f"  Symlinks created: {linked}")
    print(f"  Staging folder: {LONGTAIL}")
    print()
    return crit


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Kinematic Profiling
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        8,
    "axes.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  6.5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

CLR = {
    "ego": "#4C78A8", "lead": "#E45756", "steer": "#F58518",
    "brake": "#E45756", "gas": "#54A24B", "lane": "#72B7B2",
    "jerk": "#B279A2", "grey": "#888888",
}


def _generate_clip_figures(clip_dir: Path, row: dict) -> dict:
    """Generate fig1 + fig3 for one clip. Returns kinematic metrics."""
    result = {
        "clip_dir": str(clip_dir),
        "fig1_ok": False, "fig3_ok": False,
        "extreme_jerk": False, "jerk_max_5s": np.nan,
        "steer_rate_max_5s": np.nan, "peak_decel_5s": np.nan,
    }

    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return result
    with open(meta_path) as f:
        meta = json.load(f)

    log_hz = meta.get("log_hz", 20)
    first_to = row.get("first_takeover_s", _NAN)
    if np.isnan(first_to):
        first_to = meta.get("video_time_s", _NAN)
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

    # Compute jerk on full window
    jerk_ts = None
    cs_post = cs[(cs["t_rel"] >= 0) & (cs["t_rel"] <= POST_S)]
    if len(cs_post) > 5 and "aEgo" in cs_post.columns:
        t_arr = cs_post["time_s"].values
        a_sm = smooth_signal(cs_post["aEgo"].values, log_hz)
        dt = np.diff(t_arr); dt[dt < EPS] = EPS
        jerk_abs = np.abs(np.diff(a_sm) / dt)
        result["jerk_max_5s"] = min(float(np.nanmax(jerk_abs)), 50.0)
        result["extreme_jerk"] = result["jerk_max_5s"] > 5.0
        result["peak_decel_5s"] = float(np.nanmin(a_sm))

        cs_full = cs[(cs["t_rel"] >= -PRE_S) & (cs["t_rel"] <= POST_S)]
        if len(cs_full) > 5:
            t_f = cs_full["time_s"].values
            a_f = smooth_signal(cs_full["aEgo"].values, log_hz)
            dt_f = np.diff(t_f); dt_f[dt_f < EPS] = EPS
            jerk_ts = (cs_full["t_rel"].values[1:], np.diff(a_f) / dt_f)

    if len(cs_post) > 5 and "steeringAngleDeg" in cs_post.columns:
        t_arr = cs_post["time_s"].values
        s_sm = smooth_signal(cs_post["steeringAngleDeg"].values, log_hz)
        dt = np.diff(t_arr); dt[dt < EPS] = EPS
        result["steer_rate_max_5s"] = min(float(np.nanmax(np.abs(np.diff(s_sm) / dt))), 500.0)

    ttc_val = row.get("min_ttc_pre", row.get("ttc_at_takeover", np.nan))
    thw_val = row.get("thw_at_min_ttc", row.get("thw_at_takeover", np.nan))

    # ── Fig 1: Longitudinal Context ────────────────────────────────────
    try:
        fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.0), sharex=True,
                                 gridspec_kw={"hspace": 0.10})

        parts = []
        if not np.isnan(ttc_val):
            parts.append(f"TTC={ttc_val:.1f}s")
        if not np.isnan(thw_val):
            parts.append(f"THW={thw_val:.2f}s")
        label = ", ".join(parts)

        ax = axes[0]
        ax.plot(cs_w["t_rel"], cs_w["vEgo"] * 3.6, color=CLR["ego"], lw=1.0, label="Ego")
        if not radar_w.empty and "leadOne.vLead" in radar_w.columns:
            rv = radar_w[radar_w.get("leadOne.status", False) == True] if "leadOne.status" in radar_w.columns else radar_w
            if not rv.empty:
                ax.plot(rv["t_rel"], rv["leadOne.vLead"] * 3.6,
                        color=CLR["lead"], lw=1.0, ls="--", label="Lead")
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_ylabel("Speed (km/h)")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title(f"(a) Speed — {label}", fontsize=8, fontweight="bold", loc="left")

        ax = axes[1]
        if not radar_w.empty and "leadOne.dRel" in radar_w.columns:
            rv = radar_w[radar_w.get("leadOne.status", False) == True] if "leadOne.status" in radar_w.columns else radar_w
            if not rv.empty:
                ax.plot(rv["t_rel"], rv["leadOne.dRel"], color=CLR["lead"], lw=1.0)
                ax.fill_between(rv["t_rel"], 0, rv["leadOne.dRel"], color=CLR["lead"], alpha=0.06)
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.set_ylabel("Rel. Distance (m)")
        ax.set_title("(b) Distance to Lead", fontsize=8, fontweight="bold", loc="left")

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
        fig.suptitle(f"Longitudinal Context — {row.get('car_model', '')} clip {row.get('clip_id', '')}",
                     fontsize=9, fontweight="bold", y=1.01)
        fig.savefig(clip_dir / "fig1_longitudinal_context.png")
        plt.close(fig)
        result["fig1_ok"] = True
    except Exception as e:
        print(f"  [ERR] fig1: {e}")

    # ── Fig 3: Lateral + Jerk ──────────────────────────────────────────
    try:
        n_panels = 4 if jerk_ts is not None else 3
        fig, axes = plt.subplots(n_panels, 1, figsize=(5.5, 1.6 * n_panels + 0.5),
                                 sharex=True, gridspec_kw={"hspace": 0.10})

        ax = axes[0]
        if not model_w.empty and "laneLineMeta.leftY" in model_w.columns:
            offset = (model_w["laneLineMeta.leftY"] + model_w["laneLineMeta.rightY"]) / 2.0
            ax.plot(model_w["t_rel"], offset, color=CLR["lane"], lw=1.0)
            ax.fill_between(model_w["t_rel"], 0, offset, color=CLR["lane"], alpha=0.10)
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(0, color=CLR["grey"], lw=0.4, alpha=0.5)
        ax.set_ylabel("Lane Offset (m)")
        ax.set_title("(a) Lane Center Offset", fontsize=8, fontweight="bold", loc="left")

        ax = axes[1]
        if "steeringAngleDeg" in cs_w.columns:
            ax.plot(cs_w["t_rel"], cs_w["steeringAngleDeg"], color=CLR["steer"], lw=0.9)
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(0, color=CLR["grey"], lw=0.4, alpha=0.5)
        ax.set_ylabel("Steer Angle (°)")
        ax.set_title("(b) Steering Angle", fontsize=8, fontweight="bold", loc="left")

        ax = axes[2]
        if "steeringTorque" in cs_w.columns:
            ax.plot(cs_w["t_rel"], cs_w["steeringTorque"], color=CLR["steer"], lw=0.9, alpha=0.7)
        ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(0, color=CLR["grey"], lw=0.4, alpha=0.5)
        ax.set_ylabel("Steer Torque (Nm)")
        ax.set_title("(c) Steering Torque", fontsize=8, fontweight="bold", loc="left")

        if jerk_ts is not None:
            ax = axes[3]
            t_j, j_v = jerk_ts
            ax.plot(t_j, j_v, color=CLR["jerk"], lw=0.8, alpha=0.85)
            ax.axhline(5.0, color=CLR["brake"], lw=0.6, ls="--", alpha=0.7, label="±5 m/s³")
            ax.axhline(-5.0, color=CLR["brake"], lw=0.6, ls="--", alpha=0.7)
            extreme_mask = np.abs(j_v) > 5.0
            if extreme_mask.any():
                ax.fill_between(t_j, -50, 50, where=extreme_mask, color=CLR["brake"], alpha=0.08)
            ax.axvline(0, color="k", lw=0.7, ls=":", alpha=0.6)
            ax.set_ylabel("Jerk (m/s³)")
            ax.set_xlabel("Time relative to takeover (s)")
            ax.set_title("(d) Longitudinal Jerk", fontsize=8, fontweight="bold", loc="left")
            j_lim = min(30, max(10, np.nanpercentile(np.abs(j_v), 99) * 1.3))
            ax.set_ylim(-j_lim, j_lim)
            ax.legend(loc="upper right", frameon=False)
        else:
            axes[-1].set_xlabel("Time relative to takeover (s)")

        for a in axes:
            a.set_xlim(-PRE_S, POST_S)
        jerk_tag = "  [EXTREME JERK]" if result["extreme_jerk"] else ""
        fig.suptitle(f"Lateral + Jerk — {row.get('car_model', '')} clip {row.get('clip_id', '')}{jerk_tag}",
                     fontsize=9, fontweight="bold", y=1.01)
        fig.savefig(clip_dir / "fig3_lateral_steering_jerk.png")
        plt.close(fig)
        result["fig3_ok"] = True
    except Exception as e:
        print(f"  [ERR] fig3: {e}")

    return result


def phase2_kinematic(crit: pd.DataFrame) -> list[dict]:
    print("═" * 70)
    print("PHASE 2 — Kinematic Profiling")
    print("═" * 70)

    rows = crit.to_dict("records")
    if TEST_MODE:
        rows = rows[:1]
        print("  [TEST_MODE] 1 clip")

    results = []
    if TEST_MODE or len(rows) <= 4:
        for r in rows:
            results.append(_generate_clip_figures(Path(r["clip_dir"]), r))
    else:
        print(f"  Processing {len(rows)} clips ({N_WORKERS} workers)...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_generate_clip_figures, Path(r["clip_dir"]), r): i
                       for i, r in enumerate(rows)}
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception:
                    pass

    done1 = sum(1 for r in results if r["fig1_ok"])
    done3 = sum(1 for r in results if r["fig3_ok"])
    n_ej  = sum(1 for r in results if r["extreme_jerk"])
    print(f"  fig1: {done1}/{len(rows)}  |  fig3: {done3}/{len(rows)}  |  extreme jerk: {n_ej}")
    print()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: VLM Self-Consistency Annotation (GPT-4.1, N rounds + consensus)
# ══════════════════════════════════════════════════════════════════════════════

# ── Prompt for each independent round (receives images + sensor context) ──
ROUND_PROMPT_TEMPLATE = """\
You are an expert in autonomous driving safety.

## Sensor context (from CAN bus — these are GROUND TRUTH, not estimates):
{sensor_block}

## Task
You are viewing 3 dashcam frames taken at T−5 s, T−3 s, and T−1 s before the \
human driver frantically took over control from the ADAS.

Analyze the spatiotemporal progression of the scene. Cross-check what you see \
with the sensor data above. If the visual evidence contradicts the sensor data, \
trust the sensors and note the discrepancy.

Answer EXACTLY in JSON with these keys:
1. "lateral_and_lane_status": Lane line visibility, road curvature, surface quality, \
and whether the ego vehicle appears to be drifting or lane keeping is failing.
2. "longitudinal_traffic_hazard": Lead vehicle behavior, obstacles, cut-ins, hard \
braking, or other hazards causing the low TTC/THW.
3. "early_warning_hypothesis": Based on T-5s and T-3s frames, what early visual \
cues existed that a predictive model could have used to warn the driver before \
the emergency at T-1s?
4. "visual_sensor_consistency": Do your visual observations match the sensor data? \
Note any discrepancies.

Return ONLY the JSON object, no markdown fences.\
"""

# ── Prompt for the consensus meta-review call (text-only, no images) ──
CONSENSUS_PROMPT_TEMPLATE = """\
You are a senior autonomous driving safety analyst performing a meta-review.

Below are {n_rounds} INDEPENDENT annotations of the same critical takeover event \
(TTC < 3.0s or THW < 0.8s). Each annotation was produced by a separate analysis \
of the same dashcam frames and sensor data.

## Sensor context (ground truth):
{sensor_block}

## Independent Annotations:
{annotations_block}

## Your Task
1. Identify points of AGREEMENT across all annotations — these are high-confidence.
2. Identify points of DISAGREEMENT — flag these explicitly.
3. Produce a FINAL consolidated annotation.

Answer EXACTLY in JSON with these keys:
1. "lateral_and_lane_status": Consensus description of lane/road conditions and \
whether lane keeping was failing.
2. "longitudinal_traffic_hazard": Consensus description of the traffic hazard.
3. "early_warning_hypothesis": Consensus explanation of what early cues a \
predictive model could have used.
4. "confidence_score": Float 0.0–1.0 reflecting inter-annotator agreement \
(1.0 = perfect agreement, <0.5 = major disagreements).
5. "disagreements": List of strings describing any disagreements between rounds. \
Empty list if all rounds agree.
6. "risk_factors": List of short risk-factor tags from this set: \
["Cut-in", "Hard Braking", "Faded Lane Lines", "Sharp Curve", "Obstruction", \
"Slow Vehicle", "Pedestrian", "Merging", "Construction", "Glare/Weather", \
"Occlusion", "Tailgating", "Night Driving", "Wet Road", "Congestion"]. Only \
include factors supported by at least 2 of the {n_rounds} annotations.

Return ONLY the JSON object, no markdown fences.\
"""


def _build_sensor_block(row: dict) -> str:
    parts = []
    # TTC — show both values
    ttc_pre = row.get("min_ttc_pre", np.nan)
    ttc_inst = row.get("ttc_at_takeover", np.nan)
    thw_pre = row.get("thw_at_min_ttc", np.nan)
    thw_inst = row.get("thw_at_takeover", np.nan)

    vEgo = row.get("vEgo_at_min_ttc", row.get("vEgo_at_takeover", np.nan))
    vLead = row.get("vLead_at_min_ttc", row.get("vLead_at_takeover", np.nan))
    dRel = row.get("dRel_at_min_ttc", row.get("dRel_at_takeover", np.nan))
    closing = row.get("closing_at_min_ttc", np.nan)

    if not np.isnan(vEgo):
        parts.append(f"- Ego speed: {vEgo:.1f} m/s ({vEgo*3.6:.0f} km/h)")
    if not np.isnan(vLead):
        parts.append(f"- Lead vehicle speed: {vLead:.1f} m/s ({vLead*3.6:.0f} km/h)")
    if not np.isnan(dRel):
        parts.append(f"- Distance to lead: {dRel:.1f} m")
    if not np.isnan(closing):
        parts.append(f"- Closing speed: {closing:.1f} m/s")

    ttc_show = min(x for x in [ttc_pre, ttc_inst] if not np.isnan(x)) if any(not np.isnan(x) for x in [ttc_pre, ttc_inst]) else np.nan
    thw_show = min(x for x in [thw_pre, thw_inst] if not np.isnan(x)) if any(not np.isnan(x) for x in [thw_pre, thw_inst]) else np.nan

    if not np.isnan(ttc_show):
        crit_tag = " ← CRITICAL" if ttc_show < 3.0 else ""
        parts.append(f"- Time-to-Collision (TTC): {ttc_show:.2f} s{crit_tag}")
    if not np.isnan(thw_show):
        crit_tag = " ← CRITICAL" if thw_show < 0.8 else ""
        parts.append(f"- Time Headway (THW): {thw_show:.2f} s{crit_tag}")

    offset = row.get("lane_offset_at_takeover", np.nan)
    if not np.isnan(offset):
        parts.append(f"- Lane center offset: {offset:.3f} m ({'left' if offset > 0 else 'right'})")
    width = row.get("lane_width_at_takeover", np.nan)
    if not np.isnan(width):
        parts.append(f"- Lane width: {width:.2f} m")

    trigger = row.get("primary_trigger", "Unknown")
    parts.append(f"- Driver's first action: {trigger}")
    parts.append(f"- Lead vehicle confirmed by radar: {row.get('lead_valid', row.get('lead_status_at_takeover', False))}")

    return "\n".join(parts) if parts else "- No sensor data available"


def _extract_frames(video_path: Path, meta: dict, first_takeover_s: float
                    ) -> list[tuple[float, np.ndarray]]:
    clip_start = meta["clip_start_s"]
    camera_fps = meta.get("camera_fps", 20)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or camera_fps

    frames = []
    for offset in FRAME_OFFSETS_S:
        abs_time = first_takeover_s - offset
        vid_second = abs_time - clip_start
        frame_num = int(vid_second * vid_fps)
        frame_num = max(0, min(frame_num, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append((offset, frame))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_num - 1))
            ret2, frame2 = cap.read()
            if ret2:
                frames.append((offset, frame2))

    cap.release()
    return frames


def _frame_to_base64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _build_image_message_parts(frames: list[tuple[float, np.ndarray]],
                               prompt_text: str) -> list[dict]:
    """Build multimodal content parts: text prompt + interleaved images."""
    parts = [{"type": "text", "text": prompt_text}]
    for offset, frame in frames:
        b64 = _frame_to_base64(frame)
        parts.append({
            "type": "text",
            "text": f"\n[Frame at T−{offset:.0f}s before takeover]"
        })
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
        })
    return parts


def _call_openai(client, messages: list[dict], temperature: float,
                 max_tokens: int = GPT_MAX_TOKENS,
                 verbose: bool = False, label: str = "") -> tuple[str, dict]:
    """Call OpenAI API with retries. Returns (text, usage_dict)."""
    for attempt in range(1, GPT_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "model": response.model,
            }

            if verbose:
                print(f"\n    ── {label} RESPONSE ──")
                print(f"    Model: {usage['model']}")
                print(f"    Tokens: {usage['prompt_tokens']} in, "
                      f"{usage['completion_tokens']} out")
                for line in text.split("\n"):
                    print(f"      {line}")

            return text, usage

        except Exception as e:
            import openai
            wait = GPT_RETRY_WAIT * attempt
            etype = type(e).__name__
            if isinstance(e, openai.RateLimitError):
                print(f"    [{label} RATE LIMIT] attempt {attempt}/{GPT_MAX_RETRIES}, "
                      f"waiting {wait}s...")
            elif isinstance(e, openai.APITimeoutError):
                print(f"    [{label} TIMEOUT] attempt {attempt}/{GPT_MAX_RETRIES}, "
                      f"waiting {wait}s...")
            else:
                print(f"    [{label} {etype}] attempt {attempt}/{GPT_MAX_RETRIES}: {e}")

            if attempt < GPT_MAX_RETRIES:
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"{label} failed after {GPT_MAX_RETRIES} retries")


def _parse_vlm_response(text: str) -> dict:
    """Parse JSON from VLM response, handling markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}
    return data


RISK_KEYWORDS = {
    "Cut-in":           r"cut[\-\s]?in",
    "Hard Braking":     r"hard\s*brak|sudden\s*brak|abrupt\s*brak",
    "Faded Lane Lines": r"fad|worn|unclear|poor\s*lane|missing\s*lane|no\s*lane",
    "Sharp Curve":      r"sharp\s*curve|tight\s*curve|curve|bend",
    "Obstruction":      r"obstruct|debris|object\s*in|block",
    "Slow Vehicle":     r"slow\s*vehicle|slow\s*moving|stopped\s*vehicle",
    "Pedestrian":       r"pedestrian|person|jaywalking",
    "Merging":          r"merg|lane\s*change|lane\s*switch",
    "Construction":     r"construct|road\s*work|cone",
    "Glare/Weather":    r"glare|sun|rain|fog|wet|snow|weather",
    "Occlusion":        r"occlud|occlu|hidden|blocked\s*view",
    "Tailgating":       r"tailgat|too\s*close|following\s*too",
    "Night Driving":    r"night|dark|low[\-\s]light|poor\s*visibility",
    "Congestion":       r"congest|traffic\s*jam|queue|stop[\-\s]and[\-\s]go",
}


def _extract_risk_factors(parsed: dict) -> list[str]:
    combined = " ".join(str(v) for v in parsed.values()).lower()
    return [label for label, pat in RISK_KEYWORDS.items()
            if re.search(pat, combined, re.IGNORECASE)]


def phase3_vlm(crit: pd.DataFrame) -> list[dict]:
    """N-round self-consistency VLM annotation with consensus meta-review.

    For each clip:
      1. Run N_VLM_ROUNDS independent GPT-4.1 calls (temp=0.7)
      2. Run 1 consensus call that sees all N responses (temp=0.0)
      3. Save all round outputs + consensus to clip directory
    """
    print("═" * 70)
    print(f"PHASE 3 — VLM Self-Consistency ({GPT_MODEL}, {N_VLM_ROUNDS} rounds + consensus)")
    print("═" * 70)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  [SKIP] OPENAI_API_KEY not set — skipping VLM phase")
        print("  To enable: export OPENAI_API_KEY='sk-...'")
        print()
        return []

    import openai
    client = openai.OpenAI(api_key=api_key, timeout=GPT_TIMEOUT)

    rows = crit.to_dict("records")
    if TEST_MODE:
        rows = rows[:1]
        print("  [TEST_MODE] Processing 1 clip only")

    total_tokens_in = 0
    total_tokens_out = 0
    vlm_results = []

    n_skipped = 0
    for i, row in enumerate(rows):
        clip_dir = Path(row["clip_dir"])

        # ── Checkpoint: skip if already annotated ──────────────────────
        consensus_file = clip_dir / "vlm_consensus.txt"
        if consensus_file.exists() and consensus_file.stat().st_size > 10:
            # Load cached result
            consensus_text = consensus_file.read_text(encoding="utf-8")
            consensus_parsed = _parse_vlm_response(consensus_text)
            risk_factors = consensus_parsed.get("risk_factors", [])
            if not risk_factors or not isinstance(risk_factors, list):
                risk_factors = _extract_risk_factors(consensus_parsed)
            confidence = consensus_parsed.get("confidence_score", np.nan)
            disagreements = consensus_parsed.get("disagreements", [])
            if isinstance(disagreements, list):
                disagreements = "; ".join(str(d) for d in disagreements)
            vlm_results.append({
                "clip_dir":                     str(clip_dir),
                "car_model":                    row.get("car_model", ""),
                "clip_id":                      row.get("clip_id", ""),
                "min_ttc_pre":                  row.get("min_ttc_pre", np.nan),
                "criticality_reason":           row.get("criticality_reason", ""),
                "lateral_and_lane_status":      consensus_parsed.get("lateral_and_lane_status", ""),
                "longitudinal_traffic_hazard":  consensus_parsed.get("longitudinal_traffic_hazard", ""),
                "early_warning_hypothesis":     consensus_parsed.get("early_warning_hypothesis", ""),
                "confidence_score":             confidence,
                "disagreements":                disagreements,
                "risk_factors":                 "; ".join(risk_factors) if isinstance(risk_factors, list) else str(risk_factors),
                "n_rounds":                     N_VLM_ROUNDS,
                "n_rounds_ok":                  N_VLM_ROUNDS,  # assume all OK from cache
            })
            n_skipped += 1
            continue

        video_path = clip_dir / "takeover.mp4"
        if not video_path.exists():
            print(f"  [{i+1}/{len(rows)}] SKIP — no takeover.mp4")
            continue

        with open(clip_dir / "meta.json") as f:
            meta = json.load(f)

        first_to = row.get("first_takeover_s", meta["video_time_s"])
        if np.isnan(first_to):
            first_to = meta["video_time_s"]

        ttc_candidates = [x for x in [row.get("min_ttc_pre", np.nan), row.get("ttc_at_takeover", np.nan)] if not np.isnan(x)]
        ttc_show = min(ttc_candidates) if ttc_candidates else 999.0
        remaining = len(rows) - n_skipped
        print(f"  [{i+1}/{len(rows)}] {row.get('car_model', '')} clip {row.get('clip_id', '')} "
              f"(TTC={ttc_show:.2f}s, reason: {row.get('criticality_reason', '')})"
              + (f"  [cached: {n_skipped}]" if i == n_skipped else ""))

        # ── Extract frames ─────────────────────────────────────────────
        try:
            frames = _extract_frames(video_path, meta, first_to)
        except Exception as e:
            print(f"    Frame extraction failed: {e}")
            continue
        if not frames:
            print(f"    No frames extracted")
            continue

        if TEST_MODE:
            print(f"    Extracted {len(frames)} frames:")
            for offset, frame in frames:
                vid_sec = (first_to - offset) - meta["clip_start_s"]
                print(f"      T−{offset:.0f}s → video second {vid_sec:.2f}, "
                      f"shape={frame.shape}")

        # Save frames as PNGs
        for offset, frame in frames:
            cv2.imwrite(str(clip_dir / f"frame_T-{int(offset)}s.png"), frame)

        # ── Build sensor context ───────────────────────────────────────
        sensor_block = _build_sensor_block(row)

        if TEST_MODE:
            print(f"\n    ── SENSOR CONTEXT ──")
            for line in sensor_block.split("\n"):
                print(f"      {line}")

        # ── N independent annotation rounds ────────────────────────────
        round_prompt = ROUND_PROMPT_TEMPLATE.format(sensor_block=sensor_block)
        img_parts = _build_image_message_parts(frames, round_prompt)
        messages_round = [{"role": "user", "content": img_parts}]

        if TEST_MODE:
            print(f"\n    ── ROUND PROMPT ({len(round_prompt)} chars) ──")
            for line in round_prompt.split("\n")[:5]:
                print(f"      {line}")
            print(f"      ... ({len(round_prompt.splitlines())} lines total)")

        round_texts = []
        round_parsed = []
        for r in range(1, N_VLM_ROUNDS + 1):
            try:
                text, usage = _call_openai(
                    client, messages_round,
                    temperature=VLM_TEMPERATURE,
                    verbose=TEST_MODE,
                    label=f"Round {r}/{N_VLM_ROUNDS}",
                )
                total_tokens_in += usage["prompt_tokens"]
                total_tokens_out += usage["completion_tokens"]
                round_texts.append(text)
                round_parsed.append(_parse_vlm_response(text))
            except Exception as e:
                print(f"    Round {r} failed: {e}")
                round_texts.append("")
                round_parsed.append({})

            # Brief pause between rounds
            if r < N_VLM_ROUNDS:
                time.sleep(1.0)

        # ── Consensus meta-review ──────────────────────────────────────
        annotations_block = ""
        for r, txt in enumerate(round_texts, 1):
            annotations_block += f"\n### Annotation {r}:\n{txt}\n"

        consensus_prompt = CONSENSUS_PROMPT_TEMPLATE.format(
            n_rounds=N_VLM_ROUNDS,
            sensor_block=sensor_block,
            annotations_block=annotations_block,
        )

        # Consensus call is TEXT-ONLY (no images needed — it has all annotations)
        messages_consensus = [{"role": "user", "content": consensus_prompt}]

        consensus_text = ""
        consensus_parsed = {}
        try:
            consensus_text, usage = _call_openai(
                client, messages_consensus,
                temperature=CONSENSUS_TEMPERATURE,
                max_tokens=1000,
                verbose=TEST_MODE,
                label="CONSENSUS",
            )
            total_tokens_in += usage["prompt_tokens"]
            total_tokens_out += usage["completion_tokens"]
            consensus_parsed = _parse_vlm_response(consensus_text)
        except Exception as e:
            print(f"    Consensus call failed: {e}")

        # ── Extract risk factors ───────────────────────────────────────
        risk_factors = consensus_parsed.get("risk_factors", [])
        if not risk_factors or not isinstance(risk_factors, list):
            risk_factors = _extract_risk_factors(consensus_parsed)

        confidence = consensus_parsed.get("confidence_score", np.nan)
        disagreements = consensus_parsed.get("disagreements", [])
        if isinstance(disagreements, list):
            disagreements = "; ".join(str(d) for d in disagreements)

        # ── Save all outputs ───────────────────────────────────────────
        for r, txt in enumerate(round_texts, 1):
            (clip_dir / f"vlm_round_{r}.txt").write_text(txt, encoding="utf-8")

        (clip_dir / "vlm_consensus.txt").write_text(consensus_text, encoding="utf-8")

        combined = f"=== CONSENSUS (confidence={confidence}) ===\n{consensus_text}\n\n"
        for r, txt in enumerate(round_texts, 1):
            combined += f"=== ROUND {r} ===\n{txt}\n\n"
        (clip_dir / "vlm_annotation.txt").write_text(combined, encoding="utf-8")

        # ── Collect result ─────────────────────────────────────────────
        vlm_results.append({
            "clip_dir":                     str(clip_dir),
            "car_model":                    row.get("car_model", ""),
            "clip_id":                      row.get("clip_id", ""),
            "min_ttc_pre":                  row.get("min_ttc_pre", np.nan),
            "criticality_reason":           row.get("criticality_reason", ""),
            "lateral_and_lane_status":      consensus_parsed.get("lateral_and_lane_status", ""),
            "longitudinal_traffic_hazard":  consensus_parsed.get("longitudinal_traffic_hazard", ""),
            "early_warning_hypothesis":     consensus_parsed.get("early_warning_hypothesis", ""),
            "confidence_score":             confidence,
            "disagreements":                disagreements,
            "risk_factors":                 "; ".join(risk_factors) if isinstance(risk_factors, list) else str(risk_factors),
            "n_rounds":                     N_VLM_ROUNDS,
            "n_rounds_ok":                  sum(1 for t in round_texts if t),
        })

        if TEST_MODE:
            print(f"\n    ── FINAL CONSENSUS ──")
            print(f"    Confidence: {confidence}")
            print(f"    Risk factors: {risk_factors}")
            print(f"    Disagreements: {disagreements}")
            for k in ["lateral_and_lane_status", "longitudinal_traffic_hazard",
                       "early_warning_hypothesis"]:
                print(f"    {k}: {consensus_parsed.get(k, '')}")

        # Delay between clips
        if not TEST_MODE and i < len(rows) - 1:
            time.sleep(1.0)

    print(f"\n  VLM annotations complete: {len(vlm_results)}/{len(rows)}")
    if n_skipped:
        print(f"  Cached (skipped): {n_skipped}, New: {len(vlm_results) - n_skipped}")
    print(f"  Total API usage: {total_tokens_in:,} prompt + "
          f"{total_tokens_out:,} completion tokens")
    print()
    return vlm_results


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Aggregation
# ══════════════════════════════════════════════════════════════════════════════
def phase4_aggregate(crit: pd.DataFrame, kin: list[dict], vlm: list[dict]):
    print("═" * 70)
    print("PHASE 4 — Insight Aggregation")
    print("═" * 70)

    summary = crit.copy()

    if kin:
        kin_df = pd.DataFrame(kin)
        kin_cols = ["clip_dir", "fig1_ok", "fig3_ok", "extreme_jerk",
                    "jerk_max_5s", "steer_rate_max_5s", "peak_decel_5s"]
        merge = kin_df[[c for c in kin_cols if c in kin_df.columns]]
        summary = summary.merge(merge, on="clip_dir", how="left")

    if vlm:
        vlm_df = pd.DataFrame(vlm)
        vlm_cols = ["clip_dir", "lateral_and_lane_status",
                    "longitudinal_traffic_hazard", "early_warning_hypothesis",
                    "risk_factors", "confidence_score", "disagreements",
                    "n_rounds", "n_rounds_ok"]
        merge = vlm_df[[c for c in vlm_cols if c in vlm_df.columns]]
        summary = summary.merge(merge, on="clip_dir", how="left")

    # Drop helper column
    summary.drop(columns=["_ttc_min"], errors="ignore", inplace=True)

    csv_out = LONGTAIL / "longtail_analysis_summary.csv"
    summary.to_csv(csv_out, index=False)
    print(f"  Saved {csv_out}")
    print(f"  Rows: {len(summary)}, Columns: {len(summary.columns)}")

    # Stats
    print(f"\n  Summary:")
    print(f"    Total critical clips: {len(summary)}")
    ttc_vals = summary[["min_ttc_pre", "ttc_at_takeover"]].min(axis=1).dropna()
    if len(ttc_vals):
        print(f"    TTC range: {ttc_vals.min():.2f} – {ttc_vals.max():.2f} s")
    if "extreme_jerk" in summary.columns:
        print(f"    Extreme jerk (>5 m/s³): {summary['extreme_jerk'].sum()}")

    if "risk_factors" in summary.columns:
        factors = summary["risk_factors"].dropna().str.split("; ").explode()
        factors = factors[factors != ""]
        if len(factors):
            print(f"\n    Risk factors:")
            for f, c in factors.value_counts().head(10).items():
                print(f"      {f}: {c}")

    # Markdown report
    lines = [
        "# Long-Tail Critical Takeover Analysis (v2)\n",
        f"- **Criteria**: TTC < {TTC_THRESHOLD}s OR THW < {THW_THRESHOLD}s (radar-verified)",
        f"- **Critical clips**: {len(summary)}",
        f"- **Model**: {GPT_MODEL}",
        f"- **Self-consistency**: {N_VLM_ROUNDS} rounds + consensus\n",
    ]

    # Trigger breakdown
    if "primary_trigger" in summary.columns:
        lines.append("## Trigger Modality\n")
        lines.append("| Trigger | Count | % |")
        lines.append("|---------|------:|--:|")
        for trig, cnt in summary["primary_trigger"].value_counts().items():
            lines.append(f"| {trig} | {cnt} | {100*cnt/len(summary):.1f}% |")
        lines.append("")

    # Kinematic highlights
    if "jerk_max_5s" in summary.columns:
        j = summary["jerk_max_5s"].dropna()
        if len(j) > 0:
            lines.append("## Post-Takeover Kinematic Profile\n")
            lines.append(f"- Jerk max: median {j.median():.1f}, "
                         f"P95 {j.quantile(0.95):.1f} m/s³")
        sr = summary.get("steer_rate_max_5s", pd.Series(dtype=float)).dropna()
        if len(sr) > 0:
            lines.append(f"- Steer rate max: median {sr.median():.1f}, "
                         f"P95 {sr.quantile(0.95):.1f} °/s")
        pd_col = summary.get("peak_decel_5s", pd.Series(dtype=float)).dropna()
        if len(pd_col) > 0:
            lines.append(f"- Peak decel: median {pd_col.median():.2f}, "
                         f"P5 {pd_col.quantile(0.05):.2f} m/s²")
        lines.append("")

    # VLM consensus quality
    if "confidence_score" in summary.columns:
        conf = summary["confidence_score"].dropna()
        if len(conf) > 0:
            lines.append("## VLM Consensus Quality\n")
            lines.append(f"- **Annotation rounds per clip**: {N_VLM_ROUNDS}")
            lines.append(f"- **Model**: {GPT_MODEL}")
            lines.append(f"- **Median confidence**: {conf.median():.2f}")
            lines.append(f"- **Low confidence (<0.5)**: "
                         f"{(conf < 0.5).sum()} clips")
            lines.append(f"- **High confidence (≥0.8)**: "
                         f"{(conf >= 0.8).sum()} clips")
            lines.append("")

    # Risk factors
    if "risk_factors" in summary.columns and len(factors):
        lines.append("## VLM-Identified Risk Factors (consensus)\n")
        lines.append("| Factor | Count | % of annotated |")
        lines.append("|--------|------:|---------------:|")
        n_ann = summary["risk_factors"].notna().sum()
        for f, c in factors.value_counts().items():
            lines.append(f"| {f} | {c} | {100*c/max(1,n_ann):.1f}% |")
        lines.append("")

    (LONGTAIL / "longtail_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Saved longtail_report.md")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  LONG-TAIL v2: TTC<3s OR THW<0.8s ({GPT_MODEL}, {N_VLM_ROUNDS}R+consensus)  ║")
    if TEST_MODE:
        print("║  >>> TEST MODE — 1 clip only <<<                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    crit = phase1_curate()
    if TEST_MODE:
        crit = crit.head(1)

    # Phase 2 (kinematic figures) skipped — use regenerate_longtail_figures.py
    kin = []
    vlm = phase3_vlm(crit)
    phase4_aggregate(crit, kin, vlm)

    print("Done.")
    if TEST_MODE:
        print("\n  Set TEST_MODE = False and re-run for all clips.")


if __name__ == "__main__":
    main()
