#!/usr/bin/env python3
"""
classify_ego_nonego.py (v7)
===========================
Binary classification of takeover clips: Ego / Non-ego.

Key improvements vs v6:
  1) BLINKER OVERRIDES CONFLICT: blinker + any ego detector → Ego regardless of
     conflict flags.  Blinker = driver intent = strongest ego signal.
  2) Stationary rule tightened: pre_speed < 0.5 m/s (was 1.0) AND post evidence.
  3) JUNCTION_NEAR_STOP_MPS reverted to 3.0 (was 4.5 — too aggressive).
  4) Junction override reverted: only override close_lead, NOT TTC/DRAC/FCW.
  5) Post window 10s, curvature features, left-turn detection (from v6).
  6) Sharp curve speed gate: curves never slow below 4.47 m/s (10 mph).

Outputs:
  DatasetClassification/ego_nonego_labels.csv
  DatasetClassification/{Ego,Non-ego}/   — symlinks (default) or copies
  DatasetClassification/ego_nonego_report.md

Usage:
  python3 classify_ego_nonego.py
  python3 classify_ego_nonego.py --repo_root ROOT --out_root DIR --copy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
DEFAULT_OUT = DEFAULT_ROOT / "Code" / "DatasetClassification"
MASTER_CSV = DEFAULT_ROOT / "Code" / "stats_output" / "analysis_master.csv"
N_WORKERS = 12

# Windows (seconds relative to event time)
WPRE_S = 3.0
WPOST_SHORT_S = 5.0   # short window (for tight confirmation)
WPOST_LONG_S = 10.0   # extended window (catches late maneuvers)

# ═══════════════════════════════════════════════════════════════════════
# THRESHOLDS  (v6)
# ═══════════════════════════════════════════════════════════════════════

# ── Stationary / near-stop ──
STATIONARY_MAX_SPEED_MPS = 0.5     # < 0.5 m/s ≈ truly stationary (tightened from 1.0)

# ── Lane change (Detector A) ──
LC_BLINKER_MIN_DURATION_S = 0.5
LC_ALERT_MATCH = "Changing Lanes"
LC_PRE_STEER_MAX_DEG = 15.0
LC_STEER_RETURN_RATIO = 0.40
LC_POST_STEER_MIN_DEG = 8.0        # in short window
LC_A1_MIN_POST_STEER_PEAK_DEG = 8.0  # raised from 5 to avoid subtle false positives
LC_A1_MAX_ONE_SIDED_RATIO = 0.92
LC_A1_MAX_DUR_STEER_GT20_S = 3.5   # extended for 10s window (was 2.5)
LC_A1_MAX_POST_DUR_STRONG_CURV_S = 1.5  # slightly relaxed for 10s window
LC_A1_MAX_POST_MEAN_ABS_CURV = 0.015

# Curve override
CURVE_OVERRIDE_SPEED_MPS = 10.0
CURVE_OVERRIDE_MIN_DUR_STEER_GT20_S = 3.0  # raised for 10s window
CURVE_OVERRIDE_MIN_ONE_SIDED_RATIO = 0.93
CURVE_OVERRIDE_MIN_PRE_MEAN_CURV = 0.003

# ── Junction/intersection turn (Detector B) ──
JUNCTION_NEAR_STOP_MPS = 3.0       # ≈ 6.7 mph — reverted from 4.5 (too aggressive)
JUNCTION_LOW_SPEED_MPS = 8.0
JUNCTION_POST_STEER_MIN = 15.0     # lowered from 20 to catch gentler turns
TURN_MIN_DUR_STEER_GT20_S = 1.5
TURN_MIN_ONE_SIDED_RATIO = 0.90
TURN_MIN_POST_DUR_STRONG_CURV_S = 1.0
TURN_MIN_POST_MAX_ABS_CURV = 0.02
TURN_MIN_CURV_DEVIATION = 0.005
# Sharp curve speed gate: curves maintain speed > 10 mph (4.47 m/s)
SHARP_CURVE_MIN_SPEED_MPS = 4.47

# ── Discretionary acceleration (Detector C) ──
ACCEL_POST_SPEED_DELTA_MIN_MPS = 2.0  # reverted to 2.0 (3.0 was too strict)
ACCEL_MAX_RISK_SCORE = 0.3
ACCEL_POST_STEER_MAX_DEG = 4.0
ACCEL_MAX_POST_MEAN_ABS_CURV = 0.003
ACCEL_MIN_POST_STEER_PEAK_DEG = 0.0   # no steer requirement for accel
ACCEL_MIN_POST_SPEED_DELTA_STRONG_MPS = 5.0  # strong accel needs less evidence

# ── Conflict / reactive (Detector D) ──
TTC_CRITICAL_S = 1.5
THW_CRITICAL_S = 0.8
DRAC_CRITICAL_MPS2 = 3.0
CLOSE_DREL_M = 10.0

# ── Curve boundary (Detector E) ──
CURVE_PRE_CURVATURE_MIN = 0.02
CURVE_PRE_STEER_RATE_MIN = 50.0
CURVE_LANE_PROB_MAX = 0.3

# ── System alerts (Detector F) ──
SYSTEM_ALERTS = [
    "TAKE CONTROL IMMEDIATELY",
    "Dashcam Mode",
    "openpilot Unavailable",
    "Steering Temporarily Unavailable",
]


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════
def _duration_true(t: np.ndarray, mask: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    dt = np.diff(t)
    dt = np.clip(dt, 0.0, np.inf)
    return float(np.sum(dt[mask[:-1]]))


def _sign_changes(x: np.ndarray, deadband: float = 1.0) -> int:
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    s[np.abs(x) < deadband] = 0.0
    nz = s[s != 0]
    if len(nz) < 2:
        return 0
    return int(np.sum(nz[1:] != nz[:-1]))


def _one_sided_ratio(x: np.ndarray, eps: float = 1e-6) -> float:
    x = np.asarray(x, dtype=float)
    denom = np.sum(np.abs(x)) + eps
    return float(np.abs(np.sum(x)) / denom)


# ═══════════════════════════════════════════════════════════════════════
# RAW FEATURE EXTRACTION (parallel) — dual-window: 5s + 10s
# ═══════════════════════════════════════════════════════════════════════
def _resolve_event_time_s_from_mono(df: pd.DataFrame, event_mono: int) -> tuple[float, float]:
    if df is None or df.empty:
        return (np.nan, np.nan)
    if "logMonoTime" not in df.columns or "time_s" not in df.columns:
        return (np.nan, np.nan)
    mono = df["logMonoTime"].to_numpy()
    try:
        mono = mono.astype(np.int64, copy=False)
    except Exception:
        mono = mono.astype(np.int64)
    idx = int(np.argmin(np.abs(mono - int(event_mono))))
    err = float(mono[idx] - int(event_mono))
    t0 = float(df["time_s"].iloc[idx])
    return (t0, err)


def _read_meta_event(clip_dir: Path) -> tuple[float, int | None]:
    meta_path = clip_dir / "meta.json"
    video_time_s = 10.0
    event_mono = None
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            video_time_s = float(meta.get("video_time_s", 10.0))
            if "event_mono" in meta:
                event_mono = int(meta["event_mono"])
        except Exception:
            pass
    return video_time_s, event_mono


def _extract_raw_features(args: tuple) -> dict:
    """
    Extract per-clip features from raw CSVs.  Uses dual windows:
      - Short (5s): tight confirmation features
      - Long (10s): catches late maneuvers
    """
    clip_dir_str, video_time_s, event_mono = args
    clip_dir = Path(clip_dir_str)

    result = {
        "clip_dir": clip_dir_str,
        "event_time_s": np.nan,
        "event_time_source": "unknown",
        "event_mono": int(event_mono) if event_mono is not None else np.nan,
        "event_mono_err_ns": np.nan,
        # Blinker
        "blinker_any_pre": False, "blinker_any_post": False,
        "blinker_left_pre": False, "blinker_right_pre": False,
        "blinker_left_post": False, "blinker_right_post": False,
        "blinker_pre_duration_s": 0.0, "blinker_post_duration_s": 0.0,
        # Steering — short window (0-5s)
        "post5_steer_peak_deg": 0.0, "post5_steer_end_deg": 0.0,
        "post5_dur_abs_steer_gt10_s": 0.0, "post5_dur_abs_steer_gt20_s": 0.0,
        "post5_sign_changes": 0, "post5_one_sided_ratio": 0.0,
        # Steering — long window (0-10s)
        "post10_steer_peak_deg": 0.0, "post10_steer_end_deg": 0.0,
        "post10_dur_abs_steer_gt10_s": 0.0, "post10_dur_abs_steer_gt20_s": 0.0,
        "post10_sign_changes": 0, "post10_one_sided_ratio": 0.0,
        # Steer return (computed on long window)
        "steer_return": False, "steer_return_ratio": 1.0,
        # Steer direction (for left-turn detection)
        "post10_steer_mean_sign": 0.0,  # >0 = mostly left, <0 = mostly right
        # Steering — pre window
        "pre_dur_abs_steer_gt20_s": 0.0, "pre_one_sided_ratio": 0.0,
        # Speed
        "post5_speed_min_mps": np.nan, "post10_speed_min_mps": np.nan,
        "pre_speed_min_mps": np.nan,
        "post5_speed_delta_mps": np.nan, "post10_speed_delta_mps": np.nan,
        # Curvature — short window (0-5s)
        "post5_max_abs_curv": 0.0, "post5_mean_abs_curv": 0.0,
        "post5_dur_strong_curv_s": 0.0,
        "post5_max_curv_deviation": 0.0,
        "post5_curv_sign_consistency": 0.0,
        # Curvature — long window (0-10s)
        "post10_max_abs_curv": 0.0, "post10_mean_abs_curv": 0.0,
        "post10_dur_moderate_curv_s": 0.0, "post10_dur_strong_curv_s": 0.0,
        "post10_max_curv_deviation": 0.0, "post10_mean_curv_deviation": 0.0,
        "post10_curv_sign_consistency": 0.0,
        # Curvature — pre
        "pre_max_abs_curv": 0.0, "pre_mean_abs_curv": 0.0,
        "pre_max_abs_desired_curv": 0.0,
    }

    # ── Read carState ──
    cs_path = clip_dir / "carState.csv"
    cs = pd.DataFrame()
    if cs_path.exists():
        for cols in [
            ["time_s", "logMonoTime", "steeringAngleDeg", "vEgo", "leftBlinker", "rightBlinker"],
            ["time_s", "logMonoTime", "steeringAngleDeg", "vEgo"],
            ["time_s", "steeringAngleDeg", "vEgo"],
        ]:
            try:
                cs = pd.read_csv(cs_path, usecols=cols, low_memory=False)
                break
            except Exception:
                cs = pd.DataFrame()

    # ── Event time ──
    event_time_s = np.nan
    if event_mono is not None and not cs.empty and "logMonoTime" in cs.columns:
        t0, err_ns = _resolve_event_time_s_from_mono(cs, int(event_mono))
        if np.isfinite(t0):
            event_time_s = t0
            result["event_time_source"] = "mono_aligned"
            result["event_mono_err_ns"] = err_ns
    if not np.isfinite(event_time_s):
        event_time_s = float(video_time_s) if np.isfinite(video_time_s) else 10.0
        result["event_time_source"] = "video_time_fallback" if np.isfinite(video_time_s) else "default10"
    result["event_time_s"] = float(event_time_s)

    # ── Blinker ──
    def _blinker_from_df(df, tcol, lcol, rcol):
        if df is None or df.empty or not {tcol, lcol, rcol}.issubset(df.columns):
            return
        tmp = df[[tcol, lcol, rcol]].copy()
        for col in (lcol, rcol):
            tmp[col] = tmp[col].astype(str).str.strip().str.lower() == "true"
        # Use extended post window for blinker detection
        pre = tmp[(tmp[tcol] >= event_time_s - WPRE_S) & (tmp[tcol] <= event_time_s)]
        post = tmp[(tmp[tcol] >= event_time_s) & (tmp[tcol] <= event_time_s + WPOST_LONG_S)]

        def _dur(w):
            if w is None or w.empty or len(w) < 2:
                return 0.0
            return _duration_true(w[tcol].to_numpy(float), (w[lcol].to_numpy(bool) | w[rcol].to_numpy(bool)))

        if not pre.empty:
            result["blinker_left_pre"] = bool(pre[lcol].any())
            result["blinker_right_pre"] = bool(pre[rcol].any())
            result["blinker_any_pre"] = bool((pre[lcol] | pre[rcol]).any())
            result["blinker_pre_duration_s"] = float(_dur(pre))
        if not post.empty:
            result["blinker_left_post"] = bool(post[lcol].any())
            result["blinker_right_post"] = bool(post[rcol].any())
            result["blinker_any_post"] = bool((post[lcol] | post[rcol]).any())
            result["blinker_post_duration_s"] = float(_dur(post))

    cc_path = clip_dir / "carControl.csv"
    cc = pd.DataFrame()
    if cc_path.exists():
        try:
            cc = pd.read_csv(cc_path, usecols=["time_s", "leftBlinker", "rightBlinker"], low_memory=False)
        except Exception:
            cc = pd.DataFrame()
    if not cc.empty:
        _blinker_from_df(cc, "time_s", "leftBlinker", "rightBlinker")
    elif not cs.empty and "leftBlinker" in cs.columns:
        _blinker_from_df(cs, "time_s", "leftBlinker", "rightBlinker")

    # ── Steering waveform (dual window) ──
    if not cs.empty and {"time_s", "steeringAngleDeg", "vEgo"}.issubset(cs.columns) and len(cs) >= 5:
        pre_cs = cs[(cs["time_s"] >= event_time_s - WPRE_S) & (cs["time_s"] <= event_time_s)]
        post5 = cs[(cs["time_s"] >= event_time_s) & (cs["time_s"] <= event_time_s + WPOST_SHORT_S)]
        post10 = cs[(cs["time_s"] >= event_time_s) & (cs["time_s"] <= event_time_s + WPOST_LONG_S)]

        # Pre
        if not pre_cs.empty:
            result["pre_speed_min_mps"] = float(pre_cs["vEgo"].min())
        if len(pre_cs) >= 5:
            t = pre_cs["time_s"].to_numpy(float)
            steer = pre_cs["steeringAngleDeg"].to_numpy(float)
            result["pre_dur_abs_steer_gt20_s"] = _duration_true(t, np.abs(steer) > 20.0)
            result["pre_one_sided_ratio"] = _one_sided_ratio(steer)

        # Short window (5s)
        if len(post5) >= 5:
            t = post5["time_s"].to_numpy(float)
            steer = post5["steeringAngleDeg"].to_numpy(float)
            result["post5_steer_peak_deg"] = float(np.max(np.abs(steer)))
            result["post5_speed_min_mps"] = float(post5["vEgo"].min())
            v = post5["vEgo"].to_numpy(float)
            result["post5_speed_delta_mps"] = float(v[-1] - v[0]) if len(v) >= 2 else 0.0
            result["post5_dur_abs_steer_gt10_s"] = _duration_true(t, np.abs(steer) > 10.0)
            result["post5_dur_abs_steer_gt20_s"] = _duration_true(t, np.abs(steer) > 20.0)
            result["post5_sign_changes"] = _sign_changes(steer, deadband=1.0)
            result["post5_one_sided_ratio"] = _one_sided_ratio(steer)
            # end steer
            late5 = cs[(cs["time_s"] >= event_time_s + 4.0) & (cs["time_s"] <= event_time_s + 5.5)]
            if late5.empty:
                late5 = post5.tail(max(1, len(post5) // 5))
            result["post5_steer_end_deg"] = float(np.mean(np.abs(late5["steeringAngleDeg"].to_numpy(float))))

        # Long window (10s)
        if len(post10) >= 5:
            t = post10["time_s"].to_numpy(float)
            steer = post10["steeringAngleDeg"].to_numpy(float)
            peak_abs = float(np.max(np.abs(steer)))
            result["post10_steer_peak_deg"] = peak_abs
            result["post10_speed_min_mps"] = float(post10["vEgo"].min())
            v = post10["vEgo"].to_numpy(float)
            result["post10_speed_delta_mps"] = float(v[-1] - v[0]) if len(v) >= 2 else 0.0
            result["post10_dur_abs_steer_gt10_s"] = _duration_true(t, np.abs(steer) > 10.0)
            result["post10_dur_abs_steer_gt20_s"] = _duration_true(t, np.abs(steer) > 20.0)
            result["post10_sign_changes"] = _sign_changes(steer, deadband=1.0)
            result["post10_one_sided_ratio"] = _one_sided_ratio(steer)
            # Mean steer sign — positive = mostly left
            result["post10_steer_mean_sign"] = float(np.mean(steer))
            # end steer (last 1-2s of 10s window)
            late10 = cs[(cs["time_s"] >= event_time_s + 9.0) & (cs["time_s"] <= event_time_s + 10.5)]
            if late10.empty:
                late10 = post10.tail(max(1, len(post10) // 5))
            end_abs = float(np.mean(np.abs(late10["steeringAngleDeg"].to_numpy(float))))
            result["post10_steer_end_deg"] = end_abs

            # Steer return (based on long window)
            if peak_abs >= 5.0:
                ratio = end_abs / max(peak_abs, 1e-6)
                result["steer_return_ratio"] = float(ratio)
                result["steer_return"] = bool(ratio < LC_STEER_RETURN_RATIO)
            else:
                result["steer_return"] = True
                result["steer_return_ratio"] = 0.0

    # ── Curvature (dual window) ──
    ctrl_path = clip_dir / "controlsState.csv"
    if ctrl_path.exists():
        ctrl = pd.DataFrame()
        try:
            ctrl = pd.read_csv(ctrl_path, usecols=["time_s", "curvature", "desiredCurvature"], low_memory=False)
        except Exception:
            try:
                ctrl = pd.read_csv(ctrl_path, low_memory=False)
                need = {"time_s", "curvature", "desiredCurvature"}
                ctrl = ctrl[list(need)] if need.issubset(ctrl.columns) else pd.DataFrame()
            except Exception:
                ctrl = pd.DataFrame()

        if not ctrl.empty and len(ctrl) >= 3:
            ctrl["curvature"] = pd.to_numeric(ctrl["curvature"], errors="coerce")
            ctrl["desiredCurvature"] = pd.to_numeric(ctrl["desiredCurvature"], errors="coerce")

            pre_ctrl = ctrl[(ctrl["time_s"] >= event_time_s - WPRE_S) & (ctrl["time_s"] <= event_time_s)]
            post5_ctrl = ctrl[(ctrl["time_s"] >= event_time_s) & (ctrl["time_s"] <= event_time_s + WPOST_SHORT_S)]
            post10_ctrl = ctrl[(ctrl["time_s"] >= event_time_s) & (ctrl["time_s"] <= event_time_s + WPOST_LONG_S)]

            def _curv_features(window, prefix):
                if len(window) < 3:
                    return
                curv = window["curvature"].to_numpy(float)
                dcurv = window["desiredCurvature"].to_numpy(float)
                valid = np.isfinite(curv) & np.isfinite(dcurv)
                if valid.sum() < 3:
                    return
                t_v = window["time_s"].to_numpy(float)[valid]
                c_v = curv[valid]
                d_v = dcurv[valid]
                result[f"{prefix}_max_abs_curv"] = float(np.max(np.abs(c_v)))
                result[f"{prefix}_mean_abs_curv"] = float(np.mean(np.abs(c_v)))
                dev = np.abs(c_v - d_v)
                result[f"{prefix}_max_curv_deviation"] = float(np.max(dev))
                result[f"{prefix}_mean_curv_deviation"] = float(np.mean(dev))
                result[f"{prefix}_dur_moderate_curv_s"] = _duration_true(t_v, np.abs(c_v) > 0.005)
                result[f"{prefix}_dur_strong_curv_s"] = _duration_true(t_v, np.abs(c_v) > 0.01)
                result[f"{prefix}_curv_sign_consistency"] = _one_sided_ratio(c_v)

            # Pre
            if len(pre_ctrl) >= 3:
                curv = pre_ctrl["curvature"].to_numpy(float)
                dcurv = pre_ctrl["desiredCurvature"].to_numpy(float)
                valid_c = curv[np.isfinite(curv)]
                valid_d = dcurv[np.isfinite(dcurv)]
                if len(valid_c) > 0:
                    result["pre_max_abs_curv"] = float(np.max(np.abs(valid_c)))
                    result["pre_mean_abs_curv"] = float(np.mean(np.abs(valid_c)))
                if len(valid_d) > 0:
                    result["pre_max_abs_desired_curv"] = float(np.max(np.abs(valid_d)))

            _curv_features(post5_ctrl, "post5")
            _curv_features(post10_ctrl, "post10")

    return result


def extract_all_raw_features(master: pd.DataFrame, root: Path) -> pd.DataFrame:
    tasks = []
    for _, row in master.iterrows():
        clip_dir = (root / str(row["car_model"]) / str(row["dongle_id"])
                    / str(row["route_id"]) / str(int(row["clip_id"])))
        video_time_s, event_mono = _read_meta_event(clip_dir)
        tasks.append((str(clip_dir), video_time_s, event_mono))

    print(f"  Extracting raw features from {len(tasks):,} clips ({N_WORKERS} workers)...")
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_extract_raw_features, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                results.append(fut.result())
            except Exception:
                t = futures[fut]
                results.append({"clip_dir": t[0]})
            if done % 3000 == 0:
                print(f"    {done:,}/{len(tasks):,} done")
    print(f"    {done:,}/{len(tasks):,} done")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════

def detect_stationary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detector G: Stationary takeover (Ego).
    If the vehicle is essentially stopped before the event, the driver intentionally
    took over (e.g., at a red light, stop sign, parking).  Always Ego.
    Tightened in v7: require BOTH pre_speed_mean AND pre_speed_min < 0.5 m/s,
    plus some post-event driver action (steer, gas, or speed change).
    """
    near_zero_mean = df["pre_speed_mean_mps"].fillna(999) < STATIONARY_MAX_SPEED_MPS
    near_zero_min = df["pre_speed_min_mps"].fillna(999) < STATIONARY_MAX_SPEED_MPS
    both_near_zero = near_zero_mean & near_zero_min

    # Require some post-event evidence (driver did something after takeover)
    has_post_steer = df["post10_steer_peak_deg"].fillna(0) > 3.0
    has_post_speed = df["post10_speed_delta_mps"].fillna(0) > 0.5
    has_gas = df["trig_gas"].fillna(False).astype(bool)
    has_steer_trig = df["trig_steer"].fillna(False).astype(bool)
    has_post_evidence = has_post_steer | has_post_speed | has_gas | has_steer_trig

    flag = both_near_zero & has_post_evidence

    conf = pd.Series("none", index=df.index)
    conf[flag] = "high"

    return pd.DataFrame({"flag_stationary": flag, "conf_stationary": conf})


def detect_lane_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detector A: Lane-change (Ego).
    Uses BOTH short (5s) and long (10s) windows.
    A1: blinker + LC dynamics (in either window) + curvature OK
    A2: no blinker + planned_lc + steer return + visible steer + curvature OK
    """
    is_lc = df["post_maneuver_type"] == "lane_change"

    blinker_active = (
        (df["blinker_any_pre"] & (df["blinker_pre_duration_s"] >= LC_BLINKER_MIN_DURATION_S))
        | (df["blinker_any_post"] & (df["blinker_post_duration_s"] >= LC_BLINKER_MIN_DURATION_S))
    )

    # Curve override (sustained one-sided at moderate+ speed)
    # IMPORTANT: use 5s window, NOT 10s — in 10s many real LC show sustained one-sided
    # because the driver stays in new lane after the S-curve
    not_junction_speed = df["pre_speed_mean_mps"].fillna(0) > CURVE_OVERRIDE_SPEED_MPS
    sustained_steer = (
        (df["post5_dur_abs_steer_gt20_s"].fillna(0) > 2.5)  # 2.5s in 5s window (was 3.0 in 10s)
        & (df["post5_one_sided_ratio"].fillna(0) > CURVE_OVERRIDE_MIN_ONE_SIDED_RATIO)
    )
    pre_already_curving = df["pre_mean_abs_curv"].fillna(0) > CURVE_OVERRIDE_MIN_PRE_MEAN_CURV
    curve_override = not_junction_speed & (sustained_steer | pre_already_curving)

    # Curvature gates (use short-window curvature for tighter check)
    curv_brief = df["post5_dur_strong_curv_s"].fillna(999) <= LC_A1_MAX_POST_DUR_STRONG_CURV_S
    curv_low = df["post5_mean_abs_curv"].fillna(999) <= LC_A1_MAX_POST_MEAN_ABS_CURV
    curv_ok = curv_brief & curv_low

    # LC dynamics — check BOTH windows and take the best
    def _lc_dynamics(peak_col, osr_col, dur_col, sign_col):
        peak_ok = df[peak_col].fillna(0) >= LC_A1_MIN_POST_STEER_PEAK_DEG
        not_one_sided = df[osr_col].fillna(1.0) <= LC_A1_MAX_ONE_SIDED_RATIO
        shortish = df[dur_col].fillna(999) <= LC_A1_MAX_DUR_STEER_GT20_S
        lc_shape = (df[sign_col].fillna(0).astype(int) >= 1) | df["steer_return"].fillna(False).astype(bool)
        return peak_ok & not_one_sided & shortish & lc_shape

    lc_dyn_5 = _lc_dynamics("post5_steer_peak_deg", "post5_one_sided_ratio",
                            "post5_dur_abs_steer_gt20_s", "post5_sign_changes")
    lc_dyn_10 = _lc_dynamics("post10_steer_peak_deg", "post10_one_sided_ratio",
                             "post10_dur_abs_steer_gt20_s", "post10_sign_changes")
    lc_dynamics = lc_dyn_5 | lc_dyn_10

    # A1: blinker + (LC maneuver OR LC dynamics) + curvature + NOT curve override
    a1 = blinker_active & (is_lc | lc_dynamics) & curv_ok & ~curve_override

    # A2: no blinker + planned LC + straight pre + steer return + visible steer + curv OK
    planned_lc = df["scenario"] == "planned_lane_change"
    pre_not_sustained = df["pre_dur_abs_steer_gt20_s"].fillna(0) < 0.75
    returns = df["steer_return"].fillna(False).astype(bool)
    # Visible steer in either window
    visible = (
        (df["post5_steer_peak_deg"].fillna(0) >= LC_POST_STEER_MIN_DEG)
        | (df["post10_steer_peak_deg"].fillna(0) >= LC_POST_STEER_MIN_DEG)
    )
    a2 = is_lc & ~blinker_active & planned_lc & pre_not_sustained & returns & visible & curv_ok & ~curve_override

    # A3 (v7): LC maneuver + steer return + visible steer + curv OK
    # Catches conflict-triggered lane changes (scenario != planned_lane_change but
    # post_maneuver_type == lane_change with good LC dynamics)
    a3 = is_lc & ~blinker_active & ~planned_lc & returns & visible & curv_ok & ~curve_override & lc_dynamics & ~a1 & ~a2

    flag = a1 | a2 | a3
    conf = pd.Series("none", index=df.index)
    conf[a3] = "medium"
    conf[a2] = "medium"
    conf[a1] = "high"

    return pd.DataFrame({
        "flag_lc": flag, "conf_lc": conf,
        "flag_lc_a1": a1, "flag_lc_a2": a2,
        "flag_lc_curve_override": curve_override & (blinker_active | is_lc),
    })


def detect_junction_turn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detector B: Junction/intersection turn (Ego).

    Key insight: sharp curves NEVER slow below ~4.5 m/s (10 mph),
    but junction turns DO decelerate to near-stop.

    Uses 10s post window for catching late-onset turns.

    B1 (high):  near-stop (< 4.5 m/s) + visible steer in 10s window
    B2 (medium): turn_ramp maneuver + low speed + curvature evidence
    B3 (medium): low speed + blinker + visible steer in 10s window
    B4 (low):   low speed + left turn (left turns are almost always Ego)
    """
    is_turn = df["post_maneuver_type"] == "turn_ramp"
    near_stop = df["pre_speed_mean_mps"].fillna(999) < JUNCTION_NEAR_STOP_MPS
    low_speed = df["pre_speed_mean_mps"].fillna(999) < JUNCTION_LOW_SPEED_MPS

    # Visible steer in either window
    visible_5 = df["post5_steer_peak_deg"].fillna(0) >= JUNCTION_POST_STEER_MIN
    visible_10 = df["post10_steer_peak_deg"].fillna(0) >= JUNCTION_POST_STEER_MIN
    visible_any = visible_5 | visible_10
    # Also check analysis_master steer
    visible_master = df["post_max_abs_steer_angle_deg"].fillna(0) > 10.0

    # Sustained steering (one-sided) in either window
    sustained_5 = (
        (df["post5_dur_abs_steer_gt20_s"].fillna(0) >= TURN_MIN_DUR_STEER_GT20_S)
        & (df["post5_one_sided_ratio"].fillna(0) >= TURN_MIN_ONE_SIDED_RATIO)
    )
    sustained_10 = (
        (df["post10_dur_abs_steer_gt20_s"].fillna(0) >= TURN_MIN_DUR_STEER_GT20_S)
        & (df["post10_one_sided_ratio"].fillna(0) >= TURN_MIN_ONE_SIDED_RATIO)
    )
    sustained_steer = sustained_5 | sustained_10

    # Curvature evidence
    strong_curv_5 = df["post5_dur_strong_curv_s"].fillna(0) >= TURN_MIN_POST_DUR_STRONG_CURV_S
    strong_curv_10 = df["post10_dur_strong_curv_s"].fillna(0) >= TURN_MIN_POST_DUR_STRONG_CURV_S
    high_curv_peak = (
        (df["post5_max_abs_curv"].fillna(0) >= TURN_MIN_POST_MAX_ABS_CURV)
        | (df["post10_max_abs_curv"].fillna(0) >= TURN_MIN_POST_MAX_ABS_CURV)
    )
    curv_evidence = (strong_curv_5 | strong_curv_10) & high_curv_peak

    # Pre not already in sustained curve
    pre_not_sustained = df["pre_dur_abs_steer_gt20_s"].fillna(0) < 1.0

    # Speed check: sharp curves maintain speed > 4.47 m/s (10 mph)
    # If post speed drops below this, it's NOT a sharp curve → likely a turn
    post_speed_low_5 = df["post5_speed_min_mps"].fillna(999) < SHARP_CURVE_MIN_SPEED_MPS
    post_speed_low_10 = df["post10_speed_min_mps"].fillna(999) < SHARP_CURVE_MIN_SPEED_MPS
    post_speed_drops = post_speed_low_5 | post_speed_low_10

    # Left turn detection: steering mean > 0 indicates left (in most openpilot conventions)
    # Left turns are almost always Ego
    is_left_turn = (
        (df["post10_steer_mean_sign"].fillna(0) > 5.0)  # mean steer > 5° left
        & (df["post10_steer_peak_deg"].fillna(0) > 20.0)  # visible steer
    )

    blinker = df["blinker_any_pre"].fillna(False).astype(bool) | df["blinker_any_post"].fillna(False).astype(bool)
    left_blinker = df["blinker_left_pre"].fillna(False).astype(bool) | df["blinker_left_post"].fillna(False).astype(bool)

    # B1: near-stop + (visible steer OR curvature evidence OR sustained steer)
    # Near-stop means speed < 4.5 m/s → definitely not a sharp curve
    b1 = near_stop & (visible_any | visible_master | curv_evidence | sustained_steer)

    # B2: turn_ramp + low speed + curvature + pre not sustained
    b2 = is_turn & low_speed & curv_evidence & pre_not_sustained & ~b1

    # B3: low speed + blinker + (visible steer OR curv evidence)
    b3 = low_speed & blinker & (visible_any | curv_evidence) & ~b1 & ~b2

    # B4: left turn (left blinker OR left steer direction) + low speed
    b4 = low_speed & (left_blinker | is_left_turn) & (visible_any | visible_master) & ~b1 & ~b2 & ~b3

    flag = b1 | b2 | b3 | b4
    conf = pd.Series("none", index=df.index)
    conf[b4] = "low"
    conf[b3] = "medium"
    conf[b2] = "medium"
    conf[b1] = "high"

    return pd.DataFrame({
        "flag_turn": flag, "conf_turn": conf,
        "flag_turn_from_stop": b1,
    })


def detect_discretionary_accel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detector C: Discretionary acceleration (Ego).
    Tightened: requires stronger speed evidence to avoid false positives
    on trivial or system-initiated speed changes.
    """
    is_accel = df["post_maneuver_type"] == "acceleration"
    # Use analysis_master speed delta (5s window) — require significant
    speed_up = df["post_speed_delta_mps"].fillna(0) > ACCEL_POST_SPEED_DELTA_MIN_MPS
    gas_trigger = df["trig_gas"].fillna(False).astype(bool)
    low_risk = df["risk_score"].fillna(1) < ACCEL_MAX_RISK_SCORE
    no_conflict = ~df["flag_longitudinal_conflict"].fillna(False).astype(bool)
    safe_scenario = df["scenario"].isin(["planned_acceleration", "discretionary"])
    straight_steer = df["post_max_abs_steer_angle_deg"].fillna(999) < ACCEL_POST_STEER_MAX_DEG
    straight_curv = df["post5_mean_abs_curv"].fillna(999) < ACCEL_MAX_POST_MEAN_ABS_CURV

    c1 = is_accel & speed_up & gas_trigger & low_risk & no_conflict & safe_scenario & straight_steer & straight_curv

    conf = pd.Series("none", index=df.index)
    conf[c1] = "high"
    return pd.DataFrame({"flag_accel": c1, "conf_accel": conf})


def detect_conflict_reactive(df: pd.DataFrame) -> pd.DataFrame:
    """Detector D: Conflict/reactive (Non-ego evidence)."""
    ttc_crit = df["pre_ttc_min_capped_s"].fillna(999) < TTC_CRITICAL_S
    thw_crit = df["pre_thw_min_s"].fillna(999) < THW_CRITICAL_S
    drac_crit = df["pre_drac_max_capped_mps2"].fillna(0) > DRAC_CRITICAL_MPS2
    fcw = df["pre_fcw_present"].fillna(False).astype(str).str.lower() == "true"
    close_lead = (
        (df["pre_lead_present_rate"].fillna(0) > 0.3)
        & (df["pre_min_drel_m"].fillna(999) < CLOSE_DREL_M)
    )
    any_conflict = ttc_crit | thw_crit | drac_crit | fcw | close_lead

    n_ind = (ttc_crit.astype(int) + thw_crit.astype(int) + drac_crit.astype(int)
             + fcw.astype(int) + close_lead.astype(int))
    conf = pd.Series("none", index=df.index)
    conf[any_conflict & (n_ind == 1)] = "low"
    conf[any_conflict & (n_ind == 2)] = "medium"
    conf[any_conflict & (n_ind >= 3)] = "high"

    return pd.DataFrame({
        "flag_conflict": any_conflict, "conf_conflict": conf,
        "conflict_ttc": ttc_crit, "conflict_thw": thw_crit,
        "conflict_drac": drac_crit, "conflict_fcw": fcw,
        "conflict_close_lead": close_lead,
    })


def detect_curve_boundary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detector E: Curve/ODD boundary (Non-ego).
    Enhanced: also check post speed — if speed drops below 10 mph,
    it's NOT a sharp curve (might be a turn).
    """
    high_curv = df.get("pre_max_abs_curvature", pd.Series(0, index=df.index)).fillna(0) > CURVE_PRE_CURVATURE_MIN
    high_steer_rate = df.get("pre_steer_rate_max_deg_per_s", pd.Series(0, index=df.index)).fillna(0) > CURVE_PRE_STEER_RATE_MIN
    no_blinker = ~df["blinker_any_pre"].fillna(False).astype(bool)
    not_lc = df["post_maneuver_type"] != "lane_change"

    has_lane = df.get("pre_has_lane_probs", pd.Series(False, index=df.index)).fillna(False).astype(str).str.lower() == "true"
    low_lane_l = df.get("pre_lane_left_prob_mean", pd.Series(1.0, index=df.index)).fillna(1.0) < CURVE_LANE_PROB_MAX
    low_lane_r = df.get("pre_lane_right_prob_mean", pd.Series(1.0, index=df.index)).fillna(1.0) < CURVE_LANE_PROB_MAX
    low_lane = has_lane & low_lane_l & low_lane_r

    pre_sustained = (
        (df["pre_dur_abs_steer_gt20_s"].fillna(0) > 1.5)
        & (df["pre_one_sided_ratio"].fillna(0) > 0.92)
    )
    pre_high_curv = df["pre_mean_abs_curv"].fillna(0) > CURVE_OVERRIDE_MIN_PRE_MEAN_CURV

    # Speed gate: sharp curves maintain speed above 10 mph
    # If speed drops below, it's more likely a junction turn, not a curve
    speed_stays_high_5 = df["post5_speed_min_mps"].fillna(999) >= SHARP_CURVE_MIN_SPEED_MPS
    speed_stays_high_10 = df["post10_speed_min_mps"].fillna(999) >= SHARP_CURVE_MIN_SPEED_MPS
    speed_stays_high = speed_stays_high_5 & speed_stays_high_10

    e1 = (high_curv | pre_sustained | pre_high_curv) & no_blinker & not_lc & speed_stays_high
    e2 = high_steer_rate & no_blinker & not_lc & low_lane & speed_stays_high & ~e1

    flag = e1 | e2
    conf = pd.Series("none", index=df.index)
    conf[e2] = "medium"
    conf[e1] = "high"

    return pd.DataFrame({"flag_curve": flag, "conf_curve": conf})


def detect_system_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """Detector F: System-initiated / unknown (Non-ego)."""
    alert_text = df["pre_alert_text"].fillna("")
    system_alert = pd.Series(False, index=df.index)
    for pat in SYSTEM_ALERTS:
        system_alert |= alert_text.str.contains(pat, case=False, na=False)

    no_driver = (
        ~df["trig_steer"].fillna(False).astype(bool)
        & ~df["trig_brake"].fillna(False).astype(bool)
        & ~df["trig_gas"].fillna(False).astype(bool)
    )

    f1 = system_alert & no_driver
    f2 = system_alert & ~f1
    flag = f1 | f2
    conf = pd.Series("none", index=df.index)
    conf[f2] = "medium"
    conf[f1] = "high"

    return pd.DataFrame({
        "flag_system": flag, "conf_system": conf,
        "flag_system_alert": system_alert,
        "flag_no_driver_trigger": no_driver,
    })


# ═══════════════════════════════════════════════════════════════════════
# CLASSIFICATION — Binary: Ego / Non-ego
# ═══════════════════════════════════════════════════════════════════════
CONF_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary classification: Ego / Non-ego  (v7).

    Priority order:
      1) Stationary → Ego (always)
      2) Junction turn from near-stop with borderline conflict → Ego
      3) BLINKER + ego detector → Ego (overrides conflict, NOT system/curve)
      4) Clean ego evidence (LC/turn/accel) without non-ego → Ego
      5) Ego detector with only close_lead conflict → Ego
      6) Everything else → Non-ego
    """
    label = pd.Series("Non-ego", index=df.index)
    ego_reason = pd.Series("", index=df.index)
    nonego_reason = pd.Series("other", index=df.index)
    confidence = pd.Series("low", index=df.index)

    has_stationary = df["flag_stationary"].fillna(False).astype(bool)
    has_lc = df["flag_lc"].fillna(False).astype(bool)
    has_turn = df["flag_turn"].fillna(False).astype(bool)
    has_accel = df["flag_accel"].fillna(False).astype(bool)
    any_ego = has_lc | has_turn | has_accel | has_stationary

    has_conflict = df["flag_conflict"].fillna(False).astype(bool)
    has_curve = df["flag_curve"].fillna(False).astype(bool)
    has_system = df["flag_system"].fillna(False).astype(bool)
    any_nonego = has_conflict | has_curve | has_system

    blinker_active = (
        df["blinker_any_pre"].fillna(False).astype(bool)
        | df["blinker_any_post"].fillna(False).astype(bool)
    )

    # Helper for borderline conflict (close_lead only, no TTC/DRAC/FCW)
    borderline_conflict = (
        has_conflict
        & df.get("conflict_close_lead", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        & ~df.get("conflict_ttc", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        & ~df.get("conflict_drac", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        & ~df.get("conflict_fcw", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    )

    # ── 1. Stationary → Ego always ──
    label[has_stationary] = "Ego"
    ego_reason[has_stationary] = "stationary_takeover"
    confidence[has_stationary] = "high"

    # ── 2. Ego + blinker → Ego (overrides ALL conflict, NOT system/curve) ──
    # Blinker = strongest ego signal.  Ego detector + blinker overrides conflict.
    not_yet_ego = label == "Non-ego"
    ego_blinker = (has_lc | has_turn | has_accel) & blinker_active & ~has_system & ~has_curve & not_yet_ego
    label[ego_blinker] = "Ego"
    ego_reason[ego_blinker & has_lc] = "lane_change"
    ego_reason[ego_blinker & has_turn & ~has_lc] = "junction_turn"
    ego_reason[ego_blinker & has_accel & ~has_lc & ~has_turn] = "discretionary_accel"
    confidence[ego_blinker] = "high"

    # ── 3. Turn from near-stop → Ego (overrides ALL conflict) ──
    not_yet_ego = label == "Non-ego"
    from_stop = df.get("flag_turn_from_stop", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    junction_override = from_stop & ~has_system & ~has_curve & not_yet_ego
    label[junction_override] = "Ego"
    ego_reason[junction_override] = "junction_turn"
    confidence[junction_override] = "high"

    # ── 3b. Discretionary accel → Ego (overrides conflict) ──
    # Gas-trigger acceleration is almost always driver-initiated even with lead car
    not_yet_ego = label == "Non-ego"
    accel_override = has_accel & ~has_system & ~has_curve & not_yet_ego
    label[accel_override] = "Ego"
    ego_reason[accel_override] = "discretionary_accel"
    confidence[accel_override] = "medium"

    # ── 4. Clean ego evidence, no non-ego ──
    not_yet_ego = label == "Non-ego"
    ego_clean = (has_lc | has_turn | has_accel) & ~any_nonego & not_yet_ego
    label[ego_clean] = "Ego"
    ego_reason[ego_clean & has_lc] = "lane_change"
    ego_reason[ego_clean & has_turn & ~has_lc] = "junction_turn"
    ego_reason[ego_clean & has_accel & ~has_lc & ~has_turn] = "discretionary_accel"
    for src, cc in [("flag_lc", "conf_lc"), ("flag_turn", "conf_turn"), ("flag_accel", "conf_accel")]:
        mask = ego_clean & df[src].fillna(False).astype(bool)
        confidence[mask] = df.loc[mask, cc]

    # ── 5. Ego detector with borderline conflict → Ego ──
    not_yet_ego = label == "Non-ego"
    ego_border = (has_lc | has_turn | has_accel) & borderline_conflict & ~has_curve & ~has_system & not_yet_ego
    label[ego_border] = "Ego"
    ego_reason[ego_border & has_lc] = "lane_change"
    ego_reason[ego_border & has_turn & ~has_lc] = "junction_turn"
    ego_reason[ego_border & has_accel & ~has_lc & ~has_turn] = "discretionary_accel"
    confidence[ego_border] = "medium"

    # ── 6. Blinker alone → Ego if no conflict at all ──
    not_yet_ego = label == "Non-ego"
    blinker_only = blinker_active & ~has_system & ~has_curve & ~has_conflict & not_yet_ego
    label[blinker_only] = "Ego"
    ego_reason[blinker_only] = "blinker_intent"
    confidence[blinker_only] = "medium"

    # ── 7. Non-ego reasons ──
    is_nonego = label == "Non-ego"
    nonego_reason[is_nonego & has_conflict] = "conflict_reactive"
    nonego_reason[is_nonego & has_curve & ~has_conflict] = "curve_boundary"
    nonego_reason[is_nonego & has_system & ~has_conflict & ~has_curve] = "system_unknown"
    mixed = any_ego & any_nonego & is_nonego
    nonego_reason[mixed] = "mixed_ego_nonego"

    for src, cc in [("flag_conflict", "conf_conflict"), ("flag_curve", "conf_curve"), ("flag_system", "conf_system")]:
        mask = is_nonego & df[src].fillna(False).astype(bool)
        upgrade = mask & (df[cc].map(CONF_RANK) > confidence.map(CONF_RANK))
        confidence[upgrade] = df.loc[upgrade, cc]

    return pd.DataFrame({
        "label": label,
        "ego_reason": ego_reason,
        "nonego_reason": nonego_reason.where(label == "Non-ego", ""),
        "confidence": confidence,
    })


# ═══════════════════════════════════════════════════════════════════════
# SYMLINK / COPY
# ═══════════════════════════════════════════════════════════════════════
def create_links(df: pd.DataFrame, root: Path, out_root: Path, mode: str = "link"):
    dirs = {}
    for lbl in ("Ego", "Non-ego"):
        d = out_root / lbl
        d.mkdir(parents=True, exist_ok=True)
        for item in d.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        dirs[lbl] = d

    # Clean old Uncertain folder
    unc_dir = out_root / "Uncertain"
    if unc_dir.exists():
        for item in unc_dir.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()

    created = {k: 0 for k in dirs}
    errors = 0
    for _, row in df.iterrows():
        clip_dir = (root / str(row["car_model"]) / str(row["dongle_id"])
                    / str(row["route_id"]) / str(int(row["clip_id"])))
        if not clip_dir.exists():
            errors += 1
            continue
        lbl = row["label"]
        target_dir = dirs[lbl]
        dongle_short = str(row["dongle_id"])[:8]
        link_name = f"{row['car_model']}__{dongle_short}__{row['route_id']}__{int(row['clip_id'])}"
        link_path = target_dir / link_name
        if mode == "copy":
            if not link_path.exists():
                shutil.copytree(clip_dir, link_path)
        else:
            try:
                rel = os.path.relpath(clip_dir, target_dir)
                link_path.symlink_to(rel)
            except OSError:
                link_path.symlink_to(clip_dir)
        created[lbl] += 1

    print(f"  Created {created['Ego']:,} Ego, {created['Non-ego']:,} Non-ego "
          f"{'symlinks' if mode == 'link' else 'copies'} ({errors} missing dirs)")


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════
def generate_report(df: pd.DataFrame, out_root: Path):
    lines = []
    N = len(df)
    lines.append("# Ego vs Non-ego Classification Report (v7)\n")
    lines.append(f"**Total clips**: {N:,}\n")

    lines.append("## Label Distribution\n")
    lines.append("| Label | Count | % |")
    lines.append("|-------|------:|--:|")
    for lbl in ["Ego", "Non-ego"]:
        cnt = (df["label"] == lbl).sum()
        lines.append(f"| {lbl} | {cnt:,} | {cnt / N * 100:.1f}% |")

    lines.append("\n## Ego Reasons\n")
    ego = df[df["label"] == "Ego"]
    lines.append("| Reason | Count | % of Ego |")
    lines.append("|--------|------:|---------:|")
    for r in ego["ego_reason"].value_counts().index:
        cnt = (ego["ego_reason"] == r).sum()
        lines.append(f"| {r} | {cnt:,} | {cnt / max(len(ego), 1) * 100:.1f}% |")

    lines.append("\n## Non-ego Reasons\n")
    ne = df[df["label"] == "Non-ego"]
    lines.append("| Reason | Count | % of Non-ego |")
    lines.append("|--------|------:|-------------:|")
    for r in ne["nonego_reason"].value_counts().index:
        cnt = (ne["nonego_reason"] == r).sum()
        lines.append(f"| {r} | {cnt:,} | {cnt / max(len(ne), 1) * 100:.1f}% |")

    report_path = out_root / "ego_nonego_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Report saved → {report_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    global N_WORKERS
    parser = argparse.ArgumentParser(description="Classify takeover clips: Ego / Non-ego (v7)")
    parser.add_argument("--repo_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--master_csv", type=Path, default=MASTER_CSV)
    parser.add_argument("--workers", type=int, default=N_WORKERS)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--link", action="store_true", help="Create symlinks (default).")
    group.add_argument("--copy", action="store_true", help="Copy clip directories.")
    args = parser.parse_args()

    N_WORKERS = int(args.workers)
    mode = "copy" if args.copy else "link"
    root = args.repo_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Loading analysis_master.csv...")
    master = pd.read_csv(args.master_csv, low_memory=False)
    print(f"  {len(master):,} clips loaded, {len(master.columns)} columns")

    # 2. Extract raw features (dual-window: 5s + 10s)
    print("Extracting raw features (dual-window 5s+10s, blinker, steer, curvature)...")
    master["_clip_dir"] = master.apply(
        lambda r: str(root / str(r["car_model"]) / str(r["dongle_id"])
                      / str(r["route_id"]) / str(int(r["clip_id"]))),
        axis=1,
    )
    raw_df = extract_all_raw_features(master, root)
    raw_df = raw_df.rename(columns={"clip_dir": "_clip_dir"})
    master = master.merge(raw_df, on="_clip_dir", how="left")

    # Fill NaN defaults
    bool_cols = [
        "blinker_any_pre", "blinker_any_post", "blinker_left_pre", "blinker_right_pre",
        "blinker_left_post", "blinker_right_post", "steer_return",
    ]
    for col in bool_cols:
        if col in master.columns:
            master[col] = master[col].fillna(False)

    float_cols = [c for c in master.columns if c.startswith(("post5_", "post10_", "pre_")) and master[c].dtype != object]
    for col in float_cols:
        master[col] = master[col].fillna(0.0)
    for col in ["blinker_pre_duration_s", "blinker_post_duration_s",
                "event_time_s", "event_mono_err_ns"]:
        if col in master.columns:
            master[col] = master[col].fillna(0.0)
    if "steer_return_ratio" in master.columns:
        master["steer_return_ratio"] = master["steer_return_ratio"].fillna(1.0)

    # 3. Run detectors
    print("Running detectors...")
    det_stat = detect_stationary(master)
    det_lc = detect_lane_change(master)
    det_turn = detect_junction_turn(master)
    det_accel = detect_discretionary_accel(master)
    det_conflict = detect_conflict_reactive(master)
    det_curve = detect_curve_boundary(master)
    det_system = detect_system_unknown(master)

    for det_df in [det_stat, det_lc, det_turn, det_accel, det_conflict, det_curve, det_system]:
        for col in det_df.columns:
            master[col] = det_df[col].values

    # 4. Classify
    print("Classifying...")
    cls = classify(master)
    for col in cls.columns:
        master[col] = cls[col].values

    # 5. Save CSV
    label_cols = [
        "car_model", "brand", "dongle_id", "route_id", "clip_id",
        "source", "source_group",
        "label", "ego_reason", "nonego_reason", "confidence",
        "event_time_s", "event_time_source",
        # Blinker
        "blinker_any_pre", "blinker_any_post",
        "blinker_left_pre", "blinker_right_pre", "blinker_left_post", "blinker_right_post",
        "blinker_pre_duration_s", "blinker_post_duration_s",
        # Steering (short)
        "post5_steer_peak_deg", "post5_dur_abs_steer_gt20_s", "post5_one_sided_ratio",
        # Steering (long)
        "post10_steer_peak_deg", "post10_dur_abs_steer_gt20_s", "post10_one_sided_ratio",
        "post10_steer_mean_sign",
        "steer_return", "steer_return_ratio",
        "pre_dur_abs_steer_gt20_s", "pre_one_sided_ratio",
        # Curvature (short)
        "post5_max_abs_curv", "post5_mean_abs_curv", "post5_dur_strong_curv_s",
        "post5_max_curv_deviation", "post5_curv_sign_consistency",
        # Curvature (long)
        "post10_max_abs_curv", "post10_mean_abs_curv", "post10_dur_strong_curv_s",
        "post10_max_curv_deviation", "post10_curv_sign_consistency",
        "pre_max_abs_curv", "pre_mean_abs_curv",
        # Speed
        "post5_speed_min_mps", "post10_speed_min_mps", "pre_speed_min_mps",
        "post5_speed_delta_mps", "post10_speed_delta_mps",
        # Detector flags
        "flag_stationary", "conf_stationary",
        "flag_lc", "flag_lc_a1", "flag_lc_a2", "flag_lc_curve_override", "conf_lc",
        "flag_turn", "flag_turn_from_stop", "conf_turn",
        "flag_accel", "conf_accel",
        "flag_conflict", "conf_conflict",
        "conflict_ttc", "conflict_thw", "conflict_drac", "conflict_fcw", "conflict_close_lead",
        "flag_curve", "conf_curve",
        "flag_system", "conf_system",
        # Context
        "post_maneuver_type", "scenario", "risk_score",
        "pre_speed_mean_mps", "post_speed_delta_mps",
        "pre_max_abs_steer_angle_deg", "post_max_abs_steer_angle_deg",
        "pre_max_abs_curvature", "post_max_abs_curvature",
        "trig_steer", "trig_brake", "trig_gas", "primary_trigger",
        "pre_alert_text",
    ]
    label_cols = [c for c in label_cols if c in master.columns]
    csv_path = out_root / "ego_nonego_labels.csv"
    master[label_cols].to_csv(csv_path, index=False)
    print(f"  Labels saved → {csv_path}")

    # 6. Symlinks
    print(f"Creating {mode}s...")
    create_links(master, root, out_root, mode=mode)

    # 7. Report
    print("Generating report...")
    generate_report(master, out_root)

    # 8. Summary
    N = len(master)
    n_ego = int((master["label"] == "Ego").sum())
    n_nonego = int((master["label"] == "Non-ego").sum())
    print("\n" + "=" * 65)
    print("  EGO vs NON-EGO CLASSIFICATION SUMMARY (v7)")
    print("=" * 65)
    print(f"  Total clips:  {N:,}")
    print(f"  Ego:          {n_ego:,}  ({n_ego / N * 100:.1f}%)")
    print(f"  Non-ego:      {n_nonego:,}  ({n_nonego / N * 100:.1f}%)")
    print()
    ego_df = master[master["label"] == "Ego"]
    print("  Ego breakdown:")
    for r, cnt in ego_df["ego_reason"].value_counts().items():
        print(f"    {r:25s}: {cnt:,}")
    ne_df = master[master["label"] == "Non-ego"]
    print("  Non-ego breakdown:")
    for r, cnt in ne_df["nonego_reason"].value_counts().items():
        print(f"    {r:25s}: {cnt:,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
