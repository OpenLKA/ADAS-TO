#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_derived_signals_v3.py
=============================
Stage 1 v3: Compute per-clip safety metrics on a common resampled time grid.

Key upgrades over v2:
  1. Resampling: all topics interpolated to a common 20 Hz grid using
     timestamp-based linear interpolation (continuous) or forward-fill
     (booleans/categoricals). This makes derivative features comparable
     across qlog (10 Hz) and rlog (100 Hz) clips.
  2. Robust TTC/DRAC: closing-speed floor (0.5 m/s), dRel floor (5 m),
     and capped variants to suppress extreme tail artifacts.
  3. Richer summaries: quantiles (P5/P50/P95), exposure durations
     (time below TTC threshold), time-to-extrema, lead continuity,
     severity integrals.
  4. Stabilization for both [0,5]s and [0,10]s windows.
  5. Anomaly flags saved to a separate CSV for triage.
  6. Accelerometer roughness on native timestamps (not resampled) because
     qlog has only ~1 Hz accelerometer — resampling would fabricate data.

Run:
    python3 compute_derived_signals_v3.py              # full run
    python3 compute_derived_signals_v3.py --spot 100   # spot-check 100 clips

Outputs:
    stats_output/derived_signals_v3.parquet  (+ .csv)
    stats_output/anomaly_flags.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  Paths & Config
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
OUT  = CODE / "stats_output"
OUT.mkdir(parents=True, exist_ok=True)

with open(CODE / "configs" / "analysis_thresholds.yaml") as f:
    CFG = yaml.safe_load(f)

EPS       = CFG["eps"]
N_WORKERS = CFG["n_workers"]
SEED      = CFG["random_seed"]
np.random.seed(SEED)

# Windows
W_PRE   = CFG["pre_window"]      # [-5, 0]
W_POST  = CFG["post_window"]     # [0, 5]
W_FULL  = CFG["full_window"]     # [-10, 10]

# Resampling
RESAMPLE_HZ    = CFG["resampling"]["hz"]
RESAMPLE_DT    = 1.0 / RESAMPLE_HZ

# Smoothing
SMOOTH_CFG     = CFG["smoothing"]
SMOOTH_WIN_S   = SMOOTH_CFG["savgol_window_s"]
SMOOTH_POLY    = SMOOTH_CFG["savgol_polyorder"]

# Robust TTC/DRAC
RTTC           = CFG["robust_ttc_drac"]
CLOSING_MIN    = RTTC["closing_speed_min_mps"]
DREL_MIN       = RTTC["drel_min_m"]
TTC_CAP        = RTTC["ttc_cap_s"]
DRAC_CAP       = RTTC["drac_cap_mps2"]

# Exposure
EXPOSURE       = CFG["exposure_thresholds"]

# Stabilization
STAB           = CFG["stabilization"]

# Anomaly
ANOM           = CFG["anomaly"]

MIN_V_THW      = CFG["min_speed_for_thw_mps"]
_NAN           = float("nan")

# ──────────────────────────────────────────────────────────────────────────────
#  I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
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


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Resampling
# ──────────────────────────────────────────────────────────────────────────────
def resample_topic(df: pd.DataFrame, t_grid: np.ndarray,
                   continuous_cols: list[str],
                   bool_cols: list[str] | None = None,
                   cat_cols: list[str] | None = None) -> pd.DataFrame:
    """Resample a topic DataFrame onto a common time grid.

    - continuous_cols: linear interpolation
    - bool_cols: forward-fill then cast to bool
    - cat_cols: forward-fill (string)
    Time column must be 'time_s'.
    Returns DataFrame indexed by t_grid with requested columns.
    """
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame({"time_s": t_grid})

    df = df.sort_values("time_s").copy()
    out = pd.DataFrame({"time_s": t_grid})

    # Convert columns to numeric where needed
    for c in continuous_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Linear interpolation for continuous
    for c in continuous_cols:
        if c not in df.columns:
            out[c] = _NAN
            continue
        valid = df[["time_s", c]].dropna()
        if len(valid) < 2:
            out[c] = valid[c].iloc[0] if len(valid) == 1 else _NAN
            continue
        out[c] = np.interp(t_grid, valid["time_s"].values, valid[c].values,
                           left=_NAN, right=_NAN)

    # Boolean: forward-fill
    for c in (bool_cols or []):
        if c not in df.columns:
            out[c] = False
            continue
        bvals = parse_bool(df[c])
        # merge_asof-style: for each grid point, take last known value
        merged = pd.merge_asof(
            out[["time_s"]], df[["time_s"]].assign(**{c: bvals}),
            on="time_s", direction="backward")
        out[c] = merged[c].fillna(False).astype(bool)

    # Categorical: forward-fill
    for c in (cat_cols or []):
        if c not in df.columns:
            out[c] = ""
            continue
        merged = pd.merge_asof(
            out[["time_s"]], df[["time_s", c]],
            on="time_s", direction="backward")
        out[c] = merged[c].fillna("").astype(str)

    return out


def build_resampled_topics(clip_dir: Path, event_t: float) -> dict[str, pd.DataFrame]:
    """Read all topic CSVs, resample onto a common grid, return dict."""
    t_lo = event_t + W_FULL[0]
    t_hi = event_t + W_FULL[1]
    t_grid = np.arange(t_lo, t_hi + RESAMPLE_DT / 2, RESAMPLE_DT)

    topics = {}

    # carState
    cs_raw = safe_read_csv(clip_dir / "carState.csv")
    topics["carState"] = resample_topic(
        cs_raw, t_grid,
        continuous_cols=["vEgo", "aEgo", "steeringAngleDeg", "steeringTorque",
                         "gas", "brake", "cruiseState.speed"],
        bool_cols=["steeringPressed", "brakePressed", "gasPressed",
                   "cruiseState.enabled", "standstill"],
    )

    # radarState
    radar_raw = safe_read_csv(clip_dir / "radarState.csv")
    topics["radarState"] = resample_topic(
        radar_raw, t_grid,
        continuous_cols=["leadOne.dRel", "leadOne.vRel", "leadOne.aRel",
                         "leadOne.vLead"],
        bool_cols=["leadOne.status"],
    )

    # controlsState
    ctrl_raw = safe_read_csv(clip_dir / "controlsState.csv")
    topics["controlsState"] = resample_topic(
        ctrl_raw, t_grid,
        continuous_cols=["curvature", "desiredCurvature", "vCruise"],
        bool_cols=["enabled", "active"],
        cat_cols=["alertText1"],
    )

    # longitudinalPlan
    lp_raw = safe_read_csv(clip_dir / "longitudinalPlan.csv")
    topics["longitudinalPlan"] = resample_topic(
        lp_raw, t_grid,
        continuous_cols=["aTarget"],
        bool_cols=["fcw", "hasLead", "shouldStop"],
    )

    # drivingModelData
    dm_raw = safe_read_csv(clip_dir / "drivingModelData.csv")
    topics["drivingModelData"] = resample_topic(
        dm_raw, t_grid,
        continuous_cols=["laneLineMeta.leftProb", "laneLineMeta.rightProb"],
    )

    # accelerometer — NOT resampled (qlog has ~1Hz; resampling fabricates)
    accel_raw = safe_read_csv(clip_dir / "accelerometer.csv")
    topics["accelerometer_native"] = accel_raw

    return topics


# ──────────────────────────────────────────────────────────────────────────────
#  Smoothing + differentiation at resampled rate
# ──────────────────────────────────────────────────────────────────────────────
def smooth_savgol(values: np.ndarray) -> np.ndarray:
    """SavGol at RESAMPLE_HZ with config window."""
    n = len(values)
    if n < 5:
        return values.copy()
    win = max(3, int(SMOOTH_WIN_S * RESAMPLE_HZ))
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


def diff_dt(values: np.ndarray) -> np.ndarray:
    """Derivative using constant RESAMPLE_DT."""
    return np.diff(values) / RESAMPLE_DT

# ──────────────────────────────────────────────────────────────────────────────
#  Window slicer on resampled grid
# ──────────────────────────────────────────────────────────────────────────────
def win_slice(df: pd.DataFrame, event_t: float, w: list) -> pd.DataFrame:
    """Slice resampled DataFrame to [event_t + w[0], event_t + w[1]]."""
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame()
    lo = event_t + w[0]
    hi = event_t + w[1]
    return df[(df["time_s"] >= lo - EPS) & (df["time_s"] <= hi + EPS)].copy()


def quantiles_or_nan(arr, probs=(0.05, 0.50, 0.95)):
    """Return dict of quantiles, or NaN if empty."""
    a = arr[np.isfinite(arr)] if len(arr) > 0 else np.array([])
    if len(a) == 0:
        return {f"p{int(p*100)}": _NAN for p in probs}
    return {f"p{int(p*100)}": float(np.quantile(a, p)) for p in probs}

# ──────────────────────────────────────────────────────────────────────────────
#  Per-window feature computation on resampled grid
# ──────────────────────────────────────────────────────────────────────────────
def compute_safety_proxies(cs: pd.DataFrame, radar: pd.DataFrame,
                           event_t: float, prefix: str) -> dict:
    """Lead-vehicle conflict proxies: THW, TTC, DRAC with robust caps."""
    p = prefix
    feat = {}

    # Defaults
    for k in ["thw_min_s", "thw_p5_s", "thw_p50_s", "thw_p95_s",
              "ttc_min_raw_s", "ttc_min_capped_s", "ttc_p5_s", "ttc_p50_s", "ttc_p95_s",
              "drac_max_raw_mps2", "drac_max_capped_mps2", "drac_p50_mps2", "drac_p95_mps2",
              "min_drel_m", "p5_drel_m",
              "lead_present_rate", "lead_drop_count", "longest_cont_lead_s",
              "time_of_min_ttc_s", "time_of_max_drac_s"]:
        feat[f"{p}{k}"] = _NAN
    feat[f"{p}n_lead_samples"] = 0

    # Exposure durations default
    for th in EXPOSURE["ttc_thresholds_s"]:
        feat[f"{p}time_below_ttc_{th}s"] = 0.0
        feat[f"{p}severity_integral_ttc_{th}s"] = 0.0
    for th in EXPOSURE["thw_thresholds_s"]:
        feat[f"{p}time_below_thw_{th}s"] = 0.0
    for th in EXPOSURE["drac_thresholds_mps2"]:
        feat[f"{p}time_above_drac_{th}mps2"] = 0.0

    if radar.empty or "leadOne.status" not in radar.columns:
        return feat

    lead_status = radar["leadOne.status"].values if "leadOne.status" in radar.columns else np.array([])
    if len(lead_status) == 0:
        return feat

    n_total = len(lead_status)
    n_present = int(lead_status.sum())
    feat[f"{p}lead_present_rate"] = n_present / max(n_total, 1)
    feat[f"{p}n_lead_samples"] = n_present

    # Lead continuity: count transitions True→False
    if n_present > 0:
        transitions = np.diff(lead_status.astype(int))
        feat[f"{p}lead_drop_count"] = int((transitions == -1).sum())
        # Longest continuous lead segment
        runs = []
        run_start = None
        t_radar = radar["time_s"].values
        for i, s in enumerate(lead_status):
            if s:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    runs.append(t_radar[i-1] - t_radar[run_start])
                    run_start = None
        if run_start is not None:
            runs.append(t_radar[-1] - t_radar[run_start])
        feat[f"{p}longest_cont_lead_s"] = float(max(runs)) if runs else 0.0

    if n_present == 0 or "leadOne.dRel" not in radar.columns:
        return feat

    # Extract lead-present rows
    lead_mask = lead_status.astype(bool)
    t_lead = radar.loc[lead_mask, "time_s"].values
    drel = radar.loc[lead_mask, "leadOne.dRel"].values.astype(float)

    feat[f"{p}min_drel_m"] = float(np.nanmin(drel)) if len(drel) > 0 else _NAN
    drel_q = quantiles_or_nan(drel, (0.05,))
    feat[f"{p}p5_drel_m"] = drel_q["p5"]

    # Get ego speed aligned to lead timestamps
    if cs.empty or "vEgo" not in cs.columns:
        return feat
    v_ego_all = cs["vEgo"].values.astype(float)
    t_cs = cs["time_s"].values
    v_ego = np.interp(t_lead, t_cs, v_ego_all)

    # ── THW ──────────────────────────────────────────────────────────
    valid_v = v_ego > MIN_V_THW
    if valid_v.any():
        thw = drel[valid_v] / v_ego[valid_v]
        thw = thw[np.isfinite(thw) & (thw > 0)]
        if len(thw) > 0:
            feat[f"{p}thw_min_s"] = float(np.min(thw))
            q = quantiles_or_nan(thw)
            feat[f"{p}thw_p5_s"]  = q["p5"]
            feat[f"{p}thw_p50_s"] = q["p50"]
            feat[f"{p}thw_p95_s"] = q["p95"]
            # Exposure durations
            dt_resamp = RESAMPLE_DT
            for th in EXPOSURE["thw_thresholds_s"]:
                # Compute full THW time series on lead-present resampled points
                thw_full = drel / np.maximum(v_ego, MIN_V_THW)
                feat[f"{p}time_below_thw_{th}s"] = float(
                    np.sum(thw_full[np.isfinite(thw_full)] < th) * dt_resamp)

    # ── TTC (closing only) ───────────────────────────────────────────
    if "leadOne.vRel" in radar.columns:
        vrel = radar.loc[lead_mask, "leadOne.vRel"].values.astype(float)
        closing = vrel < -CLOSING_MIN
        if closing.any():
            d_c = drel[closing]
            v_c = -vrel[closing]
            t_c = t_lead[closing]

            ttc_raw = d_c / np.maximum(v_c, EPS)
            ttc_raw = ttc_raw[np.isfinite(ttc_raw) & (ttc_raw > 0)]

            if len(ttc_raw) > 0:
                feat[f"{p}ttc_min_raw_s"] = float(np.min(ttc_raw))
                ttc_capped = np.clip(ttc_raw, 0, TTC_CAP)
                feat[f"{p}ttc_min_capped_s"] = float(np.min(ttc_capped))
                q = quantiles_or_nan(ttc_capped)
                feat[f"{p}ttc_p5_s"]  = q["p5"]
                feat[f"{p}ttc_p50_s"] = q["p50"]
                feat[f"{p}ttc_p95_s"] = q["p95"]
                # Time of min TTC
                idx_min = np.argmin(ttc_raw)
                feat[f"{p}time_of_min_ttc_s"] = float(t_c[idx_min] - event_t)

                # Exposure durations + severity integral
                dt_resamp = RESAMPLE_DT
                for th in EXPOSURE["ttc_thresholds_s"]:
                    below = ttc_capped < th
                    feat[f"{p}time_below_ttc_{th}s"] = float(below.sum() * dt_resamp)
                    excess = np.maximum(th - ttc_capped, 0)
                    feat[f"{p}severity_integral_ttc_{th}s"] = float(excess.sum() * dt_resamp)

            # ── DRAC ─────────────────────────────────────────────────
            drac_mask = d_c > DREL_MIN
            if drac_mask.any():
                drac_raw = (v_c[drac_mask] ** 2) / (2 * d_c[drac_mask])
                drac_raw = drac_raw[np.isfinite(drac_raw)]
                if len(drac_raw) > 0:
                    feat[f"{p}drac_max_raw_mps2"] = float(np.max(drac_raw))
                    drac_capped = np.clip(drac_raw, 0, DRAC_CAP)
                    feat[f"{p}drac_max_capped_mps2"] = float(np.max(drac_capped))
                    q = quantiles_or_nan(drac_capped, (0.50, 0.95))
                    feat[f"{p}drac_p50_mps2"] = q["p50"]
                    feat[f"{p}drac_p95_mps2"] = q["p95"]
                    # Time of max DRAC
                    idx_max = np.argmin(drac_raw)  # already filtered
                    t_drac = t_c[drac_mask]
                    feat[f"{p}time_of_max_drac_s"] = float(t_drac[np.argmax(drac_raw)] - event_t)

                    for th in EXPOSURE["drac_thresholds_mps2"]:
                        feat[f"{p}time_above_drac_{th}mps2"] = float(
                            (drac_capped > th).sum() * RESAMPLE_DT)

    return feat


def compute_dynamics(cs: pd.DataFrame, event_t: float, prefix: str) -> dict:
    """Ego dynamics: accel, jerk, speed, steering."""
    p = prefix
    feat = {}

    # Defaults
    for k in ["min_accel_mps2", "max_accel_mps2",
              "accel_p5_mps2", "accel_p50_mps2", "accel_p95_mps2",
              "max_abs_jerk_mps3", "jerk_p50_mps3", "jerk_p95_mps3",
              "time_of_peak_decel_s", "time_of_peak_jerk_s",
              "max_abs_steer_angle_deg", "max_abs_steer_torque",
              "steer_rate_max_deg_per_s", "steer_rate_p95_deg_per_s",
              "time_of_peak_steer_rate_s",
              "speed_mean_mps", "speed_delta_mps",
              "max_abs_curvature", "max_abs_desired_curvature"]:
        feat[f"{p}{k}"] = _NAN

    if cs.empty:
        return feat

    t = cs["time_s"].values

    # Speed
    if "vEgo" in cs.columns:
        v = cs["vEgo"].values.astype(float)
        valid = np.isfinite(v)
        if valid.any():
            feat[f"{p}speed_mean_mps"] = float(np.nanmean(v[valid]))
            if valid.sum() >= 2:
                feat[f"{p}speed_delta_mps"] = float(v[valid][-1] - v[valid][0])

    # Acceleration + jerk
    if "aEgo" in cs.columns:
        a = cs["aEgo"].values.astype(float)
        valid = np.isfinite(a)
        if valid.any():
            feat[f"{p}min_accel_mps2"] = float(np.nanmin(a[valid]))
            feat[f"{p}max_accel_mps2"] = float(np.nanmax(a[valid]))
            q = quantiles_or_nan(a[valid])
            feat[f"{p}accel_p5_mps2"]  = q["p5"]
            feat[f"{p}accel_p50_mps2"] = q["p50"]
            feat[f"{p}accel_p95_mps2"] = q["p95"]
            feat[f"{p}time_of_peak_decel_s"] = float(
                t[valid][np.argmin(a[valid])] - event_t)

            if valid.sum() >= 5:
                a_smooth = smooth_savgol(a[valid])
                jerk = diff_dt(a_smooth)
                abs_jerk = np.abs(jerk)
                feat[f"{p}max_abs_jerk_mps3"] = float(np.max(abs_jerk))
                q = quantiles_or_nan(abs_jerk, (0.50, 0.95))
                feat[f"{p}jerk_p50_mps3"] = q["p50"]
                feat[f"{p}jerk_p95_mps3"] = q["p95"]
                t_jerk = (t[valid][:-1] + t[valid][1:]) / 2
                feat[f"{p}time_of_peak_jerk_s"] = float(
                    t_jerk[np.argmax(abs_jerk)] - event_t)

    # Steering
    if "steeringAngleDeg" in cs.columns:
        sa = cs["steeringAngleDeg"].values.astype(float)
        valid = np.isfinite(sa)
        if valid.any():
            feat[f"{p}max_abs_steer_angle_deg"] = float(np.max(np.abs(sa[valid])))
            if valid.sum() >= 5:
                sa_smooth = smooth_savgol(sa[valid])
                sr = np.abs(diff_dt(sa_smooth))
                feat[f"{p}steer_rate_max_deg_per_s"] = float(np.max(sr))
                q = quantiles_or_nan(sr, (0.95,))
                feat[f"{p}steer_rate_p95_deg_per_s"] = q["p95"]
                t_sr = (t[valid][:-1] + t[valid][1:]) / 2
                feat[f"{p}time_of_peak_steer_rate_s"] = float(
                    t_sr[np.argmax(sr)] - event_t)

    if "steeringTorque" in cs.columns:
        st = cs["steeringTorque"].values.astype(float)
        valid = np.isfinite(st)
        if valid.any():
            feat[f"{p}max_abs_steer_torque"] = float(np.max(np.abs(st[valid])))

    # Curvature (if available)
    for col, key in [("curvature", "max_abs_curvature"),
                     ("desiredCurvature", "max_abs_desired_curvature")]:
        # curvature is in controlsState, but we pass cs (carState) here;
        # caller can merge if needed. Check if present:
        if col in cs.columns:
            vals = cs[col].values.astype(float)
            valid = np.isfinite(vals)
            if valid.any():
                feat[f"{p}{key}"] = float(np.max(np.abs(vals[valid])))

    return feat


def compute_roughness(accel_native: pd.DataFrame, event_t: float,
                      w: list, prefix: str) -> dict:
    """Roughness from accelerometer on NATIVE timestamps (not resampled)."""
    p = prefix
    feat = {
        f"{p}roughness_rms_mps2": _NAN,
        f"{p}roughness_pp_mps2": _NAN,
        f"{p}roughness_z_rms_mps2": _NAN,
        f"{p}accel_native_hz": _NAN,
    }

    if accel_native.empty or "acceleration.v" not in accel_native.columns:
        return feat
    if "time_s" not in accel_native.columns:
        return feat

    # Window slice on native
    lo = event_t + w[0]
    hi = event_t + w[1]
    mask = (accel_native["time_s"] >= lo) & (accel_native["time_s"] <= hi)
    aw = accel_native[mask]
    if len(aw) < 3:
        return feat

    # Estimate native Hz
    t_a = aw["time_s"].values
    if len(t_a) >= 2:
        dt_m = np.median(np.diff(t_a))
        feat[f"{p}accel_native_hz"] = float(1.0 / max(dt_m, EPS))

    try:
        norms = []
        z_vals = []
        for v in aw["acceleration.v"].dropna():
            parsed = json.loads(str(v)) if isinstance(v, str) else v
            if isinstance(parsed, list) and len(parsed) >= 3:
                x, y, z = float(parsed[0]), float(parsed[1]), float(parsed[2])
                norms.append(np.sqrt(x*x + y*y + z*z))
                z_vals.append(z)
        if len(norms) >= 3:
            a_norm = np.array(norms)
            a_detrend = a_norm - np.mean(a_norm)
            feat[f"{p}roughness_rms_mps2"] = float(np.sqrt(np.mean(a_detrend**2)))
            feat[f"{p}roughness_pp_mps2"] = float(np.ptp(a_detrend))
            z_arr = np.array(z_vals)
            z_dt = z_arr - np.mean(z_arr)
            feat[f"{p}roughness_z_rms_mps2"] = float(np.sqrt(np.mean(z_dt**2)))
    except Exception:
        pass

    return feat


def compute_alerts(ctrl: pd.DataFrame, lplan: pd.DataFrame, prefix: str) -> dict:
    """FCW and alert flags."""
    p = prefix
    feat = {
        f"{p}fcw_present": False,
        f"{p}fcw_source": "none",
        f"{p}alert_present": False,
        f"{p}alert_text": "",
        f"{p}has_lane_probs": False,
        f"{p}lane_left_prob_mean": _NAN,
        f"{p}lane_right_prob_mean": _NAN,
    }

    # FCW from longitudinalPlan
    if not lplan.empty and "fcw" in lplan.columns:
        if lplan["fcw"].any():
            feat[f"{p}fcw_present"] = True
            feat[f"{p}fcw_source"] = "explicit"

    # Alerts from controlsState
    if not ctrl.empty and "alertText1" in ctrl.columns:
        alerts = ctrl["alertText1"].astype(str).str.strip()
        non_empty = alerts[~alerts.isin(["", "nan", "None", "NaN"])]
        if len(non_empty) > 0:
            feat[f"{p}alert_present"] = True
            feat[f"{p}alert_text"] = str(non_empty.iloc[-1])
            # Detect FCW-like from alert text if not already found
            if not feat[f"{p}fcw_present"]:
                fcw_keywords = ["forward collision", "fcw", "brake!", "collision"]
                if any(kw in non_empty.str.lower().str.cat(sep=" ") for kw in fcw_keywords):
                    feat[f"{p}fcw_present"] = True
                    feat[f"{p}fcw_source"] = "alert_text"

    return feat


def compute_stabilization(cs: pd.DataFrame, event_t: float,
                          max_s: float) -> tuple[float, bool]:
    """Stabilization time within [0, max_s]."""
    a0 = STAB["accel_threshold_mps2"]
    j0 = STAB["jerk_threshold_mps3"]
    sustain = STAB["sustain_duration_s"]

    if cs.empty or "aEgo" not in cs.columns:
        return (_NAN, True)

    # Slice post-takeover
    post = cs[(cs["time_s"] >= event_t - EPS) &
              (cs["time_s"] <= event_t + max_s + EPS)].copy()
    if len(post) < 5:
        return (_NAN, True)

    t = post["time_s"].values
    a = post["aEgo"].values.astype(float)
    valid = np.isfinite(a)
    if valid.sum() < 5:
        return (_NAN, True)

    t = t[valid]
    a = a[valid]
    t0 = event_t

    a_smooth = smooth_savgol(a)
    jerk = diff_dt(a_smooth)
    t_jerk = (t[:-1] + t[1:]) / 2

    stable = (np.abs(a_smooth[:-1]) < a0) & (np.abs(jerk) < j0)
    if not stable.any():
        return (max_s, True)

    run_start = None
    for i in range(len(stable)):
        if stable[i]:
            if run_start is None:
                run_start = i
            if t_jerk[i] - t_jerk[run_start] >= sustain:
                return (float(t_jerk[run_start] - t0), False)
        else:
            run_start = None

    return (max_s, True)


def classify_post_maneuver(feat: dict) -> str:
    sr = feat.get("post_steer_rate_max_deg_per_s", _NAN)
    sa = feat.get("post_max_abs_steer_angle_deg", _NAN)
    spd_d = feat.get("post_speed_delta_mps", _NAN)
    min_a = feat.get("post_min_accel_mps2", _NAN)
    if not np.isnan(sr) and sr > 20 and not np.isnan(sa) and sa > 15:
        return "lane_change"
    if not np.isnan(sa) and sa > 30:
        return "turn_ramp"
    if not np.isnan(spd_d) and spd_d > 2.0:
        return "acceleration"
    if not np.isnan(min_a) and min_a < -2.0:
        return "braking"
    return "stabilize"


# ──────────────────────────────────────────────────────────────────────────────
#  Anomaly flags
# ──────────────────────────────────────────────────────────────────────────────
def compute_anomaly_flags(rec: dict) -> dict:
    """Boolean anomaly flags + diagnostic fields."""
    flags = {
        "anomaly_jerk_extreme": False,
        "anomaly_steer_rate_extreme": False,
        "anomaly_ttc_tail": False,
        "anomaly_drac_tail": False,
        "anomaly_lead_dropout": False,
        "anomaly_stabilization_censored": False,
        "anomaly_nan_rate_high": False,
        "anomaly_any": False,
        "anomaly_reason": "",
    }
    reasons = []

    jerk = rec.get("pre_max_abs_jerk_mps3", _NAN)
    if not np.isnan(jerk) and jerk > ANOM["jerk_abs_threshold_mps3"]:
        flags["anomaly_jerk_extreme"] = True
        reasons.append("extreme_jerk")

    sr = rec.get("pre_steer_rate_max_deg_per_s", _NAN)
    if not np.isnan(sr) and sr > ANOM["steer_rate_abs_threshold_deg_s"]:
        flags["anomaly_steer_rate_extreme"] = True
        reasons.append("extreme_steer_rate")

    ttc = rec.get("pre_ttc_min_raw_s", _NAN)
    if not np.isnan(ttc) and ttc > ANOM["ttc_raw_max_valid_s"]:
        flags["anomaly_ttc_tail"] = True
        reasons.append("ttc_tail")

    drac = rec.get("pre_drac_max_raw_mps2", _NAN)
    if not np.isnan(drac) and drac > ANOM["drac_raw_max_valid_mps2"]:
        flags["anomaly_drac_tail"] = True
        reasons.append("drac_tail")

    drops = rec.get("pre_lead_drop_count", 0)
    if not np.isnan(drops) and drops > ANOM["lead_dropout_max"]:
        flags["anomaly_lead_dropout"] = True
        reasons.append("lead_dropout")

    if rec.get("stabilization_5s_censored", True) and rec.get("stabilization_10s_censored", True):
        flags["anomaly_stabilization_censored"] = True
        reasons.append("stab_censored")

    # NaN rate: fraction of key columns that are NaN
    key_cols = ["pre_speed_mean_mps", "pre_min_accel_mps2", "pre_lead_present_rate"]
    nan_count = sum(1 for k in key_cols if np.isnan(rec.get(k, _NAN)))
    if nan_count / max(len(key_cols), 1) > ANOM["nan_rate_threshold"]:
        flags["anomaly_nan_rate_high"] = True
        reasons.append("high_nan_rate")

    flags["anomaly_any"] = any(v for k, v in flags.items()
                                if k.startswith("anomaly_") and isinstance(v, bool) and v)
    flags["anomaly_reason"] = "; ".join(reasons)

    return flags


# ──────────────────────────────────────────────────────────────────────────────
#  Process one clip (main entry point per worker)
# ──────────────────────────────────────────────────────────────────────────────
def process_clip_v3(clip_dir: Path) -> dict | None:
    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    event_t = float(meta["video_time_s"])
    log_hz = int(meta.get("log_hz", 20))

    rec = {
        "car_model": meta["car_model"],
        "dongle_id": meta["dongle_id"],
        "route_id":  meta["route_id"],
        "clip_id":   int(meta["clip_id"]),
        "log_kind":  meta.get("log_kind", ""),
        "log_hz":    log_hz,
        "resample_hz": RESAMPLE_HZ,
    }

    # ── Build resampled topics ──────────────────────────────────────────
    topics = build_resampled_topics(clip_dir, event_t)
    cs_full   = topics["carState"]
    radar_full = topics["radarState"]
    ctrl_full  = topics["controlsState"]
    lplan_full = topics["longitudinalPlan"]
    dm_full    = topics["drivingModelData"]
    accel_nat  = topics["accelerometer_native"]

    # Merge curvature from controlsState into carState for convenience
    for curvCol in ["curvature", "desiredCurvature"]:
        if curvCol in ctrl_full.columns and curvCol not in cs_full.columns:
            cs_full[curvCol] = ctrl_full[curvCol].values

    # ── Pre-window features ─────────────────────────────────────────────
    cs_pre    = win_slice(cs_full, event_t, W_PRE)
    radar_pre = win_slice(radar_full, event_t, W_PRE)
    ctrl_pre  = win_slice(ctrl_full, event_t, W_PRE)
    lplan_pre = win_slice(lplan_full, event_t, W_PRE)
    dm_pre    = win_slice(dm_full, event_t, W_PRE)

    rec.update(compute_safety_proxies(cs_pre, radar_pre, event_t, "pre_"))
    rec.update(compute_dynamics(cs_pre, event_t, "pre_"))
    rec.update(compute_roughness(accel_nat, event_t, W_PRE, "pre_"))
    rec.update(compute_alerts(ctrl_pre, lplan_pre, "pre_"))

    # Lane probs (from drivingModelData)
    if not dm_pre.empty:
        for col, key in [("laneLineMeta.leftProb", "pre_lane_left_prob_mean"),
                         ("laneLineMeta.rightProb", "pre_lane_right_prob_mean")]:
            if col in dm_pre.columns:
                vals = dm_pre[col].values.astype(float)
                valid = np.isfinite(vals)
                if valid.any():
                    rec[key] = float(np.nanmean(vals[valid]))
                    rec["pre_has_lane_probs"] = True

    # ── Post-window features ────────────────────────────────────────────
    cs_post    = win_slice(cs_full, event_t, W_POST)
    radar_post = win_slice(radar_full, event_t, W_POST)
    ctrl_post  = win_slice(ctrl_full, event_t, W_POST)
    lplan_post = win_slice(lplan_full, event_t, W_POST)

    rec.update(compute_safety_proxies(cs_post, radar_post, event_t, "post_"))
    rec.update(compute_dynamics(cs_post, event_t, "post_"))
    rec.update(compute_roughness(accel_nat, event_t, W_POST, "post_"))
    rec.update(compute_alerts(ctrl_post, lplan_post, "post_"))

    # ── Post-maneuver classification ────────────────────────────────────
    rec["post_maneuver_type"] = classify_post_maneuver(rec)

    # ── Stabilization (two windows) ─────────────────────────────────────
    s5, c5 = compute_stabilization(cs_full, event_t, STAB["max_search_s_short"])
    rec["stabilization_5s_time_s"]   = s5
    rec["stabilization_5s_censored"] = c5

    s10, c10 = compute_stabilization(cs_full, event_t, STAB["max_search_s_long"])
    rec["stabilization_10s_time_s"]   = s10
    rec["stabilization_10s_censored"] = c10

    # ── Anomaly flags ───────────────────────────────────────────────────
    rec.update(compute_anomaly_flags(rec))

    return rec


# ──────────────────────────────────────────────────────────────────────────────
#  Clip discovery
# ──────────────────────────────────────────────────────────────────────────────
def find_clips_from_per_clip() -> list[Path]:
    """Use per_clip.csv as definitive clip list."""
    pc_path = CODE / CFG["paths"]["per_clip"]
    if not pc_path.exists():
        print(f"  WARNING: {pc_path} not found, falling back to rglob")
        return _find_all_clips_rglob()

    pc = pd.read_csv(pc_path, usecols=["car_model", "dongle_id", "route_id", "clip_id"],
                     low_memory=False)
    clips = []
    for _, row in pc.iterrows():
        # Reconstruct clip path
        clip_dir = ROOT / str(row["car_model"]) / str(row["dongle_id"]) / \
                   str(row["route_id"]) / str(int(row["clip_id"]))
        if (clip_dir / "meta.json").exists():
            clips.append(clip_dir)
    return sorted(clips)


def _find_all_clips_rglob() -> list[Path]:
    clips = []
    for meta in ROOT.rglob("meta.json"):
        if "Code" in meta.parts:
            continue
        clips.append(meta.parent)
    return sorted(clips)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Stage 1 v3: compute derived signals")
    parser.add_argument("--spot", type=int, default=0,
                        help="Spot-check mode: process only N clips")
    args = parser.parse_args()

    print("Finding clips …")
    clips = find_clips_from_per_clip()
    print(f"  Found {len(clips):,} clips")

    if args.spot > 0:
        rng = np.random.RandomState(SEED)
        clips = list(rng.choice(clips, size=min(args.spot, len(clips)), replace=False))
        print(f"  Spot-check mode: processing {len(clips)} clips")

    workers = min(N_WORKERS, len(clips))
    print(f"Processing with {workers} workers (resample_hz={RESAMPLE_HZ}) …")

    results: list[dict] = []
    errors = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_clip_v3, c): c for c in clips}
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  [WARN] {futures[fut]}: {e}")
            if done % 2000 == 0:
                print(f"  {done:,}/{len(clips):,} done")

    print(f"  {done:,}/{len(clips):,} done  ({errors} errors)")

    df = pd.DataFrame(results)

    # ── Save ────────────────────────────────────────────────────────────
    # Parquet (primary)
    parquet_path = CODE / CFG["paths"]["derived_signals_v3"]
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved {parquet_path}")
    except Exception:
        print(f"  [WARN] Parquet save failed, CSV only")

    # CSV (convenience)
    csv_path = CODE / CFG["paths"]["derived_signals_v3_csv"]
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}  ({len(df):,} rows, {len(df.columns)} columns)")

    # ── Anomaly flags (separate file) ───────────────────────────────────
    anom_cols = (["car_model", "dongle_id", "route_id", "clip_id",
                  "log_kind", "log_hz"] +
                 [c for c in df.columns if c.startswith("anomaly_")])
    anom_cols = [c for c in anom_cols if c in df.columns]
    anom_df = df[anom_cols]
    anom_path = CODE / CFG["paths"]["anomaly_flags"]
    anom_df.to_csv(anom_path, index=False)
    n_anom = anom_df["anomaly_any"].sum() if "anomaly_any" in anom_df.columns else 0
    print(f"  Saved {anom_path}  ({n_anom:,} clips flagged)")

    # ── Summary report ──────────────────────────────────────────────────
    _print_summary(df)


def _print_summary(df: pd.DataFrame):
    N = len(df)
    print(f"\n{'═'*70}")
    print(f"  STAGE 1 v3 SUMMARY  ({N:,} clips, {len(df.columns)} columns)")
    print(f"{'═'*70}")

    # Key metrics
    for col in ["pre_ttc_min_capped_s", "pre_thw_min_s", "pre_drac_max_capped_mps2",
                "pre_max_abs_jerk_mps3", "pre_roughness_rms_mps2",
                "stabilization_5s_time_s", "stabilization_10s_time_s"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"  {col}:")
                print(f"    n={len(s):,}  mean={s.mean():.3f}  "
                      f"median={s.median():.3f}  p5={s.quantile(0.05):.3f}  "
                      f"p95={s.quantile(0.95):.3f}")

    # Robust vs raw TTC/DRAC comparison
    print(f"\n{'─'*70}")
    print("  ROBUST vs RAW (tail reduction)")
    print(f"{'─'*70}")
    for raw, cap in [("pre_ttc_min_raw_s", "pre_ttc_min_capped_s"),
                     ("pre_drac_max_raw_mps2", "pre_drac_max_capped_mps2")]:
        if raw in df.columns and cap in df.columns:
            r = df[raw].dropna()
            c = df[cap].dropna()
            if len(r) > 0:
                print(f"  {raw}: mean={r.mean():.2f}  p99={r.quantile(0.99):.2f}  max={r.max():.2f}")
                print(f"  {cap}: mean={c.mean():.2f}  p99={c.quantile(0.99):.2f}  max={c.max():.2f}")

    # qlog vs rlog after resampling
    print(f"\n{'─'*70}")
    print("  QLOG vs RLOG SENSITIVITY (post-resampling)")
    print(f"{'─'*70}")
    for col in ["pre_max_abs_jerk_mps3", "pre_steer_rate_max_deg_per_s"]:
        if col in df.columns:
            for lk in ["qlog", "rlog"]:
                sub = df[df["log_kind"] == lk][col].dropna()
                if len(sub) > 0:
                    print(f"  {col} [{lk}]: n={len(sub):,}  "
                          f"median={sub.median():.3f}  p95={sub.quantile(0.95):.3f}")

    # Exposure durations
    print(f"\n{'─'*70}")
    print("  EXPOSURE DURATIONS (pre-window)")
    print(f"{'─'*70}")
    for col in df.columns:
        if col.startswith("pre_time_below_") or col.startswith("pre_time_above_"):
            s = df[col].dropna()
            n_nonzero = (s > 0).sum()
            if len(s) > 0:
                print(f"  {col}: mean={s.mean():.3f}s  "
                      f"nonzero={n_nonzero:,}/{len(s):,}")

    # Anomalies
    print(f"\n{'─'*70}")
    print("  ANOMALY FLAGS")
    print(f"{'─'*70}")
    for col in df.columns:
        if col.startswith("anomaly_") and col != "anomaly_reason" and col != "anomaly_any":
            if df[col].dtype == bool:
                n = df[col].sum()
                print(f"  {col}: {n:,}  ({n/N*100:.1f}%)")
    if "anomaly_any" in df.columns:
        n_any = df["anomaly_any"].sum()
        print(f"  anomaly_any: {n_any:,}  ({n_any/N*100:.1f}%)")

    # Post-maneuver
    print(f"\n{'─'*70}")
    print("  POST-MANEUVER DISTRIBUTION")
    print(f"{'─'*70}")
    if "post_maneuver_type" in df.columns:
        for val, cnt in df["post_maneuver_type"].value_counts().items():
            print(f"  {val:20s}: {cnt:,}")


if __name__ == "__main__":
    main()
