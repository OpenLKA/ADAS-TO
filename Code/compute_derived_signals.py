#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_derived_signals.py
==========================
Compute per-clip safety metrics and derived signals for Section IV analysis.

For each clip, computes features in pre-takeover [-5, 0]s and post-takeover
[0, +5]s windows: TTC, THW, DRAC, jerk, steering rate, roughness,
stabilization time, etc.

Key design choices (addressing reviewer concerns):
  - All derivative metrics (jerk, steer_rate) use timestamp-based dt (not
    sample-index differencing), with Savitzky-Golay smoothing before
    differentiation to reduce noise amplification.
  - Roughness uses acceleration norm ||a|| (not just z-axis), detrended,
    making it robust to device orientation differences.
  - Stabilization time defined as first continuous 1.0s window where
    |accel| < 0.5 m/s² AND |jerk| < 1.0 m/s³; right-censored at 5.0s.
  - log_hz stored per clip for qlog vs rlog sensitivity reporting.

Run:
    python3 compute_derived_signals.py

Output:
    stats_output/derived_signals.csv
"""
from __future__ import annotations

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

PRE_BEFORE  = abs(CFG["pre_window"][0])   # 5.0
PRE_AFTER   = CFG["pre_window"][1]         # 0.0
POST_BEFORE = CFG["post_window"][0]        # 0.0
POST_AFTER  = CFG["post_window"][1]        # 5.0
EPS         = CFG["eps"]
N_WORKERS   = CFG["n_workers"]
MIN_V_THW   = CFG["min_speed_for_thw_mps"]

# Smoothing config
SMOOTH_CFG  = CFG["smoothing"]
SMOOTH_WINDOW_S = SMOOTH_CFG["savgol_window_s"]
SMOOTH_POLY     = SMOOTH_CFG["savgol_polyorder"]

# Stabilization config
STAB_CFG = CFG["stabilization"]

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers (reused from dataset_statistics.py)
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


def parse_bool_col(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def time_window(df: pd.DataFrame, event_t: float,
                before_s: float, after_s: float) -> pd.DataFrame:
    """Slice rows within [event_t - before_s, event_t + after_s] using time_s."""
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame()
    lo = event_t - before_s
    hi = event_t + after_s
    mask = (df["time_s"] >= lo) & (df["time_s"] <= hi)
    return df[mask].copy()


def safe_diff_dt(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """First derivative with time-based dt, avoiding division by zero."""
    dt = np.diff(times)
    dt[dt < EPS] = EPS
    return np.diff(values) / dt


def smooth_signal(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with config-driven window.

    Window size is computed from SMOOTH_WINDOW_S * sample_rate_hz, forced odd.
    Falls back to rolling mean if signal is too short for SavGol.
    """
    n = len(values)
    if n < 5:
        return values
    # Compute window length in samples
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
    # Fallback: simple rolling mean
    return pd.Series(values).rolling(3, center=True, min_periods=1).mean().values


_NAN = float("nan")

# ──────────────────────────────────────────────────────────────────────────────
#  Per-window feature extraction
# ──────────────────────────────────────────────────────────────────────────────
def compute_window_features(
    cs_w: pd.DataFrame,    # carState window
    radar_w: pd.DataFrame, # radarState window
    ctrl_w: pd.DataFrame,  # controlsState window
    accel_w: pd.DataFrame, # accelerometer window
    lplan_w: pd.DataFrame, # longitudinalPlan window
    model_w: pd.DataFrame, # drivingModelData window
    prefix: str,           # "pre_" or "post_"
    sample_rate_hz: float = 20.0,
) -> dict:
    """Compute all features for one time window."""
    feat: dict = {}
    p = prefix

    # ── Lead-vehicle safety metrics (radarState) ────────────────────────
    feat[f"{p}thw_min_s"]         = _NAN
    feat[f"{p}ttc_min_s"]         = _NAN
    feat[f"{p}drac_max_mps2"]     = _NAN
    feat[f"{p}min_drel_m"]        = _NAN
    feat[f"{p}lead_present_rate"] = _NAN
    feat[f"{p}n_lead_samples"]    = 0

    if not radar_w.empty and "leadOne.status" in radar_w.columns:
        lead_status = parse_bool_col(radar_w["leadOne.status"])
        n_total = len(lead_status)
        n_present = int(lead_status.sum())
        feat[f"{p}lead_present_rate"] = n_present / max(n_total, 1)
        feat[f"{p}n_lead_samples"]    = n_present

        if n_present > 0 and "leadOne.dRel" in radar_w.columns:
            lead = radar_w[lead_status].copy()
            drel = pd.to_numeric(lead["leadOne.dRel"], errors="coerce")
            feat[f"{p}min_drel_m"] = float(drel.min()) if drel.notna().any() else _NAN

            # Need speed for THW
            if not cs_w.empty and "vEgo" in cs_w.columns:
                lead_t = lead["time_s"].values
                cs_t = cs_w["time_s"].values
                cs_v = pd.to_numeric(cs_w["vEgo"], errors="coerce").values
                idxs = np.searchsorted(cs_t, lead_t, side="left").clip(0, len(cs_t) - 1)
                v_ego = cs_v[idxs]

                d = drel.values.astype(float)
                # THW = dRel / vEgo (only when vEgo > MIN_V_THW)
                valid_v = v_ego > MIN_V_THW
                if valid_v.any():
                    thw = d[valid_v] / v_ego[valid_v]
                    thw_valid = thw[np.isfinite(thw) & (thw > 0)]
                    if len(thw_valid) > 0:
                        feat[f"{p}thw_min_s"] = float(np.nanmin(thw_valid))

                # TTC = dRel / max(-vRel, eps), closing only (vRel < 0)
                if "leadOne.vRel" in lead.columns:
                    vrel = pd.to_numeric(lead["leadOne.vRel"], errors="coerce").values
                    closing = vrel < -EPS
                    if closing.any():
                        ttc = d[closing] / np.maximum(-vrel[closing], EPS)
                        ttc_valid = ttc[(ttc > 0) & np.isfinite(ttc)]
                        if len(ttc_valid) > 0:
                            feat[f"{p}ttc_min_s"] = float(np.min(ttc_valid))

                        d_closing = d[closing]
                        v_closing = -vrel[closing]
                        drac = (v_closing ** 2) / np.maximum(2 * d_closing, EPS)
                        drac_valid = drac[np.isfinite(drac)]
                        if len(drac_valid) > 0:
                            feat[f"{p}drac_max_mps2"] = float(np.max(drac_valid))

    # ── Ego dynamics (carState) ─────────────────────────────────────────
    feat[f"{p}min_accel_mps2"]          = _NAN
    feat[f"{p}max_accel_mps2"]          = _NAN
    feat[f"{p}max_abs_jerk_mps3"]       = _NAN
    feat[f"{p}max_abs_steer_angle_deg"] = _NAN
    feat[f"{p}max_abs_steer_torque"]    = _NAN
    feat[f"{p}steer_rate_max_deg_per_s"] = _NAN
    feat[f"{p}speed_mean_mps"]          = _NAN
    feat[f"{p}speed_delta_mps"]         = _NAN

    if not cs_w.empty:
        t = cs_w["time_s"].values

        # Estimate actual sample rate from timestamps
        if len(t) >= 2:
            dt_median = np.median(np.diff(t))
            actual_hz = 1.0 / max(dt_median, EPS)
        else:
            actual_hz = sample_rate_hz

        if "vEgo" in cs_w.columns:
            v = pd.to_numeric(cs_w["vEgo"], errors="coerce").values
            valid = np.isfinite(v)
            if valid.any():
                feat[f"{p}speed_mean_mps"] = float(np.nanmean(v[valid]))
                if valid.sum() >= 2:
                    feat[f"{p}speed_delta_mps"] = float(v[valid][-1] - v[valid][0])

        if "aEgo" in cs_w.columns:
            a = pd.to_numeric(cs_w["aEgo"], errors="coerce").values
            valid = np.isfinite(a)
            if valid.any():
                feat[f"{p}min_accel_mps2"] = float(np.nanmin(a[valid]))
                feat[f"{p}max_accel_mps2"] = float(np.nanmax(a[valid]))

                # Jerk: smooth with SavGol THEN differentiate using dt
                if valid.sum() >= 5:
                    a_smooth = smooth_signal(a[valid], actual_hz)
                    t_v = t[valid]
                    jerk = safe_diff_dt(a_smooth, t_v)
                    jerk_valid = jerk[np.isfinite(jerk)]
                    if len(jerk_valid) > 0:
                        feat[f"{p}max_abs_jerk_mps3"] = float(np.max(np.abs(jerk_valid)))

        if "steeringAngleDeg" in cs_w.columns:
            sa = pd.to_numeric(cs_w["steeringAngleDeg"], errors="coerce").values
            valid = np.isfinite(sa)
            if valid.any():
                feat[f"{p}max_abs_steer_angle_deg"] = float(np.max(np.abs(sa[valid])))
                # Steer rate: smooth THEN differentiate using dt
                if valid.sum() >= 5:
                    sa_smooth = smooth_signal(sa[valid], actual_hz)
                    sr = safe_diff_dt(sa_smooth, t[valid])
                    sr_valid = sr[np.isfinite(sr)]
                    if len(sr_valid) > 0:
                        feat[f"{p}steer_rate_max_deg_per_s"] = float(np.max(np.abs(sr_valid)))

        if "steeringTorque" in cs_w.columns:
            st = pd.to_numeric(cs_w["steeringTorque"], errors="coerce").values
            valid = np.isfinite(st)
            if valid.any():
                feat[f"{p}max_abs_steer_torque"] = float(np.max(np.abs(st[valid])))

    # ── Roughness (acceleration norm, not just z-axis) ──────────────────
    # Use ||a|| to be robust to device orientation differences across cars.
    # Also compute z-axis only as secondary feature for comparison.
    feat[f"{p}roughness_rms_mps2"]   = _NAN
    feat[f"{p}roughness_pp_mps2"]    = _NAN
    feat[f"{p}roughness_z_rms_mps2"] = _NAN

    if not accel_w.empty and "acceleration.v" in accel_w.columns:
        try:
            norms = []
            z_vals = []
            for v in accel_w["acceleration.v"].dropna():
                parsed = json.loads(str(v)) if isinstance(v, str) else v
                if isinstance(parsed, list) and len(parsed) >= 3:
                    x, y, z = float(parsed[0]), float(parsed[1]), float(parsed[2])
                    norms.append(np.sqrt(x*x + y*y + z*z))
                    z_vals.append(z)
            if len(norms) >= 5:
                # Norm-based roughness (primary): detrend by subtracting mean
                a_norm = np.array(norms)
                a_detrend = a_norm - np.mean(a_norm)
                feat[f"{p}roughness_rms_mps2"] = float(np.sqrt(np.mean(a_detrend ** 2)))
                feat[f"{p}roughness_pp_mps2"]  = float(np.ptp(a_detrend))
                # Z-axis only (secondary)
                z_arr = np.array(z_vals)
                z_detrend = z_arr - np.mean(z_arr)
                feat[f"{p}roughness_z_rms_mps2"] = float(np.sqrt(np.mean(z_detrend ** 2)))
        except Exception:
            pass

    # ── FCW & alerts ────────────────────────────────────────────────────
    feat[f"{p}fcw_present"]   = False
    feat[f"{p}alert_present"] = False

    if not lplan_w.empty and "fcw" in lplan_w.columns:
        feat[f"{p}fcw_present"] = bool(parse_bool_col(lplan_w["fcw"]).any())

    if not ctrl_w.empty and "alertText1" in ctrl_w.columns:
        alerts = ctrl_w["alertText1"].astype(str).str.strip()
        non_empty = alerts[~alerts.isin(["", "nan", "None", "NaN"])]
        feat[f"{p}alert_present"] = len(non_empty) > 0

    # ── Lane line probabilities (optional — only if fields exist) ───────
    feat[f"{p}lane_left_prob_mean"]  = _NAN
    feat[f"{p}lane_right_prob_mean"] = _NAN
    feat[f"{p}has_lane_probs"]       = False

    if not model_w.empty:
        for col, key in [("laneLineMeta.leftProb", f"{p}lane_left_prob_mean"),
                         ("laneLineMeta.rightProb", f"{p}lane_right_prob_mean")]:
            if col in model_w.columns:
                vals = pd.to_numeric(model_w[col], errors="coerce")
                if vals.notna().any():
                    feat[key] = float(vals.mean())
                    feat[f"{p}has_lane_probs"] = True

    return feat


# ──────────────────────────────────────────────────────────────────────────────
#  Stabilization time metric (post-takeover)
# ──────────────────────────────────────────────────────────────────────────────
def compute_stabilization_time(cs_post: pd.DataFrame,
                               sample_rate_hz: float) -> tuple[float, bool]:
    """Compute time to stabilization after takeover.

    Stabilization = first continuous window of SUSTAIN_DUR seconds where
    |accel| < a0 AND |jerk| < j0.

    Returns:
        (stabilization_time_s, is_censored)
        If not found within max_search_s, returns (max_search_s, True).
    """
    a0 = STAB_CFG["accel_threshold_mps2"]
    j0 = STAB_CFG["jerk_threshold_mps3"]
    sustain = STAB_CFG["sustain_duration_s"]
    max_s = STAB_CFG["max_search_s"]

    if cs_post.empty or "aEgo" not in cs_post.columns or "time_s" not in cs_post.columns:
        return (_NAN, True)

    t = cs_post["time_s"].values
    a = pd.to_numeric(cs_post["aEgo"], errors="coerce").values
    valid = np.isfinite(a) & np.isfinite(t)
    if valid.sum() < 5:
        return (_NAN, True)

    t = t[valid]
    a = a[valid]
    t0 = t[0]  # start of post-window = takeover moment

    # Smooth and compute jerk
    a_smooth = smooth_signal(a, sample_rate_hz)
    jerk = safe_diff_dt(a_smooth, t)
    # Jerk has one fewer element; align with midpoint times
    t_jerk = (t[:-1] + t[1:]) / 2

    # For each time point, check if conditions are met
    stable_mask = np.abs(a_smooth[:-1]) < a0  # aligned with jerk
    stable_mask &= np.abs(jerk) < j0

    if not stable_mask.any():
        return (max_s, True)

    # Find first continuous run of 'sustain' seconds
    # Walk through stable segments
    run_start = None
    for i in range(len(stable_mask)):
        if stable_mask[i]:
            if run_start is None:
                run_start = i
            elapsed = t_jerk[i] - t_jerk[run_start]
            if elapsed >= sustain:
                return (float(t_jerk[run_start] - t0), False)
        else:
            run_start = None

    return (max_s, True)


# ──────────────────────────────────────────────────────────────────────────────
#  Post-takeover maneuver classification
# ──────────────────────────────────────────────────────────────────────────────
def classify_post_maneuver(feat: dict) -> str:
    """Classify post-takeover maneuver from post_ features."""
    sr = feat.get("post_steer_rate_max_deg_per_s", _NAN)
    sa = feat.get("post_max_abs_steer_angle_deg", _NAN)
    spd_delta = feat.get("post_speed_delta_mps", _NAN)
    min_a = feat.get("post_min_accel_mps2", _NAN)

    # Lane change: high steering rate + large angle
    if not np.isnan(sr) and sr > 20 and not np.isnan(sa) and sa > 15:
        return "lane_change"

    # Turn/ramp: sustained large angle
    if not np.isnan(sa) and sa > 30:
        return "turn_ramp"

    # Acceleration: notable positive speed change
    if not np.isnan(spd_delta) and spd_delta > 2.0:
        return "acceleration"

    # Braking: notable deceleration
    if not np.isnan(min_a) and min_a < -2.0:
        return "braking"

    return "stabilize"


# ──────────────────────────────────────────────────────────────────────────────
#  Process one clip
# ──────────────────────────────────────────────────────────────────────────────
def process_clip(clip_dir: Path) -> dict | None:
    """Compute all derived signals for a single clip."""
    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    event_t = float(meta["video_time_s"])
    log_hz = int(meta.get("log_hz", 20))

    # Clip identity + metadata for sensitivity analysis
    rec = dict(
        car_model = meta["car_model"],
        dongle_id = meta["dongle_id"],
        route_id  = meta["route_id"],
        clip_id   = int(meta["clip_id"]),
        log_kind  = meta.get("log_kind", ""),
        log_hz    = log_hz,
    )

    # ── Read CSVs ───────────────────────────────────────────────────────
    cs_df = safe_read_csv(clip_dir / "carState.csv")
    radar_df = safe_read_csv(clip_dir / "radarState.csv")
    ctrl_df = safe_read_csv(clip_dir / "controlsState.csv")
    accel_df = safe_read_csv(clip_dir / "accelerometer.csv")
    lplan_df = safe_read_csv(clip_dir / "longitudinalPlan.csv")
    model_df = safe_read_csv(clip_dir / "drivingModelData.csv")

    # Sort by time_s for all
    for df in [cs_df, radar_df, ctrl_df, accel_df, lplan_df, model_df]:
        if not df.empty and "time_s" in df.columns:
            df.sort_values("time_s", inplace=True)

    # ── Pre-window ──────────────────────────────────────────────────────
    pre_feat = compute_window_features(
        cs_w    = time_window(cs_df, event_t, PRE_BEFORE, PRE_AFTER),
        radar_w = time_window(radar_df, event_t, PRE_BEFORE, PRE_AFTER),
        ctrl_w  = time_window(ctrl_df, event_t, PRE_BEFORE, PRE_AFTER),
        accel_w = time_window(accel_df, event_t, PRE_BEFORE, PRE_AFTER),
        lplan_w = time_window(lplan_df, event_t, PRE_BEFORE, PRE_AFTER),
        model_w = time_window(model_df, event_t, PRE_BEFORE, PRE_AFTER),
        prefix  = "pre_",
        sample_rate_hz = float(log_hz),
    )
    rec.update(pre_feat)

    # ── Post-window ─────────────────────────────────────────────────────
    post_feat = compute_window_features(
        cs_w    = time_window(cs_df, event_t, POST_BEFORE, POST_AFTER),
        radar_w = time_window(radar_df, event_t, POST_BEFORE, POST_AFTER),
        ctrl_w  = time_window(ctrl_df, event_t, POST_BEFORE, POST_AFTER),
        accel_w = time_window(accel_df, event_t, POST_BEFORE, POST_AFTER),
        lplan_w = time_window(lplan_df, event_t, POST_BEFORE, POST_AFTER),
        model_w = time_window(model_df, event_t, POST_BEFORE, POST_AFTER),
        prefix  = "post_",
        sample_rate_hz = float(log_hz),
    )
    rec.update(post_feat)

    # ── Post-maneuver classification ────────────────────────────────────
    rec["post_maneuver_type"] = classify_post_maneuver(rec)

    # ── Stabilization time ──────────────────────────────────────────────
    cs_post = time_window(cs_df, event_t, 0.0, STAB_CFG["max_search_s"])
    stab_time, stab_censored = compute_stabilization_time(cs_post, float(log_hz))
    rec["stabilization_time_s"] = stab_time
    rec["stabilization_censored"] = stab_censored

    return rec


# ──────────────────────────────────────────────────────────────────────────────
#  Find all clips
# ──────────────────────────────────────────────────────────────────────────────
def find_all_clips() -> list[Path]:
    clips = []
    for meta in ROOT.rglob("meta.json"):
        if "Code" in meta.parts:
            continue
        clips.append(meta.parent)
    return sorted(clips)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Finding all clips …")
    clips = find_all_clips()
    print(f"  Found {len(clips):,} clips")

    print(f"Processing with {N_WORKERS} workers …")
    results: list[dict] = []
    errors = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_clip, c): c for c in clips}
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
    out_path = OUT / "derived_signals.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  ({len(df):,} rows, {len(df.columns)} columns)")

    # ── Summary statistics ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("DERIVED SIGNAL SUMMARY")
    print(f"{'─'*60}")
    for col in ["pre_ttc_min_s", "pre_thw_min_s", "pre_drac_max_mps2",
                "pre_max_abs_jerk_mps3", "pre_roughness_rms_mps2",
                "stabilization_time_s"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"  {col}: n={len(s):,}  mean={s.mean():.3f}  "
                      f"median={s.median():.3f}  p5={s.quantile(0.05):.3f}  "
                      f"p95={s.quantile(0.95):.3f}")

    # ── qlog vs rlog sensitivity for derivative metrics ─────────────────
    print(f"\n{'─'*60}")
    print("QLOG vs RLOG SENSITIVITY (derivative metrics)")
    print(f"{'─'*60}")
    for col in ["pre_max_abs_jerk_mps3", "post_max_abs_jerk_mps3",
                "pre_steer_rate_max_deg_per_s", "post_steer_rate_max_deg_per_s"]:
        if col in df.columns:
            for lk, hz in [("qlog", 10), ("rlog", 100)]:
                sub = df[df["log_kind"] == lk][col].dropna()
                if len(sub) > 0:
                    print(f"  {col} [{lk}/{hz}Hz]: n={len(sub):,}  "
                          f"median={sub.median():.3f}  p95={sub.quantile(0.95):.3f}")

    # ── Post-maneuver distribution ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print("POST-MANEUVER DISTRIBUTION")
    print(f"{'─'*60}")
    if "post_maneuver_type" in df.columns:
        for val, cnt in df["post_maneuver_type"].value_counts().items():
            print(f"  {val:20s}: {cnt:,}")

    # ── Stabilization time ──────────────────────────────────────────────
    if "stabilization_time_s" in df.columns:
        stab = df["stabilization_time_s"].dropna()
        censored = df["stabilization_censored"].sum() if "stabilization_censored" in df.columns else 0
        print(f"\n  Stabilization time: n={len(stab):,}  "
              f"median={stab.median():.2f}s  censored={censored:,}")


if __name__ == "__main__":
    main()
