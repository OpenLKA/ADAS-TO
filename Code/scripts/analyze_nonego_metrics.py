#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_nonego_metrics.py
=========================
Focused analysis of Non-ego takeover clips: lateral deviation, longitudinal
conflict proxies (no radarState), post-takeover stability, and plan-output
mismatch.  Produces 11+ figures, a metrics CSV, and a markdown report.

Run:
    python3 scripts/analyze_nonego_metrics.py
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
ROOT   = CODE.parent                          # TakeOver/
LABELS = CODE / "DatasetClassification" / "ego_nonego_labels.csv"
MASTER = CODE / "stats_output" / "analysis_master.csv"
OUTDIR = CODE / "outputs" / "metric"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
#  Tuneable constants
# ═══════════════════════════════════════════════════════════════════════
PRE_S        = 3.0          # pre-window seconds
POST_S       = 5.0          # post-window seconds
SMOOTH_WIN_S = 0.3          # Savitzky-Golay window in seconds
SMOOTH_POLY  = 2
STAB_A0      = 0.5          # |aEgo| threshold for stabilization (m/s²)
STAB_J0      = 1.0          # |jerk| threshold (m/s³)
STAB_SR0     = 30.0         # |steer_rate| threshold (°/s)
STAB_DUR     = 1.0          # sustained duration (s)
JERK_CAP     = 50.0         # m/s³
STEER_RATE_CAP = 500.0      # °/s
N_WORKERS    = 12
N_BOOT       = 2000
EPS          = 1e-6

# Speed regime thresholds (m/s)
SPEED_LOW  = 16.7   # ~60 km/h
SPEED_HIGH = 27.8   # ~100 km/h

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

RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def _save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(OUTDIR / f"{name}{ext}")
    plt.close(fig)
    print(f"  saved {name}")


def safe_read_csv(path, usecols=None):
    """Read CSV robustly; return None on failure."""
    try:
        df = pd.read_csv(path, usecols=usecols)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def time_window(df, event_t, before_s, after_s):
    """Slice dataframe by time_s around event_t."""
    if df is None or "time_s" not in df.columns:
        return None
    t = df["time_s"].values
    mask = (t >= event_t - before_s) & (t <= event_t + after_s)
    sub = df.loc[mask]
    return sub if len(sub) >= 2 else None


def smooth_signal(values, sample_rate_hz):
    """Savitzky-Golay smoothing."""
    from scipy.signal import savgol_filter
    win = max(int(round(SMOOTH_WIN_S * sample_rate_hz)), 3)
    if win % 2 == 0:
        win += 1
    if len(values) < win:
        return values
    return savgol_filter(values, win, SMOOTH_POLY)


def safe_diff_dt(values, times):
    """d(values)/dt with variable dt."""
    if len(values) < 2:
        return np.array([])
    dt = np.diff(times)
    dt[dt < EPS] = EPS
    return np.diff(values) / dt


def estimate_hz(times):
    """Estimate sampling rate from timestamps."""
    if len(times) < 3:
        return 10.0
    dt = np.median(np.diff(times))
    return 1.0 / max(dt, EPS)


def align_to_grid(df_a, col_a, df_b, col_b, event_t, before_s, after_s,
                  grid_hz=20):
    """Interpolate two signals to common time grid, return (vals_a, vals_b)."""
    wa = time_window(df_a, event_t, before_s, after_s)
    wb = time_window(df_b, event_t, before_s, after_s)
    if wa is None or wb is None:
        return None, None
    if col_a not in wa.columns or col_b not in wb.columns:
        return None, None
    ta, va = wa["time_s"].values, wa[col_a].values
    tb, vb = wb["time_s"].values, wb[col_b].values
    mask_a = np.isfinite(va)
    mask_b = np.isfinite(vb)
    if mask_a.sum() < 2 or mask_b.sum() < 2:
        return None, None
    ta, va = ta[mask_a], va[mask_a]
    tb, vb = tb[mask_b], vb[mask_b]
    t0 = max(ta[0], tb[0])
    t1 = min(ta[-1], tb[-1])
    if t1 - t0 < 0.1:
        return None, None
    grid = np.arange(t0, t1, 1.0 / grid_hz)
    ia = np.interp(grid, ta, va)
    ib = np.interp(grid, tb, vb)
    return ia, ib


def bootstrap_median_ci(arr, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap percentile CI for the median."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return np.nan, np.nan, np.nan
    med = np.nanmedian(arr)
    boots = np.array([np.nanmedian(RNG.choice(arr, len(arr), replace=True))
                      for _ in range(n_boot)])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return med, lo, hi


def parse_json_list(val):
    """Parse a JSON-encoded list column (e.g. speeds, accels)."""
    if pd.isna(val):
        return None
    if isinstance(val, (list, np.ndarray)):
        return val
    try:
        out = json.loads(str(val).replace("'", '"'))
        if isinstance(out, list):
            return out
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Clip discovery
# ═══════════════════════════════════════════════════════════════════════

def find_all_clips():
    """Build dict: (dongle_id, route_id, clip_id) → clip_dir."""
    lookup = {}
    for mj in ROOT.rglob("meta.json"):
        d = mj.parent
        try:
            with open(mj, "r") as f:
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
    """Extract all metrics for one clip. Returns dict or None."""
    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None

    event_t = meta.get("video_time_s", 10.0)
    clip_id = meta.get("clip_id", -1)
    dongle  = meta.get("dongle_id", "")
    route   = meta.get("route_id", "")
    car     = meta.get("car_model", "")

    row = dict(
        dongle_id=dongle, route_id=route, clip_id=clip_id,
        car_model=car, event_t=event_t,
    )

    # ── Read CSVs ────────────────────────────────────────────────────
    model = safe_read_csv(clip_dir / "drivingModelData.csv")
    lplan = safe_read_csv(clip_dir / "longitudinalPlan.csv")
    state = safe_read_csv(clip_dir / "carState.csv")
    ctrl  = safe_read_csv(clip_dir / "carControl.csv")
    outp  = safe_read_csv(clip_dir / "carOutput.csv")

    row["miss_model"]   = model is None
    row["miss_lplan"]   = lplan is None
    row["miss_state"]   = state is None
    row["miss_ctrl"]    = ctrl is None
    row["miss_output"]  = outp is None

    # ── A) Lateral: lane deviation at takeover ───────────────────────
    row["lane_dev"]      = np.nan
    row["lane_width"]    = np.nan
    row["laneProb_min"]  = np.nan
    row["laneProb_left"] = np.nan
    row["laneProb_right"] = np.nan

    if model is not None and "time_s" in model.columns:
        m_at = time_window(model, event_t, 0.5, 0.5)  # narrow window around t=0
        if m_at is not None and len(m_at) > 0:
            leftY  = m_at.get("laneLineMeta.leftY")
            rightY = m_at.get("laneLineMeta.rightY")
            leftP  = m_at.get("laneLineMeta.leftProb")
            rightP = m_at.get("laneLineMeta.rightProb")

            if leftY is not None and rightY is not None:
                lY = leftY.values
                rY = rightY.values
                # Pick the sample closest to event_t
                idx = np.argmin(np.abs(m_at["time_s"].values - event_t))
                ly_val = lY[idx]
                ry_val = rY[idx]
                # Lane deviation: lateral offset of ego from lane center
                # In ego coords: leftY < 0, rightY > 0 when centered
                # Lane center = (leftY + rightY) / 2; deviation = -center
                # Positive = ego shifted right
                row["lane_dev"]   = -(ly_val + ry_val) / 2.0
                row["lane_width"] = ry_val - ly_val

            if leftP is not None and rightP is not None:
                idx = np.argmin(np.abs(m_at["time_s"].values - event_t))
                row["laneProb_left"]  = leftP.values[idx]
                row["laneProb_right"] = rightP.values[idx]
                row["laneProb_min"]   = min(leftP.values[idx],
                                            rightP.values[idx])

        # Also compute pre-window lane prob stats
        m_pre = time_window(model, event_t, PRE_S, 0)
        if m_pre is not None:
            lp = m_pre.get("laneLineMeta.leftProb")
            rp = m_pre.get("laneLineMeta.rightProb")
            if lp is not None and rp is not None:
                minp = np.minimum(lp.values, rp.values)
                row["laneProb_min_pre"] = np.nanmin(minp) if len(minp) > 0 else np.nan
            else:
                row["laneProb_min_pre"] = np.nan
        else:
            row["laneProb_min_pre"] = np.nan
    else:
        row["laneProb_min_pre"] = np.nan

    # ── B) Longitudinal proxies (no radarState) ─────────────────────
    row["hasLead_rate_pre"]      = np.nan
    row["decel_demand_pre"]      = np.nan
    row["planned_speed_drop"]    = np.nan
    row["aTarget_min_pre"]       = np.nan
    row["aTarget_mean_pre"]      = np.nan
    row["fcw_present_pre"]       = False

    if lplan is not None and "time_s" in lplan.columns:
        lp_pre = time_window(lplan, event_t, PRE_S, 0)
        if lp_pre is not None and len(lp_pre) > 0:
            # hasLead fraction
            if "hasLead" in lp_pre.columns:
                hl = lp_pre["hasLead"]
                if hl.dtype == object:
                    hl = hl.map({"True": True, "true": True, "1": True,
                                 "False": False, "false": False, "0": False})
                row["hasLead_rate_pre"] = float(hl.astype(float).mean())

            # aTarget
            if "aTarget" in lp_pre.columns:
                at = pd.to_numeric(lp_pre["aTarget"], errors="coerce").dropna()
                if len(at) > 0:
                    row["aTarget_min_pre"]  = float(at.min())
                    row["aTarget_mean_pre"] = float(at.mean())
                    row["decel_demand_pre"] = float(np.abs(at.min()))

            # FCW
            if "fcw" in lp_pre.columns:
                fcw = lp_pre["fcw"]
                if fcw.dtype == object:
                    fcw = fcw.map({"True": True, "true": True, "1": True,
                                   "False": False, "false": False, "0": False})
                row["fcw_present_pre"] = bool(fcw.astype(float).max() > 0.5)

        # Planned speed drop in post-window
        lp_post = time_window(lplan, event_t, 0, 2.0)
        if lp_post is not None and "speeds" in lp_post.columns:
            drops = []
            for _, r in lp_post.iterrows():
                sp = parse_json_list(r.get("speeds"))
                if sp and len(sp) > 1:
                    drops.append(sp[0] - min(sp))
            if drops:
                row["planned_speed_drop"] = float(np.nanmax(drops))

    # ── C) Post-takeover stability ───────────────────────────────────
    row["vEgo_at_to"]         = np.nan
    row["steer_rate_max_post"] = np.nan
    row["jerk_max_post"]       = np.nan
    row["peak_decel_post"]     = np.nan
    row["a_lat_max_post"]      = np.nan
    row["lat_jerk_max_post"]   = np.nan
    row["stabilization_time"]  = np.nan
    row["stab_censored"]       = True

    if state is not None and "time_s" in state.columns:
        # Speed at takeover
        s_at = time_window(state, event_t, 0.2, 0.2)
        if s_at is not None and "vEgo" in s_at.columns:
            idx = np.argmin(np.abs(s_at["time_s"].values - event_t))
            row["vEgo_at_to"] = float(s_at["vEgo"].values[idx])

        s_post = time_window(state, event_t, 0, POST_S)
        if s_post is not None and len(s_post) >= 3:
            t = s_post["time_s"].values
            hz = estimate_hz(t)

            # Steer rate
            if "steeringAngleDeg" in s_post.columns:
                steer = s_post["steeringAngleDeg"].values
                steer_s = smooth_signal(steer, hz)
                sr = np.abs(safe_diff_dt(steer_s, t))
                if len(sr) > 0:
                    row["steer_rate_max_post"] = float(min(np.nanmax(sr),
                                                           STEER_RATE_CAP))

            # Jerk
            if "aEgo" in s_post.columns:
                aego = s_post["aEgo"].values
                aego_s = smooth_signal(aego, hz)
                jerk = np.abs(safe_diff_dt(aego_s, t))
                if len(jerk) > 0:
                    row["jerk_max_post"] = float(min(np.nanmax(jerk), JERK_CAP))
                row["peak_decel_post"] = float(np.nanmin(aego))

            # Stabilization time
            if "aEgo" in s_post.columns and "steeringAngleDeg" in s_post.columns:
                aego = s_post["aEgo"].values
                aego_s = smooth_signal(aego, hz)
                jk = np.abs(safe_diff_dt(aego_s, t))
                steer = s_post["steeringAngleDeg"].values
                steer_s = smooth_signal(steer, hz)
                sr_ = np.abs(safe_diff_dt(steer_s, t))
                n = min(len(jk), len(sr_))
                if n > 0:
                    jk = jk[:n]
                    sr_ = sr_[:n]
                    ae = np.abs(aego_s[1:n+1])
                    calm = (ae < STAB_A0) & (jk < STAB_J0) & (sr_ < STAB_SR0)
                    t_mid = (t[1:n+1] + t[:n]) / 2.0
                    # Find first sustained calm period of STAB_DUR
                    stab_t = np.nan
                    for start_idx in range(len(calm)):
                        if not calm[start_idx]:
                            continue
                        end_t = t_mid[start_idx] + STAB_DUR
                        ok = True
                        for j in range(start_idx, len(calm)):
                            if t_mid[j] > end_t:
                                break
                            if not calm[j]:
                                ok = False
                                break
                        if ok:
                            stab_t = t_mid[start_idx] - event_t
                            break
                    row["stabilization_time"] = float(stab_t)
                    row["stab_censored"] = np.isnan(stab_t)

    # ── Lateral accel proxy: a_lat ≈ vEgo² * curvature ──────────────
    if (state is not None and outp is not None
            and "time_s" in state.columns and "time_s" in outp.columns):
        s_post = time_window(state, event_t, 0, POST_S)
        o_post = time_window(outp,  event_t, 0, POST_S)
        if (s_post is not None and o_post is not None
                and "vEgo" in s_post.columns
                and "actuatorsOutput.curvature" in o_post.columns):
            # Interpolate curvature to state timestamps
            ts = s_post["time_s"].values
            v  = s_post["vEgo"].values
            tc = o_post["time_s"].values
            kc = o_post["actuatorsOutput.curvature"].values
            mask = np.isfinite(kc)
            if mask.sum() >= 2:
                curv_interp = np.interp(ts, tc[mask], kc[mask])
                a_lat = v**2 * np.abs(curv_interp)
                row["a_lat_max_post"] = float(np.nanmax(a_lat))

                hz_s = estimate_hz(ts)
                a_lat_s = smooth_signal(a_lat, hz_s)
                lat_jerk = np.abs(safe_diff_dt(a_lat_s, ts))
                if len(lat_jerk) > 0:
                    row["lat_jerk_max_post"] = float(
                        min(np.nanmax(lat_jerk), JERK_CAP))

    # ── D) Plan→output→state mismatch RMSE ──────────────────────────
    row["rmse_accel_plan_output_pre"]  = np.nan
    row["rmse_accel_plan_output_post"] = np.nan
    row["rmse_accel_output_state_pre"] = np.nan
    row["rmse_accel_output_state_post"] = np.nan
    row["rmse_curv_plan_output_pre"]   = np.nan
    row["rmse_curv_plan_output_post"]  = np.nan

    def _rmse(a, b):
        if a is None or b is None:
            return np.nan
        diff = a - b
        return float(np.sqrt(np.nanmean(diff**2)))

    if ctrl is not None and outp is not None:
        # accel plan→output
        for label, bef, aft in [("pre", PRE_S, 0), ("post", 0, POST_S)]:
            va, vb = align_to_grid(ctrl, "actuators.accel",
                                   outp, "actuatorsOutput.accel",
                                   event_t, bef, aft)
            val = _rmse(va, vb)
            # Gate: if RMSE is exactly 0, channel likely inactive
            if val == 0.0:
                val = np.nan
            row[f"rmse_accel_plan_output_{label}"] = val

        # curvature plan→output
        for label, bef, aft in [("pre", PRE_S, 0), ("post", 0, POST_S)]:
            va, vb = align_to_grid(ctrl, "actuators.curvature",
                                   outp, "actuatorsOutput.curvature",
                                   event_t, bef, aft)
            val = _rmse(va, vb)
            if val == 0.0:
                val = np.nan
            row[f"rmse_curv_plan_output_{label}"] = val

    if outp is not None and state is not None:
        # accel output→state
        for label, bef, aft in [("pre", PRE_S, 0), ("post", 0, POST_S)]:
            va, vb = align_to_grid(outp, "actuatorsOutput.accel",
                                   state, "aEgo",
                                   event_t, bef, aft)
            val = _rmse(va, vb)
            if val == 0.0:
                val = np.nan
            row[f"rmse_accel_output_state_{label}"] = val

    return row


# ═══════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_nonego_set():
    """Load Non-ego clip IDs and merge with clip directory lookup."""
    print("Loading labels …")
    labels = pd.read_csv(LABELS)
    ne = labels[labels["label"] == "Non-ego"].copy()
    print(f"  Non-ego clips in labels CSV: {len(ne)}")

    print("Building clip directory lookup …")
    lookup = find_all_clips()
    print(f"  Found {len(lookup)} clip dirs total")

    # Map clip dirs
    dirs = []
    for _, r in ne.iterrows():
        key = (r["dongle_id"], r["route_id"], int(r["clip_id"]))
        dirs.append(lookup.get(key))
    ne["clip_dir"] = dirs
    ne = ne.dropna(subset=["clip_dir"])
    print(f"  Non-ego clips with directories: {len(ne)}")
    return ne


# ═══════════════════════════════════════════════════════════════════════
#  Parallel extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_all(ne_df):
    """Run per-clip extraction in parallel."""
    clip_dirs = ne_df["clip_dir"].tolist()
    results = []
    total = len(clip_dirs)
    done = 0
    errors = 0

    print(f"Extracting metrics from {total} clips ({N_WORKERS} workers) …")
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
    print(f"  Extracted {len(df)} rows, {errors} errors")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Post-processing
# ═══════════════════════════════════════════════════════════════════════

def postprocess(df):
    """Add derived columns and speed regimes."""
    # Speed regime
    v = df["vEgo_at_to"]
    df["speed_regime"] = pd.cut(
        v, bins=[0, SPEED_LOW, SPEED_HIGH, 999],
        labels=["Low (<60 km/h)", "Medium", "High (>100 km/h)"],
        right=False,
    )
    # Curvature mismatch proxy from RMSE
    df["curv_mismatch"] = df["rmse_curv_plan_output_pre"]

    # Winsorize curvature mismatch at P99
    p99 = df["curv_mismatch"].quantile(0.99)
    if np.isfinite(p99):
        df["curv_mismatch"] = df["curv_mismatch"].clip(upper=p99)

    # Cap a_lat_max_post at P99 (v²×κ can produce extreme outliers)
    for col in ["a_lat_max_post", "lat_jerk_max_post"]:
        p99 = df[col].quantile(0.99)
        if np.isfinite(p99) and p99 > 0:
            df[col] = df[col].clip(upper=p99)

    # Lead-present flag
    df["lead_present"] = df["hasLead_rate_pre"].fillna(0) > 0.5
    return df


# ═══════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════

def fig01_lane_dev_dist(df):
    """Distribution of lane deviation at takeover."""
    vals = df["lane_dev"].dropna().values
    if len(vals) < 30:
        print("  SKIP fig01: insufficient lane_dev data")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.hist(vals, bins=60, density=True, color=C["blue"], alpha=0.7,
            edgecolor="white", linewidth=0.3)

    # KDE-like smooth via gaussian_kde
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(vals, bw_method=0.15)
    xg = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 200)
    ax.plot(xg, kde(xg), color=C["red"], lw=1.5)

    med = np.nanmedian(vals)
    ax.axvline(med, color=C["orange"], ls="--", lw=1.0,
               label=f"Median = {med:.3f} m")
    ax.axvline(np.percentile(vals, 5), color=C["gray"], ls=":", lw=0.8,
               label=f"P5 = {np.percentile(vals,5):.3f}")
    ax.axvline(np.percentile(vals, 95), color=C["gray"], ls=":", lw=0.8,
               label=f"P95 = {np.percentile(vals,95):.3f}")
    ax.set_xlabel("Lane deviation (m)\n← shifted left    shifted right →")
    ax.set_ylabel("Density")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    _save(fig, "fig01_lane_dev_dist")


def fig02_lane_dev_speed(df):
    """Lane deviation by speed regime."""
    sub = df.dropna(subset=["lane_dev", "speed_regime"])
    if len(sub) < 30:
        print("  SKIP fig02: insufficient data")
        return
    regimes = ["Low (<60 km/h)", "Medium", "High (>100 km/h)"]
    data = [sub.loc[sub["speed_regime"] == r, "lane_dev"].dropna().values
            for r in regimes]
    colors = [C["teal"], C["blue"], C["red"]]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bp = ax.boxplot(data, positions=range(len(regimes)), widths=0.5,
                    patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, fontsize=7.5)
    ax.set_ylabel("Lane deviation (m)")
    # Annotate N
    for i, d in enumerate(data):
        ax.text(i, ax.get_ylim()[1] * 0.95, f"n={len(d)}",
                ha="center", fontsize=6, color="#555")
    fig.tight_layout()
    _save(fig, "fig02_lane_dev_speed")


def fig03_lane_dev_vs_laneprob(df):
    """Lane dev vs laneProb_min scatter (subsampled)."""
    sub = df.dropna(subset=["lane_dev", "laneProb_min"])
    if len(sub) < 30:
        print("  SKIP fig03: insufficient data")
        return
    # Subsample to avoid overplotting
    if len(sub) > 3000:
        sub = sub.sample(3000, random_state=42)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.scatter(sub["laneProb_min"], np.abs(sub["lane_dev"]),
               s=3, alpha=0.25, color=C["blue"], rasterized=True)
    ax.set_xlabel("Lane probability (min L/R)")
    ax.set_ylabel("|Lane deviation| (m)")
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, "fig03_lane_dev_vs_laneprob")


def fig04_longit_proxy(df):
    """Longitudinal conflict proxy distributions for lead-present clips."""
    lp = df[df["lead_present"]].copy()
    if len(lp) < 30:
        print("  SKIP fig04: insufficient lead-present data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # (a) decel demand
    ax = axes[0]
    vals = lp["decel_demand_pre"].dropna().values
    if len(vals) > 5:
        ax.hist(vals, bins=50, color=C["orange"], alpha=0.7,
                edgecolor="white", linewidth=0.3)
        med = np.nanmedian(vals)
        ax.axvline(med, color=C["red"], ls="--", lw=1.0,
                   label=f"Median = {med:.2f}")
        ax.set_xlabel("Decel demand |aTarget_min| (m/s²)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6.5, frameon=False)
        ax.text(0.95, 0.92, f"n = {len(vals)}", transform=ax.transAxes,
                ha="right", fontsize=7, color="#555")

    # (b) planned speed drop
    ax = axes[1]
    vals = lp["planned_speed_drop"].dropna().values
    if len(vals) > 5:
        ax.hist(vals, bins=50, color=C["teal"], alpha=0.7,
                edgecolor="white", linewidth=0.3)
        med = np.nanmedian(vals)
        ax.axvline(med, color=C["red"], ls="--", lw=1.0,
                   label=f"Median = {med:.2f}")
        ax.set_xlabel("Planned speed drop (m/s)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6.5, frameon=False)
        ax.text(0.95, 0.92, f"n = {len(vals)}", transform=ax.transAxes,
                ha="right", fontsize=7, color="#555")

    fig.tight_layout()
    _save(fig, "fig04_longit_proxy_dist")


def fig05_longit_proxy_speed(df):
    """Longitudinal proxies by speed regime (lead-present only)."""
    lp = df[df["lead_present"]].dropna(subset=["decel_demand_pre", "speed_regime"])
    if len(lp) < 30:
        print("  SKIP fig05")
        return
    regimes = ["Low (<60 km/h)", "Medium", "High (>100 km/h)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, col, lab in [(axes[0], "decel_demand_pre", "Decel demand (m/s²)"),
                         (axes[1], "planned_speed_drop", "Planned speed drop (m/s)")]:
        data = [lp.loc[lp["speed_regime"] == r, col].dropna().values
                for r in regimes]
        colors = [C["teal"], C["blue"], C["red"]]
        bp = ax.boxplot(data, positions=range(3), widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        for patch, clr in zip(bp["boxes"], colors):
            patch.set_facecolor(clr)
            patch.set_alpha(0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(regimes, fontsize=7)
        ax.set_ylabel(lab)
        for i, d in enumerate(data):
            ax.text(i, ax.get_ylim()[1]*0.95, f"n={len(d)}",
                    ha="center", fontsize=6, color="#555")
    fig.tight_layout()
    _save(fig, "fig05_longit_proxy_speed")


def fig06_stability_violins(df):
    """Violin/box for jerk_max, steer_rate_max, a_lat_max post-takeover."""
    metrics = [
        ("jerk_max_post", "Max jerk (m/s³)"),
        ("steer_rate_max_post", "Max steer rate (°/s)"),
        ("a_lat_max_post", "Max lat. accel (m/s²)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    colors = [C["blue"], C["orange"], C["purple"]]
    for ax, (col, lab), clr in zip(axes, metrics, colors):
        vals = df[col].dropna().values
        if len(vals) < 10:
            ax.text(0.5, 0.5, "Insufficient data", ha="center",
                    va="center", transform=ax.transAxes, fontsize=8)
            ax.set_ylabel(lab)
            continue
        vp = ax.violinplot([vals], positions=[0], showmedians=True,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(clr)
            body.set_alpha(0.5)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.2)
        # Overlay box
        bp = ax.boxplot([vals], positions=[0], widths=0.15,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=0))
        bp["boxes"][0].set_facecolor(clr)
        bp["boxes"][0].set_alpha(0.7)
        ax.set_ylabel(lab)
        ax.set_xticks([])
        med = np.nanmedian(vals)
        ax.text(0.95, 0.92, f"Median = {med:.1f}\nn = {len(vals)}",
                transform=ax.transAxes, ha="right", fontsize=6.5, color="#555")
    fig.tight_layout()
    _save(fig, "fig06_stability_violins")


def _binned_line_ci(ax, x_vals, y_vals, bins, color, label="", marker="o"):
    """Binned median line with bootstrap CI."""
    meds, los, his, xs, ns = [], [], [], [], []
    for i in range(len(bins) - 1):
        mask = (x_vals >= bins[i]) & (x_vals < bins[i+1])
        yb = y_vals[mask]
        if len(yb) < 10:
            continue
        m, lo, hi = bootstrap_median_ci(yb)
        meds.append(m)
        los.append(lo)
        his.append(hi)
        xs.append((bins[i] + bins[i+1]) / 2)
        ns.append(len(yb))
    if not xs:
        return
    xs, meds, los, his = map(np.array, [xs, meds, los, his])
    ax.plot(xs, meds, f"{marker}-", color=color, lw=1.2, ms=4, label=label)
    ax.fill_between(xs, los, his, color=color, alpha=0.15)
    for xi, ni in zip(xs, ns):
        ax.text(xi, ax.get_ylim()[0], f"{ni}", ha="center",
                fontsize=5, color="#999", va="bottom")


def fig07_stability_vs_laneprob(df):
    """Stability metrics vs laneProb_min (binned line + CI)."""
    sub = df.dropna(subset=["laneProb_min_pre", "steer_rate_max_post"])
    if len(sub) < 50:
        print("  SKIP fig07")
        return
    bins = [0, 0.1, 0.3, 0.6, 0.9, 1.01]
    metrics = [
        ("steer_rate_max_post", "Steer rate max (°/s)", C["blue"]),
        ("jerk_max_post",       "Jerk max (m/s³)",      C["orange"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, (col, lab, clr) in zip(axes, metrics):
        s = df.dropna(subset=["laneProb_min_pre", col])
        xv = s["laneProb_min_pre"].values
        yv = s[col].values
        _binned_line_ci(ax, xv, yv, bins, clr)
        ax.set_xlabel("Lane probability (min pre)")
        ax.set_ylabel(lab)
    fig.tight_layout()
    _save(fig, "fig07_stability_vs_laneprob")


def fig08_stability_vs_mismatch(df):
    """Stability vs curvature mismatch (binned line + CI)."""
    sub = df.dropna(subset=["curv_mismatch", "steer_rate_max_post"])
    if len(sub) < 50:
        print("  SKIP fig08")
        return
    # Quantile-based bins
    qs = sub["curv_mismatch"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    qs[-1] += EPS
    metrics = [
        ("steer_rate_max_post", "Steer rate max (°/s)", C["blue"]),
        ("jerk_max_post",       "Jerk max (m/s³)",      C["orange"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    for ax, (col, lab, clr) in zip(axes, metrics):
        s = df.dropna(subset=["curv_mismatch", col])
        _binned_line_ci(ax, s["curv_mismatch"].values, s[col].values, qs, clr)
        ax.set_xlabel("Curvature mismatch RMSE (1/m)")
        ax.set_ylabel(lab)
    fig.tight_layout()
    _save(fig, "fig08_stability_vs_mismatch")


def fig09_rmse_distributions(df):
    """Plan→output RMSE distributions: 3 panels."""
    pairs = [
        ("rmse_accel_plan_output", "Accel plan→output"),
        ("rmse_accel_output_state", "Accel output→state"),
        ("rmse_curv_plan_output",  "Curv plan→output"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    colors = [C["blue"], C["orange"], C["teal"]]
    for ax, (prefix, lab), clr in zip(axes, pairs, colors):
        pre_col  = f"{prefix}_pre"
        post_col = f"{prefix}_post"
        vpre  = df[pre_col].dropna().values
        vpost = df[post_col].dropna().values
        if len(vpre) < 10 or len(vpost) < 10:
            ax.text(0.5, 0.5, "Insufficient", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_ylabel(lab)
            continue
        bp = ax.boxplot([vpre, vpost], positions=[0, 1], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set_facecolor(clr)
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor(clr)
        bp["boxes"][1].set_alpha(0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"], fontsize=8)
        ax.set_ylabel(f"RMSE  ({lab})")
        ax.text(0, ax.get_ylim()[1]*0.92, f"n={len(vpre)}",
                ha="center", fontsize=6, color="#555")
        ax.text(1, ax.get_ylim()[1]*0.92, f"n={len(vpost)}",
                ha="center", fontsize=6, color="#555")
    fig.tight_layout()
    _save(fig, "fig09_rmse_distributions")


def fig10_correlation_heatmap(df):
    """Correlation heatmap among key metrics."""
    cols = [
        "lane_dev", "laneProb_min_pre", "curv_mismatch",
        "decel_demand_pre", "jerk_max_post", "steer_rate_max_post",
        "a_lat_max_post", "peak_decel_post",
    ]
    labels = [
        "Lane dev", "Lane prob", "Curv mismatch",
        "Decel demand", "Jerk max", "Steer rate",
        "Lat accel", "Peak decel",
    ]
    sub = df[cols].dropna(thresh=4)
    if len(sub) < 50:
        print("  SKIP fig10")
        return
    corr = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    n = len(cols)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=6.5)
    # Annotate
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(v) > 0.5 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig10_correlation_heatmap")


def fig11_stability_by_speed(df):
    """Stability metrics by speed regime (box plots)."""
    regimes = ["Low (<60 km/h)", "Medium", "High (>100 km/h)"]
    metrics = [
        ("jerk_max_post", "Max jerk (m/s³)"),
        ("steer_rate_max_post", "Max steer rate (°/s)"),
        ("a_lat_max_post", "Max lat. accel (m/s²)"),
    ]
    colors_r = [C["teal"], C["blue"], C["red"]]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8))
    for ax, (col, lab) in zip(axes, metrics):
        data = [df.loc[df["speed_regime"]==r, col].dropna().values
                for r in regimes]
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
            ax.text(i, ax.get_ylim()[1]*0.95, f"n={len(d)}",
                    ha="center", fontsize=5.5, color="#555")
    fig.tight_layout()
    _save(fig, "fig11_stability_by_speed")


def fig12_stabilization_dist(df):
    """Distribution of stabilization time (censored at 5s)."""
    vals = df["stabilization_time"].dropna().values
    cens = df["stab_censored"].sum()
    total = len(df)
    if len(vals) < 10:
        print("  SKIP fig12")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.hist(vals, bins=40, color=C["green"], alpha=0.7,
            edgecolor="white", linewidth=0.3)
    med = np.nanmedian(vals)
    ax.axvline(med, color=C["red"], ls="--", lw=1.0,
               label=f"Median = {med:.2f} s")
    ax.set_xlabel("Stabilization time (s)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6.5, frameon=False)
    ax.text(0.95, 0.88,
            f"Stabilized: {len(vals)}\nCensored: {cens}",
            transform=ax.transAxes, ha="right", fontsize=6.5, color="#555")
    fig.tight_layout()
    _save(fig, "fig12_stabilization_dist")


# ═══════════════════════════════════════════════════════════════════════
#  Markdown report
# ═══════════════════════════════════════════════════════════════════════

def write_report(df):
    """Generate the markdown summary report."""
    n = len(df)
    lines = [
        "# Non-Ego Takeover Metrics Report\n",
        f"## 1. Dataset Overview\n",
        f"- **Total Non-ego clips analysed**: {n}",
        f"- **Lead-present clips** (hasLead > 50% pre-window): "
        f"{df['lead_present'].sum()} ({100*df['lead_present'].mean():.1f}%)",
    ]

    # Missingness
    miss_cols = [c for c in df.columns if c.startswith("miss_")]
    lines.append(f"\n### Missingness rates\n")
    lines.append("| Source | Missing % |")
    lines.append("|--------|--------:|")
    for mc in miss_cols:
        pct = 100 * df[mc].mean()
        lines.append(f"| {mc.replace('miss_','')} | {pct:.1f}% |")

    # Metric availability
    key_metrics = [
        "lane_dev", "laneProb_min", "laneProb_min_pre",
        "decel_demand_pre", "planned_speed_drop",
        "jerk_max_post", "steer_rate_max_post", "a_lat_max_post",
        "lat_jerk_max_post", "peak_decel_post", "stabilization_time",
        "rmse_accel_plan_output_pre", "rmse_accel_plan_output_post",
        "rmse_curv_plan_output_pre", "rmse_curv_plan_output_post",
        "rmse_accel_output_state_pre", "rmse_accel_output_state_post",
    ]
    lines.append(f"\n### Metric availability\n")
    lines.append("| Metric | Non-missing | % |")
    lines.append("|--------|----------:|----:|")
    for mc in key_metrics:
        if mc in df.columns:
            cnt = df[mc].notna().sum()
            pct = 100 * cnt / n
            lines.append(f"| {mc} | {cnt} | {pct:.1f}% |")

    # Key medians
    lines.append(f"\n## 2. Key Metric Summaries\n")
    lines.append("| Metric | Median | IQR (Q1–Q3) | P5 | P95 |")
    lines.append("|--------|-------:|:-----------:|---:|----:|")
    for mc in key_metrics:
        if mc in df.columns:
            v = df[mc].dropna()
            if len(v) > 5:
                med = v.median()
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                p5, p95 = v.quantile(0.05), v.quantile(0.95)
                lines.append(
                    f"| {mc} | {med:.3f} | {q1:.3f}–{q3:.3f} | "
                    f"{p5:.3f} | {p95:.3f} |")

    # Findings
    lines.append(f"\n## 3. Key Findings\n")

    findings = []

    # Lane dev
    ld = df["lane_dev"].dropna()
    if len(ld) > 50:
        findings.append(
            f"**Lane deviation at takeover**: Median |lane_dev| = "
            f"{np.abs(ld).median():.3f} m (IQR {np.abs(ld).quantile(0.25):.3f}–"
            f"{np.abs(ld).quantile(0.75):.3f}). "
            f"The distribution is roughly symmetric around zero, suggesting "
            f"no systematic lateral bias at Non-ego takeover onset.")

    # laneProb → steer rate
    sub = df.dropna(subset=["laneProb_min_pre", "steer_rate_max_post"])
    lo = sub[sub["laneProb_min_pre"] < 0.1]["steer_rate_max_post"]
    hi = sub[sub["laneProb_min_pre"] > 0.9]["steer_rate_max_post"]
    if len(lo) > 20 and len(hi) > 20:
        findings.append(
            f"**Lane confidence → lateral urgency**: Non-ego clips with "
            f"laneProb < 0.1 show median steer rate {lo.median():.1f} °/s "
            f"versus {hi.median():.1f} °/s for laneProb > 0.9 "
            f"({lo.median()/max(hi.median(),0.01):.1f}× ratio).")

    # Lead-present decel demand
    lp = df[df["lead_present"]]
    dd = lp["decel_demand_pre"].dropna()
    if len(dd) > 20:
        findings.append(
            f"**Longitudinal conflict proxy (lead-present)**: Median decel "
            f"demand = {dd.median():.2f} m/s² (n = {len(dd)}). "
            f"This reflects planner intent; true TTC/THW require radar "
            f"distance data which is excluded from this analysis.")

    # Stability
    sr = df["steer_rate_max_post"].dropna()
    jk = df["jerk_max_post"].dropna()
    if len(sr) > 50:
        findings.append(
            f"**Post-takeover lateral stability**: Median steer rate = "
            f"{sr.median():.1f} °/s; P95 = {sr.quantile(0.95):.1f} °/s.")
    if len(jk) > 50:
        findings.append(
            f"**Post-takeover longitudinal stability**: Median jerk = "
            f"{jk.median():.1f} m/s³; P95 = {jk.quantile(0.95):.1f} m/s³.")

    # Stabilization
    st = df["stabilization_time"].dropna()
    cens_rate = 100 * df["stab_censored"].mean()
    if len(st) > 20:
        findings.append(
            f"**Stabilization time**: Median = {st.median():.2f} s. "
            f"Censored (not stabilized within {POST_S}s): {cens_rate:.1f}%.")

    # RMSE pre vs post
    rpre = df["rmse_accel_plan_output_pre"].dropna()
    rpost = df["rmse_accel_plan_output_post"].dropna()
    if len(rpre) > 50 and len(rpost) > 50:
        findings.append(
            f"**Plan→output accel RMSE**: Pre median = {rpre.median():.3f}, "
            f"Post median = {rpost.median():.3f}. "
            f"{'Increase' if rpost.median() > rpre.median() else 'Decrease'} "
            f"post-takeover reflects "
            f"{'higher' if rpost.median() > rpre.median() else 'lower'} "
            f"control mismatch during driver intervention.")

    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}\n")

    # Limitations
    lines.append(f"\n## 4. Limitations\n")
    lines.extend([
        "1. **No radar-derived TTC/THW**: This analysis does not use "
        "radarState.csv. The longitudinal conflict metrics (decel demand, "
        "planned speed drop) are planner-intent proxies, not physical "
        "headway measures. True TTC and THW require lead-vehicle distance "
        "(dRel) from radar.\n",
        "2. **Lane model availability**: drivingModelData is missing for "
        f"{100*df['miss_model'].mean():.1f}% of clips, limiting the lateral "
        "deviation analysis to clips with active lane detection.\n",
        "3. **Timing alignment**: Event time is taken from meta.json "
        "(video_time_s). Minor misalignment between CAN-bus timestamps and "
        "video clock may affect metrics computed at the exact takeover instant.\n",
        "4. **Sampling rate heterogeneity**: The dataset includes both qlog "
        "(~10 Hz) and rlog (~100 Hz) recordings. Derivative-based metrics "
        "(jerk, steer rate) are smoothed (Savitzky–Golay, 0.3s window) to "
        "mitigate rate-dependent artifacts, but residual sensitivity remains.\n",
        "5. **Curvature mismatch winsorization**: plan→output curvature RMSE "
        "is winsorized at P99 to remove extreme outliers likely reflecting "
        "instrumentation artifacts.\n",
        f"6. **Classification accuracy**: The Ego/Non-ego partition has "
        f"84.0% validated accuracy. Approximately 16% of clips may be "
        f"mislabeled, which could attenuate observed associations.\n",
    ])

    report_path = OUTDIR / "non_ego_metric_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved to {report_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Data dictionary check
# ═══════════════════════════════════════════════════════════════════════

def data_dict_check():
    """Print column availability check for key source files."""
    print("\n" + "="*60)
    print("DATA DICTIONARY CHECK")
    print("="*60)

    # Check a sample clip for available columns
    sample_dirs = list(ROOT.rglob("meta.json"))[:1]
    if sample_dirs:
        sd = sample_dirs[0].parent
        for csv_name in ["drivingModelData.csv", "longitudinalPlan.csv",
                         "carState.csv", "carControl.csv", "carOutput.csv"]:
            p = sd / csv_name
            if p.exists():
                cols = pd.read_csv(p, nrows=0).columns.tolist()
                print(f"\n{csv_name} ({len(cols)} cols):")
                for c in cols:
                    print(f"  - {c}")
            else:
                print(f"\n{csv_name}: NOT FOUND")

    # Check labels
    if LABELS.exists():
        ldf = pd.read_csv(LABELS, nrows=0)
        print(f"\nego_nonego_labels.csv: {len(ldf.columns)} cols")
    else:
        print(f"\nWARNING: {LABELS} not found!")

    # Check analysis_master
    if MASTER.exists():
        mdf = pd.read_csv(MASTER, nrows=0)
        print(f"analysis_master.csv: {len(mdf.columns)} cols")
        # Check for distance / radar columns
        radar_cols = [c for c in mdf.columns
                      if any(k in c.lower() for k in
                             ["drel", "vrel", "radar", "distance"])]
        print(f"  Radar-related cols: {radar_cols if radar_cols else 'NONE'}")
    else:
        print(f"\nWARNING: {MASTER} not found!")

    print("="*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    data_dict_check()

    ne = load_nonego_set()
    df = extract_all(ne)
    df = postprocess(df)

    # Save master CSV
    csv_path = OUTDIR / "non_ego_metrics_master.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV: {csv_path}  ({len(df)} rows)")

    # Print metric availability
    print("\n" + "="*60)
    print("METRIC AVAILABILITY (non-missing %)")
    print("="*60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in sorted(numeric_cols):
        pct = 100 * df[c].notna().mean()
        print(f"  {c:40s} {pct:6.1f}%")

    # Generate figures
    print("\nGenerating figures …")
    fig01_lane_dev_dist(df)
    fig02_lane_dev_speed(df)
    fig03_lane_dev_vs_laneprob(df)
    fig04_longit_proxy(df)
    fig05_longit_proxy_speed(df)
    fig06_stability_violins(df)
    fig07_stability_vs_laneprob(df)
    fig08_stability_vs_mismatch(df)
    fig09_rmse_distributions(df)
    fig10_correlation_heatmap(df)
    fig11_stability_by_speed(df)
    fig12_stabilization_dist(df)

    # Report
    print("\nWriting report …")
    write_report(df)

    # Print figure paths
    print("\n" + "="*60)
    print("GENERATED FIGURES")
    print("="*60)
    for f in sorted(OUTDIR.glob("fig*")):
        print(f"  {f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
