#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
takeover_safety_smoothness.py
=============================
Quantify safety proxies, control smoothness (plan→output→state consistency),
and post-takeover stability for each clip.  Uses 6 CSV topics per clip
(NO radarState, NO clustering).

Output:
    outputs/tables/control_safety_metrics.csv
    outputs/figures/fig_pre_post_distributions.pdf  (+ .png)
    outputs/figures/fig_perception_vs_smoothness.pdf
    outputs/figures/fig_plan_output_state_rmse.pdf
    outputs/figures/fig_smoothness_by_trigger.pdf
    outputs/figures/fig_interaction_flags.pdf
    outputs/reports/takeover_safety_stability.md

Run:
    cd "…/TakeOver/Code"
    python3 takeover_safety_smoothness.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from textwrap import dedent

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
#  Paths & Constants
# ═══════════════════════════════════════════════════════════════════════
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
OUT_TABLES  = CODE / "outputs" / "tables"
OUT_FIGS    = CODE / "outputs" / "figures"
OUT_REPORTS = CODE / "outputs" / "reports"
for d in (OUT_TABLES, OUT_FIGS, OUT_REPORTS):
    d.mkdir(parents=True, exist_ok=True)

MASTER_CSV = CODE / "stats_output" / "analysis_master.csv"

SMOOTH_WINDOW_S = 0.3
SMOOTH_POLY     = 2
EPS             = 1e-6
N_WORKERS       = 12
PRE_BEFORE      = 3.0
PRE_AFTER       = 0.0
POST_BEFORE     = 0.0
POST_AFTER      = 5.0
STAB_SUSTAIN_S  = 1.0
STAB_MAX_S      = 5.0

_NAN = float("nan")

# ═══════════════════════════════════════════════════════════════════════
#  Reused Utilities (from compute_derived_signals.py)
# ═══════════════════════════════════════════════════════════════════════
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
    if df.empty or "time_s" not in df.columns:
        return pd.DataFrame()
    lo = event_t - before_s
    hi = event_t + after_s
    mask = (df["time_s"] >= lo) & (df["time_s"] <= hi)
    return df[mask].copy()


def safe_diff_dt(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    dt = np.diff(times)
    dt[dt < EPS] = EPS
    return np.diff(values) / dt


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
    clips = []
    for meta in ROOT.rglob("meta.json"):
        if "Code" in meta.parts:
            continue
        clips.append(meta.parent)
    return sorted(clips)


def _estimate_hz(t: np.ndarray) -> float:
    if len(t) < 2:
        return 20.0
    return 1.0 / max(float(np.median(np.diff(t))), EPS)


# ═══════════════════════════════════════════════════════════════════════
#  New Helper: align two signals to a common time grid
# ═══════════════════════════════════════════════════════════════════════
def align_to_common_grid(
    df_a: pd.DataFrame, col_a: str,
    df_b: pd.DataFrame, col_b: str,
    event_t: float, before_s: float, after_s: float,
    grid_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate two signals onto a common time grid for RMSE computation."""
    wa = time_window(df_a, event_t, before_s, after_s)
    wb = time_window(df_b, event_t, before_s, after_s)
    if wa.empty or wb.empty:
        return None
    if col_a not in wa.columns or col_b not in wb.columns:
        return None

    ta = wa["time_s"].values
    va = pd.to_numeric(wa[col_a], errors="coerce").values
    tb = wb["time_s"].values
    vb = pd.to_numeric(wb[col_b], errors="coerce").values

    # Drop NaN
    ma = np.isfinite(va) & np.isfinite(ta)
    mb = np.isfinite(vb) & np.isfinite(tb)
    if ma.sum() < 3 or mb.sum() < 3:
        return None
    ta, va = ta[ma], va[ma]
    tb, vb = tb[mb], vb[mb]

    t_lo = max(ta[0], tb[0])
    t_hi = min(ta[-1], tb[-1])
    if t_hi - t_lo < 0.1:
        return None

    grid = np.arange(t_lo, t_hi, 1.0 / grid_hz)
    if len(grid) < 3:
        return None

    ia = np.interp(grid, ta, va)
    ib = np.interp(grid, tb, vb)
    return ia, ib


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ═══════════════════════════════════════════════════════════════════════
#  Per-Clip Worker
# ═══════════════════════════════════════════════════════════════════════
def process_clip(clip_dir: Path) -> dict | None:
    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    event_t = float(meta["video_time_s"])
    log_hz = int(meta.get("log_hz", 20))

    rec: dict = dict(
        car_model=meta["car_model"],
        dongle_id=meta["dongle_id"],
        route_id=meta["route_id"],
        clip_id=int(meta["clip_id"]),
    )

    # ── Read 6 CSVs ────────────────────────────────────────────────────
    cc_df   = safe_read_csv(clip_dir / "carControl.csv")
    co_df   = safe_read_csv(clip_dir / "carOutput.csv")
    cs_df   = safe_read_csv(clip_dir / "carState.csv")
    ctrl_df = safe_read_csv(clip_dir / "controlsState.csv")
    model_df = safe_read_csv(clip_dir / "drivingModelData.csv")
    lp_df   = safe_read_csv(clip_dir / "longitudinalPlan.csv")

    for df in (cc_df, co_df, cs_df, ctrl_df, model_df, lp_df):
        if not df.empty and "time_s" in df.columns:
            df.sort_values("time_s", inplace=True)

    # Missingness flags
    rec["miss_carControl"]      = cc_df.empty
    rec["miss_carOutput"]       = co_df.empty
    rec["miss_carState"]        = cs_df.empty
    rec["miss_controlsState"]   = ctrl_df.empty
    rec["miss_drivingModelData"] = model_df.empty
    rec["miss_longitudinalPlan"] = lp_df.empty

    # ── Windows ────────────────────────────────────────────────────────
    lp_pre  = time_window(lp_df,  event_t, PRE_BEFORE, PRE_AFTER)
    cc_pre  = time_window(cc_df,  event_t, PRE_BEFORE, PRE_AFTER)
    model_pre = time_window(model_df, event_t, PRE_BEFORE, PRE_AFTER)
    ctrl_pre  = time_window(ctrl_df,  event_t, PRE_BEFORE, PRE_AFTER)

    cs_post  = time_window(cs_df,  event_t, POST_BEFORE, POST_AFTER)
    ctrl_post = time_window(ctrl_df, event_t, POST_BEFORE, POST_AFTER)

    hz = float(log_hz)

    # ══════════════════════════════════════════════════════════════════
    #  LONGITUDINAL SAFETY (6 metrics) — pre-window
    # ══════════════════════════════════════════════════════════════════
    rec["hasLead_rate_pre"]         = _NAN
    rec["leadVisible_rate_pre"]     = _NAN
    rec["aTarget_min_pre"]          = _NAN
    rec["aTarget_mean_pre"]         = _NAN
    rec["planned_speed_drop_pre"]   = _NAN
    rec["lead_consistency_flag"]    = _NAN

    if not lp_pre.empty:
        if "hasLead" in lp_pre.columns:
            hl = parse_bool_col(lp_pre["hasLead"])
            rec["hasLead_rate_pre"] = float(hl.mean())
        if "aTarget" in lp_pre.columns:
            at = pd.to_numeric(lp_pre["aTarget"], errors="coerce").dropna()
            if len(at) > 0:
                rec["aTarget_min_pre"]  = float(at.min())
                rec["aTarget_mean_pre"] = float(at.mean())
        if "speeds" in lp_pre.columns:
            drops = []
            for s in lp_pre["speeds"].dropna():
                try:
                    sp = json.loads(str(s)) if isinstance(s, str) else s
                    if isinstance(sp, list) and len(sp) > 1:
                        drops.append(float(sp[0]) - float(min(sp)))
                except Exception:
                    pass
            if drops:
                rec["planned_speed_drop_pre"] = float(max(drops))

    if not cc_pre.empty and "hudControl.leadVisible" in cc_pre.columns:
        lv = parse_bool_col(cc_pre["hudControl.leadVisible"])
        rec["leadVisible_rate_pre"] = float(lv.mean())

    # Lead consistency: mismatch rate of hasLead XOR leadVisible
    if not lp_pre.empty and not cc_pre.empty:
        if "hasLead" in lp_pre.columns and "hudControl.leadVisible" in cc_pre.columns:
            try:
                lp_t = lp_pre[["time_s"]].copy()
                lp_t["hasLead"] = parse_bool_col(lp_pre["hasLead"]).values
                cc_t = cc_pre[["time_s"]].copy()
                cc_t["leadVis"] = parse_bool_col(cc_pre["hudControl.leadVisible"]).values
                merged = pd.merge_asof(
                    lp_t.sort_values("time_s"),
                    cc_t.sort_values("time_s"),
                    on="time_s", direction="nearest", tolerance=0.2,
                )
                valid = merged.dropna(subset=["leadVis"])
                if len(valid) > 0:
                    mismatch = (valid["hasLead"] != valid["leadVis"]).mean()
                    rec["lead_consistency_flag"] = float(mismatch)
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════
    #  LATERAL PERCEPTION (6 metrics) — pre-window
    # ══════════════════════════════════════════════════════════════════
    rec["laneProb_min_pre"]              = _NAN
    rec["laneProb_mean_pre"]             = _NAN
    rec["laneWidth_mean_pre"]            = _NAN
    rec["laneCenter_range_pre"]          = _NAN
    rec["curvature_mismatch_mean_pre"]   = _NAN
    rec["curvature_mismatch_max_pre"]    = _NAN

    if not model_pre.empty:
        lp_col = "laneLineMeta.leftProb"
        rp_col = "laneLineMeta.rightProb"
        ly_col = "laneLineMeta.leftY"
        ry_col = "laneLineMeta.rightY"

        if lp_col in model_pre.columns and rp_col in model_pre.columns:
            lprob = pd.to_numeric(model_pre[lp_col], errors="coerce")
            rprob = pd.to_numeric(model_pre[rp_col], errors="coerce")
            min_prob = np.minimum(lprob.values, rprob.values)
            valid = np.isfinite(min_prob)
            if valid.any():
                rec["laneProb_min_pre"]  = float(np.nanmin(min_prob[valid]))
                rec["laneProb_mean_pre"] = float(np.nanmean(min_prob[valid]))

        if ly_col in model_pre.columns and ry_col in model_pre.columns:
            ly = pd.to_numeric(model_pre[ly_col], errors="coerce")
            ry = pd.to_numeric(model_pre[ry_col], errors="coerce")
            width = ry.values - ly.values
            center = (ly.values + ry.values) / 2.0
            wv = np.isfinite(width)
            cv = np.isfinite(center)
            if wv.any():
                rec["laneWidth_mean_pre"] = float(np.nanmean(width[wv]))
            if cv.any():
                rec["laneCenter_range_pre"] = float(np.nanmax(center[cv]) - np.nanmin(center[cv]))

    # Curvature mismatch: |desiredCurvature - curvature| from controlsState
    if not ctrl_pre.empty:
        if "desiredCurvature" in ctrl_pre.columns and "curvature" in ctrl_pre.columns:
            dc = pd.to_numeric(ctrl_pre["desiredCurvature"], errors="coerce").values
            c  = pd.to_numeric(ctrl_pre["curvature"], errors="coerce").values
            diff = np.abs(dc - c)
            valid = np.isfinite(diff)
            if valid.any():
                rec["curvature_mismatch_mean_pre"] = float(np.nanmean(diff[valid]))
                rec["curvature_mismatch_max_pre"]  = float(np.nanmax(diff[valid]))

    # ══════════════════════════════════════════════════════════════════
    #  CONTROL SMOOTHNESS (9 metrics) — derivatives + cross-CSV RMSE
    # ══════════════════════════════════════════════════════════════════
    rec["jerk_max_post"]           = _NAN
    rec["steer_rate_max_post"]     = _NAN
    rec["curvature_rate_max_post"] = _NAN

    # -- Jerk (post)
    if not cs_post.empty and "aEgo" in cs_post.columns:
        t = cs_post["time_s"].values
        a = pd.to_numeric(cs_post["aEgo"], errors="coerce").values
        valid = np.isfinite(a) & np.isfinite(t)
        if valid.sum() >= 5:
            actual_hz = _estimate_hz(t[valid])
            a_s = smooth_signal(a[valid], actual_hz)
            jerk = safe_diff_dt(a_s, t[valid])
            jv = jerk[np.isfinite(jerk)]
            if len(jv) > 0:
                rec["jerk_max_post"] = min(float(np.max(np.abs(jv))), 50.0)

    # -- Steer rate (post)
    if not cs_post.empty and "steeringAngleDeg" in cs_post.columns:
        t = cs_post["time_s"].values
        sa = pd.to_numeric(cs_post["steeringAngleDeg"], errors="coerce").values
        valid = np.isfinite(sa) & np.isfinite(t)
        if valid.sum() >= 5:
            actual_hz = _estimate_hz(t[valid])
            sa_s = smooth_signal(sa[valid], actual_hz)
            sr = safe_diff_dt(sa_s, t[valid])
            srv = sr[np.isfinite(sr)]
            if len(srv) > 0:
                rec["steer_rate_max_post"] = min(float(np.max(np.abs(srv))), 500.0)

    # -- Curvature rate (post) from controlsState
    if not ctrl_post.empty and "curvature" in ctrl_post.columns:
        t = ctrl_post["time_s"].values
        c = pd.to_numeric(ctrl_post["curvature"], errors="coerce").values
        valid = np.isfinite(c) & np.isfinite(t)
        if valid.sum() >= 5:
            actual_hz = _estimate_hz(t[valid])
            c_s = smooth_signal(c[valid], actual_hz)
            cr = safe_diff_dt(c_s, t[valid])
            crv = cr[np.isfinite(cr)]
            if len(crv) > 0:
                rec["curvature_rate_max_post"] = float(np.max(np.abs(crv)))

    # -- Cross-CSV RMSE: accel plan→output, output→state, curvature plan→output
    for tag, before, after in [("pre", PRE_BEFORE, PRE_AFTER),
                                ("post", POST_BEFORE, POST_AFTER)]:
        # Accel: plan (carControl) vs output (carOutput)
        pair = align_to_common_grid(
            cc_df, "actuators.accel", co_df, "actuatorsOutput.accel",
            event_t, before, after)
        rec[f"accel_plan_output_rmse_{tag}"] = _rmse(*pair) if pair else _NAN

        # Accel: output (carOutput) vs state (carState aEgo)
        pair = align_to_common_grid(
            co_df, "actuatorsOutput.accel", cs_df, "aEgo",
            event_t, before, after)
        rec[f"accel_output_state_rmse_{tag}"] = _rmse(*pair) if pair else _NAN

        # Curvature: plan (carControl) vs output (carOutput)
        pair = align_to_common_grid(
            cc_df, "actuators.curvature", co_df, "actuatorsOutput.curvature",
            event_t, before, after)
        rec[f"curv_plan_output_rmse_{tag}"] = _rmse(*pair) if pair else _NAN

    # ══════════════════════════════════════════════════════════════════
    #  POST-TAKEOVER STABILITY (7 metrics)
    # ══════════════════════════════════════════════════════════════════
    rec["stabilization_time_5s"]  = _NAN
    rec["stabilization_censored"] = True

    # Stabilization: first τ where continuous 1.0s has |aEgo|<0.5 ∧ |jerk|<1.0 ∧ |steer_rate|<30
    if not cs_post.empty and "aEgo" in cs_post.columns:
        t = cs_post["time_s"].values
        a = pd.to_numeric(cs_post["aEgo"], errors="coerce").values
        valid = np.isfinite(a) & np.isfinite(t)
        if valid.sum() >= 5:
            tv, av = t[valid], a[valid]
            t0 = tv[0]
            actual_hz = _estimate_hz(tv)
            a_s = smooth_signal(av, actual_hz)
            jerk = safe_diff_dt(a_s, tv)
            t_jerk = (tv[:-1] + tv[1:]) / 2.0

            # Steer rate for stabilization
            sr_cond = np.ones(len(t_jerk), dtype=bool)
            if "steeringAngleDeg" in cs_post.columns:
                sa = pd.to_numeric(cs_post["steeringAngleDeg"], errors="coerce").values
                sa_v = np.isfinite(sa) & np.isfinite(t)
                if sa_v.sum() >= 5:
                    sa_s = smooth_signal(sa[sa_v], actual_hz)
                    sr = safe_diff_dt(sa_s, t[sa_v])
                    t_sr = (t[sa_v][:-1] + t[sa_v][1:]) / 2.0
                    sr_interp = np.interp(t_jerk, t_sr, np.abs(sr))
                    sr_cond = sr_interp < 30.0

            stable = (np.abs(a_s[:-1]) < 0.5) & (np.abs(jerk) < 1.0) & sr_cond

            run_start = None
            for i in range(len(stable)):
                if stable[i]:
                    if run_start is None:
                        run_start = i
                    if t_jerk[i] - t_jerk[run_start] >= STAB_SUSTAIN_S:
                        rec["stabilization_time_5s"] = float(t_jerk[run_start] - t0)
                        rec["stabilization_censored"] = False
                        break
                else:
                    run_start = None

    # Driver onset times
    rec["driver_onset_steer_s"] = _NAN
    rec["driver_onset_brake_s"] = _NAN
    rec["driver_onset_gas_s"]   = _NAN
    rec["pressed_duty_post"]    = _NAN

    if not cs_post.empty:
        t = cs_post["time_s"].values
        t0 = t[0] if len(t) > 0 else event_t

        for col, key in [("steeringPressed", "driver_onset_steer_s"),
                         ("brakePressed", "driver_onset_brake_s"),
                         ("gasPressed", "driver_onset_gas_s")]:
            if col in cs_post.columns:
                pressed = parse_bool_col(cs_post[col])
                idxs = pressed[pressed].index
                if len(idxs) > 0:
                    first_idx = idxs[0]
                    loc = cs_post.index.get_loc(first_idx)
                    rec[key] = float(t[loc] - t0)

        # Pressed duty: fraction of post samples with any pressed
        any_pressed = pd.Series(np.zeros(len(cs_post), dtype=bool))
        for col in ["steeringPressed", "brakePressed", "gasPressed"]:
            if col in cs_post.columns:
                any_pressed = any_pressed | parse_bool_col(cs_post[col]).values
        rec["pressed_duty_post"] = float(any_pressed.mean())

    return rec


# ═══════════════════════════════════════════════════════════════════════
#  Style Setup
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


def _save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(OUT_FIGS / f"{name}{ext}")
    plt.close(fig)
    print(f"  Saved {name}")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 1: Pre/Post Distributions (2×3 violins)
# ═══════════════════════════════════════════════════════════════════════
def fig_pre_post_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0))

    specs = [
        # row 0: pre-window
        ("aTarget_min_pre", "aTarget min (m/s²)", C["blue"]),
        ("laneProb_min_pre", "Lane prob. min", C["teal"]),
        ("curvature_mismatch_max_pre", "Curv. mismatch max", C["purple"]),
        # row 1: post-window
        ("jerk_max_post", "Jerk max (m/s³)", C["red"]),
        ("steer_rate_max_post", "Steer rate max (°/s)", C["orange"]),
        ("stabilization_time_5s", "Stabilization (s)", C["green"]),
    ]

    for i, (col, label, color) in enumerate(specs):
        ax = axes[i // 3, i % 3]
        vals = df[col].dropna().values
        if len(vals) < 5:
            ax.set_title(label)
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            continue
        parts = ax.violinplot(vals, positions=[0], showmedians=True,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color(C["orange"])
        ax.set_title(label)
        ax.set_xticks([])
        # Add n annotation
        ax.text(0.95, 0.95, f"n={len(vals):,}", ha="right", va="top",
                transform=ax.transAxes, fontsize=6.5)

    fig.tight_layout()
    _save(fig, "fig_pre_post_distributions")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 2: Perception vs Smoothness (1×2 scatter)
# ═══════════════════════════════════════════════════════════════════════
def fig_perception_vs_smoothness(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

    pairs = [
        ("laneProb_min_pre", "jerk_max_post", "Lane prob. min (pre)", "Jerk max (post, m/s³)"),
        ("laneProb_min_pre", "stabilization_time_5s", "Lane prob. min (pre)", "Stabilization (s)"),
    ]

    for ax, (xc, yc, xl, yl) in zip(axes, pairs):
        sub = df[[xc, yc]].dropna()
        if len(sub) < 5:
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            continue
        ax.scatter(sub[xc], sub[yc], s=4, alpha=0.25, color=C["blue"],
                   edgecolors="none", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.text(0.95, 0.95, f"n={len(sub):,}", ha="right", va="top",
                transform=ax.transAxes, fontsize=6.5)

    fig.tight_layout()
    _save(fig, "fig_perception_vs_smoothness")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 3: Plan→Output→State RMSE (1×3 paired box pre vs post)
# ═══════════════════════════════════════════════════════════════════════
def fig_plan_output_state_rmse(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    metrics = [
        ("accel_plan_output_rmse", "Accel plan→output"),
        ("accel_output_state_rmse", "Accel output→state"),
        ("curv_plan_output_rmse", "Curv. plan→output"),
    ]

    for ax, (base, title) in zip(axes, metrics):
        pre_vals = df[f"{base}_pre"].dropna().values
        post_vals = df[f"{base}_post"].dropna().values
        data = []
        labels = []
        if len(pre_vals) >= 5:
            data.append(pre_vals)
            labels.append("Pre")
        if len(post_vals) >= 5:
            data.append(post_vals)
            labels.append("Post")
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                            widths=0.5, showfliers=False,
                            medianprops=dict(color=C["orange"], linewidth=1.2))
            colors = [C["blue"], C["red"]]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_title(title)
        ax.set_ylabel("RMSE")

    fig.tight_layout()
    _save(fig, "fig_plan_output_state_rmse")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 4: Smoothness by Trigger (1×2 grouped box)
# ═══════════════════════════════════════════════════════════════════════
def fig_smoothness_by_trigger(df: pd.DataFrame):
    if "primary_trigger" not in df.columns:
        print("  [SKIP] fig_smoothness_by_trigger: no primary_trigger column")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    triggers = df["primary_trigger"].dropna().unique()
    triggers = sorted(triggers)
    colors_map = {"Steering Override": C["blue"], "Brake Override": C["teal"],
                  "Gas Override": C["green"]}

    for ax, (col, ylabel) in zip(axes, [
        ("jerk_max_post", "Jerk max (m/s³)"),
        ("steer_rate_max_post", "Steer rate max (°/s)"),
    ]):
        data = []
        labs = []
        box_colors = []
        for trig in triggers:
            vals = df.loc[df["primary_trigger"] == trig, col].dropna().values
            if len(vals) >= 5:
                data.append(vals)
                labs.append(trig.replace(" Override", ""))
                box_colors.append(colors_map.get(trig, C["gray"]))
        if data:
            bp = ax.boxplot(data, labels=labs, patch_artist=True,
                            widths=0.5, showfliers=False,
                            medianprops=dict(color=C["orange"], linewidth=1.2))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    _save(fig, "fig_smoothness_by_trigger")


# ═══════════════════════════════════════════════════════════════════════
#  Figure 5: Interaction Flags (bar chart)
# ═══════════════════════════════════════════════════════════════════════
def fig_interaction_flags(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Flag 1: Lead inconsistency > 0.10
    f1 = df["lead_consistency_flag"].dropna()
    rate1 = (f1 > 0.10).mean() if len(f1) > 0 else 0.0

    # Flag 2: Low lane + high curv output
    lp = df["laneProb_min_pre"].dropna()
    cm = df["curvature_mismatch_max_pre"].dropna()
    if len(cm) > 0:
        cm_p75 = cm.quantile(0.75)
        both = df[["laneProb_min_pre", "curvature_mismatch_max_pre"]].dropna()
        rate2 = ((both["laneProb_min_pre"] < 0.3) &
                 (both["curvature_mismatch_max_pre"] > cm_p75)).mean() if len(both) > 0 else 0.0
    else:
        rate2 = 0.0

    # Flag 3: Plan-output accel mismatch outlier (> P90)
    apo = df["accel_plan_output_rmse_pre"].dropna()
    if len(apo) > 0:
        rate3 = (apo > apo.quantile(0.90)).mean()
    else:
        rate3 = 0.0

    names = ["Lead\ninconsistency", "Low lane +\nhigh curv.", "Plan-output\nmismatch"]
    rates = [rate1, rate2, rate3]
    colors = [C["red"], C["purple"], C["orange"]]

    bars = ax.bar(names, rates, color=colors, width=0.55, alpha=0.8)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{r:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(rates) * 1.3 + 0.02)
    ax.set_title("Interaction Issue Flags")
    fig.tight_layout()
    _save(fig, "fig_interaction_flags")


# ═══════════════════════════════════════════════════════════════════════
#  Report Generation
# ═══════════════════════════════════════════════════════════════════════
def generate_report(df: pd.DataFrame):
    N = len(df)
    lines = [
        "# Takeover Safety & Stability Report",
        "",
        "## 1. Dataset Overview",
        "",
        f"- **Total clips:** {N:,}",
    ]

    # Missingness
    miss_cols = [c for c in df.columns if c.startswith("miss_")]
    if miss_cols:
        lines.append("- **Missingness rates:**")
        for mc in miss_cols:
            rate = df[mc].mean()
            topic = mc.replace("miss_", "")
            lines.append(f"  - {topic}: {rate:.1%}")

    lines += ["", "## 2. Safety Proxies (Pre-Window)", ""]

    safety_cols = ["hasLead_rate_pre", "leadVisible_rate_pre",
                   "aTarget_min_pre", "aTarget_mean_pre",
                   "planned_speed_drop_pre", "laneProb_min_pre",
                   "laneProb_mean_pre", "curvature_mismatch_mean_pre",
                   "curvature_mismatch_max_pre"]
    lines.append("| Metric | N | Median | P5 | P95 |")
    lines.append("|--------|---|--------|----|----|")
    for col in safety_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                lines.append(
                    f"| {col} | {len(s):,} | {s.median():.4f} "
                    f"| {s.quantile(0.05):.4f} | {s.quantile(0.95):.4f} |")

    lines += ["", "## 3. Control Smoothness", ""]

    smooth_cols = ["jerk_max_post", "steer_rate_max_post", "curvature_rate_max_post",
                   "accel_plan_output_rmse_pre", "accel_plan_output_rmse_post",
                   "accel_output_state_rmse_pre", "accel_output_state_rmse_post",
                   "curv_plan_output_rmse_pre", "curv_plan_output_rmse_post"]
    lines.append("| Metric | N | Median | P5 | P95 |")
    lines.append("|--------|---|--------|----|----|")
    for col in smooth_cols:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                lines.append(
                    f"| {col} | {len(s):,} | {s.median():.4f} "
                    f"| {s.quantile(0.05):.4f} | {s.quantile(0.95):.4f} |")

    # Pre→Post changes
    lines += ["", "### RMSE Pre→Post Changes", ""]
    for base in ["accel_plan_output_rmse", "accel_output_state_rmse", "curv_plan_output_rmse"]:
        pre_c = f"{base}_pre"
        post_c = f"{base}_post"
        if pre_c in df.columns and post_c in df.columns:
            pre = df[pre_c].dropna()
            post = df[post_c].dropna()
            if len(pre) > 0 and len(post) > 0:
                lines.append(f"- **{base}**: pre median={pre.median():.4f}, "
                             f"post median={post.median():.4f}, "
                             f"Δ={post.median() - pre.median():+.4f}")

    lines += ["", "## 4. Post-Takeover Stability", ""]
    if "stabilization_time_5s" in df.columns:
        st = df["stabilization_time_5s"].dropna()
        cens = df["stabilization_censored"].sum() if "stabilization_censored" in df.columns else 0
        lines += [
            f"- **N with stabilization data:** {len(st):,}",
            f"- **Median stabilization time:** {st.median():.2f} s",
            f"- **P95 stabilization time:** {st.quantile(0.95):.2f} s",
            f"- **Censored (not stabilized in 5s):** {cens:,} ({cens / max(N, 1):.1%})",
        ]

    onset_cols = ["driver_onset_steer_s", "driver_onset_brake_s", "driver_onset_gas_s"]
    for oc in onset_cols:
        if oc in df.columns:
            s = df[oc].dropna()
            if len(s) > 0:
                lines.append(f"- **{oc}**: n={len(s):,}, median={s.median():.2f} s")

    if "pressed_duty_post" in df.columns:
        pd_col = df["pressed_duty_post"].dropna()
        if len(pd_col) > 0:
            lines.append(f"- **Pressed duty (post):** median={pd_col.median():.2%}")

    lines += ["", "## 5. Interaction Flags", ""]
    # Repeat flag computations
    f1 = df["lead_consistency_flag"].dropna()
    r1 = (f1 > 0.10).mean() if len(f1) > 0 else 0.0
    lines.append(f"- **Lead inconsistency (>10%):** {r1:.1%} of clips")

    cm = df["curvature_mismatch_max_pre"].dropna()
    if len(cm) > 0:
        cm_p75 = cm.quantile(0.75)
        both = df[["laneProb_min_pre", "curvature_mismatch_max_pre"]].dropna()
        r2 = ((both["laneProb_min_pre"] < 0.3) &
              (both["curvature_mismatch_max_pre"] > cm_p75)).mean() if len(both) > 0 else 0.0
    else:
        r2 = 0.0
    lines.append(f"- **Low lane + high curv. mismatch:** {r2:.1%} of clips")

    apo = df["accel_plan_output_rmse_pre"].dropna()
    r3 = (apo > apo.quantile(0.90)).mean() if len(apo) > 0 else 0.0
    lines.append(f"- **Plan-output accel mismatch outlier (>P90):** {r3:.1%} of clips")

    lines += [
        "",
        "## 6. Limitations",
        "",
        "- No radar-derived TTC/THW (radarState excluded from this analysis).",
        "- Safety proxies (hasLead, leadVisible, aTarget) are indirect; they reflect "
        "planner intent rather than physical headway.",
        "- Curvature mismatch uses controlsState desired vs actual curvature, which "
        "may differ in meaning across vehicle platforms.",
        "- Stabilization metric is sensitive to smoothing parameters and sample rate "
        "(qlog 10 Hz vs rlog 100 Hz).",
        "- RMSE metrics depend on temporal alignment via linear interpolation at 20 Hz; "
        "aliasing may affect high-frequency dynamics.",
        "",
    ]

    report_path = OUT_REPORTS / "takeover_safety_stability.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved report: {report_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
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

    # Merge primary_trigger from analysis_master if available
    if MASTER_CSV.exists():
        try:
            master = pd.read_csv(MASTER_CSV,
                                 usecols=["dongle_id", "route_id", "clip_id", "primary_trigger"],
                                 low_memory=False)
            df = df.merge(master, on=["dongle_id", "route_id", "clip_id"], how="left")
            print(f"  Merged primary_trigger from analysis_master ({df['primary_trigger'].notna().sum():,} matched)")
        except Exception as e:
            print(f"  [WARN] Could not merge analysis_master: {e}")

    # Save CSV
    csv_path = OUT_TABLES / "control_safety_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}  ({len(df):,} rows, {len(df.columns)} columns)")

    # ── Figures ────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    fig_pre_post_distributions(df)
    fig_perception_vs_smoothness(df)
    fig_plan_output_state_rmse(df)
    fig_smoothness_by_trigger(df)
    fig_interaction_flags(df)

    # ── Report ────────────────────────────────────────────────────────
    print("\nGenerating report …")
    generate_report(df)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("SUMMARY")
    print(f"{'─' * 60}")
    for col in ["aTarget_min_pre", "laneProb_min_pre", "jerk_max_post",
                "steer_rate_max_post", "stabilization_time_5s",
                "accel_plan_output_rmse_pre", "accel_plan_output_rmse_post"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                print(f"  {col}: n={len(s):,}  median={s.median():.3f}  "
                      f"p5={s.quantile(0.05):.3f}  p95={s.quantile(0.95):.3f}")

    miss_cols = [c for c in df.columns if c.startswith("miss_")]
    if miss_cols:
        print(f"\nMissingness:")
        for mc in miss_cols:
            print(f"  {mc}: {df[mc].mean():.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
