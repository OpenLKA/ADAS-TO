#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_statistics.py
=====================
Comprehensive statistics for the ADAS TakeOver dataset.

Run:
    python3 dataset_statistics.py

Outputs (all under Code/stats_output/):
    summary_report.txt   – full text report for paper reference
    per_clip.csv         – per-clip feature table (one row per event)
    model_stats.csv      – per car-model aggregate statistics
    brand_stats.csv      – per brand aggregate statistics
    figs/                – PNG figures (12 plots)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
OUT  = ROOT / "Code" / "stats_output"
FIGS = OUT / "figs"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Trigger detection window around event_mono
# ──────────────────────────────────────────────────────────────────────────────
TRIGGER_BEFORE_S = 0.2   # tight window: only the causal triggering action
TRIGGER_AFTER_S  = 0.5   # signal registration lag on CAN bus
SIMUL_TOL_NS = int(0.05 * 1e9)  # 50 ms tolerance for simultaneous onsets

# ──────────────────────────────────────────────────────────────────────────────
#  Explicit powertrain classification
#  (keyword matching is unreliable for ambiguous names like FORD_MAVERICK_MK1)
# ──────────────────────────────────────────────────────────────────────────────
_BEV: set[str] = {
    "CHEVROLET_BOLT_EUV", "CHEVROLET_BOLT_EUV_2022",
    "FORD_F_150_LIGHTNING_MK1", "FORD_MUSTANG_MACH_E_MK1",
    "GENESIS_GV60_EV_1ST_GEN",
    "HYUNDAI_IONIQ_5", "HYUNDAI_IONIQ_5_2022",
    "HYUNDAI_IONIQ_6_2023", "HYUNDAI_IONIQ_ELECTRIC_2020",
    "HYUNDAI_KONA_ELECTRIC_2019",
    "KIA_EV6", "KIA_EV6_2022",
    "KIA_NIRO_EV", "KIA_NIRO_EV_2020", "KIA_NIRO_EV_2ND_GEN",
    "NISSAN_LEAF",
    "TESLA_AP3_MODEL_3", "TESLA_MODEL_S_RAVEN", "TESLA_MODELY",
}

_HEV: set[str] = {
    "CHEVROLET_VOLT_PREMIER_2017", "CHEVROLET_VOLT_PREMIER_2018",  # PHEV
    "FORD_MAVERICK_MK1",           # standard-fit 2.5L HEV
    "HONDA_ACCORD_HYBRID_2018",
    "HONDA_CLARITY", "HONDA_CLARITY_2018",                         # PHEV
    "HYUNDAI_IONIQ_PHEV_2020",
    "HYUNDAI_SONATA_HEV_2024", "HYUNDAI_SONATA_HYBRID",
    "HYUNDAI_TUCSON_HYBRID_4TH_GEN",
    "KIA_K8_HYBRID_1ST_GEN",
    "KIA_NIRO_HYBRID_2ND_GEN", "KiaNiro2023",
    "LEXUS_RX_HYBRID_2017",
    "TOYOTA_HIGHLANDER_HYBRID_2020",
    "TOYOTA_PRIUS",
    "TOYOTA_RAV4_HYBRID_2023", "TOYOTA_RAV4_PRIME",                # PRIME = PHEV
}

def classify_powertrain(model: str) -> str:
    if model in _BEV:
        return "BEV"
    if model in _HEV:
        return "HEV/PHEV"
    return "ICE"

# ──────────────────────────────────────────────────────────────────────────────
#  Brand extraction
# ──────────────────────────────────────────────────────────────────────────────
_BRAND_MAP: dict[str, str] = {
    "ACURA": "Acura", "AUDI": "Audi", "CHEVROLET": "Chevrolet",
    "FORD": "Ford", "GENESIS": "Genesis", "HONDA": "Honda",
    "HYUNDAI": "Hyundai", "JEEP": "Jeep", "KIA": "Kia",
    "KiaNiro2023": "Kia",
    "LEXUS": "Lexus", "MAZDA": "Mazda", "NISSAN": "Nissan",
    "PORSCHE": "Porsche", "RAM": "Ram", "SKODA": "Skoda",
    "SUBARU": "Subaru", "TESLA": "Tesla", "TOYOTA": "Toyota",
    "VOLKSWAGEN": "Volkswagen", "VOLVO": "Volvo",
}

def extract_brand(model: str) -> str:
    if model in _BRAND_MAP:
        return _BRAND_MAP[model]
    prefix = model.split("_")[0]
    return _BRAND_MAP.get(prefix, prefix.title())

# ──────────────────────────────────────────────────────────────────────────────
#  CSV helpers
# ──────────────────────────────────────────────────────────────────────────────
def safe_read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read CSV, gracefully handling missing files or missing columns."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except (ValueError, Exception):
        # usecols might include columns absent in this file; fall back
        try:
            df = pd.read_csv(path, low_memory=False)
            if usecols:
                existing = [c for c in usecols if c in df.columns]
                return df[existing] if existing else pd.DataFrame()
            return df
        except Exception:
            return pd.DataFrame()

def parse_bool_col(series: pd.Series) -> pd.Series:
    """Convert 'True'/'False' strings (or 0/1) to boolean."""
    return (
        series.astype(str).str.strip().str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )

def closest_row(df: pd.DataFrame, mono: int) -> pd.Series | None:
    """Return the DataFrame row whose logMonoTime is nearest to mono."""
    if df.empty or "logMonoTime" not in df.columns:
        return None
    idx = (df["logMonoTime"].astype(np.int64) - mono).abs().idxmin()
    return df.loc[idx]

def window_around(df: pd.DataFrame, mono: int,
                  before_s: float, after_s: float) -> pd.DataFrame:
    """Slice rows within [mono - before_s, mono + after_s]."""
    if df.empty or "logMonoTime" not in df.columns:
        return pd.DataFrame()
    lo = mono - int(before_s * 1e9)
    hi = mono + int(after_s  * 1e9)
    mask = (df["logMonoTime"].astype(np.int64) >= lo) & \
           (df["logMonoTime"].astype(np.int64) <= hi)
    return df[mask].copy()

# ──────────────────────────────────────────────────────────────────────────────
#  Per-clip feature extraction
# ──────────────────────────────────────────────────────────────────────────────
_NAN = float("nan")

def process_clip(meta: dict, clip_dir: Path) -> dict:
    event_mono = int(meta["event_mono"])

    rec: dict = dict(
        car_model       = meta["car_model"],
        brand           = extract_brand(meta["car_model"]),
        powertrain      = classify_powertrain(meta["car_model"]),
        dongle_id       = meta["dongle_id"],
        route_id        = meta["route_id"],
        clip_id         = int(meta["clip_id"]),
        log_kind        = meta["log_kind"],
        log_hz          = int(meta["log_hz"]),
        vid_kind        = meta["vid_kind"],
        clip_dur_s      = float(meta["clip_dur_s"]),
        n_segs          = len(meta.get("seg_nums_used", [])),
        # carState
        speed_mps       = _NAN,
        speed_kmh       = _NAN,
        accel_mps2      = _NAN,
        steer_angle_deg = _NAN,
        steer_torque    = _NAN,
        cruise_speed_kmh= _NAN,
        # driver inputs (trigger window)
        trig_steer      = False,
        trig_brake      = False,
        trig_gas        = False,
        primary_trigger = "System / Unknown",
        # lead vehicle
        has_lead        = False,
        lead_drel_m     = _NAN,
        lead_vrel_mps   = _NAN,
        # alert
        alert1          = "",
        alert2          = "",
        # CSV completeness
        has_carState    = False,
        has_controls    = False,
        has_radar       = False,
        has_carControl  = False,
        has_carOutput   = False,
        has_modelData   = False,
        has_longPlan    = False,
        has_accel       = False,
    )

    # ── CSV completeness flags ────────────────────────────────────────────────
    for fname, flag in [
        ("carState.csv",          "has_carState"),
        ("controlsState.csv",     "has_controls"),
        ("radarState.csv",        "has_radar"),
        ("carControl.csv",        "has_carControl"),
        ("carOutput.csv",         "has_carOutput"),
        ("drivingModelData.csv",  "has_modelData"),
        ("longitudinalPlan.csv",  "has_longPlan"),
        ("accelerometer.csv",     "has_accel"),
    ]:
        rec[flag] = (clip_dir / fname).exists()

    # ── carState.csv ──────────────────────────────────────────────────────────
    cs_df = safe_read_csv(clip_dir / "carState.csv", usecols=[
        "logMonoTime", "vEgo", "aEgo", "steeringAngleDeg", "steeringTorque",
        "steeringPressed", "brakePressed", "gasPressed", "cruiseState.speed",
    ])
    if not cs_df.empty:
        row = closest_row(cs_df, event_mono)
        if row is not None:
            def _f(k):
                try:
                    v = row[k]
                    return float(v) if not pd.isna(v) else _NAN
                except Exception:
                    return _NAN
            rec["speed_mps"]        = _f("vEgo")
            rec["speed_kmh"]        = rec["speed_mps"] * 3.6 if not np.isnan(rec["speed_mps"]) else _NAN
            rec["accel_mps2"]       = _f("aEgo")
            rec["steer_angle_deg"]  = _f("steeringAngleDeg")
            rec["steer_torque"]     = _f("steeringTorque")
            cs_v = _f("cruiseState.speed")
            rec["cruise_speed_kmh"] = cs_v * 3.6 if (not np.isnan(cs_v) and cs_v > 0) else _NAN

        # trigger window — tight [-0.2, +0.5]s to capture only the causal action
        w = window_around(cs_df, event_mono, TRIGGER_BEFORE_S, TRIGGER_AFTER_S)
        if not w.empty:
            rec["trig_steer"] = bool(parse_bool_col(w["steeringPressed"]).any())
            rec["trig_brake"] = bool(parse_bool_col(w["brakePressed"]).any())
            rec["trig_gas"]   = bool(parse_bool_col(w["gasPressed"]).any())

            # Find first onset timestamp for each channel
            onsets: dict[str, int] = {}
            for col, label in [("steeringPressed", "Steering"),
                                ("brakePressed",    "Brake"),
                                ("gasPressed",      "Gas")]:
                bools = parse_bool_col(w[col])
                active = w.loc[bools, "logMonoTime"]
                if not active.empty:
                    onsets[label] = int(active.iloc[0])

            if onsets:
                min_onset = min(onsets.values())
                # Signals within 50 ms of the earliest are simultaneous
                first_signals = [k for k, v in onsets.items()
                                 if v - min_onset <= SIMUL_TOL_NS]
                if len(first_signals) > 1:
                    rec["primary_trigger"] = "Mixed"
                else:
                    rec["primary_trigger"] = first_signals[0]

    # ── controlsState.csv ─────────────────────────────────────────────────────
    # Alert extraction: take the LAST non-empty alert in the [-10s, 0s] window
    # before event_mono, rather than a single snapshot at event_mono.
    # This captures "alert shown → driver reacts → takeover" causal chains.
    ctrl_df = safe_read_csv(clip_dir / "controlsState.csv", usecols=[
        "logMonoTime", "alertText1", "alertText2",
    ])
    if not ctrl_df.empty:
        w_ctrl = window_around(ctrl_df, event_mono, before_s=10.0, after_s=0.0)
        if not w_ctrl.empty:
            for col, key in [("alertText1", "alert1"), ("alertText2", "alert2")]:
                if col not in w_ctrl.columns:
                    continue
                col_str = w_ctrl[col].astype(str).str.strip()
                valid = col_str[~col_str.isin(["", "nan", "None", "NaN"])]
                rec[key] = valid.iloc[-1] if not valid.empty else ""

    # ── radarState.csv ────────────────────────────────────────────────────────
    radar_df = safe_read_csv(clip_dir / "radarState.csv", usecols=[
        "logMonoTime", "leadOne.status", "leadOne.dRel", "leadOne.vRel",
    ])
    if not radar_df.empty:
        row = closest_row(radar_df, event_mono)
        if row is not None:
            rec["has_lead"] = str(row.get("leadOne.status", "False")).lower() == "true"
            if rec["has_lead"]:
                def _rf(k):
                    try:
                        v = row[k]
                        return float(v) if not pd.isna(v) else _NAN
                    except Exception:
                        return _NAN
                rec["lead_drel_m"]   = _rf("leadOne.dRel")
                rec["lead_vrel_mps"] = _rf("leadOne.vRel")

    return rec

# ──────────────────────────────────────────────────────────────────────────────
#  Statistics helper
# ──────────────────────────────────────────────────────────────────────────────
def stat_line(series: pd.Series, unit: str = "") -> str:
    s = series.dropna()
    if len(s) == 0:
        return "  (no valid data)"
    return (
        f"  n={len(s):,}  mean={s.mean():.2f}{unit}  median={s.median():.2f}{unit}"
        f"  std={s.std():.2f}{unit}"
        f"  [p5={s.quantile(0.05):.2f}, p95={s.quantile(0.95):.2f}]{unit}"
    )

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── 1. Collect all meta.json paths ────────────────────────────────────────
    print("Scanning for meta.json files …")
    meta_paths = sorted(
        p for p in ROOT.rglob("meta.json")
        if "Code" not in p.parts
    )
    print(f"  Found {len(meta_paths):,} clips.")

    # ── 2. Build per-clip DataFrame ───────────────────────────────────────────
    records = []
    for mp in tqdm(meta_paths, desc="Extracting features", ncols=80):
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            records.append(process_clip(meta, mp.parent))
        except Exception as e:
            print(f"  [WARN] {mp}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(OUT / "per_clip_raw.csv", index=False)
    print(f"  Saved per_clip_raw.csv ({len(df):,} rows)")

    # ── 3. Derived columns ────────────────────────────────────────────────────
    df["n_triggers"] = df[["trig_steer", "trig_brake", "trig_gas"]].sum(axis=1)

    # Quality flag: mark noise clips that are NOT real driving takeovers
    _NOISE_ALERTS = [
        "Joystick Mode",
        "WARNING: This branch is not tested",
        "Gear not D",
        "Reverse\nGear",
        "CAN Error: Check Connections",
    ]
    df["is_noise"] = df["alert1"].isin(_NOISE_ALERTS)
    n_noise = int(df["is_noise"].sum())

    # Filtered dataset (used for all subsequent statistics)
    df_all = df.copy()                 # keep full set for reference
    df     = df[~df["is_noise"]].copy()
    df.to_csv(OUT / "per_clip.csv", index=False)
    print(f"  Saved per_clip.csv ({len(df):,} clean rows, {n_noise} noise removed)")
    n = len(df)

    # ── 4. Build summary report ───────────────────────────────────────────────
    lines: list[str] = []
    def p(s: str = "") -> None:
        lines.append(s)

    p("=" * 72)
    p("  ADAS TAKEOVER DATASET — COMPREHENSIVE STATISTICS REPORT")
    p("=" * 72)
    p()

    # ── Section 1: Overview ───────────────────────────────────────────────────
    total_h = df["clip_dur_s"].sum() / 3600
    csv_cols = ["has_carState","has_controls","has_radar",
                "has_carControl","has_carOutput","has_modelData",
                "has_longPlan","has_accel"]
    n_complete = int((df[csv_cols].all(axis=1)).sum())

    p("─" * 72)
    p("1. DATASET OVERVIEW")
    p("─" * 72)
    p(f"  Raw clips (before filtering)  : {len(df_all):,}")
    p(f"  Noise clips removed           : {n_noise:,}  ({', '.join(_NOISE_ALERTS)})")
    p(f"  Clean clips (after filtering) : {n:,}")
    p(f"  Unique car models             : {df['car_model'].nunique()}")
    p(f"  Unique brands                 : {df['brand'].nunique()}")
    p(f"  Unique vehicles (dongle_id)   : {df['dongle_id'].nunique()}")
    p(f"  Unique routes                 : {df['route_id'].nunique()}")
    p(f"  Total video duration          : {total_h:.2f} h  ({total_h*60:.1f} min)")
    p(f"  Clips with all 8 CSVs present : {n_complete:,}  ({n_complete/n*100:.1f}%)")
    p()
    p("  Clip duration statistics:")
    p(stat_line(df["clip_dur_s"], " s"))
    p(f"    Full 20-second clips   : {(df['clip_dur_s'] >= 19.9).sum():,}  ({(df['clip_dur_s'] >= 19.9).mean()*100:.1f}%)")
    p(f"    Short clips (<10 s)    : {(df['clip_dur_s'] < 10.0).sum():,}  ({(df['clip_dur_s'] < 10.0).mean()*100:.1f}%)")
    p()

    # ── Section 2: Vehicle Diversity ──────────────────────────────────────────
    p("─" * 72)
    p("2. VEHICLE DIVERSITY")
    p("─" * 72)

    p("  Powertrain type:")
    pt_cnts = df.groupby("powertrain").size().sort_values(ascending=False)
    for pt, cnt in pt_cnts.items():
        p(f"    {pt:<12}: {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p()

    p("  Clips by brand (all brands):")
    brand_cnts = df.groupby("brand").size().sort_values(ascending=False)
    for brand, cnt in brand_cnts.items():
        p(f"    {brand:<15}: {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p()

    p("  Top 20 car models by clip count:")
    model_cnts = df.groupby("car_model").size().sort_values(ascending=False)
    for i, (m, cnt) in enumerate(model_cnts.head(20).items(), 1):
        p(f"    {i:>2}. {m:<42}: {cnt:>5,}")
    p()

    # ── Section 3: Recording Quality ──────────────────────────────────────────
    p("─" * 72)
    p("3. RECORDING QUALITY")
    p("─" * 72)
    p("  Log type:")
    for k, cnt in df["log_kind"].value_counts().items():
        p(f"    {k:<8}: {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p("  Log sampling rate:")
    for k, cnt in df["log_hz"].value_counts().sort_index().items():
        p(f"    {k:>3} Hz : {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p("  Video type:")
    for k, cnt in df["vid_kind"].value_counts().items():
        p(f"    {k:<10}: {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p()
    p("  Per-CSV completeness:")
    csv_labels = {
        "has_carState":   "carState.csv       ",
        "has_controls":   "controlsState.csv  ",
        "has_radar":      "radarState.csv     ",
        "has_carControl": "carControl.csv     ",
        "has_carOutput":  "carOutput.csv      ",
        "has_modelData":  "drivingModelData.csv",
        "has_longPlan":   "longitudinalPlan.csv",
        "has_accel":      "accelerometer.csv  ",
    }
    for col, label in csv_labels.items():
        cnt = int(df[col].sum())
        p(f"    {label}: {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p()

    # ── Section 4: Driving Conditions at Takeover ─────────────────────────────
    p("─" * 72)
    p("4. DRIVING CONDITIONS AT TAKEOVER MOMENT")
    p("─" * 72)

    spd = df["speed_kmh"].dropna()
    p(f"  Speed (km/h):")
    p(stat_line(df["speed_kmh"], " km/h"))
    p(f"    Standstill (< 2 km/h)       : {(spd <   2).sum():>5,}  ({(spd <   2).mean()*100:.1f}%)")
    p(f"    Urban     (2 – 60 km/h)     : {((spd>=2) & (spd<60)).sum():>5,}  ({((spd>=2)&(spd<60)).mean()*100:.1f}%)")
    p(f"    Rural     (60 – 100 km/h)   : {((spd>=60)&(spd<100)).sum():>5,}  ({((spd>=60)&(spd<100)).mean()*100:.1f}%)")
    p(f"    Highway   (≥ 100 km/h)      : {(spd >= 100).sum():>5,}  ({(spd >= 100).mean()*100:.1f}%)")
    p()

    p(f"  Acceleration (m/s²):")
    p(stat_line(df["accel_mps2"], " m/s²"))
    acc = df["accel_mps2"].dropna()
    p(f"    Braking   (a < −0.5 m/s²)   : {(acc < -0.5).sum():>5,}  ({(acc < -0.5).mean()*100:.1f}%)")
    p(f"    Coasting  (|a| ≤ 0.5 m/s²)  : {(acc.abs() <= 0.5).sum():>5,}  ({(acc.abs() <= 0.5).mean()*100:.1f}%)")
    p(f"    Accel.    (a >  0.5 m/s²)   : {(acc >  0.5).sum():>5,}  ({(acc >  0.5).mean()*100:.1f}%)")
    p()

    p(f"  Steering angle (°, signed):")
    p(stat_line(df["steer_angle_deg"], "°"))
    p(f"  Steering angle (°, absolute):")
    p(stat_line(df["steer_angle_deg"].abs(), "°"))
    p()

    p(f"  Steering torque:")
    p(stat_line(df["steer_torque"]))
    p()

    p(f"  Set cruise speed at takeover (km/h):")
    p(stat_line(df["cruise_speed_kmh"], " km/h"))
    p()

    # ── Section 5: Lead Vehicle ────────────────────────────────────────────────
    p("─" * 72)
    p("5. LEAD VEHICLE STATISTICS")
    p("─" * 72)
    has_lead_n = int(df["has_lead"].sum())
    p(f"  Clips with lead vehicle    : {has_lead_n:>5,}  ({has_lead_n/n*100:.1f}%)")
    p(f"  Clips without lead vehicle : {n-has_lead_n:>5,}  ({(n-has_lead_n)/n*100:.1f}%)")
    lead_df = df[df["has_lead"]]
    p(f"  Lead distance dRel (m):")
    p(stat_line(lead_df["lead_drel_m"], " m"))
    drel = lead_df["lead_drel_m"].dropna()
    p(f"    Close-range (< 20 m)   : {(drel <  20).sum():>5,}  ({(drel <  20).mean()*100:.1f}%)")
    p(f"    Mid-range (20 – 60 m)  : {((drel>=20)&(drel<60)).sum():>5,}  ({((drel>=20)&(drel<60)).mean()*100:.1f}%)")
    p(f"    Far (≥ 60 m)           : {(drel >= 60).sum():>5,}  ({(drel >= 60).mean()*100:.1f}%)")
    p(f"  Lead relative speed (m/s):")
    p(stat_line(lead_df["lead_vrel_mps"], " m/s"))
    p()

    # ── Section 6: Takeover Trigger Analysis ──────────────────────────────────
    p("─" * 72)
    p("6. TAKEOVER TRIGGER ANALYSIS  (window: [−0.2 s, +0.5 s] around event)")
    p("─" * 72)
    trig_s   = int(df["trig_steer"].sum())
    trig_b   = int(df["trig_brake"].sum())
    trig_g   = int(df["trig_gas"].sum())
    trig_any = int((df["trig_steer"] | df["trig_brake"] | df["trig_gas"]).sum())
    trig_non = int((~df["trig_steer"] & ~df["trig_brake"] & ~df["trig_gas"]).sum())

    p(f"  Steering override   : {trig_s:>5,}  ({trig_s/n*100:.1f}%)")
    p(f"  Brake override      : {trig_b:>5,}  ({trig_b/n*100:.1f}%)")
    p(f"  Gas override        : {trig_g:>5,}  ({trig_g/n*100:.1f}%)")
    p(f"  Any driver input    : {trig_any:>5,}  ({trig_any/n*100:.1f}%)")
    p(f"  No driver input     : {trig_non:>5,}  ({trig_non/n*100:.1f}%)  (system-/alert-initiated)")
    p()
    p("  Number of simultaneous trigger inputs:")
    for k, cnt in df["n_triggers"].value_counts().sort_index().items():
        p(f"    {int(k)} input(s) : {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p()
    p("  Primary trigger (highest-priority single label):")
    for k, cnt in df["primary_trigger"].value_counts().items():
        p(f"    {k:<22}: {cnt:>5,}  ({cnt/n*100:.1f}%)")
    p()

    # ── Section 7: Alert Texts ─────────────────────────────────────────────────
    p("─" * 72)
    p("7. ALERT TEXTS AT TAKEOVER MOMENT (alertText1, top 25)")
    p("─" * 72)
    alert_cnts = df["alert1"].fillna("").value_counts()
    p(f"  Unique alert texts    : {len(alert_cnts)}")
    empty_cnt = int(alert_cnts.get("", 0))
    p(f"  Empty string ('')     : {empty_cnt:>5,}  ({empty_cnt/n*100:.1f}%)  ← typically driver-initiated")
    p()
    p("  Non-empty alert texts (top 25):")
    non_empty = alert_cnts[alert_cnts.index != ""]
    for txt, cnt in non_empty.head(25).items():
        label = (txt[:60] + "…") if len(txt) > 60 else txt
        p(f"    {cnt:>5,}  {label!r}")
    p()

    # ── Section 8: Route & Segment Statistics ─────────────────────────────────
    p("─" * 72)
    p("8. ROUTE AND SEGMENT STATISTICS")
    p("─" * 72)
    cpr = df.groupby("route_id").size()
    p(f"  Unique routes with ≥1 clip  : {cpr.count():,}")
    p(f"  Clips per route:")
    p(stat_line(cpr.astype(float)))
    p(f"    Routes with exactly 1 clip : {(cpr==1).sum():>5,}  ({(cpr==1).mean()*100:.1f}%)")
    p(f"    Routes with 2–5 clips      : {((cpr>=2)&(cpr<=5)).sum():>5,}  ({((cpr>=2)&(cpr<=5)).mean()*100:.1f}%)")
    p(f"    Routes with > 5 clips      : {(cpr >5).sum():>5,}  ({(cpr >5).mean()*100:.1f}%)")
    p(f"    Max clips in one route     : {cpr.max()}")
    p()
    p(f"  Segments used per clip:")
    p(stat_line(df["n_segs"].astype(float)))
    p()

    # ── Section 9: Per-model aggregate table ──────────────────────────────────
    p("─" * 72)
    p("9. PER-MODEL AGGREGATE STATISTICS (sorted by clip count)")
    p("─" * 72)
    model_agg = df.groupby("car_model").agg(
        powertrain   = ("powertrain",    "first"),
        n_clips      = ("clip_id",       "count"),
        n_routes     = ("route_id",      "nunique"),
        n_vehicles   = ("dongle_id",     "nunique"),
        spd_mean     = ("speed_kmh",     "mean"),
        spd_std      = ("speed_kmh",     "std"),
        acc_mean     = ("accel_mps2",    "mean"),
        pct_lead     = ("has_lead",      "mean"),
        pct_steer    = ("trig_steer",    "mean"),
        pct_brake    = ("trig_brake",    "mean"),
        pct_gas      = ("trig_gas",      "mean"),
    ).sort_values("n_clips", ascending=False)

    # format for display
    ma = model_agg.copy()
    for col in ["pct_lead", "pct_steer", "pct_brake", "pct_gas"]:
        ma[col] = (ma[col] * 100).round(1)
    ma["spd_mean"] = ma["spd_mean"].round(1)
    ma["spd_std"]  = ma["spd_std"].round(1)
    ma["acc_mean"] = ma["acc_mean"].round(3)

    hdr = (f"  {'Model':<42} {'PT':<9} {'N':>5} {'Rts':>4} "
           f"{'Veh':>4} {'Spd±σ (km/h)':>14} {'Acc (m/s²)':>10} "
           f"{'Lead%':>6} {'Steer%':>7} {'Brake%':>7}")
    p(hdr)
    p("  " + "─" * (len(hdr) - 2))
    for model, row in ma.iterrows():
        spd_str = f"{row.spd_mean:.1f}±{row.spd_std:.1f}"
        p(f"  {model:<42} {row.powertrain:<9} {int(row.n_clips):>5} "
          f"{int(row.n_routes):>4} {int(row.n_vehicles):>4} "
          f"{spd_str:>14} {row.acc_mean:>10.3f} "
          f"{row.pct_lead:>6.1f} {row.pct_steer:>7.1f} {row.pct_brake:>7.1f}")
    p()

    # save tables
    model_agg.round(3).to_csv(OUT / "model_stats.csv")
    brand_agg = df.groupby("brand").agg(
        n_clips      = ("clip_id",    "count"),
        n_models     = ("car_model",  "nunique"),
        n_vehicles   = ("dongle_id",  "nunique"),
        spd_mean     = ("speed_kmh",  "mean"),
        spd_std      = ("speed_kmh",  "std"),
        pct_lead     = ("has_lead",   "mean"),
        pct_steer    = ("trig_steer", "mean"),
        pct_brake    = ("trig_brake", "mean"),
    ).sort_values("n_clips", ascending=False).round(3)
    brand_agg.to_csv(OUT / "brand_stats.csv")
    p(f"  (detailed tables saved to model_stats.csv and brand_stats.csv)")
    p()

    report_text = "\n".join(lines)
    (OUT / "summary_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)

    # ── 5. Figures ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("seaborn-whitegrid")

        DPI = 150
        W, H = 10, 6

        def save(name: str) -> None:
            plt.tight_layout()
            plt.savefig(FIGS / name, dpi=DPI, bbox_inches="tight")
            plt.close()

        # ── Fig 01: Top-20 models (horizontal bar) ────────────────────────────
        fig, ax = plt.subplots(figsize=(W, 9), dpi=DPI)
        top20 = model_cnts.head(20).sort_values()
        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(top20)))
        bars = ax.barh(top20.index, top20.values, color=colors)
        ax.set_xlabel("Number of Takeover Clips", fontsize=12)
        ax.set_title("Top 20 Vehicle Models by Takeover Clip Count", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for bar, v in zip(bars, top20.values):
            ax.text(v + 4, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=8)
        save("01_top20_models.png")

        # ── Fig 02: Brand distribution (bar) ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(W, 6), dpi=DPI)
        bc = brand_cnts
        ax.bar(bc.index, bc.values, color=plt.cm.tab20.colors[:len(bc)], edgecolor="white")
        ax.set_xlabel("Brand", fontsize=12)
        ax.set_ylabel("Number of Clips", fontsize=12)
        ax.set_title("Takeover Clips by Vehicle Brand", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        save("02_brand_distribution.png")

        # ── Fig 03: Powertrain pie ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 7), dpi=DPI)
        pt_order = ["BEV", "HEV/PHEV", "ICE"]
        pt_vals  = [pt_cnts.get(p, 0) for p in pt_order]
        pt_colors = ["#4CAF50", "#2196F3", "#FF9800"]
        wedges, texts, autotexts = ax.pie(
            pt_vals, labels=pt_order, autopct="%1.1f%%",
            colors=pt_colors, startangle=140, pctdistance=0.78,
            textprops={"fontsize": 12},
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
        ax.set_title("Powertrain Distribution", fontsize=13, fontweight="bold")
        save("03_powertrain_pie.png")

        # ── Fig 04: Speed histogram ───────────────────────────────────────────
        spd_v = df["speed_kmh"].dropna()
        spd_v = spd_v[spd_v >= 0]
        fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
        ax.hist(spd_v, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(spd_v.mean(),   color="red",    linestyle="--", linewidth=1.8,
                   label=f"Mean   = {spd_v.mean():.1f} km/h")
        ax.axvline(spd_v.median(), color="orange", linestyle="--", linewidth=1.8,
                   label=f"Median = {spd_v.median():.1f} km/h")
        ax.set_xlabel("Speed at Takeover Moment (km/h)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Speed Distribution at Takeover Moment", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        save("04_speed_histogram.png")

        # ── Fig 05: Speed CDF ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
        sorted_spd = np.sort(spd_v.values)
        pvals = np.arange(1, len(sorted_spd) + 1) / len(sorted_spd) * 100
        ax.plot(sorted_spd, pvals, color="steelblue", linewidth=2)
        for pct, col in [(50, "gray"), (90, "orange"), (95, "red")]:
            pct_val = float(np.percentile(sorted_spd, pct))
            ax.axhline(pct, color=col, linestyle="--", linewidth=1,
                       label=f"P{pct} = {pct_val:.1f} km/h")
        ax.set_xlabel("Speed (km/h)", fontsize=12)
        ax.set_ylabel("Cumulative Percentage (%)", fontsize=12)
        ax.set_title("CDF of Speed at Takeover Moment", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)
        save("05_speed_cdf.png")

        # ── Fig 06: Acceleration histogram ───────────────────────────────────
        acc_v = df["accel_mps2"].dropna()
        fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
        ax.hist(acc_v, bins=80, color="darkorange", edgecolor="white", linewidth=0.3)
        ax.axvline(0,           color="black", linewidth=1.2, linestyle="-")
        ax.axvline(acc_v.mean(), color="red",  linewidth=1.8, linestyle="--",
                   label=f"Mean = {acc_v.mean():.3f} m/s²")
        ax.set_xlabel("Acceleration at Takeover Moment (m/s²)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Acceleration Distribution at Takeover Moment", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        save("06_acceleration_histogram.png")

        # ── Fig 07: Steering angle histogram ─────────────────────────────────
        steer_v = df["steer_angle_deg"].dropna()
        fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
        ax.hist(steer_v, bins=100, color="seagreen", edgecolor="white", linewidth=0.2)
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Steering Angle at Takeover Moment (°)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Steering Angle Distribution at Takeover Moment", fontsize=13, fontweight="bold")
        save("07_steer_angle_histogram.png")

        # ── Fig 08: Trigger distribution (bar) ───────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 6), dpi=DPI)
        pt_counts = df["primary_trigger"].value_counts()
        _trig_order = ["Steering", "Brake", "Gas", "Mixed", "System / Unknown"]
        trigger_vals = {k: pt_counts.get(k, 0) for k in _trig_order}
        # Drop categories with 0 count
        trigger_vals = {k: v for k, v in trigger_vals.items() if v > 0}
        _tcolors_map = {"Steering": "#2196F3", "Brake": "#f44336",
                        "Gas": "#4CAF50", "Mixed": "#B279A2",
                        "System / Unknown": "#9E9E9E"}
        tcolors = [_tcolors_map[k] for k in trigger_vals]
        bars = ax.bar(trigger_vals.keys(), trigger_vals.values(), color=tcolors, width=0.55)
        ax.set_ylabel("Number of Clips", fontsize=12)
        ax.set_title("Takeover Trigger Distribution", fontsize=13, fontweight="bold")
        for bar, v in zip(bars, trigger_vals.values()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30,
                    f"{v:,}\n({v/n*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        save("08_trigger_distribution.png")

        # ── Fig 09: Lead vehicle distance histogram ───────────────────────────
        drel_v = lead_df["lead_drel_m"].dropna()
        drel_v = drel_v[(drel_v > 0) & (drel_v < 200)]
        fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
        ax.hist(drel_v, bins=60, color="mediumpurple", edgecolor="white", linewidth=0.3)
        ax.axvline(drel_v.mean(),   color="red",    linestyle="--", linewidth=1.8,
                   label=f"Mean   = {drel_v.mean():.1f} m")
        ax.axvline(drel_v.median(), color="orange", linestyle="--", linewidth=1.8,
                   label=f"Median = {drel_v.median():.1f} m")
        ax.set_xlabel("Distance to Lead Vehicle (m)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Lead Vehicle Distance Distribution at Takeover", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        save("09_lead_distance_histogram.png")

        # ── Fig 10: Alert text top 15 ─────────────────────────────────────────
        ne_alerts = alert_cnts[alert_cnts.index != ""].head(15)
        if not ne_alerts.empty:
            fig, ax = plt.subplots(figsize=(11, 7), dpi=DPI)
            labels = [(t[:55] + "…") if len(t) > 55 else t for t in ne_alerts.index]
            ax.barh(labels[::-1], ne_alerts.values[::-1], color="coral", edgecolor="white")
            ax.set_xlabel("Number of Clips", fontsize=12)
            ax.set_title("Top 15 Alert Texts at Takeover Moment (alertText1)", fontsize=13, fontweight="bold")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            save("10_alert_texts_top15.png")

        # ── Fig 11: Speed boxplot by powertrain ───────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
        pt_ord   = [p for p in ["BEV", "HEV/PHEV", "ICE"] if p in df["powertrain"].values]
        pt_data  = [df[df["powertrain"] == p]["speed_kmh"].dropna().values for p in pt_ord]
        pt_cols  = {"BEV": "#4CAF50", "HEV/PHEV": "#2196F3", "ICE": "#FF9800"}
        bp = ax.boxplot(pt_data, labels=pt_ord, patch_artist=True,
                        medianprops=dict(color="red", linewidth=2.0),
                        flierprops=dict(marker="o", markersize=2, alpha=0.3))
        for patch, pt in zip(bp["boxes"], pt_ord):
            patch.set_facecolor(pt_cols[pt])
            patch.set_alpha(0.7)
        ax.set_xlabel("Powertrain Type", fontsize=12)
        ax.set_ylabel("Speed at Takeover (km/h)", fontsize=12)
        ax.set_title("Speed at Takeover by Powertrain Type", fontsize=13, fontweight="bold")
        save("11_speed_by_powertrain_boxplot.png")

        # ── Fig 12: Clips-per-route distribution ──────────────────────────────
        fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
        cpr_vc = cpr.value_counts().sort_index()
        # group "≥6" together
        cpr_plot = cpr_vc[cpr_vc.index <= 5].copy()
        rest = int(cpr_vc[cpr_vc.index > 5].sum())
        if rest > 0:
            cpr_plot["≥6"] = rest
        ax.bar([str(k) for k in cpr_plot.index], cpr_plot.values,
               color="teal", edgecolor="white")
        ax.set_xlabel("Takeover Clips per Route", fontsize=12)
        ax.set_ylabel("Number of Routes", fontsize=12)
        ax.set_title("Distribution of Takeover Events per Route", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for i, v in enumerate(cpr_plot.values):
            ax.text(i, v + 5, str(v), ha="center", fontsize=9)
        save("12_clips_per_route.png")

        print(f"\nFigures saved to: {FIGS}/")
        for i in range(1, 13):
            print(f"  {FIGS.name}/{i:02d}_*.png")

    except ImportError as e:
        print(f"[WARN] matplotlib not available ({e}); skipping figures.")

    print(f"\nAll outputs saved under: {OUT}")
    print(f"  summary_report.txt")
    print(f"  per_clip.csv")
    print(f"  model_stats.csv")
    print(f"  brand_stats.csv")
    print(f"  figs/01_*.png … figs/12_*.png")


if __name__ == "__main__":
    main()
