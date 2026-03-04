#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_scenarios.py
==================
Rule-based scenario classification for each takeover clip.
Merges per_clip.csv + derived_signals.csv + engagement_source.csv into
a single analysis_master.csv.

Produces TWO levels of labels (addressing reviewer concerns about
"first-match-wins" losing information):

  1. multi_flags  — boolean columns for each rule, may overlap
  2. primary_context_label  — single-class from priority tree
     + mixed_flag = True when >=2 major rule groups are active
     + uncertain_mixed when evidence is insufficient or conflicting

All thresholds come from configs/analysis_thresholds.yaml.
Lane-probability features are optional (only used if present and valid).

Run:
    python3 label_scenarios.py

Outputs:
    stats_output/scenario_labels.csv   — per-clip scenario + all rule flags
    stats_output/analysis_master.csv   — full merged table for analysis
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  Paths & Config
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
OUT  = CODE / "stats_output"

with open(CODE / "configs" / "analysis_thresholds.yaml") as f:
    CFG = yaml.safe_load(f)

# Thresholds
TTC_CRIT   = CFG["ttc_critical_s"]
THW_CRIT   = CFG["thw_critical_s"]
DRAC_CRIT  = CFG["drac_critical_mps2"]
SR_THRESH  = CFG["steer_rate_threshold_deg_per_s"]
ROUGH_RMS  = CFG["roughness_rms_threshold_mps2"]
CLOSE_DREL = CFG["close_drel_m"]
LANE_LOW   = CFG["lane_prob_low"]
SPD_INC    = CFG["speed_increase_threshold_mps"]

# ──────────────────────────────────────────────────────────────────────────────
#  Load data
# ──────────────────────────────────────────────────────────────────────────────
def load_and_merge() -> pd.DataFrame:
    pc = pd.read_csv(OUT / "per_clip.csv", low_memory=False)

    # Prefer v3 derived signals; fall back to v2
    ds_v3_path = OUT / "derived_signals_v3.csv"
    ds_v2_path = OUT / "derived_signals.csv"
    if ds_v3_path.exists():
        ds = pd.read_csv(ds_v3_path, low_memory=False)
        # Create backward-compatible aliases for v3 column names
        _v3_aliases = {
            "pre_ttc_min_capped_s": "pre_ttc_min_s",
            "post_ttc_min_capped_s": "post_ttc_min_s",
            "pre_drac_max_capped_mps2": "pre_drac_max_mps2",
            "post_drac_max_capped_mps2": "post_drac_max_mps2",
            "stabilization_5s_time_s": "stabilization_time_s",
            "stabilization_5s_censored": "stabilization_censored",
        }
        for v3_name, alias in _v3_aliases.items():
            if v3_name in ds.columns and alias not in ds.columns:
                ds[alias] = ds[v3_name]
    elif ds_v2_path.exists():
        ds = pd.read_csv(ds_v2_path, low_memory=False)
    else:
        raise FileNotFoundError(f"No derived signals found at {ds_v3_path} or {ds_v2_path}")

    es = pd.read_csv(OUT / "engagement_source.csv", low_memory=False)

    keys = ["car_model", "dongle_id", "route_id", "clip_id"]

    for df in [pc, ds, es]:
        df["clip_id"] = pd.to_numeric(df["clip_id"], errors="coerce").astype("Int64")

    # Merge derived signals
    merged = pc.merge(ds, on=keys, how="left", suffixes=("", "_ds"))

    # Merge engagement source (select key columns)
    es_cols = keys + [c for c in ["source", "ctrl_enabled_pct", "cruise_enabled_pct"]
                      if c in es.columns]
    merged = merged.merge(es[es_cols], on=keys, how="left", suffixes=("", "_es"))

    # Add source_group for OEM vs OP analysis
    if "source" in merged.columns:
        def _source_group(s):
            s = str(s)
            if s in ("openpilot_only", "openpilot_primary"):
                return "openpilot"
            elif s in ("oem_only", "oem_primary"):
                return "OEM"
            elif s == "both":
                return "both_active"
            return "other"
        merged["source_group"] = merged["source"].apply(_source_group)

    print(f"  Merged: {len(merged):,} rows, {len(merged.columns)} columns")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def col(df, name):
    """Numeric column with NaN fallback."""
    return pd.to_numeric(df.get(name, pd.Series(np.nan, index=df.index)),
                         errors="coerce")

def bcol(df, name):
    """Boolean column with False fallback."""
    s = df.get(name, pd.Series(False, index=df.index))
    if s.dtype == object:
        return s.astype(str).str.strip().str.lower().isin(["true", "1"])
    return s.fillna(False).astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
#  Rule-based scenario classification
# ──────────────────────────────────────────────────────────────────────────────
def classify_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Apply multi-flag rules + priority-based primary label."""
    n = len(df)

    # ── Pre-window metrics ──────────────────────────────────────────────
    pre_ttc = col(df, "pre_ttc_min_s")
    pre_thw = col(df, "pre_thw_min_s")
    pre_drac = col(df, "pre_drac_max_mps2")
    pre_drel = col(df, "pre_min_drel_m")
    pre_lead_rate = col(df, "pre_lead_present_rate")
    pre_fcw = bcol(df, "pre_fcw_present")
    pre_steer_rate = col(df, "pre_steer_rate_max_deg_per_s")
    pre_lane_left = col(df, "pre_lane_left_prob_mean")
    pre_lane_right = col(df, "pre_lane_right_prob_mean")
    has_lane_probs = bcol(df, "pre_has_lane_probs")
    pre_roughness = col(df, "pre_roughness_rms_mps2")
    pre_alert = bcol(df, "pre_alert_present")
    pre_speed = col(df, "pre_speed_mean_mps")
    pre_max_steer_angle = col(df, "pre_max_abs_steer_angle_deg")

    # Post-window features
    post_maneuver = df.get("post_maneuver_type", pd.Series("", index=df.index))
    post_speed_delta = col(df, "post_speed_delta_mps")

    # Triggers from per_clip
    trig_steer = bcol(df, "trig_steer")
    trig_brake = bcol(df, "trig_brake")
    trig_gas = bcol(df, "trig_gas")
    has_lead = bcol(df, "has_lead")

    # ══════════════════════════════════════════════════════════════════════
    #  LAYER 1: Multi-flag rules (all independent, may overlap)
    # ══════════════════════════════════════════════════════════════════════
    # --- Longitudinal conflict indicators ---
    df["rule_ttc_critical"]   = pre_ttc < TTC_CRIT
    df["rule_thw_critical"]   = pre_thw < THW_CRIT
    df["rule_drac_critical"]  = pre_drac > DRAC_CRIT
    df["rule_fcw"]            = pre_fcw
    df["rule_close_lead"]     = (pre_lead_rate > 0.3) & (pre_drel < CLOSE_DREL)

    # --- Lateral challenge indicators ---
    df["rule_high_steer_rate"]  = pre_steer_rate > SR_THRESH
    df["rule_high_steer_angle"] = pre_max_steer_angle > 30.0
    # Lane probs only used when available (graceful degradation)
    df["rule_low_lane_probs"] = (
        has_lane_probs &
        (pre_lane_left < LANE_LOW) & (pre_lane_right < LANE_LOW)
    )

    # --- Other indicators ---
    df["rule_high_roughness"] = pre_roughness > ROUGH_RMS
    df["rule_alert"]          = pre_alert
    df["rule_gas_trigger"]    = trig_gas
    df["rule_speed_increase"] = post_speed_delta > SPD_INC
    df["rule_lc_maneuver"]    = post_maneuver == "lane_change"
    df["rule_near_stop"]      = pre_speed < 2.0

    # ── Composite group flags (for multi-flag output) ───────────────────
    df["flag_longitudinal_conflict"] = (
        df["rule_ttc_critical"] | df["rule_thw_critical"] |
        df["rule_drac_critical"] | df["rule_fcw"] | df["rule_close_lead"]
    )

    df["flag_lateral_challenge"] = (
        df["rule_high_steer_rate"] | df["rule_high_steer_angle"] |
        df["rule_low_lane_probs"]
    )

    df["flag_roughness"] = df["rule_high_roughness"]
    df["flag_alert"]     = df["rule_alert"]

    # Count how many major groups are active
    major_flags = df[["flag_longitudinal_conflict", "flag_lateral_challenge",
                      "flag_roughness"]].sum(axis=1)
    df["n_major_flags"] = major_flags
    df["mixed_flag"] = major_flags >= 2

    # ── Continuous risk and maneuver scores (for scatter/density) ───────
    # Risk score: normalized composite of conflict evidence [0, 1]
    risk_components = pd.DataFrame(index=df.index)
    risk_components["ttc_risk"] = (TTC_CRIT - pre_ttc).clip(lower=0) / TTC_CRIT
    risk_components["thw_risk"] = (THW_CRIT - pre_thw).clip(lower=0) / THW_CRIT
    risk_components["drac_risk"] = (pre_drac / DRAC_CRIT).clip(upper=1.0)
    risk_components["fcw_risk"] = pre_fcw.astype(float)
    # Average over available components
    df["risk_score"] = risk_components.mean(axis=1, skipna=True).fillna(0.0)

    # Maneuver score: normalized composite of steering/speed activity [0, 1]
    man_sr = (col(df, "post_steer_rate_max_deg_per_s") / 60.0).clip(upper=1.0)
    man_sd = (col(df, "post_speed_delta_mps").abs() / 5.0).clip(upper=1.0)
    df["maneuver_score"] = pd.DataFrame({"sr": man_sr, "sd": man_sd}).mean(
        axis=1, skipna=True).fillna(0.0)

    # ══════════════════════════════════════════════════════════════════════
    #  LAYER 2: Primary context label (single class, priority tree)
    # ══════════════════════════════════════════════════════════════════════
    long_conflict = df["flag_longitudinal_conflict"]
    lat_challenge = df["flag_lateral_challenge"] & ~long_conflict

    low_risk = ~long_conflict & ~lat_challenge

    planned_lc = low_risk & df["rule_lc_maneuver"]
    planned_accel = (
        low_risk & ~planned_lc &
        df["rule_gas_trigger"] & df["rule_speed_increase"]
    )
    intersection_odd = (
        low_risk & ~planned_lc & ~planned_accel &
        df["rule_near_stop"] & ~has_lead
    )
    ride_discomfort = (
        low_risk & ~planned_lc & ~planned_accel & ~intersection_odd &
        df["rule_high_roughness"]
    )
    no_driver_trigger = ~trig_steer & ~trig_brake & ~trig_gas
    system_boundary = (
        low_risk & ~planned_lc & ~planned_accel & ~intersection_odd &
        ~ride_discomfort & df["rule_alert"] & no_driver_trigger
    )
    any_trigger = trig_steer | trig_brake | trig_gas
    discretionary = (
        low_risk & ~planned_lc & ~planned_accel & ~intersection_odd &
        ~ride_discomfort & ~system_boundary &
        any_trigger & ~df["rule_alert"]
    )

    # Assign: lowest priority first, highest overwrites
    scenario = pd.Series("uncertain_mixed", index=df.index, dtype=str)
    scenario[discretionary]     = "discretionary"
    scenario[system_boundary]   = "system_boundary"
    scenario[ride_discomfort]   = "ride_discomfort"
    scenario[intersection_odd]  = "intersection_odd"
    scenario[planned_accel]     = "planned_acceleration"
    scenario[planned_lc]        = "planned_lane_change"
    scenario[lat_challenge]     = "lateral_conflict"
    scenario[long_conflict]     = "longitudinal_conflict"

    df["scenario"] = scenario

    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading and merging datasets …")
    df = load_and_merge()

    print("Classifying scenarios …")
    df = classify_scenarios(df)

    # ── Save scenario labels (rule flags + labels) ──────────────────────
    label_cols = (
        ["car_model", "dongle_id", "route_id", "clip_id",
         "scenario", "post_maneuver_type",
         "mixed_flag", "n_major_flags", "risk_score", "maneuver_score"] +
        [c for c in df.columns if c.startswith("rule_")] +
        [c for c in df.columns if c.startswith("flag_")]
    )
    label_cols = [c for c in label_cols if c in df.columns]
    df[label_cols].to_csv(OUT / "scenario_labels.csv", index=False)
    print(f"  Saved scenario_labels.csv")

    # ── Save full master table ──────────────────────────────────────────
    df.to_csv(OUT / "analysis_master.csv", index=False)
    print(f"  Saved analysis_master.csv  ({len(df):,} rows, {len(df.columns)} cols)")

    # ── Summary ─────────────────────────────────────────────────────────
    N = len(df)
    print(f"\n{'═'*60}")
    print("SCENARIO DISTRIBUTION (primary label)")
    print(f"{'═'*60}")
    for cat in CFG["scenario_categories"]:
        cnt = (df["scenario"] == cat).sum()
        pct = cnt / N * 100
        print(f"  {cat:30s}: {cnt:6,}  ({pct:5.1f}%)")

    print(f"\n{'─'*60}")
    print("MULTI-FLAG OVERLAP")
    print(f"{'─'*60}")
    n_mixed = df["mixed_flag"].sum()
    print(f"  Clips with >=2 major flags (mixed_flag): {n_mixed:,} ({n_mixed/N*100:.1f}%)")
    for fg in ["flag_longitudinal_conflict", "flag_lateral_challenge", "flag_roughness"]:
        if fg in df.columns:
            cnt = df[fg].sum()
            print(f"  {fg:40s}: {cnt:6,}  ({cnt/N*100:.1f}%)")

    print(f"\n{'─'*60}")
    print("POST-MANEUVER DISTRIBUTION")
    print(f"{'─'*60}")
    if "post_maneuver_type" in df.columns:
        for val, cnt in df["post_maneuver_type"].value_counts().items():
            print(f"  {val:30s}: {cnt:6,}  ({cnt/N*100:.1f}%)")

    # ── Sanity checks ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("SANITY CHECKS")
    print(f"{'═'*60}")

    lc_df = df[df["scenario"] == "longitudinal_conflict"]
    disc_df = df[df["scenario"] == "discretionary"]
    rough_df = df[df["scenario"] == "ride_discomfort"]

    # 1. Longitudinal conflict should have lower TTC than discretionary
    if len(lc_df) > 0 and len(disc_df) > 0:
        lc_ttc = lc_df["pre_ttc_min_s"].dropna()
        disc_ttc = disc_df["pre_ttc_min_s"].dropna()
        if len(lc_ttc) > 0 and len(disc_ttc) > 0:
            ok = "PASS" if lc_ttc.median() < disc_ttc.median() else "CHECK"
            print(f"  TTC: longitudinal_conflict median={lc_ttc.median():.2f}s "
                  f"< discretionary median={disc_ttc.median():.2f}s → {ok}")

    # 2. Roughness class should have higher roughness
    if len(rough_df) > 0:
        r_rough = rough_df["pre_roughness_rms_mps2"].dropna()
        r_other = df[df["scenario"] != "ride_discomfort"]["pre_roughness_rms_mps2"].dropna()
        if len(r_rough) > 0 and len(r_other) > 0:
            ok = "PASS" if r_rough.median() > r_other.median() else "CHECK"
            print(f"  Roughness: ride_discomfort median={r_rough.median():.2f} "
                  f"> others median={r_other.median():.2f} → {ok}")

    # 3. Risk score distribution by scenario
    print(f"\n  Risk score by scenario (median):")
    for cat in CFG["scenario_categories"]:
        sub = df[df["scenario"] == cat]["risk_score"]
        if len(sub) > 0:
            print(f"    {cat:30s}: {sub.median():.3f}")


if __name__ == "__main__":
    main()
