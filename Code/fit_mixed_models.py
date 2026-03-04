#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_mixed_models.py
===================
Statistical modeling for Section IV: Take Over Analysis.

Three analysis tracks:
  (A) Binary outcome models (threshold-event approach):
      - I(TTC < 1.5) via GEE logistic with exchangeable correlation by driver
      - I(THW < 0.8)  via GEE logistic
      - I(DRAC > 3.0) via GEE logistic
      This avoids missingness/right-censoring problems that plague continuous
      TTC/THW regression.

  (B) Continuous outcome models (mixed-effects linear):
      - peak decel, peak jerk, stabilization time via LMM with driver
        random intercept.
      Reports ICC to quantify driver heterogeneity.

  (C) OEM vs OP comparison with propensity score weighting:
      - Logit propensity model: P(OP | speed, lead, dRel, log_hz, steer_angle)
      - IPW (inverse probability weighting) for ATE estimation
      - SMD balance diagnostics pre- and post-weighting
      - Overlap restriction (0.1 < propensity < 0.9)

Run:
    python3 fit_mixed_models.py

Outputs:
    stats_output/model_results.json    — all model coefficients, CIs, diagnostics
    stats_output/modeling_notes.txt    — plain-text summary of methods and caveats
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
OUT  = CODE / "stats_output"

with open(CODE / "configs" / "analysis_thresholds.yaml") as f:
    CFG = yaml.safe_load(f)

MIN_CLIPS = CFG["min_clips_per_driver"]
SEED      = CFG["random_seed"]
np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(OUT / "analysis_master.csv", low_memory=False)

    # Filter to drivers with sufficient clips for mixed models
    driver_counts = df["dongle_id"].value_counts()
    valid_drivers = driver_counts[driver_counts >= MIN_CLIPS].index
    df_model = df[df["dongle_id"].isin(valid_drivers)].copy()
    print(f"  Full dataset: {len(df):,} clips, {df['dongle_id'].nunique()} drivers")
    print(f"  Model subset (>={MIN_CLIPS} clips): {len(df_model):,} clips, "
          f"{df_model['dongle_id'].nunique()} drivers")

    # Prepare categorical variables
    if "scenario" in df_model.columns:
        df_model["scenario"] = pd.Categorical(
            df_model["scenario"], categories=CFG["scenario_categories"])

    # Speed covariate
    if "pre_speed_mean_mps" in df_model.columns:
        df_model["speed"] = pd.to_numeric(df_model["pre_speed_mean_mps"], errors="coerce")
    elif "speed_mps" in df_model.columns:
        df_model["speed"] = pd.to_numeric(df_model["speed_mps"], errors="coerce")

    # Trigger grouping
    if "primary_trigger" in df_model.columns:
        def _trig(t):
            t = str(t)
            if "Steer" in t: return "steering"
            if "Brake" in t: return "brake"
            if "Gas" in t:   return "gas"
            return "system"
        df_model["trigger"] = pd.Categorical(
            df_model["primary_trigger"].apply(_trig),
            categories=["steering", "brake", "gas", "system"])

    # Binary outcomes (threshold-event indicators)
    # Support both v3 and v2 column names
    ttc_col = next((c for c in ("pre_ttc_min_s", "pre_ttc_min_capped_s") if c in df_model.columns), None)
    thw_col = "pre_thw_min_s" if "pre_thw_min_s" in df_model.columns else None
    drac_col = next((c for c in ("pre_drac_max_mps2", "pre_drac_max_capped_mps2") if c in df_model.columns), None)

    df_model["ttc_critical"] = (
        pd.to_numeric(df_model[ttc_col], errors="coerce") < CFG["ttc_critical_s"]
    ).astype(float) if ttc_col else 0.0
    df_model["thw_critical"] = (
        pd.to_numeric(df_model[thw_col], errors="coerce") < CFG["thw_critical_s"]
    ).astype(float) if thw_col else 0.0
    df_model["drac_critical"] = (
        pd.to_numeric(df_model[drac_col], errors="coerce") > CFG["drac_critical_mps2"]
    ).astype(float) if drac_col else 0.0

    return df, df_model


# ──────────────────────────────────────────────────────────────────────────────
#  (A) GEE logistic for binary outcomes
# ──────────────────────────────────────────────────────────────────────────────
def fit_gee_binary(df: pd.DataFrame, dep_var: str, label: str,
                   subset_col: str | None = None) -> dict:
    """Fit GEE with logit link, exchangeable correlation, clustered by driver.

    Falls back to cluster-robust logistic if GEE fails to converge.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    sub = df.dropna(subset=[dep_var, "speed"]).copy()
    # Optionally restrict to subset (e.g., lead-present only)
    if subset_col:
        mask = sub[subset_col].astype(str).str.lower().isin(["true", "1", "1.0"])
        sub = sub[mask].copy()

    if len(sub) < 30 or sub["dongle_id"].nunique() < 3:
        return {"error": f"insufficient data (n={len(sub)})"}

    # Build design matrix
    y = sub[dep_var].values
    # Only model if outcome has variation
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return {"error": f"insufficient outcome variation (events={int(y.sum())}/{len(y)})"}

    # Covariates: scenario dummies + speed + lead_present_rate
    X_parts = [pd.get_dummies(sub["scenario"], drop_first=True, prefix="scen")]
    X_parts.append(sub[["speed"]].rename(columns={"speed": "speed_mps"}))
    if "pre_lead_present_rate" in sub.columns:
        X_parts.append(sub[["pre_lead_present_rate"]].fillna(0))
    if "pre_alert_present" in sub.columns:
        alert = sub["pre_alert_present"].astype(str).str.lower().isin(["true", "1"]).astype(float)
        X_parts.append(alert.to_frame("alert_present"))

    X = pd.concat(X_parts, axis=1).astype(float)
    X = sm.add_constant(X)

    # Drop columns with zero variance
    X = X.loc[:, X.std() > 0]

    groups = sub["dongle_id"].values

    try:
        model = GEE(y, X, groups=groups,
                     family=Binomial(),
                     cov_struct=Exchangeable())
        result = model.fit(maxiter=100)

        coefs = {}
        params = result.params
        conf = result.conf_int()
        pvals = result.pvalues

        for name in params.index:
            # Odds ratios for logistic
            est = float(params[name])
            coefs[name] = {
                "estimate": est,
                "odds_ratio": float(np.exp(est)),
                "ci_lower": float(conf.loc[name, 0]),
                "ci_upper": float(conf.loc[name, 1]),
                "or_ci_lower": float(np.exp(conf.loc[name, 0])),
                "or_ci_upper": float(np.exp(conf.loc[name, 1])),
                "p_value": float(pvals[name]),
            }

        return {
            "method": "GEE_logistic_exchangeable",
            "label": label,
            "formula": f"{dep_var} ~ scenario + speed + covariates",
            "n_obs": len(sub),
            "n_events": int(y.sum()),
            "event_rate": float(y.mean()),
            "n_clusters": int(sub["dongle_id"].nunique()),
            "coefficients": coefs,
        }
    except Exception as e:
        # Fallback: cluster-robust logistic
        try:
            from statsmodels.genmod.generalized_linear_model import GLM
            model = GLM(y, X, family=Binomial())
            result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

            coefs = {}
            for name in result.params.index:
                est = float(result.params[name])
                conf = result.conf_int()
                coefs[name] = {
                    "estimate": est,
                    "odds_ratio": float(np.exp(est)),
                    "ci_lower": float(conf.loc[name, 0]),
                    "ci_upper": float(conf.loc[name, 1]),
                    "or_ci_lower": float(np.exp(conf.loc[name, 0])),
                    "or_ci_upper": float(np.exp(conf.loc[name, 1])),
                    "p_value": float(result.pvalues[name]),
                }
            return {
                "method": "GLM_logistic_cluster_robust",
                "label": label,
                "n_obs": len(sub),
                "n_events": int(y.sum()),
                "n_clusters": int(sub["dongle_id"].nunique()),
                "coefficients": coefs,
                "note": f"GEE failed ({e}), fell back to cluster-robust GLM",
            }
        except Exception as e2:
            return {"error": f"GEE: {e}; GLM fallback: {e2}"}


# ──────────────────────────────────────────────────────────────────────────────
#  (B) Mixed-effects linear models for continuous outcomes
# ──────────────────────────────────────────────────────────────────────────────
def fit_lmm(df: pd.DataFrame, formula: str, dep_var: str, label: str) -> dict:
    """Fit linear mixed-effects model with driver random intercept."""
    import statsmodels.formula.api as smf

    # Drop NaN in dep_var AND all formula covariates to keep alignment
    drop_cols = [dep_var]
    if "speed" in df.columns:
        drop_cols.append("speed")
    if "scenario" in df.columns:
        drop_cols.append("scenario")
    sub = df.dropna(subset=[c for c in drop_cols if c in df.columns]).copy().reset_index(drop=True)
    if len(sub) < 30 or sub["dongle_id"].nunique() < 3:
        return {"error": f"insufficient data (n={len(sub)})"}

    try:
        model = smf.mixedlm(formula, sub, groups=sub["dongle_id"])
        result = model.fit(reml=True, maxiter=200)

        coefs = {}
        conf = result.conf_int()
        for name in result.params.index:
            if name == "Group Var":
                continue
            coefs[name] = {
                "estimate": float(result.params[name]),
                "ci_lower": float(conf.loc[name, 0]) if name in conf.index else _NAN,
                "ci_upper": float(conf.loc[name, 1]) if name in conf.index else _NAN,
                "p_value": float(result.pvalues[name]) if name in result.pvalues.index else _NAN,
            }

        re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, "iloc") else float(result.cov_re)
        resid_var = float(result.scale)
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0.0

        return {
            "method": "LMM_REML",
            "label": label,
            "formula": formula,
            "n_obs": len(sub),
            "n_groups": int(sub["dongle_id"].nunique()),
            "icc": icc,
            "random_intercept_var": re_var,
            "residual_var": resid_var,
            "converged": result.converged,
            "coefficients": coefs,
        }
    except Exception as e:
        # Fallback: OLS with cluster-robust SE (statsmodels MixedLM has
        # known index bugs in some versions)
        try:
            ols_result = smf.ols(formula, sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["dongle_id"]})
            coefs = {}
            conf = ols_result.conf_int()
            for name in ols_result.params.index:
                coefs[name] = {
                    "estimate": float(ols_result.params[name]),
                    "ci_lower": float(conf.loc[name, 0]),
                    "ci_upper": float(conf.loc[name, 1]),
                    "p_value": float(ols_result.pvalues[name]),
                }
            return {
                "method": "OLS_cluster_robust",
                "label": label,
                "formula": formula,
                "n_obs": len(sub),
                "n_groups": int(sub["dongle_id"].nunique()),
                "note": f"LMM failed ({e}); fell back to OLS with cluster-robust SE",
                "coefficients": coefs,
            }
        except Exception as e2:
            return {"error": f"LMM: {e}; OLS fallback: {e2}"}


_NAN = float("nan")


# ──────────────────────────────────────────────────────────────────────────────
#  (C) Propensity score weighting for OEM vs OP
# ──────────────────────────────────────────────────────────────────────────────
def fit_propensity_model(df_full: pd.DataFrame) -> dict:
    """IPW comparison of OEM vs openpilot engagement sources.

    Steps:
      1. Fit logistic propensity: P(OP | covariates)
      2. Restrict to common support (0.1 < p < 0.9)
      3. Compute IPW weights
      4. Report SMD before and after weighting
      5. Estimate weighted treatment effects for key outcomes
    """
    import statsmodels.api as sm

    # Filter to OEM vs OP only
    if "source_group" not in df_full.columns:
        return {"error": "no source_group column"}

    df = df_full[df_full["source_group"].isin(["openpilot", "OEM"])].copy()
    if len(df) < 50:
        return {"error": f"insufficient OEM/OP data (n={len(df)})"}

    df["treatment"] = (df["source_group"] == "openpilot").astype(float)

    # Covariates for propensity model
    cov_names = CFG["propensity"]["covariates"]
    available_covs = [c for c in cov_names if c in df.columns]

    if len(available_covs) < 2:
        return {"error": f"insufficient covariates: {available_covs}"}

    # Prepare covariates (fill NaN with median)
    X = df[available_covs].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    X = sm.add_constant(X)

    # Drop zero-variance columns
    X = X.loc[:, X.std() > 0]

    y = df["treatment"].values

    # Fit propensity model
    try:
        ps_model = sm.GLM(y, X, family=sm.families.Binomial())
        ps_result = ps_model.fit()
        p_scores = ps_result.predict(X)
    except Exception as e:
        return {"error": f"propensity model failed: {e}"}

    df["propensity"] = p_scores.values

    # Overlap restriction
    caliper_lo = 0.1
    caliper_hi = 0.9
    overlap_mask = (df["propensity"] > caliper_lo) & (df["propensity"] < caliper_hi)
    n_excluded = (~overlap_mask).sum()
    df_overlap = df[overlap_mask].copy()

    if len(df_overlap) < 30:
        return {"error": f"too few in overlap region (n={len(df_overlap)})"}

    # IPW weights: treated get 1/p, controls get 1/(1-p)
    p = df_overlap["propensity"].values
    t = df_overlap["treatment"].values
    weights = np.where(t == 1, 1.0 / np.maximum(p, 0.01),
                       1.0 / np.maximum(1 - p, 0.01))
    # Normalize weights
    weights = weights / weights.mean()
    df_overlap["ipw"] = weights

    # ── SMD balance diagnostics ─────────────────────────────────────────
    def compute_smd(data, cov, treat_col, weight_col=None):
        """Standardized mean difference (NaN-safe)."""
        sub = data.dropna(subset=[cov])
        t1 = sub[sub[treat_col] == 1]
        t0 = sub[sub[treat_col] == 0]
        if len(t1) < 2 or len(t0) < 2:
            return 0.0
        if weight_col:
            w1 = t1[weight_col].values
            w0 = t0[weight_col].values
            m1 = np.average(t1[cov].values, weights=w1)
            m0 = np.average(t0[cov].values, weights=w0)
        else:
            m1 = t1[cov].mean()
            m0 = t0[cov].mean()
        s1 = t1[cov].std()
        s0 = t0[cov].std()
        pooled_sd = np.sqrt((s1**2 + s0**2) / 2)
        if pooled_sd < 1e-10:
            return 0.0
        return float((m1 - m0) / pooled_sd)

    smd_before = {}
    smd_after = {}
    for cov in available_covs:
        if cov in df_overlap.columns:
            vals = pd.to_numeric(df_overlap[cov], errors="coerce")
            if vals.notna().sum() > 10:
                df_overlap[cov] = vals
                smd_before[cov] = compute_smd(df_overlap, cov, "treatment")
                smd_after[cov] = compute_smd(df_overlap, cov, "treatment", "ipw")

    # ── Treatment effects for key outcomes ──────────────────────────────
    # Support both v3 and v2 column names
    ttc_out = next((c for c in ("pre_ttc_min_s", "pre_ttc_min_capped_s")
                    if c in df_overlap.columns), "pre_ttc_min_s")
    stab_out = next((c for c in ("stabilization_time_s", "stabilization_5s_time_s")
                     if c in df_overlap.columns), "stabilization_time_s")
    outcomes = [
        (ttc_out, "Min TTC (s)"),
        ("pre_thw_min_s", "Min THW (s)"),
        ("post_max_abs_jerk_mps3", "Max |Jerk| (m/s³)"),
        ("post_min_accel_mps2", "Peak decel (m/s²)"),
        (stab_out, "Stabilization time (s)"),
    ]

    effects = {}
    for out_col, out_label in outcomes:
        if out_col not in df_overlap.columns:
            continue
        vals = pd.to_numeric(df_overlap[out_col], errors="coerce")
        mask = vals.notna()
        if mask.sum() < 20:
            continue

        sub = df_overlap[mask].copy()
        sub[out_col] = vals[mask]

        # Weighted means
        op = sub[sub["treatment"] == 1]
        oem = sub[sub["treatment"] == 0]
        if len(op) < 5 or len(oem) < 5:
            continue

        op_mean = np.average(op[out_col].values, weights=op["ipw"].values)
        oem_mean = np.average(oem[out_col].values, weights=oem["ipw"].values)
        diff = op_mean - oem_mean

        # Bootstrap CI for the weighted difference
        rng = np.random.RandomState(SEED)
        boot_diffs = []
        for _ in range(1000):
            b_op = op.sample(n=len(op), replace=True, random_state=rng)
            b_oem = oem.sample(n=len(oem), replace=True, random_state=rng)
            b_diff = (np.average(b_op[out_col].values, weights=b_op["ipw"].values) -
                      np.average(b_oem[out_col].values, weights=b_oem["ipw"].values))
            boot_diffs.append(b_diff)

        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

        effects[out_col] = {
            "label": out_label,
            "op_weighted_mean": float(op_mean),
            "oem_weighted_mean": float(oem_mean),
            "difference": float(diff),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "n_op": len(op),
            "n_oem": len(oem),
        }

    return {
        "method": "IPW_ATE",
        "n_total": len(df),
        "n_overlap": len(df_overlap),
        "n_excluded_overlap": int(n_excluded),
        "overlap_bounds": [caliper_lo, caliper_hi],
        "smd_before": smd_before,
        "smd_after": smd_after,
        "smd_threshold": CFG["propensity"]["smd_threshold"],
        "max_smd_after": float(max(abs(v) for v in smd_after.values())) if smd_after else None,
        "effects": effects,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    df_full, df_model = load_data()

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    #  (A) Binary outcome GEE models
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("(A) GEE LOGISTIC MODELS — BINARY OUTCOMES")
    print(f"{'═'*60}")

    # TTC < 1.5s (on lead-present + closing subset)
    print("\n  Model A1: I(TTC < 1.5s) …")
    results["gee_ttc_critical"] = fit_gee_binary(
        df_model, "ttc_critical", "P(TTC < 1.5s | closing lead)",
        subset_col="has_lead")
    _report_gee(results["gee_ttc_critical"])

    # THW < 0.8s (on lead-present subset)
    print("  Model A2: I(THW < 0.8s) …")
    results["gee_thw_critical"] = fit_gee_binary(
        df_model, "thw_critical", "P(THW < 0.8s | lead present)",
        subset_col="has_lead")
    _report_gee(results["gee_thw_critical"])

    # DRAC > 3.0 (on lead-present + closing subset)
    print("  Model A3: I(DRAC > 3.0) …")
    results["gee_drac_critical"] = fit_gee_binary(
        df_model, "drac_critical", "P(DRAC > 3.0 | closing lead)",
        subset_col="has_lead")
    _report_gee(results["gee_drac_critical"])

    # ══════════════════════════════════════════════════════════════════════
    #  (B) Continuous outcome LMM models
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("(B) LINEAR MIXED MODELS — CONTINUOUS OUTCOMES")
    print(f"{'═'*60}")

    print("\n  Model B1: Peak decel ~ scenario + speed | driver …")
    results["lmm_peak_decel"] = fit_lmm(
        df_model, "post_min_accel_mps2 ~ C(scenario) + speed",
        "post_min_accel_mps2", "Peak deceleration (m/s²)")
    _report_lmm(results["lmm_peak_decel"])

    print("  Model B2: Peak jerk ~ scenario + speed | driver …")
    results["lmm_peak_jerk"] = fit_lmm(
        df_model, "post_max_abs_jerk_mps3 ~ C(scenario) + speed",
        "post_max_abs_jerk_mps3", "Peak |jerk| (m/s³)")
    _report_lmm(results["lmm_peak_jerk"])

    print("  Model B3: Stabilization time ~ scenario + speed | driver …")
    # Only uncensored stabilization times; support v3 and v2 column names
    stab_time_col = next((c for c in ("stabilization_time_s", "stabilization_5s_time_s")
                          if c in df_model.columns), "stabilization_time_s")
    stab_cens_col = next((c for c in ("stabilization_censored", "stabilization_5s_censored")
                          if c in df_model.columns), "stabilization_censored")
    cens_series = df_model.get(stab_cens_col, pd.Series(True, index=df_model.index))
    df_uncensored = df_model[cens_series == False].copy()
    if len(df_uncensored) > 30:
        results["lmm_stabilization"] = fit_lmm(
            df_uncensored, f"{stab_time_col} ~ C(scenario) + speed",
            stab_time_col, "Stabilization time (s, uncensored)")
    else:
        results["lmm_stabilization"] = {"error": f"too few uncensored ({len(df_uncensored)})"}
    _report_lmm(results["lmm_stabilization"])

    # ══════════════════════════════════════════════════════════════════════
    #  (C) OEM vs OP propensity-weighted comparison
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("(C) OEM vs OP — PROPENSITY SCORE WEIGHTING")
    print(f"{'═'*60}")

    results["propensity_oem_vs_op"] = fit_propensity_model(df_full)
    _report_propensity(results["propensity_oem_vs_op"])

    # ── Save ────────────────────────────────────────────────────────────
    out_path = OUT / "model_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # ── Modeling notes ──────────────────────────────────────────────────
    notes = _generate_modeling_notes(results, df_model, df_full)
    notes_path = OUT / "modeling_notes.txt"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"Saved {notes_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────
def _report_gee(res):
    if "error" in res:
        print(f"    ERROR: {res['error']}")
    else:
        print(f"    {res.get('method','?')}: n={res['n_obs']:,}, "
              f"events={res['n_events']}, rate={res['event_rate']:.3f}, "
              f"clusters={res['n_clusters']}")

def _report_lmm(res):
    if "error" in res:
        print(f"    ERROR: {res['error']}")
    else:
        print(f"    LMM: n={res['n_obs']:,}, groups={res['n_groups']}, "
              f"ICC={res['icc']:.3f}, converged={res['converged']}")

def _report_propensity(res):
    if "error" in res:
        print(f"    ERROR: {res['error']}")
        return
    print(f"    n_total={res['n_total']:,}, n_overlap={res['n_overlap']:,} "
          f"(excluded {res['n_excluded_overlap']})")
    if res.get("max_smd_after") is not None:
        print(f"    Max |SMD| after weighting: {res['max_smd_after']:.3f} "
              f"(threshold: {res['smd_threshold']})")
    for out, eff in res.get("effects", {}).items():
        print(f"    {eff['label']}: OP={eff['op_weighted_mean']:.3f}, "
              f"OEM={eff['oem_weighted_mean']:.3f}, "
              f"diff={eff['difference']:.3f} "
              f"[{eff['ci_lower']:.3f}, {eff['ci_upper']:.3f}]")


def _generate_modeling_notes(results, df_model, df_full):
    lines = []
    lines.append("=" * 72)
    lines.append("  MODELING NOTES — Take Over Analysis (Section IV)")
    lines.append("=" * 72)
    lines.append("")

    lines.append("METHOD SUMMARY")
    lines.append("-" * 40)
    lines.append("")
    lines.append("(A) Binary outcome models use GEE logistic regression with")
    lines.append("    exchangeable correlation structure, clustered by driver (dongle_id).")
    lines.append("    Fallback: cluster-robust GLM if GEE fails to converge.")
    lines.append("    Subset: lead-present clips only (TTC/DRAC further restricted to closing).")
    lines.append("    Rationale: avoids right-censoring and heavy-tailed issues with continuous")
    lines.append("    TTC/THW regression; threshold-event modeling is standard in traffic safety.")
    lines.append("")
    lines.append("(B) Continuous outcome models use linear mixed-effects models (REML)")
    lines.append("    with driver random intercept.")
    lines.append("    ICC quantifies driver-level variance share.")
    lines.append("    Stabilization time model restricted to uncensored observations.")
    lines.append("")
    lines.append("(C) OEM vs OP comparison uses inverse probability weighting (IPW).")
    lines.append(f"    Propensity covariates: {CFG['propensity']['covariates']}")
    lines.append(f"    Overlap restriction: propensity in (0.1, 0.9)")
    lines.append(f"    SMD balance threshold: {CFG['propensity']['smd_threshold']}")
    lines.append("    Bootstrap CI (n=1000) for weighted treatment effects.")
    lines.append("")

    lines.append("CAVEATS")
    lines.append("-" * 40)
    lines.append("- TTC/THW are only defined when a lead vehicle is detected;")
    lines.append("  results are conditional on lead presence.")
    lines.append("- Jerk and steer_rate are sampling-rate sensitive; qlog (10Hz)")
    lines.append("  and rlog (100Hz) clips may produce systematically different values.")
    lines.append("  See qlog vs rlog sensitivity in derived signals summary.")
    lines.append("- Propensity weighting assumes no unmeasured confounders;")
    lines.append("  self-selection into openpilot is likely correlated with")
    lines.append("  unobserved driver characteristics.")
    lines.append("- Stabilization time uses a right-censored definition at 5.0s;")
    lines.append("  censored clips are excluded from LMM (reported as descriptive).")
    lines.append("")

    # Model subset info
    n_full = len(df_full)
    n_model = len(df_model)
    lines.append("SAMPLE SIZES")
    lines.append("-" * 40)
    lines.append(f"  Full dataset: {n_full:,} clips")
    lines.append(f"  Model subset (>={MIN_CLIPS} clips/driver): {n_model:,} clips")
    lines.append(f"  Drivers in model subset: {df_model['dongle_id'].nunique()}")
    lines.append("")

    for name, res in results.items():
        lines.append(f"  {name}:")
        if "error" in res:
            lines.append(f"    ERROR: {res['error']}")
        elif "n_obs" in res:
            lines.append(f"    n_obs={res['n_obs']:,}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
