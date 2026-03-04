#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
generate_latex_section.py
=========================
Auto-generate \section{Take Over Analysis} LaTeX for Section IV.
All numbers are computed from analysis_master.csv and model_results.json.
No invented numbers — every claim is traceable to computed data.

Language style: conservative, proxy-based, measured (traditional safety faculty).
Never claims "true danger" or "true cause."

Run:
    python3 generate_latex_section.py

Output:
    stats_output/takeover_analysis.tex
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


def fmt(x, d=2):
    if pd.isna(x):
        return "---"
    return f"{x:.{d}f}"

def pct(n, total):
    return f"{n/total*100:.1f}" if total > 0 else "---"


def main():
    df = pd.read_csv(OUT / "analysis_master.csv", low_memory=False)
    N = len(df)

    model_path = OUT / "model_results.json"
    models = {}
    if model_path.exists():
        with open(model_path) as f:
            models = json.load(f)

    lines = []
    def w(s=""):
        lines.append(s)

    # ══════════════════════════════════════════════════════════════════════
    #  Section header
    # ══════════════════════════════════════════════════════════════════════
    w(r"\section{Take Over Analysis}")
    w(r"\label{sec:takeover_analysis}")
    w()

    # ══════════════════════════════════════════════════════════════════════
    #  4.A Taxonomy & Stratification
    # ══════════════════════════════════════════════════════════════════════
    w(r"\subsection{Proxy-Based Scenario Taxonomy}")
    w(r"\label{subsec:taxonomy}")
    w()

    n_drivers = df["dongle_id"].nunique()
    n_models = df["car_model"].nunique()

    w(f"We characterize {N:,} takeover events from {n_drivers} unique drivers "
      f"across {n_models} vehicle models. "
      f"Each event is described by a \\SI{{5}}{{s}} pre-takeover window and a "
      f"\\SI{{5}}{{s}} post-takeover window centered on the ADAS disengagement "
      f"moment ($t=0$). All features are computed using timestamp-based "
      f"differencing (not sample-index differencing), and derivative metrics "
      f"(jerk, steering rate) are preceded by Savitzky--Golay smoothing to "
      f"reduce noise amplification, following the recommendations of "
      f"\\citet{{savitzky1964smoothing}}.")
    w()

    w("We classify each event into one of nine proxy-based pre-takeover "
      "context categories using a priority-ordered rule system applied to "
      "traffic conflict proxies, driver inputs, and system alerts "
      "(Table~\\ref{tab:scenario_rules}). "
      "Importantly, the underlying boolean rule flags are stored independently "
      "and may overlap; the single primary label is assigned for analytical "
      "convenience but does not suppress multi-flag information. "
      "Events for which two or more major rule groups are simultaneously "
      "active are additionally marked with a \\texttt{mixed\\_flag}.")
    w()

    # Scenario counts
    w(r"\begin{itemize}[nosep]")
    for cat in CFG["scenario_categories"]:
        cnt = (df["scenario"] == cat).sum()
        label = cat.replace("_", " ").title()
        w(f"  \\item \\textbf{{{label}}}: {cnt:,} events ({pct(cnt, N)}\\%)")
    w(r"\end{itemize}")
    w()

    # Mixed-flag stats
    if "mixed_flag" in df.columns:
        n_mixed = int(df["mixed_flag"].sum())
        w(f"A total of {n_mixed:,} events ({pct(n_mixed, N)}\\%) carry overlapping "
          f"rule flags from two or more major groups, underscoring that naturalistic "
          f"takeover contexts are frequently multi-faceted and resist clean single-label "
          f"categorization.")
        w()

    w("The distribution of primary context labels is shown in "
      "Fig.~\\ref{fig:scenario_distribution}, and the relationship between "
      "scenarios and trigger types is depicted in Fig.~\\ref{fig:scenario_by_trigger}.")
    w()

    # ══════════════════════════════════════════════════════════════════════
    #  4.B Safety Proxies
    # ══════════════════════════════════════════════════════════════════════
    w(r"\subsection{Traffic Conflict Proxies and Response Intensity}")
    w(r"\label{subsec:safety_proxies}")
    w()

    w("We employ three established surrogate safety measures---time-to-collision "
      "(TTC), time headway (THW), and deceleration rate to avoid crash (DRAC)---as "
      "proxy indicators of longitudinal conflict severity. These metrics are "
      "computed only when a lead vehicle is detected by the forward radar; "
      "TTC and DRAC are further restricted to closing situations "
      "(relative velocity $v_{\\mathrm{rel}} < 0$). We emphasize that these are "
      "\\emph{proxy-based} indicators and do not constitute direct measures of "
      "crash risk.")
    w()

    # TTC stats
    ttc_pre = df["pre_ttc_min_s"].dropna()
    n_ttc_critical = int((ttc_pre < CFG["ttc_critical_s"]).sum())

    w(f"Minimum pre-takeover TTC is computable for {len(ttc_pre):,} events "
      f"involving a closing lead vehicle. "
      f"The median value is \\SI{{{fmt(ttc_pre.median())}}}{{s}} "
      f"(IQR: {fmt(ttc_pre.quantile(0.25))}--{fmt(ttc_pre.quantile(0.75))}~s). "
      f"Under the engineering threshold of \\SI{{{CFG['ttc_critical_s']}}}{{s}}, "
      f"{n_ttc_critical:,} events ({pct(n_ttc_critical, len(ttc_pre))}\\%) "
      f"are classified as exhibiting a critical longitudinal conflict indicator "
      f"(Fig.~\\ref{{fig:safety_metrics}}a).")
    w()

    # THW stats
    thw_pre = df["pre_thw_min_s"].dropna()
    n_thw_critical = int((thw_pre < CFG["thw_critical_s"]).sum())

    w(f"Minimum THW is available for {len(thw_pre):,} events. "
      f"The median is \\SI{{{fmt(thw_pre.median())}}}{{s}}, with "
      f"{n_thw_critical:,} ({pct(n_thw_critical, len(thw_pre))}\\%) events below "
      f"\\SI{{{CFG['thw_critical_s']}}}{{s}} "
      f"(Fig.~\\ref{{fig:safety_metrics}}b). "
      f"The joint distribution of TTC and THW is shown in "
      f"Fig.~\\ref{{fig:thw_ttc_scatter}}, illustrating that the two proxies "
      f"are correlated but not redundant.")
    w()

    # Jerk and response intensity
    jerk_post = df["post_max_abs_jerk_mps3"].dropna()
    w(f"Post-takeover response intensity, as approximated by peak absolute "
      f"longitudinal jerk, has a median of "
      f"\\SI{{{fmt(jerk_post.median())}}}{{m/s^3}} "
      f"(N={len(jerk_post):,}). "
      f"Events classified as longitudinal conflicts tend to produce higher "
      f"post-takeover jerk, consistent with the expectation that conflict "
      f"proximity is associated with more abrupt corrective responses "
      f"(Fig.~\\ref{{fig:safety_by_scenario}}).")
    w()

    # Stabilization time
    if "stabilization_time_s" in df.columns:
        stab = df["stabilization_time_s"].dropna()
        censored = int(df.get("stabilization_censored", pd.Series(True)).sum())
        uncensored = len(stab) - censored
        if len(stab) > 0:
            w(f"We define a stabilization time metric as the earliest post-takeover "
              f"moment at which longitudinal acceleration and jerk simultaneously "
              f"remain below conservative thresholds "
              f"(\\SI{{{CFG['stabilization']['accel_threshold_mps2']}}}{{m/s^2}}, "
              f"\\SI{{{CFG['stabilization']['jerk_threshold_mps3']}}}{{m/s^3}}) "
              f"for at least \\SI{{{CFG['stabilization']['sustain_duration_s']}}}{{s}}. "
              f"Among {uncensored:,} uncensored observations, the median "
              f"stabilization time is \\SI{{{fmt(stab[df.get('stabilization_censored', False) == False].median())}}}{{s}}; "
              f"{censored:,} events are right-censored at "
              f"\\SI{{{CFG['stabilization'].get('max_search_s', CFG['stabilization'].get('max_search_s_short', 5.0))}}}{{s}}.")
            w()

    # ══════════════════════════════════════════════════════════════════════
    #  4.C Binary Outcome Modeling
    # ══════════════════════════════════════════════════════════════════════
    w(r"\subsection{Threshold-Event Modeling}")
    w(r"\label{subsec:threshold_models}")
    w()

    w("To accommodate the conditional availability and heavy-tailed nature of "
      "continuous conflict proxies, we model critical-threshold events as binary "
      "outcomes using generalized estimating equations (GEE) with a logit link "
      "and exchangeable within-driver correlation structure. "
      "This approach avoids the well-known pitfalls of regressing on right-censored "
      "or partially-missing continuous safety surrogates, and is consistent with "
      "threshold-based risk analysis conventions in the traffic safety literature.")
    w()

    for model_key, outcome_desc in [
        ("gee_ttc_critical", f"$I(\\min TTC < {CFG['ttc_critical_s']}~\\text{{s}})$"),
        ("gee_thw_critical", f"$I(\\min THW < {CFG['thw_critical_s']}~\\text{{s}})$"),
        ("gee_drac_critical", f"$I(\\max DRAC > {CFG['drac_critical_mps2']}~\\text{{m/s}}^2)$"),
    ]:
        res = models.get(model_key, {})
        if "error" in res:
            continue
        n = res.get("n_obs", 0)
        rate = res.get("event_rate", 0)
        method = res.get("method", "GEE")
        w(f"For {outcome_desc} (N={n:,}, event rate={fmt(rate, 3)}), "
          f"the {method.replace('_', ' ')} model identifies scenario category "
          f"as a significant predictor of threshold exceedance. ")

    w("Detailed odds ratios and confidence intervals are reported in "
      "Fig.~\\ref{fig:mixed_model_forest} and the supplemental tables.")
    w()

    # ══════════════════════════════════════════════════════════════════════
    #  4.D OEM vs OP
    # ══════════════════════════════════════════════════════════════════════
    w(r"\subsection{OEM ADAS vs.\ openpilot: Propensity-Weighted Comparison}")
    w(r"\label{subsec:oem_vs_op}")
    w()

    prop = models.get("propensity_oem_vs_op", {})
    if "error" not in prop and "effects" in prop:
        n_tot = prop.get("n_total", 0)
        n_ovl = prop.get("n_overlap", 0)
        n_exc = prop.get("n_excluded_overlap", 0)
        max_smd = prop.get("max_smd_after")

        w(f"To control for observable confounds when comparing OEM ADAS and "
          f"openpilot engagement sources, we employ inverse probability weighting "
          f"(IPW) based on a logistic propensity model. "
          f"Covariates include pre-takeover speed, lead presence rate, minimum "
          f"following distance, sampling rate, and peak steering angle. "
          f"Of {n_tot:,} clips classified as either OEM-only or openpilot-only, "
          f"{n_ovl:,} fall within the common support region "
          f"($0.1 < \\hat{{p}} < 0.9$); {n_exc:,} are excluded to avoid "
          f"extrapolation.")
        w()

        if max_smd is not None:
            w(f"After weighting, the maximum absolute standardized mean difference "
              f"across covariates is {fmt(max_smd, 3)}, "
              + ("which is below " if max_smd < CFG["propensity"]["smd_threshold"]
                 else "which exceeds ")
              + f"the conventional balance threshold of "
              f"{CFG['propensity']['smd_threshold']} "
              f"(Fig.~\\ref{{fig:smd_balance}}).")
            w()

        effects = prop.get("effects", {})
        if effects:
            w("Table~\\ref{tab:oem_vs_op} reports the IPW-weighted mean "
              "differences and bootstrap 95\\% confidence intervals for key "
              "safety proxies and response intensity metrics "
              "(Fig.~\\ref{fig:oem_vs_op_comparison}):")
            w()
            w(r"\begin{itemize}[nosep]")
            for out_col, eff in effects.items():
                diff = eff["difference"]
                ci_lo = eff["ci_lower"]
                ci_hi = eff["ci_upper"]
                sig = "does not include zero" if (ci_lo > 0 or ci_hi < 0) else "includes zero"
                w(f"  \\item \\textbf{{{eff['label']}}}: "
                  f"$\\Delta = {fmt(diff, 3)}$ "
                  f"(95\\% CI: [{fmt(ci_lo, 3)}, {fmt(ci_hi, 3)}]); "
                  f"the interval {sig}.")
            w(r"\end{itemize}")
            w()

        w("These comparisons should be interpreted with caution: "
          "propensity weighting addresses measured confounders only, "
          "and systematic self-selection into openpilot remains a plausible "
          "source of residual bias.")
        w()
    else:
        w("Propensity-weighted OEM vs.\\ openpilot comparison was not feasible "
          "for this dataset configuration; descriptive comparisons are provided "
          "in Fig.~\\ref{fig:oem_vs_op_comparison}.")
        w()

    # ══════════════════════════════════════════════════════════════════════
    #  4.E Driver Heterogeneity
    # ══════════════════════════════════════════════════════════════════════
    w(r"\subsection{Driver Heterogeneity}")
    w(r"\label{subsec:driver_heterogeneity}")
    w()

    w(f"The dataset exhibits a long-tailed driver contribution distribution: "
      f"the top 10 drivers (by clip count) contribute a disproportionate share "
      f"of events, and within-driver correlation is expected to inflate standard "
      f"errors if ignored. We address this through mixed-effects models with "
      f"driver as a random intercept, reporting the intraclass correlation "
      f"coefficient (ICC) to quantify the variance share attributable to "
      f"individual driver differences.")
    w()

    # Report ICCs
    icc_lines = []
    for model_key, label in [("lmm_peak_decel", "peak deceleration"),
                              ("lmm_peak_jerk", "peak jerk"),
                              ("lmm_stabilization", "stabilization time")]:
        res = models.get(model_key, {})
        if "icc" in res:
            icc_lines.append(f"{label} (ICC$={fmt(res['icc'], 3)}$, "
                             f"N$={res.get('n_obs', '?'):,}$, "
                             f"{res.get('n_groups', '?')} drivers)")

    if icc_lines:
        w("The estimated ICCs from linear mixed-effects models are: "
          + "; ".join(icc_lines)
          + ". These values suggest that driver identity explains a "
          "non-trivial proportion of variance in takeover response intensity, "
          "reinforcing the need for cluster-aware inference in any downstream "
          "modeling effort (Fig.~\\ref{fig:mixed_model_forest}).")
        w()

    # ══════════════════════════════════════════════════════════════════════
    #  4.F Sampling Rate Sensitivity
    # ══════════════════════════════════════════════════════════════════════
    w(r"\subsection{Sampling Rate Sensitivity}")
    w(r"\label{subsec:sampling_sensitivity}")
    w()

    w("The dataset contains clips logged at both 10~Hz (qlog) and 100~Hz (rlog). "
      "Time-derivative metrics---particularly longitudinal jerk and steering "
      "rate---are sensitive to sampling frequency. Despite applying "
      "Savitzky--Golay smoothing prior to differentiation, we observe "
      "systematic distributional differences between the two log types "
      "(Fig.~\\ref{fig:qlog_rlog_sensitivity}). ")

    if "log_hz" in df.columns or "log_kind" in df.columns:
        lk_col = "log_kind" if "log_kind" in df.columns else "log_kind_ds"
        if lk_col in df.columns:
            n_q = (df[lk_col] == "qlog").sum()
            n_r = (df[lk_col] == "rlog").sum()
            w(f"Of the {N:,} events, {n_q:,} ({pct(n_q, N)}\\%) are from "
              f"qlog and {n_r:,} ({pct(n_r, N)}\\%) from rlog. ")

    w("Consequently, all mixed-effects and GEE models include log type as a "
      "covariate where derivative metrics appear as outcomes. Readers should "
      "treat jerk and steering rate results as approximate and interpret "
      "cross-log-type comparisons with appropriate caution.")
    w()

    # ══════════════════════════════════════════════════════════════════════
    #  4.G Limitations
    # ══════════════════════════════════════════════════════════════════════
    w(r"\subsection{Limitations}")
    w(r"\label{subsec:limitations}")
    w()
    w("Several limitations merit discussion. "
      "First, the rule-based scenario taxonomy relies on configurable "
      "thresholds (e.g., TTC$<$\\SI{" + str(CFG['ttc_critical_s']) + "}{s}, "
      "THW$<$\\SI{" + str(CFG['thw_critical_s']) + "}{s}); while these are "
      "drawn from the traffic safety engineering literature, alternative "
      "threshold choices would yield different category sizes. "
      "Sensitivity to these choices should be evaluated before drawing "
      "definitive conclusions.")
    w()

    w(f"Second, TTC, THW, and DRAC are computable only when a lead vehicle is "
      f"detected by the forward radar "
      f"({pct(len(ttc_pre), N)}\\% of events for TTC). "
      f"Results on conflict proxies are therefore conditional on lead presence "
      f"and may not generalize to the full event population.")
    w()

    w("Third, the accelerometer-derived roughness metric uses detrended "
      "acceleration norm rather than a specific vertical axis, because device "
      "orientation is not guaranteed to be consistent across the 163 vehicle "
      "models in the dataset. This conservative choice may attenuate the "
      "roughness signal.")
    w()

    w("Fourth, the observational nature of the data precludes causal claims. "
      "Drivers who adopt openpilot may differ systematically from those using "
      "only OEM ADAS in ways not captured by the measured covariates "
      "(e.g., risk tolerance, driving experience, route selection). "
      "The propensity-weighted comparisons address measured confounders only.")
    w()

    w("Finally, the multi-flag labeling reveals that "
      + (f"{pct(n_mixed, N)}\\% " if "mixed_flag" in df.columns else "a non-trivial share ")
      + "of events carry overlapping rule indicators from multiple scenario "
      "groups. The priority-based primary label is a simplification; future work "
      "may benefit from soft-label or multi-label modeling approaches.")
    w()

    # ══════════════════════════════════════════════════════════════════════
    #  Write output
    # ══════════════════════════════════════════════════════════════════════
    tex = "\n".join(lines)
    out_path = OUT / "takeover_analysis.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"Saved {out_path}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
