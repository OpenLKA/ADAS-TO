# Takeover Safety & Smoothness Reanalysis

## Methods

### Data and Scope

This analysis examines 15,705 takeover clips from the OpenLKA dataset.
Each clip captures a disengagement event where the human driver overrides
the Level-2 ADAS (openpilot) via steering, brake, or gas input.
Six CAN-bus topics are used per clip: carControl, carOutput, carState,
controlsState, drivingModelData, and longitudinalPlan.  radarState is
excluded; consequently, no direct time-to-collision (TTC), time headway
(THW), or deceleration-rate-to-avoid-crash (DRAC) metrics are available.
All safety-relevant quantities reported here are *proxies* derived from
planner intent and perception confidence rather than physical headway.

### Time Windows

- **Pre-takeover window**: [−3, 0] s relative to the disengagement event.
  Used for context proxies (lane confidence, curvature mismatch, planner
  targets).
- **Post-takeover window**: [0, +5] s.  Used for control-quality metrics
  (jerk, steering rate, curvature rate, RMSE, stabilization time).

### Derivative Computation

Jerk (da/dt), steering rate (dθ/dt), and curvature rate (dκ/dt) are
computed via timestamp-based finite differences after Savitzky–Golay
smoothing (window = 0.3 s, polynomial order 2).  Jerk is capped at
50 m/s³ and steering rate at 500 °/s to suppress instrumentation
artifacts.

### Control-Chain RMSE and Gating

Plan-to-output RMSE compares `actuators.accel` (carControl) against
`actuatorsOutput.accel` (carOutput) on a common 20 Hz grid via linear
interpolation.  A **gating rule** is applied: if the RMSE value is
exactly 0.0, the corresponding channel is presumed inactive and the
value is set to NA.  This is necessary because 64.3% of pre-window
and 79.7% of post-window accel plan→output RMSE values are identically
zero, indicating that the actuator command channel is not active
(e.g., the vehicle platform does not expose acceleration commands
via carOutput).  Without gating, these zeros would bias summary
statistics toward artificially low RMSE.  Curvature plan→output RMSE
has only 3.8% zeros and is retained without gating.

### Missingness

Topic-level missingness varies substantially:

- **carControl**: 0.2% missing (32/15,705)
- **carOutput**: 23.9% missing (3,751/15,705)
- **carState**: 0.2% missing (39/15,705)
- **controlsState**: 0.1% missing (11/15,705)
- **drivingModelData**: 33.9% missing (5,326/15,705)
- **longitudinalPlan**: 0.4% missing (63/15,705)

drivingModelData is absent for 33.9% of clips; metrics derived from
it (laneProb, laneCenter_range, laneWidth) have effective
N ≈ 10,200 rather than 15,700.  carOutput is absent for 23.9%;
all RMSE terms involving actuatorsOutput have effective N ≈ 11,800.
All tables report effective N alongside summary statistics.

### Winsorization

Curvature mismatch (|κ_desired − κ_actual|) exhibits extreme right
skew (P99 = 0.195 vs. P100 = 21.9).  Values above P99 are capped
(winsorized) to prevent outlier-driven distortion.  This choice is
conservative: it preserves 99% of the distribution while removing
instrumentation artifacts.

### Stabilization Time

Stabilization is defined as the first continuous 1.0 s window in the
post-takeover period where |a| < 0.5 m/s², |jerk| < 1.0 m/s³, and
|steer rate| < 30 °/s simultaneously.  The metric is right-censored
at 5.0 s: clips that do not stabilize within 5 s are marked as
censored and excluded from the uncensored distribution.

---

## Results

### 1. Pre-Takeover Context (System Load Proxies)

**Lane detection confidence.**  Among the 10,240 clips with drivingModelData, the minimum lane-line probability in the pre-window has a median of 0.326 (P5 = 0.008, P95 = 0.987).  4,989 clips (48.7%) fall below 0.30, suggesting degraded lane marking visibility or model uncertainty.

**Curvature mismatch.**  The maximum |κ_desired − κ_actual| in the pre-window (winsorized at P99) has a median of 0.000707 (N = 15,379).  The distribution is right-skewed: P95 = 0.024011.  Elevated mismatch is consistent with situations where the planner's desired path diverges from the vehicle's executed trajectory, potentially indicating challenging road geometry or controller tracking limitations.

**Longitudinal planner target.**  The minimum planned acceleration target (aTarget) in the pre-window has a median of -0.111 m/s² (N = 15,354).  3,449 clips (22.5%) show aTarget < −1.0 m/s², suggesting the planner was commanding notable deceleration before the takeover.

**Lead vehicle consistency.**  The mismatch rate between hasLead (longitudinalPlan) and hudControl.leadVisible (carControl) has a median of 0.000 (N = 15,348).  427 clips (2.8%) exhibit a mismatch rate exceeding 10%, indicating intermittent lead-vehicle detection or display discrepancies.

### 2. Post-Takeover Control Quality

**Jerk.**  Peak post-takeover jerk has a median of 4.52 m/s³ (N = 15,227, P5 = 1.66, P95 = 14.66).  The distribution is right-skewed, consistent with the expectation that most takeovers are smooth but a minority involve abrupt longitudinal corrections.

**Steering rate.**  Peak post-takeover steering rate has a median of 20.9 °/s (N = 15,227, P5 = 2.0, P95 = 223.9).  The wide P95 value suggests substantial heterogeneity in lateral control urgency across takeover events.

**Accel plan→output RMSE (gated).**  After removing inactive-channel zeros, the pre-window median is 0.0309 (N = 4,212) and the post-window median is 0.0415 (N = 2,389).  The low magnitudes suggest that when the accel command channel is active, the output closely tracks the plan.

**Accel output→state RMSE.**  This measures the gap between commanded output and realized vehicle acceleration (aEgo).  Pre-window median = 0.467 (N = 11,797), post-window median = 0.809 (N = 11,752).  The substantially larger magnitude compared to plan→output RMSE is expected: it captures both actuator lag and physical plant dynamics.

**Stabilization time.**  Of 15,705 clips, 13,315 (84.8%) are right-censored (not stabilized within 5 s).  Among the 2,390 uncensored clips, median stabilization time is 1.35 s (P5 = 0.00, P95 = 3.65).  **Caution**: the high censoring rate (84.8%) means the uncensored distribution under-represents difficult takeovers.  The median should be interpreted as a lower bound on the population stabilization time.

### 3. Behavior Validation by Trigger Type

**Post-takeover jerk by trigger type:**

| Trigger | N | Median | 95% CI |
|---------|---|--------|--------|
| Brake Override | 3,948 | 4.69 m/s³ | [4.56, 4.80] |
| Gas Override | 1,366 | 3.60 m/s³ | [3.51, 3.75] |
| Steering Override | 8,291 | 4.78 m/s³ | [4.72, 4.86] |
| System / Unknown | 1,586 | 3.73 m/s³ | [3.61, 3.85] |

**Post-takeover steering rate by trigger type:**

| Trigger | N | Median | 95% CI |
|---------|---|--------|--------|
| Brake Override | 3,948 | 15.00 °/s | [14.09, 15.95] |
| Gas Override | 1,366 | 12.34 °/s | [11.81, 13.05] |
| Steering Override | 8,291 | 28.57 °/s | [27.79, 29.35] |
| System / Unknown | 1,586 | 12.22 °/s | [11.78, 13.06] |

Steering–Brake median difference in jerk: Δ = +0.09, 95% CI [-0.05, +0.24].
Steering–Brake median difference in steering rate: Δ = +13.56, 95% CI [+12.32, +14.79].

Steering-override takeovers exhibit higher median steering rates
than brake-override takeovers, consistent with the biomechanical
expectation that steering interventions involve rapid lateral
corrections.  The effect-size confidence intervals exclude zero,
suggesting a robust difference.  However, this comparison reflects
the trigger modality itself and should not be interpreted as
evidence that one trigger type is inherently safer.

### 4. Diagnostic Interaction Flags

Five diagnostic flags identify clips exhibiting potentially
problematic perception–control interactions.  Each flag is
defined by explicit thresholds chosen from the data distribution
or engineering practice.  For each flag, we report prevalence
(among clips with available data) and the uplift in median
post-takeover steering rate (flagged vs. unflagged), with
bootstrap 95% CI for the difference.

| Flag | Rule | Prevalence | Steer rate uplift (Δ median) |
|------|------|------------|------------------------------|
| lead_inconsistency | `lead_consistency_flag > 0.10` | 2.7% | +1.2 °/s [-1.5, +4.0] |
| low_lane_prob | `laneProb_min_pre < 0.30` | 31.8% | +13.9 °/s [+12.4, +15.5] |
| high_curv_mismatch | `curvature_mismatch_max_pre > P95 (winsorized)` | 4.9% | +102.9 °/s [+82.4, +116.3] |
| output_aggressive_low_lane | `laneProb_min_pre < 0.30 AND curvature_rate_max_post > P95` | 2.4% | +260.6 °/s [+246.5, +270.6] |
| plan_output_outlier | `accel_plan_output_rmse_pre (gated) > P95` | 1.3% | +5.6 °/s [+1.3, +12.4] |

### 5. Limitations

1. **No radar-derived safety metrics.**  Without radarState, this
   analysis cannot compute TTC, THW, or DRAC.  The reported proxies
   (hasLead rate, aTarget, planned speed drop) reflect planner intent
   rather than physical headway.

2. **Proxy nature of all pre-takeover metrics.**  Lane probability,
   curvature mismatch, and lead-visibility rates are model-internal
   quantities.  Their relationship to objective road conditions is
   mediated by the perception model's accuracy, which may vary across
   vehicle platforms and lighting conditions.

3. **RMSE gating heuristic.**  The zero-RMSE gating rule (Section
   Methods) is a post-hoc heuristic.  A more rigorous approach would
   check raw channel activity (mean |signal| > ε) per clip, requiring
   a separate pass over raw CSVs.

4. **Stabilization censoring.**  The high censoring rate (≈85%)
   means the reported stabilization-time distribution is conditional
   on stabilization occurring within 5 s.  Causal or population-level
   inferences from this metric require survival-analysis methods.

5. **Sample-rate sensitivity.**  The dataset contains both qlog
   (10 Hz) and rlog (100 Hz) recordings.  Derivative metrics (jerk,
   steering rate) may exhibit systematic differences across log types.

6. **No causal claims.**  All reported associations (e.g., low lane
   probability → higher steering rate) are observational.
   Confounding by road type, weather, or vehicle platform cannot
   be ruled out without additional covariate adjustment.

7. **drivingModelData missingness.**  One-third of clips lack
   drivingModelData; lane-probability and lane-width metrics are
   computed on the remaining 66%.  If missingness is non-random
   (e.g., correlated with vehicle platform or log type), the
   reported distributions may not generalize to the full dataset.
