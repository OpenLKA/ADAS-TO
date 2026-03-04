# Figure Refinement Notes

## Data Checks

- **Total clips**: 15,705
- **laneProb_min_pre**: range [0.0003, 0.9981] — confirmed ∈ [0, 1]. Missing for 34.8% of clips (drivingModelData absent).
- **steer_rate_max_post**: range [0, 500] °/s (capped). Missing for 3.0%.
- **jerk_max_post**: range [0, 50] m/s³ (capped). Missing for 3.0%.
- **curvature_mismatch_max_pre**: P99 = 0.195, P100 = 21.9 before winsorization. Winsorized at P99 (154 values capped). Units: 1/m (|κ_desired − κ_actual|).
- **primary_trigger**: Steering Override (8,541), Brake Override (4,059), Gas Override (1,380), System/Unknown (1,679), missing (46).
- **No radarState-derived columns present** (verified: no dRel, vRel, radar, thw, ttc, headway, or distance columns found).

## Gating and Winsorization

- **RMSE gating**: `accel_plan_output_rmse_pre`: 7,583 / 11,795 values (64.3%) were exactly 0.0 → set to NA (channel inactive). `accel_plan_output_rmse_post`: 9,361 / 11,750 (79.7%) → NA.
- **Curvature mismatch winsorization**: capped at P99 ≈ 0.195. Justification: the 1% tail extends to 21.9, >100× the P99 value, likely reflecting instrumentation artifacts or edge-case controller transients. Winsorization preserves 99% of the distribution.

## Findings (for paper text)

1. **Lane confidence and lateral control urgency (Fig 1, Fig 4)**. Clips with low lane-detection confidence (laneProb < 0.1) exhibit median post-takeover steering rate of 40.4 °/s, approximately 4.0× higher than clips with high confidence (laneProb > 0.9, median 10.0 °/s). This monotonic relationship persists across speed strata (Fig 4), suggesting it is not solely a confound of low-speed driving contexts.

2. **Trigger modality (Fig 2)**. Steering-override takeovers show median steer rate of 28.6 °/s versus 15.0 °/s for brake overrides — an expected difference reflecting the biomechanics of the trigger itself rather than indicating differential safety.

3. **Curvature mismatch dose–response (Fig 3)**. Post-takeover steering rate increases monotonically across curvature-mismatch strata, consistent with the hypothesis that larger planner–vehicle path discrepancies precede more urgent lateral corrections.

4. **Lead-present planner demand (Fig 5)**. Among 7,596 clips with sustained lead-vehicle detection (hasLead > 50%), the median planner deceleration demand is 0.18 m/s². Higher decel demand is associated with modestly elevated post-takeover jerk, consistent with more urgent longitudinal corrections. **Limitation**: this proxy reflects planner intent, not physical headway; true TTC/THW requires radar distance data (radarState), which is excluded from this analysis.

## Figure Captions

**Fig 1.** Median post-takeover steering rate (a) and longitudinal jerk (b) as a function of pre-takeover lane-detection confidence. Error bars show bootstrap 95% CI (2,000 replicates). Sample sizes per bin annotated.

**Fig 2.** Distribution of post-takeover jerk (a) and steering rate (b) stratified by takeover trigger modality. Boxes show IQR with median (orange); violins show kernel density. Sample sizes annotated.

**Fig 3.** Median post-takeover steering rate across quantile strata of pre-takeover curvature mismatch (winsorized at P99). The monotonic increase is consistent with larger path-tracking discrepancies preceding more urgent lateral corrections.

**Fig 4.** Speed-stratified replication of Fig 1: median post-takeover steering rate (a–c) and jerk (d–f) versus lane-detection confidence, shown separately for low-, medium-, and high-speed regimes. The negative lateral-control association persists across speed strata; the longitudinal (jerk) association is weaker and largely confined to low-speed driving.

**Fig 5.** Lead-vehicle-present subset analysis. (a) Distribution of planner-based deceleration demand (|aTarget min|) and planned speed drop for clips where hasLead > 50% of the pre-window. (b) Median post-takeover jerk as a function of binned deceleration demand. Note: this is a planner-intent proxy, not a radar-derived headway metric; true TTC/THW cannot be computed without radarState.

**Fig 6.** Stratified robustness check: within each speed regime, the laneProb → steer\_rate relationship is shown separately for clips with sustained lead-vehicle detection (hasLead > 50\%, red) and without (teal). The association persists in both subgroups, suggesting it is not confounded by lead-vehicle presence.
