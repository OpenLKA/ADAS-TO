# Takeover Safety & Stability Report

## 1. Dataset Overview

- **Total clips:** 15,705
- **Missingness rates:**
  - carControl: 0.2%
  - carOutput: 23.9%
  - carState: 0.2%
  - controlsState: 0.1%
  - drivingModelData: 33.9%
  - longitudinalPlan: 0.4%

## 2. Safety Proxies (Pre-Window)

| Metric | N | Median | P5 | P95 |
|--------|---|--------|----|----|
| hasLead_rate_pre | 15,354 | 0.5000 | 0.0000 | 1.0000 |
| leadVisible_rate_pre | 15,357 | 0.4700 | 0.0000 | 1.0000 |
| aTarget_min_pre | 15,354 | -0.1109 | -2.0000 | 0.1891 |
| aTarget_mean_pre | 15,354 | 0.0000 | -1.3951 | 0.5400 |
| planned_speed_drop_pre | 15,237 | 0.7740 | 0.0000 | 4.9138 |
| laneProb_min_pre | 10,240 | 0.3258 | 0.0075 | 0.9866 |
| laneProb_mean_pre | 10,240 | 0.6144 | 0.0249 | 0.9913 |
| curvature_mismatch_mean_pre | 15,379 | 0.0003 | 0.0000 | 0.0089 |
| curvature_mismatch_max_pre | 15,379 | 0.0007 | 0.0000 | 0.0240 |

## 3. Control Smoothness

| Metric | N | Median | P5 | P95 |
|--------|---|--------|----|----|
| jerk_max_post | 15,227 | 4.5224 | 1.6553 | 14.6642 |
| steer_rate_max_post | 15,227 | 20.8816 | 2.0299 | 223.9126 |
| curvature_rate_max_post | 15,226 | 0.0063 | 0.0007 | 0.0816 |
| accel_plan_output_rmse_pre | 11,795 | 0.0000 | 0.0000 | 0.4967 |
| accel_plan_output_rmse_post | 11,750 | 0.0000 | 0.0000 | 0.1622 |
| accel_output_state_rmse_pre | 11,797 | 0.4667 | 0.0727 | 2.0195 |
| accel_output_state_rmse_post | 11,752 | 0.8094 | 0.1716 | 2.2300 |
| curv_plan_output_rmse_pre | 11,795 | 0.0000 | 0.0000 | 0.0015 |
| curv_plan_output_rmse_post | 11,750 | 0.0001 | 0.0000 | 0.0026 |

### RMSE Pre→Post Changes

- **accel_plan_output_rmse**: pre median=0.0000, post median=0.0000, Δ=+0.0000
- **accel_output_state_rmse**: pre median=0.4667, post median=0.8094, Δ=+0.3428
- **curv_plan_output_rmse**: pre median=0.0000, post median=0.0001, Δ=+0.0000

## 4. Post-Takeover Stability

- **N with stabilization data:** 2,390
- **Median stabilization time:** 1.35 s
- **P95 stabilization time:** 3.65 s
- **Censored (not stabilized in 5s):** 13,315 (84.8%)
- **driver_onset_steer_s**: n=11,406, median=0.20 s
- **driver_onset_brake_s**: n=8,410, median=0.00 s
- **driver_onset_gas_s**: n=8,570, median=0.20 s
- **Pressed duty (post):** median=96.00%

## 5. Interaction Flags

- **Lead inconsistency (>10%):** 2.8% of clips
- **Low lane + high curv. mismatch:** 14.2% of clips
- **Plan-output accel mismatch outlier (>P90):** 10.0% of clips

## 6. Limitations

- No radar-derived TTC/THW (radarState excluded from this analysis).
- Safety proxies (hasLead, leadVisible, aTarget) are indirect; they reflect planner intent rather than physical headway.
- Curvature mismatch uses controlsState desired vs actual curvature, which may differ in meaning across vehicle platforms.
- Stabilization metric is sensitive to smoothing parameters and sample rate (qlog 10 Hz vs rlog 100 Hz).
- RMSE metrics depend on temporal alignment via linear interpolation at 20 Hz; aliasing may affect high-frequency dynamics.
