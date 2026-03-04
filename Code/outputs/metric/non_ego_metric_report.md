# Non-Ego Takeover Metrics Report

## 1. Dataset Overview

- **Total Non-ego clips analysed**: 12683
- **Lead-present clips** (hasLead > 50% pre-window): 5796 (45.7%)

### Missingness rates

| Source | Missing % |
|--------|--------:|
| model | 34.1% |
| lplan | 0.4% |
| state | 0.3% |
| ctrl | 0.2% |
| output | 23.9% |

### Metric availability

| Metric | Non-missing | % |
|--------|----------:|----:|
| lane_dev | 8228 | 64.9% |
| laneProb_min | 8228 | 64.9% |
| laneProb_min_pre | 8260 | 65.1% |
| decel_demand_pre | 12409 | 97.8% |
| planned_speed_drop | 12221 | 96.4% |
| jerk_max_post | 12320 | 97.1% |
| steer_rate_max_post | 12320 | 97.1% |
| a_lat_max_post | 9512 | 75.0% |
| lat_jerk_max_post | 9512 | 75.0% |
| peak_decel_post | 12320 | 97.1% |
| stabilization_time | 4191 | 33.0% |
| rmse_accel_plan_output_pre | 3310 | 26.1% |
| rmse_accel_plan_output_post | 1921 | 15.1% |
| rmse_curv_plan_output_pre | 9208 | 72.6% |
| rmse_curv_plan_output_post | 9316 | 73.5% |
| rmse_accel_output_state_pre | 9539 | 75.2% |
| rmse_accel_output_state_post | 9500 | 74.9% |

## 2. Key Metric Summaries

| Metric | Median | IQR (Q1–Q3) | P5 | P95 |
|--------|-------:|:-----------:|---:|----:|
| lane_dev | 0.014 | -0.156–0.188 | -0.506 | 0.614 |
| laneProb_min | 0.636 | 0.137–0.957 | 0.016 | 0.993 |
| laneProb_min_pre | 0.388 | 0.072–0.908 | 0.009 | 0.988 |
| decel_demand_pre | 0.230 | 0.000–0.919 | 0.000 | 2.006 |
| planned_speed_drop | 0.829 | 0.225–2.660 | 0.000 | 5.037 |
| jerk_max_post | 4.465 | 3.026–6.844 | 1.718 | 13.726 |
| steer_rate_max_post | 18.501 | 8.489–42.657 | 2.263 | 204.793 |
| a_lat_max_post | 0.327 | 0.102–0.929 | 0.000 | 2.683 |
| lat_jerk_max_post | 0.874 | 0.349–1.861 | 0.000 | 4.458 |
| peak_decel_post | -1.348 | -2.152–-0.630 | -3.453 | -0.002 |
| stabilization_time | 3.809 | 1.576–4.761 | 0.064 | 4.918 |
| rmse_accel_plan_output_pre | 0.027 | 0.006–0.154 | 0.000 | 1.278 |
| rmse_accel_plan_output_post | 0.038 | 0.007–0.149 | 0.001 | 0.810 |
| rmse_curv_plan_output_pre | 0.000 | 0.000–0.000 | 0.000 | 0.001 |
| rmse_curv_plan_output_post | 0.000 | 0.000–0.000 | 0.000 | 0.002 |
| rmse_accel_output_state_pre | 0.443 | 0.200–0.894 | 0.076 | 2.059 |
| rmse_accel_output_state_post | 0.792 | 0.423–1.339 | 0.175 | 2.253 |

## 3. Key Findings

1. **Lane deviation at takeover**: Median |lane_dev| = 0.171 m (IQR 0.077–0.327). The distribution is roughly symmetric around zero, suggesting no systematic lateral bias at Non-ego takeover onset.

2. **Lane confidence → lateral urgency**: Non-ego clips with laneProb < 0.1 show median steer rate 35.5 °/s versus 9.6 °/s for laneProb > 0.9 (3.7× ratio).

3. **Longitudinal conflict proxy (lead-present)**: Median decel demand = 0.29 m/s² (n = 5796). This reflects planner intent; true TTC/THW require radar distance data which is excluded from this analysis.

4. **Post-takeover lateral stability**: Median steer rate = 18.5 °/s; P95 = 204.8 °/s.

5. **Post-takeover longitudinal stability**: Median jerk = 4.5 m/s³; P95 = 13.7 m/s³.

6. **Stabilization time**: Median = 3.81 s. Censored (not stabilized within 5.0s): 67.0%.

7. **Plan→output accel RMSE**: Pre median = 0.027, Post median = 0.038. Increase post-takeover reflects higher control mismatch during driver intervention.


## 4. Limitations

1. **No radar-derived TTC/THW**: This analysis does not use radarState.csv. The longitudinal conflict metrics (decel demand, planned speed drop) are planner-intent proxies, not physical headway measures. True TTC and THW require lead-vehicle distance (dRel) from radar.

2. **Lane model availability**: drivingModelData is missing for 34.1% of clips, limiting the lateral deviation analysis to clips with active lane detection.

3. **Timing alignment**: Event time is taken from meta.json (video_time_s). Minor misalignment between CAN-bus timestamps and video clock may affect metrics computed at the exact takeover instant.

4. **Sampling rate heterogeneity**: The dataset includes both qlog (~10 Hz) and rlog (~100 Hz) recordings. Derivative-based metrics (jerk, steer rate) are smoothed (Savitzky–Golay, 0.3s window) to mitigate rate-dependent artifacts, but residual sensitivity remains.

5. **Curvature mismatch winsorization**: plan→output curvature RMSE is winsorized at P99 to remove extreme outliers likely reflecting instrumentation artifacts.

6. **Classification accuracy**: The Ego/Non-ego partition has 84.0% validated accuracy. Approximately 16% of clips may be mislabeled, which could attenuate observed associations.
