# Data Dictionary: Stage 1 Derived Signals (v3)

All columns in `derived_signals_v3.parquet` / `derived_signals_v3.csv`.

Prefixes: `pre_` = pre-takeover window [-5, 0] s; `post_` = post-takeover
window [0, 5] s.

---

## Identifier Columns

| Column | Type | Description |
|--------|------|-------------|
| `car_model` | str | Vehicle model (e.g., "ACURA_INTEGRA") |
| `dongle_id` | str | Unique comma device ID (driver proxy) |
| `route_id` | str | Route identifier within the dongle |
| `clip_id` | int | Clip index within the route |
| `log_kind` | str | "rlog" (100 Hz native) or "qlog" (10 Hz native) |
| `log_hz` | int | Native carState sample rate (100 or 10) |
| `resample_hz` | int | Common grid frequency used (always 20) |

---

## Safety Proxies (per window: `pre_` / `post_`)

### Time Headway (THW)

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_thw_min_s` | s | Minimum THW in window | NaN if no lead or vEgo < 0.5 m/s |
| `{pre/post}_thw_p5_s` | s | 5th percentile of THW | NaN if < 2 valid THW samples |
| `{pre/post}_thw_p50_s` | s | Median THW | NaN if < 2 valid THW samples |
| `{pre/post}_thw_p95_s` | s | 95th percentile of THW | NaN if < 2 valid THW samples |

**Formula**: THW = dRel / vEgo (when leadOne.status = True and vEgo > 0.5 m/s)

### Time to Collision (TTC)

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_ttc_min_raw_s` | s | Minimum raw TTC (uncapped) | NaN if no closing approach |
| `{pre/post}_ttc_min_capped_s` | s | Minimum TTC capped at 100 s | NaN if no closing approach |
| `{pre/post}_ttc_p5_s` | s | 5th percentile of capped TTC | NaN if < 2 valid TTC samples |
| `{pre/post}_ttc_p50_s` | s | Median capped TTC | NaN if < 2 valid TTC samples |
| `{pre/post}_ttc_p95_s` | s | 95th percentile of capped TTC | NaN if < 2 valid TTC samples |
| `{pre/post}_time_of_min_ttc_s` | s | Time of min TTC relative to t=0 | NaN if no valid TTC |

**Formula**: TTC = dRel / max(-vRel, ε) (only when vRel < -0.5 m/s, i.e., closing)

### Deceleration Rate to Avoid Crash (DRAC)

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_drac_max_raw_mps2` | m/s² | Maximum raw DRAC (uncapped) | NaN if no closing + dRel > 5m |
| `{pre/post}_drac_max_capped_mps2` | m/s² | Maximum DRAC capped at 50 m/s² | NaN if no closing + dRel > 5m |
| `{pre/post}_drac_p50_mps2` | m/s² | Median DRAC | NaN if < 2 valid samples |
| `{pre/post}_drac_p95_mps2` | m/s² | 95th percentile of DRAC | NaN if < 2 valid samples |
| `{pre/post}_time_of_max_drac_s` | s | Time of max DRAC relative to t=0 | NaN if no valid DRAC |

**Formula**: DRAC = vRel² / (2 × dRel) (only when |vRel| > 0.5 m/s and dRel > 5.0 m)

### Lead Vehicle Proximity

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_min_drel_m` | m | Minimum following distance | NaN if no lead present |
| `{pre/post}_p5_drel_m` | m | 5th percentile of dRel | NaN if no lead present |
| `{pre/post}_lead_present_rate` | ratio | Fraction of samples with lead present | NaN if radar topic missing |
| `{pre/post}_n_lead_samples` | count | Number of samples with lead present | 0 if no lead |
| `{pre/post}_lead_drop_count` | count | Number of True→False lead transitions | NaN if no lead |
| `{pre/post}_longest_cont_lead_s` | s | Longest continuous lead tracking segment | 0 if no lead |

---

## Exposure Duration Features (per window)

Total time (seconds) within the window that the metric crosses the threshold.

### TTC exposure

| Column | Unit | Description |
|--------|------|-------------|
| `{pre/post}_time_below_ttc_1.5s` | s | Time with TTC < 1.5 s |
| `{pre/post}_time_below_ttc_2.0s` | s | Time with TTC < 2.0 s |
| `{pre/post}_time_below_ttc_3.0s` | s | Time with TTC < 3.0 s |

### TTC severity integrals

| Column | Unit | Description |
|--------|------|-------------|
| `{pre/post}_severity_integral_ttc_1.5s` | s² | ∫ max(1.5 - TTC, 0) dt |
| `{pre/post}_severity_integral_ttc_2.0s` | s² | ∫ max(2.0 - TTC, 0) dt |
| `{pre/post}_severity_integral_ttc_3.0s` | s² | ∫ max(3.0 - TTC, 0) dt |

### THW exposure

| Column | Unit | Description |
|--------|------|-------------|
| `{pre/post}_time_below_thw_0.8s` | s | Time with THW < 0.8 s |
| `{pre/post}_time_below_thw_1.0s` | s | Time with THW < 1.0 s |
| `{pre/post}_time_below_thw_1.5s` | s | Time with THW < 1.5 s |

### DRAC exposure

| Column | Unit | Description |
|--------|------|-------------|
| `{pre/post}_time_above_drac_3.0mps2` | s | Time with DRAC > 3.0 m/s² |
| `{pre/post}_time_above_drac_4.0mps2` | s | Time with DRAC > 4.0 m/s² |

---

## Ego Dynamics (per window)

### Acceleration

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_min_accel_mps2` | m/s² | Minimum acceleration (most negative = hardest braking) | NaN if carState missing |
| `{pre/post}_max_accel_mps2` | m/s² | Maximum acceleration | NaN if carState missing |
| `{pre/post}_accel_p5_mps2` | m/s² | 5th percentile of acceleration | NaN |
| `{pre/post}_accel_p50_mps2` | m/s² | Median acceleration | NaN |
| `{pre/post}_accel_p95_mps2` | m/s² | 95th percentile of acceleration | NaN |
| `{pre/post}_time_of_peak_decel_s` | s | Time of hardest braking relative to t=0 | NaN |

### Jerk

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_max_abs_jerk_mps3` | m/s³ | Maximum |jerk| (SavGol-smoothed then differentiated) | NaN if < 5 valid accel samples |
| `{pre/post}_jerk_p50_mps3` | m/s³ | Median |jerk| | NaN |
| `{pre/post}_jerk_p95_mps3` | m/s³ | 95th percentile of |jerk| | NaN |
| `{pre/post}_time_of_peak_jerk_s` | s | Time of peak |jerk| relative to t=0 | NaN |

**Processing**: aEgo → SavGol(window=0.3s, polyorder=2) → diff / (1/20 Hz)

### Speed

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_speed_mean_mps` | m/s | Mean ego speed in window | NaN if vEgo missing |
| `{pre/post}_speed_delta_mps` | m/s | Speed at window end minus window start | NaN if < 2 valid speed samples |

### Steering

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_max_abs_steer_angle_deg` | ° | Maximum |steeringAngleDeg| | NaN if missing |
| `{pre/post}_max_abs_steer_torque` | Nm | Maximum |steeringTorque| | NaN if missing |
| `{pre/post}_steer_rate_max_deg_per_s` | °/s | Maximum |d(steeringAngle)/dt| (smoothed) | NaN if < 5 valid samples |
| `{pre/post}_steer_rate_p95_deg_per_s` | °/s | 95th percentile of |steering rate| | NaN |
| `{pre/post}_time_of_peak_steer_rate_s` | s | Time of peak steering rate relative to t=0 | NaN |

### Curvature

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_max_abs_curvature` | 1/m | Maximum |curvature| from controlsState | NaN if field missing |
| `{pre/post}_max_abs_desired_curvature` | 1/m | Maximum |desiredCurvature| | NaN if field missing |

---

## Roughness (per window)

Computed on **native** accelerometer timestamps (not resampled).

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_roughness_rms_mps2` | m/s² | RMS of detrended acceleration norm ‖a‖ | NaN if accelerometer missing or < 3 samples |
| `{pre/post}_roughness_pp_mps2` | m/s² | Peak-to-peak of detrended ‖a‖ | NaN |
| `{pre/post}_roughness_z_rms_mps2` | m/s² | RMS of detrended z-axis acceleration | NaN |
| `{pre/post}_accel_native_hz` | Hz | Estimated native accelerometer rate (1/median Δt) | NaN if < 2 samples |

---

## Alerts and FCW (per window)

| Column | Type | Description | Missingness |
|--------|------|-------------|-------------|
| `{pre/post}_fcw_present` | bool | Forward Collision Warning present | False if missing |
| `{pre/post}_fcw_source` | str | "explicit" (from fcw field), "alert_text" (inferred), or "none" | "none" |
| `{pre/post}_alert_present` | bool | Any non-empty alertText1 in window | False if missing |
| `{pre/post}_alert_text` | str | Last non-empty alertText1 | "" if none |
| `{pre/post}_has_lane_probs` | bool | Whether lane probability fields exist | False |
| `{pre/post}_lane_left_prob_mean` | ratio | Mean left lane line probability | NaN if not available |
| `{pre/post}_lane_right_prob_mean` | ratio | Mean right lane line probability | NaN if not available |

---

## Stabilization (clip-level)

| Column | Unit | Description | Missingness |
|--------|------|-------------|-------------|
| `stabilization_5s_time_s` | s | Time to first stable 1.0s window in [0, 5] s | NaN if carState missing; right-censored at 5.0 |
| `stabilization_5s_censored` | bool | True if threshold never met within 5 s | Always present |
| `stabilization_10s_time_s` | s | Time to first stable 1.0s window in [0, 10] s | NaN; right-censored at 10.0 |
| `stabilization_10s_censored` | bool | True if threshold never met within 10 s | Always present |

**Stable** = |aEgo| < 0.5 m/s² AND |jerk| < 1.0 m/s³ sustained for ≥ 1.0 s.

---

## Post-Maneuver Classification (clip-level)

| Column | Type | Description |
|--------|------|-------------|
| `post_maneuver_type` | str | One of: "lane_change", "turn_ramp", "acceleration", "braking", "stabilize" |

Decision rules (first match wins):
1. **lane_change**: post steer_rate > 20°/s AND post steer_angle > 15°
2. **turn_ramp**: post steer_angle > 30°
3. **acceleration**: post speed_delta > 2.0 m/s
4. **braking**: post min_accel < -2.0 m/s²
5. **stabilize**: default

---

## Anomaly Flags (clip-level)

Also saved separately to `anomaly_flags.csv`.

| Column | Type | Condition |
|--------|------|-----------|
| `anomaly_jerk_extreme` | bool | pre_max_abs_jerk > 50 m/s³ |
| `anomaly_steer_rate_extreme` | bool | pre_steer_rate_max > 200 °/s |
| `anomaly_ttc_tail` | bool | pre_ttc_min_raw > 200 s |
| `anomaly_drac_tail` | bool | pre_drac_max_raw > 100 m/s² |
| `anomaly_lead_dropout` | bool | pre_lead_drop_count > 10 |
| `anomaly_stabilization_censored` | bool | Both 5s and 10s stabilization censored |
| `anomaly_nan_rate_high` | bool | > 50% of key sentinel columns are NaN |
| `anomaly_any` | bool | OR of all flags above |
| `anomaly_reason` | str | Semicolon-separated list of triggered flag names |

---

## Anomaly Flags CSV (`anomaly_flags.csv`)

Contains identifier columns (`car_model`, `dongle_id`, `route_id`, `clip_id`,
`log_kind`, `log_hz`) plus all `anomaly_*` columns from above.

---

## Notes on Missingness

- **NaN** indicates the feature could not be computed for the clip (missing
  topic, insufficient data, or guard condition not met).
- **0.0** for exposure durations indicates the threshold was never crossed
  (the metric was always in the "safe" range during the window).
- **False** for boolean flags indicates the condition was not detected.
- Features are computed independently; a NaN in one feature does not affect others.
- The `log_kind` and `log_hz` columns support post-hoc qlog vs rlog sensitivity
  stratification.
