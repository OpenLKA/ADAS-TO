# Stage 1 Methods: Derived Signal Computation (v3)

## Overview

Stage 1 computes per-clip safety proxies and driving-behavior features from
the raw comma/openpilot topic CSVs. Each 20-second clip is centered at the
takeover event (t = 0). Features are computed for a **pre-takeover window**
[-5, 0] s and a **post-takeover window** [0, 5] s, plus clip-level metrics
that span [0, 10] s for stabilization analysis.

The pipeline processes all 15,659 clips (327 drivers, 163 car models) using
`ProcessPoolExecutor` with 12 workers. Outputs are saved as Parquet
(primary) and CSV (convenience).

---

## 1. Resampling to a Common Time Grid

### Motivation

Raw topic CSVs are recorded at heterogeneous native sample rates:

| Topic | rlog (~100 Hz) | qlog (~10 Hz) |
|-------|---------------|---------------|
| carState | ~100 Hz (2000 samples/20s) | ~10 Hz (200 samples/20s) |
| radarState | ~20 Hz (400 samples/20s) | ~4 Hz (80 samples/20s) |
| controlsState | ~100 Hz | ~10 Hz |
| longitudinalPlan | ~20 Hz | ~4 Hz |
| drivingModelData | ~20 Hz | ~4 Hz |
| accelerometer | ~100 Hz (rlog) | **~1 Hz (qlog)** |

Derivative features (jerk, steering rate) computed on native timestamps
would be systematically lower in magnitude for qlog clips (10 Hz vs 100 Hz)
because of the coarser time step. Resampling to a common grid removes this
confound.

### Common Grid: 20 Hz

We chose **20 Hz** as the common resampling frequency because:

1. It is the highest rate shared by all topics in rlog (radar = 20 Hz,
   longitudinalPlan = 20 Hz, drivingModelData = 20 Hz).
2. Resampling above 20 Hz would fabricate information for any topic with a
   native rate below the target (e.g., qlog radar at 4 Hz, qlog
   accelerometer at ~1 Hz).
3. 20 Hz provides 0.05 s resolution — sufficient for safety-proxy
   computation (TTC, THW) and derivative estimation (jerk at ~20 Hz is
   standard in vehicle dynamics literature).

### Interpolation Methods

| Signal type | Method | Rationale |
|-------------|--------|-----------|
| Continuous (vEgo, aEgo, dRel, vRel, steeringAngleDeg, ...) | Linear interpolation (`numpy.interp`) | Preserves signal shape; no extrapolation beyond data bounds (fills NaN) |
| Boolean (leadOne.status, steeringPressed, brakePressed, ...) | Forward-fill via `pandas.merge_asof(direction='backward')` | Last known state is carried forward; no interpolation artifacts |
| Categorical (alertText1, ...) | Forward-fill | Same as boolean |

### Exception: Accelerometer

The accelerometer topic is **not resampled**. In qlog, it records at ~1 Hz
(~20 samples in 20 s). Upsampling 1 Hz data to 20 Hz would fabricate 95% of
samples. Instead, roughness is computed directly on native accelerometer
timestamps.

---

## 2. Derivative Computation

All derivative features (jerk, steering rate) are computed as:

1. **Smooth** the signal with a Savitzky-Golay filter (window = 0.3 s → 7
   samples at 20 Hz, polyorder = 2) to suppress high-frequency noise.
2. **Differentiate** using a constant Δt = 1/20 Hz = 0.05 s (from the
   resampled grid), ensuring comparability across clips.

This avoids the variable-Δt problem that plagues native-rate differentiation
when sample intervals are irregular.

---

## 3. Safety Proxies

### Time Headway (THW)

```
THW = dRel / vEgo       [seconds]
```

Computed only when:
- Lead vehicle is present (`leadOne.status == True`)
- Ego speed > 0.5 m/s (avoids huge THW at near-zero speed)

### Time to Collision (TTC)

```
TTC = dRel / (-vRel)    [seconds, closing only]
```

Computed only when:
- Lead vehicle is present
- Closing speed |vRel| > 0.5 m/s (avoids extreme TTC from tiny speed diffs)
- Capped at 100 s to suppress long-tail artifacts

Both raw and capped variants are stored (`ttc_min_raw_s`, `ttc_min_capped_s`).

### Deceleration Rate to Avoid Crash (DRAC)

```
DRAC = vRel² / (2 · dRel)    [m/s², closing only]
```

Computed only when:
- Closing speed > 0.5 m/s
- dRel > 5.0 m (avoids blow-up at very close range)
- Capped at 50 m/s²

### Exposure Durations

For each threshold, we compute the total time (seconds) within the window
that the metric exceeds/falls below the threshold:

- TTC thresholds: 1.5, 2.0, 3.0 s (time *below*)
- THW thresholds: 0.8, 1.0, 1.5 s (time *below*)
- DRAC thresholds: 3.0, 4.0 m/s² (time *above*)

### Severity Integrals

For TTC, the severity integral captures "how much and how long" TTC was
below threshold:

```
severity_integral = Σ max(threshold - TTC_i, 0) × Δt
```

### Lead Continuity

- `lead_present_rate`: fraction of window samples where lead is tracked
- `lead_drop_count`: number of True→False transitions in lead status
- `longest_cont_lead_s`: duration of the longest continuous lead segment

---

## 4. Ego Dynamics

- **Acceleration**: min, max, quantiles (P5, P50, P95)
- **Jerk**: max |jerk|, P50, P95 of |jerk| (after SavGol smoothing)
- **Steering rate**: max |dSteeringAngle/dt|, P95 (after SavGol smoothing)
- **Steering angle**: max |steeringAngleDeg|
- **Steering torque**: max |steeringTorque|
- **Speed**: mean vEgo, speed delta (end - start of window)
- **Curvature**: max |curvature|, max |desiredCurvature| (from controlsState)

### Time-to-Extrema

For key metrics (peak deceleration, peak jerk, peak steering rate), the
**time of the extremum** is recorded relative to t = 0 (seconds). This
supports temporal clustering analyses.

---

## 5. Roughness (Road Surface Proxy)

Computed on **native accelerometer timestamps** (not resampled).

- Parse the 3-axis vector (`acceleration.v = [x, y, z]`)
- Compute acceleration norm: ||a|| = √(x² + y² + z²)
- Detrend: subtract window mean from the norm
- **RMS roughness**: √(mean(detrended²))
- **Peak-to-peak roughness**: max(detrended) - min(detrended)
- **Z-axis RMS**: RMS of detrended z-axis only (secondary metric)
- **Native Hz**: estimated from median(Δt) of accelerometer timestamps

Using the norm rather than z-axis alone is robust to unknown device
orientation.

---

## 6. Alerts and FCW

- **FCW present**: True if `longitudinalPlan.fcw == True` OR alert text
  contains FCW-related keywords ("forward collision", "brake!", "fcw")
- **FCW source**: "explicit" (from `fcw` field) or "alert_text" (inferred)
- **Alert present**: any non-empty `alertText1`
- **Last alert text**: the last non-empty alert in the window

---

## 7. Post-Takeover Stabilization Time

Defined as the first continuous Δ-second window in [0, max_s] where:
- |accel| < a₀ = 0.5 m/s²  AND
- |jerk| < j₀ = 1.0 m/s³

Computed for two windows:
- **Short**: max_s = 5.0 s (primary)
- **Long**: max_s = 10.0 s (extended search)

If the threshold is never met, the stabilization time is **right-censored**
at max_s and the `_censored` flag is True.

---

## 8. Post-Maneuver Classification

Based on post-window features, each clip receives a `post_maneuver_type`:

| Type | Rule |
|------|------|
| lane_change | steer_rate > 20°/s AND steer_angle > 15° |
| turn_ramp | steer_angle > 30° |
| acceleration | speed_delta > 2.0 m/s |
| braking | min_accel < -2.0 m/s² |
| stabilize | default (none of the above) |

---

## 9. Anomaly Flags

Each clip receives boolean anomaly flags for downstream triage:

| Flag | Condition |
|------|-----------|
| `anomaly_jerk_extreme` | pre_max_abs_jerk > 50 m/s³ |
| `anomaly_steer_rate_extreme` | pre_steer_rate_max > 200 °/s |
| `anomaly_ttc_tail` | pre_ttc_min_raw > 200 s |
| `anomaly_drac_tail` | pre_drac_max_raw > 100 m/s² |
| `anomaly_lead_dropout` | pre_lead_drop_count > 10 |
| `anomaly_stabilization_censored` | both 5s and 10s censored |
| `anomaly_nan_rate_high` | > 50% of key columns are NaN |
| `anomaly_any` | OR of all above |

Anomaly flags are saved to a separate `anomaly_flags.csv` for manual review
and optional YOLO scene tagging.

---

## 10. YOLO Scene Tagging (Optional)

For clips flagged as anomalous or labeled as ambiguous scenarios
(`intersection_odd`, `uncertain_mixed`), keyframes are extracted from
`takeover.mp4` at t = {-3, -2, -1, 0, +1} s and processed with YOLOv8n:

- Target classes: traffic light, stop sign, person, bicycle, car, bus, truck
- Confidence threshold: 0.35
- Output: per-keyframe detections + per-clip presence summary

This is disabled by default (`yolo.enabled: false` in config) and requires
`pip install ultralytics opencv-python-headless`.

---

## 11. Configuration

All thresholds, windows, and parameters are centralized in
`configs/analysis_thresholds.yaml`. Key parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| resample_hz | 20 | Common grid frequency |
| pre_window | [-5, 0] s | Pre-takeover |
| post_window | [0, 5] s | Post-takeover |
| full_window | [-10, 10] s | Clip extent |
| closing_speed_min | 0.5 m/s | TTC/DRAC floor |
| drel_min | 5.0 m | DRAC floor |
| ttc_cap | 100 s | TTC ceiling |
| drac_cap | 50 m/s² | DRAC ceiling |
| savgol_window | 0.3 s | Smoothing before differentiation |
| savgol_polyorder | 2 | Polynomial order |
| stabilization sustain | 1.0 s | Required stable duration |

---

## 12. Outputs

| File | Format | Description |
|------|--------|-------------|
| `derived_signals_v3.parquet` | Parquet | Primary output (~120 columns per clip) |
| `derived_signals_v3.csv` | CSV | Convenience copy |
| `anomaly_flags.csv` | CSV | Anomaly boolean flags for triage |
| `yolo_scene_tags.csv` | CSV | Per-keyframe YOLO detections (optional) |
| `yolo_clip_summary.csv` | CSV | Per-clip YOLO presence flags (optional) |
