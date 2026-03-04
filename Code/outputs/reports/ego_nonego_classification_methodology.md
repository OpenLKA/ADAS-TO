# Ego vs. Non-Ego Takeover Classification: Methodology and Validation

## 1. Overview

This document describes the rule-based binary classification system (version 7) used to label each takeover clip in the OpenLKA TakeOver dataset as **Ego** (driver-initiated) or **Non-ego** (reactive/system-initiated). The classifier operates on per-clip features extracted from in-vehicle CAN-bus signals (carState, carControl, carOutput, controlsState) and upstream analysis metrics (analysis_master.csv). Out of 15,659 valid clips, the classifier assigns **2,976 (19.0%)** as Ego and **12,683 (81.0%)** as Non-ego.

The classifier was validated by **four human experts** on a stratified random sample of **500 clips**, achieving an overall accuracy of **84.0%**.

---

## 2. Problem Formulation

A *takeover event* occurs when the human driver disengages the Level-2 Advanced Driver Assistance System (ADAS)---specifically, openpilot---by pressing the steering wheel, brake pedal, or gas pedal. We define two categories:

| Category | Definition | Examples |
|----------|-----------|----------|
| **Ego** | The driver *intentionally initiates* the takeover to execute a planned maneuver. The takeover is *not* a response to an imminent safety-critical situation. | Lane change with blinker, junction turn from near-stop, discretionary acceleration, stationary departure |
| **Non-ego** | The takeover is *reactive*: the driver responds to a conflict, system limitation, road geometry challenge, or system-initiated disengagement. | Lead-vehicle conflict (TTC/THW/DRAC breach), FCW response, curve boundary, system alert |

This distinction is critical for downstream safety analysis because Ego and Non-ego takeovers have fundamentally different risk profiles, driver preparedness levels, and post-takeover dynamics.

---

## 3. Input Data and Feature Extraction

### 3.1 Data Sources

For each clip, the classifier consumes:

1. **analysis_master.csv** --- pre-computed per-clip features from the upstream pipeline (15,659 rows, 180+ columns), including:
   - Safety metrics: `pre_ttc_min_capped_s`, `pre_thw_min_s`, `pre_drac_max_capped_mps2`
   - Trigger flags: `trig_steer`, `trig_brake`, `trig_gas`, `primary_trigger`
   - Scenario labels: `scenario`, `post_maneuver_type`
   - Risk scores: `risk_score`, `flag_longitudinal_conflict`
   - Steering/speed statistics: `pre_speed_mean_mps`, `post_max_abs_steer_angle_deg`, etc.
   - Alert text: `pre_alert_text`

2. **Per-clip raw CSV files** (extracted from openpilot logs):
   - `carState.csv`: `steeringAngleDeg`, `vEgo`, `steeringPressed`, `gasPressed`, `brakePressed`
   - `carControl.csv`: `leftBlinker`, `rightBlinker`
   - `controlsState.csv`: `curvature`, `desiredCurvature`

### 3.2 Time Windows

The classifier uses three time windows relative to the takeover event at $t = 0$:

| Window | Range | Purpose |
|--------|-------|---------|
| Pre-event | $[-3, 0]$ s | Assess driving context before takeover |
| Post-short | $[0, +5]$ s | Tight confirmation of maneuver dynamics |
| Post-long | $[0, +10]$ s | Catch late-onset maneuvers (e.g., slow junction turns) |

### 3.3 Extracted Features (Per Clip)

A total of **42 raw features** are extracted from the per-clip CSVs using parallel processing (12 workers). Key feature groups:

**Blinker signals** (8 features):
- `blinker_any_pre`, `blinker_any_post`, `blinker_left_pre`, `blinker_right_pre`, etc.
- `blinker_pre_duration_s`, `blinker_post_duration_s` --- total duration of blinker activation

**Steering waveform** (dual-window, 16 features):
- `post{5,10}_steer_peak_deg` --- peak absolute steering angle
- `post{5,10}_dur_abs_steer_gt10_s`, `post{5,10}_dur_abs_steer_gt20_s` --- duration above threshold
- `post{5,10}_sign_changes` --- number of steering direction reversals (deadband = 1.0 deg)
- `post{5,10}_one_sided_ratio` --- $|\sum \theta| / \sum |\theta|$; high values indicate sustained one-directional steering (curve-following) vs. S-shaped lane-change pattern
- `steer_return` --- boolean: does the steering angle return to within 40% of peak?
- `steer_return_ratio` --- ratio of end-of-window steering to peak
- `post10_steer_mean_sign` --- mean signed steering for left/right turn detection

**Curvature** (dual-window, 12 features):
- `post{5,10}_max_abs_curv`, `post{5,10}_mean_abs_curv`
- `post{5,10}_dur_strong_curv_s` --- duration where $|\kappa| > 0.02$
- `post{5,10}_max_curv_deviation` --- max $|\kappa_{\text{desired}} - \kappa_{\text{actual}}|$

**Speed** (6 features):
- `pre_speed_min_mps`, `post{5,10}_speed_min_mps`
- `post{5,10}_speed_delta_mps` --- speed change over window

### 3.4 Event Time Alignment

To ensure precise temporal alignment, the classifier uses **monotonic clock alignment** (`logMonoTime` from the log vs. `event_mono` from `meta.json`) rather than relying solely on `video_time_s`. This avoids drift between video timestamps and CAN-bus timestamps. If monotonic alignment fails, the classifier falls back to `video_time_s`.

---

## 4. Detector Architecture

The classifier employs **seven detectors** (A--G), each targeting a specific takeover pattern. Three detectors produce **Ego** evidence, three produce **Non-ego** evidence, and one (Detector G) is a special-case stationary detector.

### 4.1 Ego Detectors

#### Detector G: Stationary Takeover (Highest Priority)

**Rationale**: If the vehicle is essentially stopped ($v < 0.5$ m/s) before the event, the driver intentionally took over (e.g., at a red light, stop sign, parking lot departure).

**Conditions** (all must hold):
- `pre_speed_mean_mps` < 0.5 m/s **AND** `pre_speed_min_mps` < 0.5 m/s
- At least one post-event evidence of driver action:
  - Steering peak > 3 deg, OR
  - Speed increase > 0.5 m/s, OR
  - Gas trigger active, OR
  - Steering trigger active

**Confidence**: Always *high*.

#### Detector A: Lane Change (3 sub-rules)

**Rationale**: Lane changes are intentional maneuvers, identifiable by blinker activation, S-shaped steering profiles, and curvature patterns consistent with lateral displacement.

**Sub-rule A1** (high confidence) --- Blinker + LC dynamics:
- Blinker active for $\geq$ 0.5 s (pre or post window)
- LC maneuver type OR LC dynamics in 5s/10s window:
  - Peak steering $\geq$ 8 deg
  - Not one-sided: $|\sum \theta| / \sum |\theta| \leq 0.92$
  - Duration of $|\theta| > 20$ deg $\leq$ 3.5 s
  - Steering sign changes $\geq 1$ OR steer return detected
- Curvature gate: duration of strong curvature $\leq$ 1.5 s AND mean $|\kappa| \leq$ 0.015
- NOT curve override (see below)

**Sub-rule A2** (medium confidence) --- Planned LC without blinker:
- `scenario == planned_lane_change` AND `post_maneuver_type == lane_change`
- No blinker, but: pre-steering straight ($|\theta| < 20$ deg sustained < 0.75 s), steer return detected, visible steering ($\geq 8$ deg), curvature OK

**Sub-rule A3** (medium confidence, new in v7) --- Conflict-triggered LC:
- `post_maneuver_type == lane_change` with good LC dynamics (steer return, visible steering, curvature OK)
- Not captured by A1 or A2 (no blinker, not `planned_lane_change` scenario)
- Catches cases where the driver initiates a lane change in response to a perceived (but not safety-critical) situation

**Curve Override** (prevents false positive LC labels on highway curves):
- Speed > 10 m/s AND sustained one-sided steering ($> 2.5$ s at $|\theta| > 20$ deg, one-sided ratio $> 0.93$) OR pre-existing curvature ($\bar{|\kappa|}_{\text{pre}} > 0.003$)
- If curve override fires, the clip is NOT labeled as lane change

#### Detector B: Junction / Intersection Turn (4 sub-rules)

**Rationale**: Junction turns involve intentional deceleration to near-stop followed by a sustained turn. A key insight distinguishes junction turns from sharp road curves: **sharp curves maintain speed above ~4.5 m/s (10 mph), while junction turns decelerate to near-stop**.

**Sub-rule B1** (high confidence) --- Near-stop + steering evidence:
- `pre_speed_mean_mps` < 3.0 m/s ($\approx$ 6.7 mph)
- Visible steering ($\geq 15$ deg) OR curvature evidence OR sustained steering

**Sub-rule B2** (medium confidence) --- Turn/ramp maneuver at low speed:
- `post_maneuver_type == turn_ramp` AND speed < 8.0 m/s
- Curvature evidence: strong curvature duration $\geq 1.0$ s AND peak $|\kappa| \geq 0.02$

**Sub-rule B3** (medium confidence) --- Low speed + blinker + steering:
- Speed < 8.0 m/s AND blinker active AND (visible steering OR curvature evidence)

**Sub-rule B4** (low confidence) --- Left turn detection:
- Speed < 8.0 m/s AND (left blinker OR mean steering sign > 5 deg left with peak > 20 deg)
- Left turns are almost always intentional (Ego) since they require yielding to oncoming traffic

#### Detector C: Discretionary Acceleration

**Rationale**: When the driver presses the gas pedal to accelerate beyond the ADAS set speed in a safe, straight-road context, the takeover is intentional.

**Conditions** (all must hold):
- `post_maneuver_type == acceleration`
- Speed increase > 2.0 m/s
- Gas trigger active
- Low risk score (< 0.3)
- No longitudinal conflict flag
- Safe scenario (`planned_acceleration` or `discretionary`)
- Straight steering (< 4 deg) AND low curvature ($\bar{|\kappa|} < 0.003$)

### 4.2 Non-Ego Detectors

#### Detector D: Conflict / Reactive

**Indicators** (any one triggers the flag):

| Indicator | Threshold | Source |
|-----------|-----------|--------|
| TTC (time-to-collision) | < 1.5 s | `pre_ttc_min_capped_s` |
| THW (time headway) | < 0.8 s | `pre_thw_min_s` |
| DRAC (deceleration rate to avoid crash) | > 3.0 m/s$^2$ | `pre_drac_max_capped_mps2` |
| FCW (forward collision warning) | present | `pre_fcw_present` |
| Close lead vehicle | present > 30% of pre-window AND $d_{\text{rel}} < 10$ m | `pre_lead_present_rate`, `pre_min_drel_m` |

**Confidence**: 1 indicator = *low*, 2 = *medium*, $\geq 3$ = *high*.

#### Detector E: Curve / ODD Boundary

**Rationale**: Takeovers triggered by road geometry challenges (sharp curves, poor lane markings) that cause the ADAS to reach its operational design domain (ODD) boundary.

**Conditions**:
- High pre-curvature ($|\kappa| > 0.02$) OR sustained pre-steering ($|\theta| > 20$ deg for > 1.5 s, one-sided ratio > 0.92) OR high pre-curvature ($\bar{|\kappa|}_{\text{pre}} > 0.003$)
- No blinker, not a lane change
- **Speed gate**: post-event speed stays above 4.47 m/s (10 mph) in both 5s and 10s windows (if speed drops, it is a junction turn, not a curve)

#### Detector F: System / Unknown

**Triggers on system alert text**:
- `"TAKE CONTROL IMMEDIATELY"`
- `"Dashcam Mode"`
- `"openpilot Unavailable"`
- `"Steering Temporarily Unavailable"`

**Confidence**: *high* if no driver trigger (steer/brake/gas) detected; *medium* otherwise.

---

## 5. Classification Priority Logic

When multiple detectors fire simultaneously (e.g., a lane change that also has a lead-vehicle conflict), the classifier resolves conflicts using a strict priority hierarchy:

| Priority | Rule | Label | Rationale |
|:--------:|------|:-----:|-----------|
| 1 | Stationary (Detector G) | **Ego** | Stopped vehicle = always driver-initiated |
| 2 | Ego detector + blinker active | **Ego** | Blinker is the strongest intentionality signal; overrides conflict evidence (but not system alerts or curve boundary) |
| 3a | Junction turn from near-stop (B1) | **Ego** | Near-stop turns are unambiguously intentional; overrides all conflict |
| 3b | Discretionary acceleration (C) | **Ego** | Gas-trigger acceleration is driver-initiated even with lead car present |
| 4 | Clean Ego evidence, no Non-ego flags | **Ego** | Unambiguous Ego with no contradicting evidence |
| 5 | Ego detector + borderline conflict only | **Ego** | Close-lead-only conflict (no TTC/DRAC/FCW breach) is insufficient to override strong Ego evidence |
| 6 | Blinker alone, no conflict | **Ego** | Blinker without explicit conflict suggests intentional maneuver (labeled as `blinker_intent`) |
| 7 | Everything else | **Non-ego** | Default: if no Ego pattern is detected or conflict evidence dominates |

---

## 6. Label Distribution (Full Dataset)

### 6.1 Overall

| Label | Count | Percentage |
|-------|------:|----------:|
| Ego | 2,976 | 19.0% |
| Non-ego | 12,683 | 81.0% |
| **Total** | **15,659** | **100%** |

### 6.2 Ego Sub-Categories

| Reason | Count | % of Ego |
|--------|------:|---------:|
| Lane change (Detector A) | 1,113 | 37.4% |
| Junction turn (Detector B) | 813 | 27.3% |
| Blinker intent (Priority 6) | 512 | 17.2% |
| Stationary takeover (Detector G) | 412 | 13.8% |
| Discretionary acceleration (Detector C) | 126 | 4.2% |

### 6.3 Non-Ego Sub-Categories

| Reason | Count | % of Non-ego |
|--------|------:|-------------:|
| Conflict / reactive (Detector D) | 7,996 | 63.0% |
| Other (no specific detector) | 3,567 | 28.1% |
| Mixed ego + non-ego evidence | 769 | 6.1% |
| System / unknown (Detector F) | 277 | 2.2% |
| Curve boundary (Detector E) | 74 | 0.6% |

---

## 7. Expert Validation

### 7.1 Test Set Construction

A stratified random sample of **500 clips** was drawn from the full dataset:
- **250 model-predicted Ego** and **250 model-predicted Non-ego** (balanced by model prediction to ensure adequate representation of both classes)
- Stratified across vehicle platforms: the test set spans **93 unique car models** from 15+ manufacturers (Acura, Audi, Chevrolet, Ford, Honda, Hyundai, Kia, Lexus, Nissan, Ram, Rivian, Tesla, Toyota, Volkswagen, Volvo, etc.) and **142 unique dongle IDs** (unique vehicles)

Each sampled clip was made accessible to reviewers via a directory containing:
- `takeover.mp4` --- the 20-second video clip centered on the takeover event
- All per-clip CSV files (carState, carControl, carOutput, controlsState, etc.)
- `meta.json` with clip metadata (car model, event time, log type)

### 7.2 Annotation Protocol

**Four expert annotators** independently reviewed each of the 500 clips. Each expert:

1. Watched the takeover video (`takeover.mp4`) to observe the driving context, road environment, and driver actions
2. Examined the accompanying CAN-bus signals (steering angle, speed, acceleration, blinker status) plotted against time
3. Assigned a binary label: **Ego** (driver-initiated) or **Non-ego** (reactive/system-initiated)

The **final ground truth** for each clip was determined by **majority vote** among the four annotators. In cases of a 2--2 tie, the annotators discussed the clip and reached a consensus.

The resulting ground truth distribution is: **264 Ego** and **236 Non-ego** (compared to the model's 250/250 split, reflecting that human reviewers identified slightly more Ego takeovers than the model predicted).

### 7.3 Results: Confusion Matrix

The confusion matrix for the v7 classifier evaluated against the expert ground truth is shown below.

```
                        Ground Truth (Human Label)
                        Ego              Non-ego
Model        Ego        TP = 238         FN = 54
Prediction              (47.6%)          (10.8%)
(v7)         Non-ego    FP = 26          TN = 182
                        (5.2%)           (36.4%)
```

**Overall Accuracy: 84.0% (420 / 500)**

### 7.4 Per-Class Metrics

| Metric | Ego Class | Non-ego Class | Macro Average |
|--------|:---------:|:------------:|:-------------:|
| **Precision** | 90.2% | 77.1% | 83.6% |
| **Recall** | 81.5% | 87.5% | 84.5% |
| **F1 Score** | 85.6% | 82.0% | 83.8% |

**Interpretation**:
- **Ego Precision (90.2%)**: When the model predicts Ego, it is correct 90.2% of the time. The classifier is conservative in assigning the Ego label.
- **Ego Recall (81.5%)**: The model captures 81.5% of true Ego takeovers. The 18.5% miss rate is primarily due to Ego takeovers that co-occur with conflict indicators (see error analysis below).
- **Non-ego Recall (87.5%)**: The model correctly identifies 87.5% of true Non-ego takeovers, demonstrating high sensitivity to reactive events.
- **Non-ego Precision (77.1%)**: Some true Ego clips are misclassified as Non-ego due to conservative conflict detection, resulting in a lower precision for the Non-ego class.

### 7.5 Error Analysis

A total of **80 misclassified clips** were identified (TP + TN = 420, errors = 80).

#### False Negatives (54 clips): Model predicted Ego, Ground Truth is Non-ego

These are clips where the model over-attributed driver intentionality:

| Misclassified Ego Reason | Count | Description |
|--------------------------|------:|-------------|
| Lane change | ~13 | Model detected LC dynamics (blinker + steering pattern) but the human reviewers judged the lane change was *reactive* (e.g., avoiding a merging vehicle) |
| Junction turn | ~12 | Low-speed turns where human reviewers saw evidence of system disengagement rather than intentional turning |
| Discretionary acceleration | ~11 | Gas pedal press attributed to driver intent, but reviewers saw it as a response to traffic flow |
| Blinker intent | ~10 | Blinker was active but the takeover was triggered by a system limitation, not a planned maneuver |
| Stationary takeover | ~8 | Stopped vehicle where the takeover was system-initiated (e.g., system timeout) rather than driver-initiated |

#### False Positives (26 clips): Model predicted Non-ego, Ground Truth is Ego

These are clips where the model was too conservative:

| Misclassified Non-ego Reason | Count | Description |
|------------------------------|------:|-------------|
| Conflict reactive | ~15 | A safety metric (TTC, THW, DRAC, or close lead) fired, masking the driver's intentional maneuver. Most common pattern: driver initiates a lane change while a lead vehicle is present, triggering the close-lead indicator |
| Other (no detector fired) | ~8 | Driver intention was visible in video but not captured by any detector's feature set |
| Mixed ego + non-ego | ~3 | Both Ego and Non-ego detectors fired; the priority logic defaulted to Non-ego |

#### Key Error Patterns

1. **Conflict masking Ego intent** (largest error source): When a driver initiates a planned maneuver (e.g., lane change) while a lead vehicle happens to be present, the conflict detector fires and can override the Ego label. The v7 priority logic mitigates this via the borderline-conflict override (Priority 5), but TTC/DRAC/FCW-level conflicts still take precedence.

2. **Ambiguous blinker intent**: A blinker may be active due to the driver's habit rather than a specific maneuver plan, leading to false Ego labels.

3. **Curve vs. turn boundary**: Distinguishing sharp road curves (Non-ego: system reaches ODD boundary) from intentional junction turns (Ego) remains challenging when speed is in the 4--8 m/s range.

---

## 8. Version History

The classifier underwent iterative refinement across seven versions:

| Version | Key Changes |
|---------|-------------|
| v1--v5 | Initial rule-based system with single 5s post-window; basic blinker/steering/speed features; progressively refined thresholds |
| v6 | Extended to 10s post-window to catch late-onset maneuvers; added curvature features and left-turn detection; junction speed threshold raised to 4.5 m/s |
| **v7** (current) | **Blinker overrides conflict** (strongest intentionality signal); stationary threshold tightened from 1.0 to 0.5 m/s; junction near-stop threshold reverted from 4.5 to 3.0 m/s; new A3 sub-detector for conflict-triggered lane changes; sharp curve speed gate at 4.47 m/s (10 mph) |

---

## 9. Implementation Details

- **Script**: `classify_ego_nonego.py` (v7)
- **Runtime**: ~3 minutes on 12 CPU cores (15,659 clips)
- **Feature extraction**: Parallel `ProcessPoolExecutor` with 12 workers; each worker reads carState.csv and carControl.csv, extracts 42 features per clip
- **Dependencies**: Python 3.10+, NumPy, Pandas (no ML libraries required)
- **Reproducibility**: Deterministic rule-based system; same input produces identical output

---

## 10. Limitations

1. **Rule-based system**: The classifier relies on hand-crafted rules and thresholds. While interpretable and transparent, it cannot capture complex non-linear interactions between features that a learned model might identify.

2. **No interior camera**: The classification relies on vehicle dynamics and CAN-bus signals only. Interior camera footage (driver gaze, hand position) would significantly improve classification accuracy but is not available in the openpilot dataset.

3. **Conflict masking**: The largest error source (conflict indicators overriding true Ego intent) is a fundamental limitation of using safety metrics as Non-ego evidence---a driver may intentionally maneuver while a lead vehicle happens to be present.

4. **Threshold sensitivity**: Several thresholds (e.g., `JUNCTION_NEAR_STOP_MPS = 3.0`, `LC_A1_MAX_ONE_SIDED_RATIO = 0.92`) were tuned on the training set and may not generalize perfectly to all vehicle platforms and driving contexts.

5. **Log rate sensitivity**: The dataset contains both qlog (10 Hz) and rlog (100 Hz) recordings. Steering waveform features (sign changes, one-sided ratio, steer return) may behave differently at different sampling rates.

6. **Unbalanced base rate**: Only 19% of clips are labeled Ego, reflecting the natural distribution of takeover types. The validation test set was balanced (50/50) to ensure adequate power for both classes, but real-world performance may differ due to class imbalance.

---

## 11. Conclusion

The v7 rule-based Ego/Non-ego classifier achieves **84.0% accuracy** on a 500-clip expert-validated test set, with balanced performance across both classes (F1: 85.6% for Ego, 82.0% for Non-ego). The classifier is fully interpretable, deterministic, and computationally efficient. Its primary limitation---conflict indicators masking true driver intent---reflects a fundamental ambiguity in distinguishing planned from reactive maneuvers using vehicle dynamics alone. The 19.0% Ego rate in the full dataset (2,976 clips) provides a substantial subset for downstream analysis of intentional takeover behavior and post-takeover control quality.
