# Non-Ego Radar-Based Longitudinal Analysis (TTC / THW)

## 1. Dataset Overview

- **Total Non-ego clips**: 12683
- **With radarState.csv**: 12683 (100.0%)
- **Lead-present in pre-window** (≥3 radar samples with leadOne.status=True): 6287 (49.6%)

## 2. Metric Definitions

| Metric | Formula | Guard / Notes |
|--------|---------|---------------|
| **THW** (time headway) | dRel / vEgo | vEgo > 0.5 m/s; capped at 15 s |
| **TTC** (time-to-collision) | dRel / (−vRel) | Only closing (vRel < −0.1 m/s); capped at 15 s |
| **dRel** | leadOne.dRel | Direct radar measurement (m) |
| **vRel** | leadOne.vRel | Relative velocity (m/s); negative = closing |

## 3. Key Metrics (lead-present, n = 6287)

| Metric | Median | IQR | P5 | P95 |
|--------|-------:|:---:|---:|----:|
| TTC min (s) | 12.29 | 6.19–15.00 | 2.67 | 15.00 |
| THW min (s) | 2.11 | 1.49–3.19 | 0.88 | 5.49 |
| dRel min (m) | 29.28 | 17.28–49.88 | 5.86 | 91.88 |
| dRel at takeover (m) | 32.60 | 19.18–55.74 | 6.54 | 101.63 |
| vRel at takeover (m/s) | -0.98 | -2.80–0.13 | -8.16 | 2.19 |
| vLead at takeover (m/s) | 13.30 | 6.54–20.54 | 0.11 | 30.00 |

## 4. Critical Event Rates (lead-present)

| Threshold | Count | Rate |
|-----------|------:|-----:|
| TTC < 1.5 s | 37 | 0.7% |
| TTC < 2.0 s | 111 | 2.0% |
| TTC < 3.0 s | 366 | 6.7% |
| THW < 0.8 s | 233 | 3.7% |
| THW < 1.0 s | 492 | 7.9% |
| THW < 1.5 s | 1582 | 25.4% |
| dRel < 10 m | 746 | 11.9% |

## 5. Speed-Stratified Summary (lead-present)

| Speed | n | TTC med (s) | THW med (s) | dRel med (m) | TTC<1.5s rate |
|-------|--:|:----------:|:----------:|:-----------:|:------------:|
| Low (<60 km/h) | 3600 | 8.41 | 2.36 | 21.2 | 1.0% |
| Medium | 1882 | 15.00 | 1.90 | 41.1 | 0.1% |
| High (>100 km/h) | 721 | 15.00 | 1.64 | 50.7 | 0.2% |

## 6. Key Findings

1. **TTC distribution**: Among 5424 lead-present Non-ego clips with closing dynamics, median TTC_min = 12.29 s. 37 clips (0.7%) breach the 1.5 s critical threshold.

2. **THW distribution**: Median THW_min = 2.11 s. 233 clips (3.7%) below the 0.8 s safety threshold.

3. **Lead distance at takeover**: Median = 32.6 m. 607 clips (9.7%) with dRel < 10 m, indicating very close following at the moment of takeover.

4. **Closing dynamics**: 68.9% of lead-present clips have negative vRel (closing) at takeover. Median vRel = -0.98 m/s.


## 7. Limitations

1. **Radar availability**: radarState.csv is present for 100.0% of Non-ego clips. Missing radar clips cannot contribute to TTC/THW.

2. **Lead detection gaps**: leadOne.status may be False even when a lead vehicle exists (e.g., radar occlusion, lateral offset). The lead_present_pre flag requires ≥3 active samples in [-3, 0] s.

3. **TTC only for closing**: TTC is undefined when ego is not closing on the lead vehicle. Non-closing lead-present clips are excluded from TTC statistics.

4. **THW denominator guard**: THW = dRel/vEgo is set to NaN when vEgo < 0.5 m/s (near-stationary).

5. **Single lead vehicle**: Only leadOne is used; leadTwo is ignored.
