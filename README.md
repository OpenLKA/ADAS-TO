---
license: cc-by-nc-sa-4.0
task_categories:
  - video-classification
  - time-series-forecasting
language:
  - en
tags:
  - autonomous-driving
  - ADAS
  - takeover
  - openpilot
  - driving-safety
  - CAN-bus
  - comma-ai
  - human-factors
size_categories:
  - 10K<n<100K
---

# HADAS-Takeover: A Large-Scale ADAS Take-Over Request Dataset

## Dataset Summary

**HADAS-Takeover** is a large-scale naturalistic dataset of Advanced Driver-Assistance System (ADAS) take-over events, extracted from real-world driving logs collected by the [comma.ai](https://comma.ai/) openpilot community. Each sample captures a 20-second window (±10 s) around the moment a human driver disengages the ADAS, providing synchronized front-facing video and multi-modal CAN-bus / planning signals.

| Statistic | Value |
|-----------|-------|
| Total clips | **15,705** |
| Vehicle models | **163** |
| Unique vehicles (dongles) | **338** |
| Unique driving routes | **2,326** |
| Manufacturers | **23** (Acura, Audi, BYD, Chevrolet, Ford, Genesis, Honda, Hyundai, Jeep, Kia, Lexus, Mazda, Nissan, Porsche, RAM, Rivian, Skoda, Subaru, Tesla, Toyota, Volkswagen, Volvo) |
| Video size | ~22.2 GB |
| CSV / metadata size | ~9.8 GB |
| **Total size** | **~33 GB** |

## Directory Structure

```
<CAR_MODEL>/
  <dongle_id>/
    <route_id>/
      <clip_id>/              # integer, 0-indexed per route
        meta.json             # clip-level metadata
        takeover.mp4          # 20 s front-camera video (H.264, 20 fps)
        carState.csv          # ego vehicle state
        controlsState.csv     # ADAS controller state
        carControl.csv        # control commands
        carOutput.csv         # actuator outputs
        drivingModelData.csv  # driving model predictions
        radarState.csv        # radar / lead vehicle data
        accelerometer.csv     # IMU acceleration
        longitudinalPlan.csv  # longitudinal planner output
```

## Clip Definition

- **ADAS engagement** is detected when `controlsState.enabled` OR `carState.cruiseState.enabled` is `True` (openpilot engaged).
- A **take-over event** is an ON → OFF transition with at least 2 s of preceding engagement and 2 s of subsequent disengagement (transitions within 0.5 s are merged).
- Each clip spans **±10 s** around the take-over moment, yielding a **20-second** window.

## Data Fields

### meta.json

| Field | Description |
|-------|-------------|
| `car_model` | Vehicle model identifier |
| `dongle_id` | Anonymized device identifier |
| `route_id` | Driving route identifier |
| `clip_id` | Clip index within the route (0-indexed) |
| `log_kind` | Log source (`qlog` at 10 Hz or `rlog` at 100 Hz) |
| `log_hz` | Signal sampling rate (Hz) |
| `vid_kind` | Video source (`qcamera` or `fcamera`) |
| `camera_fps` | Video frame rate (fps) |
| `event_mono` | logMonoTime timestamp of the take-over event |
| `video_time_s` | Time of the take-over event in the video stream (s) |
| `clip_start_s` | Start time of the clip in the driving session (s) |
| `clip_dur_s` | Duration of the clip (s) |
| `seg_nums_used` | List of openpilot segment numbers covering this route |

### CSV Topics

| File | Key Signals |
|------|-------------|
| `carState.csv` | `vEgo`, `aEgo`, `steeringAngleDeg`, `steeringTorque`, `steeringPressed`, `gasPressed`, `brakePressed`, `cruiseState.enabled` |
| `controlsState.csv` | `enabled`, `active`, `curvature`, `desiredCurvature`, `vCruise`, `longControlState`, `alertText1`, `alertText2` |
| `carControl.csv` | `latActive`, `longActive`, `actuators.accel`, `actuators.torque`, `actuators.curvature` |
| `carOutput.csv` | `actuatorsOutput.accel`, `actuatorsOutput.brake`, `actuatorsOutput.gas`, `actuatorsOutput.steer`, `actuatorsOutput.steeringAngleDeg` |
| `drivingModelData.csv` | `action.desiredCurvature`, `action.desiredAcceleration`, `laneLineMeta.leftProb`, `laneLineMeta.rightProb` |
| `radarState.csv` | `leadOne.dRel`, `leadOne.vRel`, `leadOne.vLead`, `leadOne.aLeadK`, `leadTwo.*` |
| `accelerometer.csv` | `acceleration.v` (3-axis IMU), `timestamp` |
| `longitudinalPlan.csv` | `aTarget`, `hasLead`, `fcw`, `speeds[]`, `accels[]` |

All CSVs include a `logMonoTime` column (nanoseconds) and a `time_s` column (seconds from route start) for temporal alignment.

## Top 20 Vehicle Models by Clip Count

| Rank | Vehicle Model | Clips |
|------|--------------|-------|
| 1 | Rivian R1 Gen1 | 2,127 |
| 2 | Acura MDX 3G MMR | 1,863 |
| 3 | Ford F-150 MK14 | 1,226 |
| 4 | Chevrolet Silverado | 639 |
| 5 | Toyota Prius | 482 |
| 6 | Honda Civic | 470 |
| 7 | Tesla AP3 Model 3 | 432 |
| 8 | Ford Maverick MK1 | 300 |
| 9 | Hyundai Ioniq 5 | 266 |
| 10 | Chevrolet Bolt EUV 2022 | 244 |
| 11 | Toyota RAV4 TSS2 2023 | 228 |
| 12 | RAM HD 5th Gen | 221 |
| 13 | Volkswagen Jetta MK7 | 215 |
| 14 | Kia EV6 | 209 |
| 15 | Volkswagen Golf MK7 | 192 |
| 16 | Kia Niro EV | 185 |
| 17 | Hyundai Ioniq 6 | 177 |
| 18 | Volkswagen Atlas MK1 | 153 |
| 19 | Tesla Model X | 152 |
| 20 | Hyundai Ioniq 5 2022 | 145 |

## Usage

### Loading a single clip (Python)

```python
import json
import pandas as pd
from pathlib import Path

clip_dir = Path("FORD_F_150_MK14/00e5e26644f0f460/00000003--257730d24c/0")

# Metadata
meta = json.loads((clip_dir / "meta.json").read_text())

# CAN-bus signals
car_state = pd.read_csv(clip_dir / "carState.csv")
controls   = pd.read_csv(clip_dir / "controlsState.csv")

# Video path
video_path = clip_dir / "takeover.mp4"
```

### Iterating over all clips

```python
from pathlib import Path
import json

root = Path(".")
for meta_file in sorted(root.glob("*/*/meta.json")):
    # actually meta.json is 4 levels deep
    pass

# Correct iteration:
for meta_file in sorted(root.rglob("meta.json")):
    meta = json.loads(meta_file.read_text())
    clip_dir = meta_file.parent
    print(meta["car_model"], meta["clip_id"], clip_dir)
```

## Data Source

All driving data originates from the [comma.ai](https://comma.ai/) openpilot platform. Openpilot is an open-source ADAS that runs on consumer vehicles equipped with a comma device. The logs are contributed by the comma community and accessed via the comma connect API.

## License

This dataset is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{hadas_takeover_2025,
  title   = {HADAS-Takeover: A Large-Scale ADAS Take-Over Request Dataset},
  author  = {Wang, Yuhang},
  year    = {2025},
  url     = {https://huggingface.co/datasets/AsianPalyer/HADAS-Takeover},
  license = {CC BY-NC-SA 4.0}
}
```
