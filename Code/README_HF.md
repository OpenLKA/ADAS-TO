---
license: cc-by-nc-4.0
task_categories:
  - time-series-forecasting
  - video-classification
tags:
  - autonomous-driving
  - ADAS
  - takeover
  - driver-behavior
  - openpilot
  - time-series
  - multimodal
  - CAN-bus
  - vehicle-dynamics
size_categories:
  - 10K<n<100K
language:
  - en
pretty_name: "HADAS-TakeOver: Human-ADAS Driving Takeover Dataset"
---

# HADAS-TakeOver: Human-ADAS Driving Takeover Dataset

## Dataset Summary

**HADAS-TakeOver** is a large-scale, multimodal dataset of **15,705 real-world ADAS takeover events** captured from **327 drivers** across **163 vehicle models** from **23 manufacturers**. Each takeover event is a 20-second clip centered on the moment a driver disengages an Advanced Driver Assistance System (ADAS), providing synchronized video, vehicle dynamics, controller state, and sensor data.

| Statistic | Value |
|---|---|
| Total takeover clips | 15,705 |
| Unique drivers | 327 |
| Unique driving routes | 2,312 |
| Vehicle models | 163 |
| Manufacturers | 23 |
| Clip duration | 20 seconds (±10 s around takeover) |
| Video resolution | Front-facing camera, 20 fps |
| CAN/sensor signals | 10–100 Hz |
| Total size | ~33 GB |

## Motivation

Understanding how and why drivers take over from automated driving systems is critical for improving ADAS safety, designing better human-machine interfaces, and developing predictive takeover models. Despite growing research interest, large-scale naturalistic takeover datasets remain scarce. HADAS-TakeOver addresses this gap by providing:

- **Scale**: Over 15K events across hundreds of drivers and vehicles
- **Diversity**: 163 vehicle models spanning sedans, SUVs, trucks, and EVs from 23 manufacturers
- **Richness**: Synchronized video + 9 time-series signal files per event
- **Naturalistic**: Real-world driving (not simulator), capturing genuine driver behavior

### Use Cases

- Takeover prediction and early warning systems
- Driver behavior modeling during ADAS transitions
- Analysis of ADAS disengagement patterns across vehicle types
- Human factors research in automated driving
- Multimodal time-series classification and forecasting

## Dataset Structure

```
HADAS-TakeOver/
  <CAR_MODEL>/                  # e.g., TOYOTA_PRIUS, TESLA_AP3_MODEL_3
    <driver_XXX>/               # anonymized driver ID
      <route_XXX>/              # anonymized route ID
        <clip_id>/              # integer (0-indexed per route)
          meta.json             # clip metadata
          takeover.mp4          # 20-second front-camera video
          carState.csv          # vehicle state signals
          controlsState.csv     # ADAS controller state
          carControl.csv        # control commands
          carOutput.csv         # actuator outputs
          drivingModelData.csv  # driving model predictions
          radarState.csv        # radar / lead vehicle data
          accelerometer.csv     # IMU accelerometer data
          longitudinalPlan.csv  # longitudinal planner outputs
```

Each clip contains **10 files**: 1 video, 1 metadata JSON, and 8 CSV time-series files.

## Takeover Event Definition

A **takeover event** is defined as an ADAS ON → OFF transition where:

- **ADAS engaged** = `controlsState.enabled` OR `carState.cruiseState.enabled`
- **Minimum ON duration**: 2 seconds before disengagement
- **Minimum OFF duration**: 2 seconds after disengagement
- **Gap merging**: Transient gaps < 0.5 s are merged (to filter sensor noise)
- **Clip window**: ±10 seconds centered on the ON→OFF transition (20 s total)

The first ~10 seconds of each clip show ADAS-engaged driving; the remaining ~10 seconds show the driver resuming manual control.

## Data Fields

### meta.json

| Field | Type | Description |
|---|---|---|
| `car_model` | string | Vehicle model identifier (e.g., `TOYOTA_PRIUS`) |
| `dongle_id` | string | Anonymized driver ID (`driver_XXX`) |
| `route_id` | string | Anonymized route ID (`route_XXX`) |
| `log_kind` | string | Log source: `qlog` (10 Hz) or `rlog` (100 Hz) |
| `log_hz` | int | Sampling rate of CAN signals (10 or 100) |
| `vid_kind` | string | Video source: `qcamera` or `fcamera` |
| `camera_fps` | int | Video frame rate (20 fps) |
| `clip_id` | int | Clip index within the route (0-indexed) |
| `event_mono` | int | Monotonic timestamp of the takeover event (nanoseconds) |
| `video_time_s` | float | Time of takeover within the full route video (seconds) |
| `clip_start_s` | float | Start time of the 20-second clip within the route (seconds) |
| `clip_dur_s` | float | Clip duration (seconds, typically 20.0) |
| `seg_nums_used` | list[int] | Openpilot segment numbers covering this route |

### carState.csv — Vehicle State

Driver inputs and ego vehicle dynamics.

| Column | Description |
|---|---|
| `vEgo` | Ego vehicle speed (m/s) |
| `aEgo` | Ego vehicle acceleration (m/s²) |
| `steeringAngleDeg` | Steering wheel angle (degrees) |
| `steeringTorque` | Steering torque applied by driver |
| `steeringPressed` | Whether driver is actively steering (boolean) |
| `gasPressed` | Whether gas pedal is pressed (boolean) |
| `brakePressed` | Whether brake pedal is pressed (boolean) |
| `cruiseState.enabled` | Whether cruise control / ADAS is engaged (boolean) |

### controlsState.csv — ADAS Controller State

| Column | Description |
|---|---|
| `enabled` | Whether openpilot ADAS is enabled (boolean) |
| `active` | Whether ADAS is actively controlling the vehicle |
| `curvature` | Current path curvature (1/m) |
| `desiredCurvature` | Target path curvature from planner |
| `vCruise` | Set cruise speed (m/s) |
| `longControlState` | Longitudinal control state (enum) |
| `alertText1` | Primary alert text displayed to driver |
| `alertText2` | Secondary alert text |

### carControl.csv — Control Commands

| Column | Description |
|---|---|
| `latActive` | Whether lateral control is active |
| `longActive` | Whether longitudinal control is active |
| `actuators.accel` | Commanded acceleration (m/s²) |
| `actuators.torque` | Commanded steering torque |
| `actuators.curvature` | Commanded path curvature |

### carOutput.csv — Actuator Outputs

| Column | Description |
|---|---|
| `actuatorsOutput.accel` | Actual acceleration output |
| `actuatorsOutput.brake` | Brake actuator output |
| `actuatorsOutput.gas` | Gas actuator output |
| `actuatorsOutput.steer` | Steering actuator output |
| `actuatorsOutput.steerOutputCan` | Raw CAN steering output |
| `actuatorsOutput.steeringAngleDeg` | Output steering angle (degrees) |

### drivingModelData.csv — Driving Model Predictions

| Column | Description |
|---|---|
| `action.desiredCurvature` | Model-predicted desired curvature |
| `action.desiredAcceleration` | Model-predicted desired acceleration |
| `laneLineMeta.leftProb` | Probability of left lane line detection |
| `laneLineMeta.rightProb` | Probability of right lane line detection |

### radarState.csv — Lead Vehicle Detection

| Column | Description |
|---|---|
| `leadOne.dRel` | Distance to primary lead vehicle (m) |
| `leadOne.vRel` | Relative velocity of lead vehicle (m/s) |
| `leadOne.vLead` | Absolute velocity of lead vehicle (m/s) |
| `leadOne.aLeadK` | Estimated acceleration of lead vehicle (m/s²) |
| `leadTwo.*` | Same fields for secondary lead vehicle |

### accelerometer.csv — IMU Data

| Column | Description |
|---|---|
| `acceleration.v` | 3-axis acceleration vector (m/s²) |
| `timestamp` | Sensor timestamp |

### longitudinalPlan.csv — Planner Outputs

| Column | Description |
|---|---|
| `aTarget` | Target acceleration from planner (m/s²) |
| `hasLead` | Whether a lead vehicle is detected (boolean) |
| `fcw` | Forward collision warning active (boolean) |
| `speeds[]` | Planned speed profile |
| `accels[]` | Planned acceleration profile |

## Top Vehicle Models

| Vehicle Model | Clips | | Vehicle Model | Clips |
|---|---|---|---|---|
| RIVIAN R1 GEN1 | 2,127 | | CHEVROLET BOLT EUV 2022 | 244 |
| ACURA MDX 3G MMR | 1,863 | | TOYOTA RAV4 TSS2 2023 | 228 |
| FORD F-150 MK14 | 1,226 | | RAM HD 5TH GEN | 221 |
| CHEVROLET SILVERADO | 639 | | VOLKSWAGEN JETTA MK7 | 215 |
| TOYOTA PRIUS | 482 | | KIA EV6 | 209 |
| HONDA CIVIC | 470 | | VOLKSWAGEN GOLF MK7 | 192 |
| TESLA AP3 MODEL 3 | 432 | | KIA NIRO EV | 185 |
| FORD MAVERICK MK1 | 300 | | HYUNDAI IONIQ 6 | 177 |
| HYUNDAI IONIQ 5 | 266 | | VOLKSWAGEN ATLAS MK1 | 153 |

## Usage

### Loading a Single Clip

```python
import json
import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "HenryYHW/ADAS-TO"
clip_path = "TOYOTA_PRIUS/driver_001/route_001/0"

# Download metadata
meta_path = hf_hub_download(repo_id, f"{clip_path}/meta.json", repo_type="dataset")
with open(meta_path) as f:
    meta = json.load(f)
print(meta)

# Load vehicle state signals
car_state_path = hf_hub_download(repo_id, f"{clip_path}/carState.csv", repo_type="dataset")
car_state = pd.read_csv(car_state_path)
print(car_state[["vEgo", "aEgo", "steeringAngleDeg", "brakePressed"]].describe())

# Load ADAS controller state
controls_path = hf_hub_download(repo_id, f"{clip_path}/controlsState.csv", repo_type="dataset")
controls = pd.read_csv(controls_path)
```

### Iterating Over All Clips

```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("HenryYHW/ADAS-TO", repo_type="dataset")
meta_files = [f for f in files if f.endswith("meta.json")]
print(f"Total clips: {len(meta_files)}")
```

### Downloading the Full Dataset

```bash
# Using huggingface-cli
huggingface-cli download HenryYHW/ADAS-TO --repo-type dataset --local-dir ./HADAS-TakeOver

# Using git-lfs
git lfs install
git clone https://huggingface.co/datasets/HenryYHW/ADAS-TO
```

## Data Collection

This dataset was built from driving logs collected by the [comma.ai](https://comma.ai/) community using [openpilot](https://github.com/commaai/openpilot), an open-source ADAS platform. Logs were processed to detect ADAS disengagement events, extract synchronized video and CAN-bus signals, and package them into standardized clips.

**Privacy**: All driver and route identifiers have been anonymized. No personally identifiable information (PII) is included. Video data shows the forward road view only.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{hadas_takeover_2025,
  title={HADAS-TakeOver: A Large-Scale Naturalistic Dataset of Human-ADAS Driving Takeover Events},
  author={Zhou, Haowei},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/HenryYHW/ADAS-TO}
}
```

## License

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is intended for academic and non-commercial research purposes.
