<div align="center">

# 🚗💨 ADAS-TO

### **A Large-Scale Multimodal Naturalistic Dataset and Empirical Characterization of Human Takeovers during ADAS Engagement**

*15,705 real-world takeover events · 327 drivers · 163 vehicle models · 23 manufacturers*

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Dataset on HF](https://img.shields.io/badge/🤗%20Dataset-ADAS--TO-blue)](https://huggingface.co/datasets/HenryYHW/ADAS-TO)
[![Clips](https://img.shields.io/badge/Clips-15%2C705-brightgreen)]()
[![Vehicles](https://img.shields.io/badge/Vehicle%20Models-163-orange)]()
[![Size](https://img.shields.io/badge/Size-~33%20GB-red)]()
[![Modality](https://img.shields.io/badge/Modality-Video%20%2B%20CAN%20%2B%20Radar%20%2B%20IMU-purple)]()

---

**When does a human driver take over from an ADAS?** **Why?** **How?**

ADAS-TO captures the critical moment of control transition — the exact instant a driver decides the automation is no longer sufficient — across thousands of real-world scenarios with synchronized front-view video, vehicle dynamics, radar, and IMU data.

</div>

---

## 🎬 Takeover Examples

<div align="center">

*Each GIF shows ±3 seconds around the takeover moment — ADAS engaged → driver takes control*

<table>
<tr>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_1.gif" width="240"/><br/><sub>On-coming Traffic</sub></td>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_2.gif" width="240"/><br/><sub>Bridge</sub></td>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_3.gif" width="240"/><br/><sub>Night Driving</sub></td>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_4.gif" width="240"/><br/><sub>Sharp Curve</sub></td>
</tr>
<tr>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_5.gif" width="240"/><br/><sub>Surrounding Car</sub></td>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_6.gif" width="240"/><br/><sub>Traffic Light</sub></td>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_7.gif" width="240"/><br/><sub>Lane Change</sub></td>
<td align="center"><img src="https://huggingface.co/datasets/HenryYHW/ADAS-TO/resolve/main/assets/takeover_8.gif" width="240"/><br/><sub>Hard Brake</sub></td>
</tr>
</table>

</div>

---

## 📊 Dataset at a Glance

<div align="center">

| | Statistic | Value |
|:---:|:---|:---|
| 🎥 | **Total takeover clips** | **15,705** |
| 👤 | **Unique drivers** | **327** |
| 🛣️ | **Unique driving routes** | **2,312** |
| 🚘 | **Vehicle models** | **163** |
| 🏭 | **Manufacturers** | **23** |
| ⏱️ | **Clip duration** | **20 seconds** (±10s around takeover) |
| 📹 | **Video** | Front-facing camera, **20 fps** |
| 📡 | **CAN / sensor signals** | **10–100 Hz** |
| 📁 | **Files per clip** | **10** (1 video + 1 meta + 8 CSV) |
| 💾 | **Total size** | **~33 GB** |

</div>

---

## 🔥 Why ADAS-TO?

> *"The takeover moment is the most safety-critical instant in human-automation interaction — yet it remains one of the least studied due to lack of data."*

### 🏆 Unprecedented Scale
Over **15,000 real-world takeover events** — orders of magnitude larger than existing datasets that typically contain hundreds of events captured in driving simulators.

### 🌍 Unmatched Diversity
**163 vehicle models** from **23 manufacturers** including Tesla, Toyota, Honda, Hyundai, Ford, Volkswagen, Rivian, and more. From compact EVs to full-size trucks — spanning the full spectrum of modern ADAS implementations.

### 🎯 Rich Multimodal Signals
Every clip contains **synchronized** front-camera video, vehicle dynamics (speed, acceleration, steering), ADAS controller state, control commands, actuator outputs, driving model predictions, radar/lead vehicle data, and IMU measurements.

### 🌐 Real-World Naturalistic Data
Collected through **online and offline autonomous driving communities** with diverse real-world driving conditions — highways, urban streets, suburbs, varying weather and lighting. No simulators. No scripted scenarios. Pure naturalistic driving behavior.

---

## 🎯 Use Cases

| Application | Description |
|:---|:---|
| 🔮 **Takeover Prediction** | Build early warning systems that predict when a driver will need to take over |
| 🧠 **Driver Behavior Modeling** | Understand human responses during control transitions |
| 📈 **ADAS Performance Analysis** | Compare disengagement patterns across vehicle types and ADAS systems |
| 🤖 **Autonomous Driving Safety** | Train and evaluate safety-critical decision-making models |
| 🧪 **Human Factors Research** | Study cognitive load, reaction times, and situational awareness |
| 📊 **Multimodal Time-Series** | Develop forecasting and classification models on rich temporal data |
| 🏗️ **HMI Design** | Design better human-machine interfaces for automated vehicles |

---

## 📁 Dataset Structure

```
ADAS-TO/
├── <CAR_MODEL>/                        # e.g., TOYOTA_PRIUS, TESLA_AP3_MODEL_3
│   └── <driver_XXX>/                   # 🔒 anonymized driver ID
│       └── <route_XXX>/                # 🔒 anonymized route ID
│           └── <clip_id>/              # integer (0-indexed per route)
│               ├── 🎥 takeover.mp4          20-second front-camera video
│               ├── 📋 meta.json             clip metadata & timing
│               ├── 🚗 carState.csv          vehicle dynamics & driver inputs
│               ├── 🤖 controlsState.csv     ADAS controller state & alerts
│               ├── 🎮 carControl.csv        lateral/longitudinal commands
│               ├── ⚙️ carOutput.csv          actuator outputs
│               ├── 🧠 drivingModelData.csv  model predictions & lane detection
│               ├── 📡 radarState.csv        lead vehicle radar data
│               ├── 📐 accelerometer.csv     IMU acceleration data
│               └── 📏 longitudinalPlan.csv  planner targets & FCW
└── ...
```

---

## 📐 Takeover Event Definition

<div align="center">

```
  ◄──────── 10 seconds ────────►◄──────── 10 seconds ────────►
  ┌──────────────────────────────┬──────────────────────────────┐
  │      🤖 ADAS ENGAGED         │      👤 MANUAL CONTROL        │
  │   (automation driving)       │   (driver takes over)        │
  └──────────────────────────────┴──────────────────────────────┘
                                 ▲
                            TAKEOVER EVENT
                         (ON → OFF transition)
```

</div>

A **takeover event** is detected as an ADAS ON → OFF transition satisfying:

| Criterion | Value |
|:---|:---|
| **ADAS engaged** | `controlsState.enabled` OR `cruiseState.enabled` |
| **Min ON duration** | ≥ 2 seconds before disengagement |
| **Min OFF duration** | ≥ 2 seconds after disengagement |
| **Gap merging** | Transient gaps < 0.5s merged (filters sensor noise) |
| **Clip window** | ±10 seconds centered on transition (20s total) |

---

## 📑 Data Fields Reference

### 📋 meta.json — Clip Metadata

| Field | Type | Description |
|:---|:---|:---|
| `car_model` | string | Vehicle model (e.g., `TOYOTA_PRIUS`) |
| `dongle_id` | string | Anonymized driver ID (`driver_XXX`) |
| `route_id` | string | Anonymized route ID (`route_XXX`) |
| `log_kind` | string | Log resolution: `qlog` (10 Hz) or `rlog` (100 Hz) |
| `log_hz` | int | CAN signal sampling rate |
| `vid_kind` | string | Camera source type |
| `camera_fps` | int | Video frame rate (20 fps) |
| `clip_id` | int | Clip index within route (0-indexed) |
| `event_mono` | int | Monotonic timestamp of takeover (ns) |
| `video_time_s` | float | Takeover time within full route video (s) |
| `clip_start_s` | float | Clip start time within route (s) |
| `clip_dur_s` | float | Clip duration (s) |

### 🚗 carState.csv — Vehicle Dynamics & Driver Inputs

| Column | Unit | Description |
|:---|:---|:---|
| `vEgo` | m/s | Ego vehicle speed |
| `aEgo` | m/s² | Ego vehicle acceleration |
| `steeringAngleDeg` | deg | Steering wheel angle |
| `steeringTorque` | N·m | Driver steering torque |
| `steeringPressed` | bool | Driver actively steering |
| `gasPressed` | bool | Gas pedal pressed |
| `brakePressed` | bool | Brake pedal pressed |
| `cruiseState.enabled` | bool | Cruise / ADAS engaged |

### 🤖 controlsState.csv — ADAS Controller

| Column | Unit | Description |
|:---|:---|:---|
| `enabled` | bool | ADAS system enabled |
| `active` | bool | ADAS actively controlling vehicle |
| `curvature` | 1/m | Current path curvature |
| `desiredCurvature` | 1/m | Target curvature from planner |
| `vCruise` | m/s | Set cruise speed |
| `longControlState` | enum | Longitudinal control state |
| `alertText1` | string | Primary driver alert |
| `alertText2` | string | Secondary driver alert |

### 🎮 carControl.csv — Control Commands

| Column | Unit | Description |
|:---|:---|:---|
| `latActive` | bool | Lateral control active |
| `longActive` | bool | Longitudinal control active |
| `actuators.accel` | m/s² | Commanded acceleration |
| `actuators.torque` | N·m | Commanded steering torque |
| `actuators.curvature` | 1/m | Commanded path curvature |

### ⚙️ carOutput.csv — Actuator Outputs

| Column | Description |
|:---|:---|
| `actuatorsOutput.accel` | Acceleration actuator output |
| `actuatorsOutput.brake` | Brake actuator output |
| `actuatorsOutput.gas` | Gas actuator output |
| `actuatorsOutput.steer` | Steering actuator output |
| `actuatorsOutput.steerOutputCan` | Raw CAN steering output |
| `actuatorsOutput.steeringAngleDeg` | Steering angle output (deg) |

### 🧠 drivingModelData.csv — Driving Model Predictions

| Column | Description |
|:---|:---|
| `action.desiredCurvature` | Model-predicted desired curvature |
| `action.desiredAcceleration` | Model-predicted desired acceleration |
| `laneLineMeta.leftProb` | Left lane line detection probability |
| `laneLineMeta.rightProb` | Right lane line detection probability |

### 📡 radarState.csv — Lead Vehicle Detection

| Column | Unit | Description |
|:---|:---|:---|
| `leadOne.dRel` | m | Distance to primary lead vehicle |
| `leadOne.vRel` | m/s | Relative velocity of lead |
| `leadOne.vLead` | m/s | Absolute velocity of lead |
| `leadOne.aLeadK` | m/s² | Lead vehicle acceleration |
| `leadTwo.*` | — | Secondary lead vehicle (same fields) |

### 📐 accelerometer.csv — IMU Data

| Column | Unit | Description |
|:---|:---|:---|
| `acceleration.v` | m/s² | 3-axis acceleration vector |
| `timestamp` | — | Sensor timestamp |

### 📏 longitudinalPlan.csv — Planner Outputs

| Column | Unit | Description |
|:---|:---|:---|
| `aTarget` | m/s² | Target acceleration |
| `hasLead` | bool | Lead vehicle detected |
| `fcw` | bool | Forward collision warning active |
| `speeds[]` | m/s | Planned speed profile |
| `accels[]` | m/s² | Planned acceleration profile |

---

## 🚘 Vehicle Coverage

<div align="center">

**23 Manufacturers · 163 Models · From Compact EVs to Full-Size Trucks**

</div>

### Top Vehicle Models by Clip Count

| # | Vehicle Model | Clips | | # | Vehicle Model | Clips |
|:---:|:---|---:|:---:|:---:|:---|---:|
| 1 | 🏆 RIVIAN R1 GEN1 | 2,127 | | 10 | CHEVROLET BOLT EUV | 244 |
| 2 | 🥈 ACURA MDX 3G | 1,863 | | 11 | TOYOTA RAV4 TSS2 | 228 |
| 3 | 🥉 FORD F-150 MK14 | 1,226 | | 12 | RAM HD 5TH GEN | 221 |
| 4 | CHEVROLET SILVERADO | 639 | | 13 | VOLKSWAGEN JETTA MK7 | 215 |
| 5 | TOYOTA PRIUS | 482 | | 14 | KIA EV6 | 209 |
| 6 | HONDA CIVIC | 470 | | 15 | VOLKSWAGEN GOLF MK7 | 192 |
| 7 | TESLA MODEL 3 | 432 | | 16 | KIA NIRO EV | 185 |
| 8 | FORD MAVERICK MK1 | 300 | | 17 | HYUNDAI IONIQ 6 | 177 |
| 9 | HYUNDAI IONIQ 5 | 266 | | 18 | VOLKSWAGEN ATLAS MK1 | 153 |

<details>
<summary>📋 <b>All 23 Manufacturers</b> (click to expand)</summary>

> Acura · Audi · BYD · Chevrolet · Ford · Genesis · Honda · Hyundai · Jeep · Kia · Lexus · Mazda · Nissan · Porsche · RAM · Rivian · Skoda · Subaru · Tesla · Toyota · Volkswagen · Volvo

</details>

---

## 🚀 Quick Start

### Loading a Single Clip

```python
import json
import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "HenryYHW/ADAS-TO"
clip_path = "TOYOTA_PRIUS/driver_001/route_001/0"

# 📋 Download metadata
meta_path = hf_hub_download(repo_id, f"{clip_path}/meta.json", repo_type="dataset")
with open(meta_path) as f:
    meta = json.load(f)

# 🚗 Load vehicle state signals
car_state = pd.read_csv(
    hf_hub_download(repo_id, f"{clip_path}/carState.csv", repo_type="dataset")
)
print(car_state[["vEgo", "aEgo", "steeringAngleDeg", "brakePressed"]].describe())

# 🤖 Load ADAS controller state
controls = pd.read_csv(
    hf_hub_download(repo_id, f"{clip_path}/controlsState.csv", repo_type="dataset")
)

# 📡 Load radar data
radar = pd.read_csv(
    hf_hub_download(repo_id, f"{clip_path}/radarState.csv", repo_type="dataset")
)
```

### Iterating Over All Clips

```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("HenryYHW/ADAS-TO", repo_type="dataset")
meta_files = [f for f in files if f.endswith("meta.json")]
print(f"Total clips: {len(meta_files)}")  # → 15,705
```

### 💾 Download the Full Dataset

```bash
# Using huggingface-cli (recommended)
huggingface-cli download HenryYHW/ADAS-TO --repo-type dataset --local-dir ./ADAS-TO

# Using git-lfs
git lfs install
git clone https://huggingface.co/datasets/HenryYHW/ADAS-TO
```

---

## 🔒 Privacy & Ethics

- **Anonymized identifiers**: All driver and route IDs are replaced with anonymous tokens (`driver_XXX`, `route_XXX`)
- **Forward-view only**: Video captures road-facing view only — no cabin or driver footage
- **No PII**: No personally identifiable information is included in any data file
- **Community-sourced**: Data collected through autonomous driving enthusiast communities with informed participation

---

## 📖 Data Collection

ADAS-TO was built from naturalistic driving logs contributed by **online and offline autonomous driving communities**. Participating drivers voluntarily shared their driving data collected through various ADAS-equipped vehicles during everyday driving. The raw logs were processed through an automated pipeline to:

1. **Detect** ADAS disengagement events (ON→OFF transitions)
2. **Extract** synchronized video and CAN-bus signals within a 20-second window
3. **Validate** each clip for signal completeness and temporal alignment
4. **Anonymize** all driver and route identifiers

This community-driven collection approach enables unprecedented scale and diversity, capturing genuine driver behavior across a wide spectrum of vehicles, road types, and driving conditions.

---

## 📝 Citation

If you use ADAS-TO in your research, please cite:

```bibtex
@dataset{adas_to_2026,
  title     = {ADAS-TO: A Large-Scale Multimodal Naturalistic Dataset and
               Empirical Characterization of Human Takeovers during ADAS Engagement},
  author    = {Anonymous Authors},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/HenryYHW/ADAS-TO}
}
```

---

## 📄 License

<div align="center">

This dataset is released under [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/).

For academic and non-commercial research purposes.

---

*Built with ❤️ for the autonomous driving research community*

</div>
