#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TakeOver Dataset Builder v2
============================
Uses openpilot LogReader directly (no legacy CAN decoders).

ADAS detection: controlsState.enabled  OR  carState.cruiseState.enabled
  → "op_enable OR acc_enable" as required.

Per route:
  1. Discover log files (prefer rlog > qlog) and video files (prefer fcamera > qcamera).
  2. Build concatenated video.mp4 in target dir (non-last segments normalised to 60 s).
  3. Pass-1  – scan ADAS topics only  → find takeover events (ON→OFF transitions).
  4. Pass-2  – stream all 8 topics   → write 8 topic CSVs per clip.
  5. Cut 20-second video clips per event.

Output naming:  {clip_id}--takeover.mp4
                {clip_id}--{topic}.csv   (8 files, one per topic)
                {clip_id}--meta.json

Memory management:
  - All log data is streamed; only tiny in-memory buffers are kept.
  - gc.collect() after every segment.
  - CSVs written row-by-row (never the full DataFrame in RAM).
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zstandard as zstd
from tqdm import tqdm

# ──────────────────────────────────────────────
#  Default paths  (override via CLI args)
# ──────────────────────────────────────────────
DEFAULT_OPENPILOT_PATH = "/home/henry/Dropbox/OP_CAN_DataProcessing"
DEFAULT_SOURCE_ROOT    = "/home/henry/Desktop/Drive/Dataset"
DEFAULT_TARGET_ROOT    = "/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver"

# ──────────────────────────────────────────────
#  Pipeline constants
# ──────────────────────────────────────────────
DEFAULT_CAMERA_FPS      = 20
SEGMENT_TARGET_SEC      = 60.0   # non-last segments are stretched/compressed to this
CLIP_HALF_WINDOW_S      = 10.0   # ± window around takeover moment
DEFAULT_MIN_ON_S        = 2.0    # minimum ADAS-ON duration to count as a real event
DEFAULT_MIN_OFF_S       = 2.0    # minimum ADAS-OFF duration after event
DEFAULT_GAP_MERGE_S     = 0.5    # merge short OFF gaps within an ON period

# ──────────────────────────────────────────────
#  8 output topics + fields
# ──────────────────────────────────────────────
TOPICS: List[str] = [
    "accelerometer",
    "carControl",
    "carOutput",
    "carState",
    "controlsState",
    "radarState",
    "drivingModelData",
    "longitudinalPlan",
]

TOPIC_FIELDS: Dict[str, List[str]] = {
    "accelerometer": [
        "source", "sensor", "type", "timestamp", "acceleration.v", "accel_status",
    ],
    "carControl": [
        "enabled", "latActive", "longActive", "leftBlinker", "rightBlinker",
        "actuators.accel", "actuators.torque", "actuators.speed", "actuators.curvature",
        "hudControl.setSpeed", "hudControl.leadVisible",
    ],
    "carOutput": [
        "actuatorsOutput.accel", "actuatorsOutput.brake", "actuatorsOutput.gas",
        "actuatorsOutput.speed", "actuatorsOutput.curvature", "actuatorsOutput.steer",
        "actuatorsOutput.steerOutputCan", "actuatorsOutput.steeringAngleDeg",
        "actuatorsOutput.longControlState",
    ],
    "carState": [
        "vEgo", "aEgo", "vEgoRaw", "standstill", "steeringAngleDeg", "steeringTorque",
        "steeringPressed", "gas", "gasPressed", "brake", "brakePressed",
        "cruiseState.enabled", "cruiseState.available", "cruiseState.speed",
    ],
    "controlsState": [
        "enabled", "active", "curvature", "desiredCurvature",
        "vCruise", "vCruiseCluster", "forceDecel", "longControlState",
        "alertText1", "alertText2",
    ],
    "radarState": [
        "cumLagMs",
        "leadOne.status", "leadOne.dRel", "leadOne.vRel", "leadOne.aRel",
        "leadOne.yRel", "leadOne.vLead", "leadOne.vLeadK", "leadOne.aLeadK",
        "leadTwo.status", "leadTwo.dRel", "leadTwo.vRel", "leadTwo.aRel",
        "leadTwo.yRel", "leadTwo.vLead", "leadTwo.vLeadK", "leadTwo.aLeadK",
    ],
    "drivingModelData": [
        "frameId", "frameDropPerc", "modelExecutionTime",
        "action.desiredAcceleration", "action.desiredCurvature", "action.shouldStop",
        "laneLineMeta.leftProb", "laneLineMeta.rightProb",
        "laneLineMeta.leftY",      "laneLineMeta.rightY",
    ],
    "longitudinalPlan": [
        "aTarget", "shouldStop", "allowBrake", "allowThrottle",
        "hasLead", "fcw", "longitudinalPlanSource", "processingDelay",
        "speeds", "accels",
    ],
}

# topics needed only for ADAS detection (Pass 1)
ADAS_DETECT_TOPICS = {"controlsState", "carState"}

TOPICS_SET = set(TOPICS)


# ══════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check
    )


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def numeric_prefix(fname: str) -> Optional[int]:
    m = re.match(r"^(\d+)--", fname)
    return int(m.group(1)) if m else None


def file_hz(log_path: Path) -> int:
    """Detect sampling rate from filename: *--rlog* → 100 Hz, else 10 Hz."""
    return 100 if "--rlog" in log_path.name else 10


def ffprobe_duration(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        cp = run_cmd([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ], check=True)
        s = cp.stdout.strip()
        return float(s) if s else None
    except Exception:
        return None


def parse_route_ids(route_dir: Path) -> Tuple[str, str, str]:
    """Extract (car_model, dongle_id, route_id) from path depth."""
    return route_dir.parent.parent.name, route_dir.parent.name, route_dir.name


def longest_consecutive_run(nums: List[int]) -> List[int]:
    if not nums:
        return []
    nums = sorted(nums)
    best: List[int] = []
    cur = [nums[0]]
    for x in nums[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [x]
    if len(cur) > len(best):
        best = cur
    return best


def flatten_payload(obj: Any, prefix: str, out: Dict[str, Any]) -> None:
    """Recursively flatten a capnp dict into dotted-key dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            flatten_payload(v, p, out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, separators=(",", ":")) if obj else None
    else:
        out[prefix] = obj


def patch_logreader_zstd(logreader_module: Any) -> None:
    """Patch zstd.decompress so bz2-free zstd files work with LogReader."""
    def _stream_decompress(dat: bytes) -> bytes:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(io.BytesIO(dat)) as reader:
            return reader.read()
    logreader_module.zstd.decompress = _stream_decompress


def load_logreader(openpilot_path: str):
    """Import and return LogReader from openpilot installation."""
    p = str(Path(openpilot_path).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    import tools.lib.logreader as lr_mod
    patch_logreader_zstd(lr_mod)
    return lr_mod.LogReader


# ══════════════════════════════════════════════════════════════
#  FILE DISCOVERY
# ══════════════════════════════════════════════════════════════

def discover_route_files(route_dir: Path) -> Dict[str, Dict[int, Path]]:
    """
    Scan route_dir (and common video sub-dirs) for:
      rlog, qlog, fcamera (.hevc), qcamera (.ts), pre-built mp4.
    Returns dict of {kind: {seg_num: Path}}.
    """
    cand_dirs = [route_dir]
    for sub in ("video", "videos", "camera", "cameras"):
        p = route_dir / sub
        if p.is_dir():
            cand_dirs.append(p)

    out: Dict[str, Dict[int, Path]] = {
        "rlog": {}, "qlog": {}, "fcamera": {}, "qcamera": {}, "mp4": {}
    }

    for d in cand_dirs:
        try:
            for fname in os.listdir(d):
                fpath = d / fname
                if not fpath.is_file():
                    continue
                n = numeric_prefix(fname)
                if n is None:
                    continue

                # rlog: raw or bz2
                if re.fullmatch(r"\d+--rlog(\.bz2)?", fname):
                    if n not in out["rlog"]:           # prefer raw over bz2
                        out["rlog"][n] = fpath
                    elif not fname.endswith(".bz2"):   # raw wins
                        out["rlog"][n] = fpath
                # qlog: raw, bz2, or zst
                elif re.fullmatch(r"\d+--qlog(\.bz2)?", fname) or \
                     re.fullmatch(r"\d+--qlog.*\.zst", fname):
                    if n not in out["qlog"]:
                        out["qlog"][n] = fpath
                    elif not (fname.endswith(".bz2") or fname.endswith(".zst")):
                        out["qlog"][n] = fpath
                # video
                elif re.fullmatch(r"\d+--fcamera\.hevc", fname):
                    out["fcamera"][n] = fpath
                elif re.fullmatch(r"\d+--qcamera\.ts", fname):
                    out["qcamera"][n] = fpath
                elif re.fullmatch(r"\d+--.*\.mp4", fname):
                    out["mp4"][n] = fpath
        except (FileNotFoundError, PermissionError):
            pass

    return out


def choose_best_pair(
    files: Dict[str, Dict[int, Path]]
) -> Optional[Tuple[str, Dict[int, Path], str, Dict[int, Path], List[int]]]:
    """
    Pick (log_kind, logs, vid_kind, vids, seg_nums) maximising:
      - longest consecutive common segment run
      - rlog > qlog
      - fcamera > qcamera > mp4
    """
    log_prio = {"rlog": 20, "qlog": 0}
    vid_prio = {"fcamera": 20, "qcamera": 10, "mp4": 0}

    log_candidates = [(k, files[k]) for k in ("rlog", "qlog") if files[k]]
    vid_candidates = [(k, files[k]) for k in ("fcamera", "qcamera", "mp4") if files[k]]

    if not log_candidates or not vid_candidates:
        return None

    best: Optional[Tuple] = None
    best_score = -1

    for lk, logs in log_candidates:
        for vk, vids in vid_candidates:
            common = sorted(set(logs) & set(vids))
            if not common:
                continue
            run = longest_consecutive_run(common)
            if not run:
                continue
            score = len(run) * 100 + log_prio[lk] + vid_prio[vk]
            if score > best_score:
                best_score = score
                best = (lk, logs, vk, vids, run)

    return best


# ══════════════════════════════════════════════════════════════
#  VIDEO PROCESSING  (kept from build_takeover_dataset.py)
# ══════════════════════════════════════════════════════════════

def transcode_keep_duration(in_path: Path, out_mp4: Path, camera_fps: int) -> None:
    """Convert to MP4, keep original duration (used for last segment)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    input_opts = ["-r", str(camera_fps)] if in_path.suffix.lower() == ".hevc" else []
    run_cmd([
        "ffmpeg", "-y",
        *input_opts, "-i", str(in_path),
        "-an",
        "-vf", f"fps={camera_fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ], check=True)


def transcode_normalize_to_60(
    in_path: Path, out_mp4: Path, camera_fps: int, target_sec: float
) -> None:
    """Convert and time-stretch to exactly target_sec (used for non-last segments)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    dur_in = ffprobe_duration(in_path)
    scale = float(target_sec) / float(dur_in) if (dur_in and dur_in >= 0.5) else 1.0
    input_opts = ["-r", str(camera_fps)] if in_path.suffix.lower() == ".hevc" else []
    run_cmd([
        "ffmpeg", "-y",
        *input_opts, "-i", str(in_path),
        "-an",
        "-vf", f"setpts=PTS*{scale:.12f},fps={camera_fps}",
        "-t", f"{target_sec:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ], check=True)


def concat_mp4_segments(mp4_paths: List[Path], out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        list_path = Path(f.name)
        for p in mp4_paths:
            f.write(f"file '{str(p)}'\n")
    try:
        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(out_path),
        ], check=True)
    finally:
        list_path.unlink(missing_ok=True)


def build_route_video_mp4(
    out_dir: Path,
    vid_kind: str,
    vids: Dict[int, Path],
    seg_nums: List[int],
    camera_fps: int,
    overwrite: bool,
    target_sec: float,
) -> Tuple[Optional[Path], Optional[float], Dict[int, float]]:
    """
    Build concatenated video.mp4 in out_dir.
    Non-last segments are normalised to target_sec; last keeps its real duration.
    Returns (video_path, total_duration_s, per_seg_duration_s).
    """
    out_video = out_dir / "video.mp4"
    if out_video.exists() and not overwrite:
        dur = ffprobe_duration(out_video)
        if dur:
            return out_video, dur, {}

    safe_mkdir(out_dir)
    tmp_dir = out_dir / ".tmp_video_parts"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    last_seg = seg_nums[-1]
    mp4_parts: List[Path] = []
    per_seg_dur: Dict[int, float] = {}

    for n in seg_nums:
        if n not in vids:
            continue
        in_path = vids[n]
        out_part = tmp_dir / f"{n:06d}.mp4"
        try:
            if n == last_seg:
                transcode_keep_duration(in_path, out_part, camera_fps)
            else:
                transcode_normalize_to_60(in_path, out_part, camera_fps, target_sec)
            d = ffprobe_duration(out_part)
            if d is not None:
                per_seg_dur[n] = d
            mp4_parts.append(out_part)
        except Exception as e:
            print(f"  [WARN] video part failed {in_path}: {e}")

    if not mp4_parts:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None, {}

    mp4_parts = sorted(mp4_parts, key=lambda p: int(p.stem))
    try:
        concat_mp4_segments(mp4_parts, out_video)
        dur = ffprobe_duration(out_video)
        return out_video, dur, per_seg_dur
    except Exception as e:
        print(f"  [WARN] concat failed: {e}")
        return None, None, {}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()


def cut_video_clip(
    in_video: Path, out_video: Path, start_s: float, dur_s: float
) -> None:
    safe_mkdir(out_video.parent)
    run_cmd([
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, start_s):.6f}",
        "-i", str(in_video),
        "-t", f"{max(0.05, dur_s):.6f}",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_video),
    ], check=True)


# ══════════════════════════════════════════════════════════════
#  PASS 1 – ADAS STATUS DETECTION
# ══════════════════════════════════════════════════════════════

def _detect_adas_pass1(
    log_files: Dict[int, Path],
    seg_nums: List[int],
    LogReader,
    min_on_s: float,
    min_off_s: float,
    gap_merge_s: float,
) -> Tuple[List[int], Dict[int, int]]:
    """
    Scan controlsState.enabled and carState.cruiseState.enabled.
    Returns:
      event_monos  – list of logMonoTime at each ON→OFF transition
      seg_first_mono – {seg_num: first logMonoTime in that segment}
    """
    ctrl_stream: List[Tuple[int, bool]] = []    # (mono, enabled)
    cruise_stream: List[Tuple[int, bool]] = []
    seg_first_mono: Dict[int, int] = {}

    for seg_num in sorted(seg_nums):
        if seg_num not in log_files:
            continue
        seg_first: Optional[int] = None
        try:
            for msg in LogReader(
                str(log_files[seg_num]),
                only_union_types=True,
                sort_by_time=True,
            ):
                topic = msg.which()
                if topic not in ADAS_DETECT_TOPICS:
                    continue
                mono = int(msg.logMonoTime)
                if seg_first is None:
                    seg_first = mono
                    seg_first_mono[seg_num] = mono

                payload = getattr(msg, topic).to_dict(verbose=True)
                flat: Dict[str, Any] = {}
                flatten_payload(payload, "", flat)

                if topic == "controlsState":
                    ctrl_stream.append((mono, bool(flat.get("enabled", False))))
                elif topic == "carState":
                    cruise_stream.append((mono, bool(flat.get("cruiseState.enabled", False))))
        except Exception as e:
            print(f"  [WARN] Pass-1 seg {seg_num}: {e}")
        finally:
            gc.collect()

    if not ctrl_stream and not cruise_stream:
        return [], seg_first_mono

    # ── merge two streams into unified (mono, enabled) timeline ──────────────
    combined: List[Tuple[int, str, bool]] = (
        [(m, "ctrl",   e) for m, e in ctrl_stream]
        + [(m, "cruise", e) for m, e in cruise_stream]
    )
    combined.sort(key=lambda x: x[0])
    del ctrl_stream, cruise_stream
    gc.collect()

    last_ctrl = last_cruise = False
    unified_mono: List[int] = []
    unified_en:   List[bool] = []
    for mono, src, en in combined:
        if src == "ctrl":
            last_ctrl = en
        else:
            last_cruise = en
        unified_mono.append(mono)
        unified_en.append(last_ctrl or last_cruise)
    del combined
    gc.collect()

    # ── merge short OFF gaps ──────────────────────────────────────────────────
    if gap_merge_s > 0 and unified_mono:
        gap_ns = int(gap_merge_s * 1e9)
        i = 0
        n = len(unified_en)
        while i < n:
            if not unified_en[i]:
                j = i
                while j < n and not unified_en[j]:
                    j += 1
                if 0 < i and j < n:
                    if unified_mono[j] - unified_mono[i] <= gap_ns:
                        for k in range(i, j):
                            unified_en[k] = True
                i = j
            else:
                i += 1

    # ── find ON→OFF transitions with debouncing ───────────────────────────────
    min_on_ns  = int(min_on_s  * 1e9)
    min_off_ns = int(min_off_s * 1e9)
    min_sep_ns = int(1.0       * 1e9)   # min gap between events

    events: List[int] = []
    on_start: Optional[int] = None

    i = 0
    n = len(unified_en)
    while i < n:
        en   = unified_en[i]
        mono = unified_mono[i]

        if en and on_start is None:
            on_start = mono
        elif not en and on_start is not None:
            on_dur = mono - on_start
            # look ahead for OFF duration
            j = i
            while j < n and not unified_en[j]:
                j += 1
            off_dur = (unified_mono[j] - mono) if j < n else min_off_ns + 1

            if on_dur >= min_on_ns and off_dur >= min_off_ns:
                if not events or (mono - events[-1] >= min_sep_ns):
                    events.append(mono)

            on_start = unified_mono[j] if j < n and unified_en[j] else None
            i = j
            continue
        i += 1

    del unified_mono, unified_en
    gc.collect()
    return events, seg_first_mono


# ══════════════════════════════════════════════════════════════
#  VIDEO TIME MAPPING
# ══════════════════════════════════════════════════════════════

def mono_to_video_time(
    event_mono: int,
    seg_first_mono: Dict[int, int],
    per_seg_dur: Dict[int, float],
    seg_nums: List[int],
    target_sec: float,
) -> Optional[float]:
    """
    Map a logMonoTime to seconds in the concatenated video.mp4.

    Non-last segments are normalised to target_sec; the last keeps its
    actual duration (per_seg_dur[last]).  Within each segment we scale
    linearly from the raw log time.
    """
    sorted_segs = sorted(seg_nums)
    last_seg = sorted_segs[-1]

    # find which segment this mono belongs to
    event_seg: Optional[int] = None
    for seg in sorted_segs:
        if seg in seg_first_mono and seg_first_mono[seg] <= event_mono:
            event_seg = seg
    if event_seg is None:
        return None

    # cumulative video offset before event_seg
    video_offset = 0.0
    for seg in sorted_segs:
        if seg >= event_seg:
            break
        seg_vid_dur = per_seg_dur.get(seg, target_sec) if seg == last_seg else target_sec
        video_offset += seg_vid_dur

    # within-segment time (raw log seconds)
    within_raw = (event_mono - seg_first_mono[event_seg]) / 1e9

    # scale to normalised video segment duration
    # (non-last segs are normalised to target_sec)
    actual_seg_dur = per_seg_dur.get(event_seg, target_sec)
    seg_target = actual_seg_dur if event_seg == last_seg else target_sec

    # estimate raw log duration for this segment (approximate)
    # use the next segment's start if available
    idx = sorted_segs.index(event_seg)
    if idx + 1 < len(sorted_segs) and sorted_segs[idx + 1] in seg_first_mono:
        raw_log_dur = (seg_first_mono[sorted_segs[idx + 1]] - seg_first_mono[event_seg]) / 1e9
    else:
        raw_log_dur = target_sec  # fallback

    scale = seg_target / raw_log_dur if raw_log_dur > 0 else 1.0
    within_vid = min(within_raw * scale, seg_target)

    return video_offset + within_vid


# ══════════════════════════════════════════════════════════════
#  PASS 2 – EXPORT 8 TOPIC CSVs
# ══════════════════════════════════════════════════════════════

def _get_segs_for_clip(
    event_mono: int,
    clip_half_ns: int,
    seg_first_mono: Dict[int, int],
    seg_nums: List[int],
) -> List[int]:
    """Return the segment numbers whose time range overlaps this clip window."""
    clip_start = event_mono - clip_half_ns
    clip_end   = event_mono + clip_half_ns
    sorted_segs = sorted(seg_nums)
    needed: List[int] = []
    for i, seg in enumerate(sorted_segs):
        if seg not in seg_first_mono:
            continue
        seg_start = seg_first_mono[seg]
        # estimate segment end from next segment's start
        if i + 1 < len(sorted_segs) and sorted_segs[i + 1] in seg_first_mono:
            seg_end = seg_first_mono[sorted_segs[i + 1]]
        else:
            seg_end = seg_start + int(70 * 1e9)   # 70-second buffer for last seg
        if seg_start <= clip_end and seg_end >= clip_start:
            needed.append(seg)
    return needed


def _export_topic_csvs(
    log_files: Dict[int, Path],
    events_info: List[Dict],
    out_dir: Path,
    seg_first_mono: Dict[int, int],
    seg_nums: List[int],
    LogReader,
    overwrite: bool,
) -> None:
    """
    Pass-2: stream through only the relevant log segments and write
    {clip_id}--{topic}.csv for each event × topic combination.
    """
    if not events_info:
        return

    clip_half_ns = int(CLIP_HALF_WINDOW_S * 1e9)
    route_first_mono: Optional[int] = (
        min(seg_first_mono.values()) if seg_first_mono else None
    )
    base_fields = ["logMonoTime", "time_s"]

    # ── open CSV writers for every (clip_id, topic) ──────────────────────────
    writers: Dict[Tuple[int, str], Tuple[Any, Any]] = {}
    for ev in events_info:
        cid = ev["clip_id"]
        clip_dir = ev["clip_dir"]
        for topic in TOPICS:
            out_path = clip_dir / f"{topic}.csv"
            if out_path.exists() and not overwrite:
                continue
            fields = base_fields + TOPIC_FIELDS[topic]
            fp = out_path.open("w", newline="", encoding="utf-8")
            w  = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            writers[(cid, topic)] = (fp, w)

    # ── group events by needed segments  ─────────────────────────────────────
    seg_to_events: Dict[int, List[Dict]] = {}
    for ev in events_info:
        for seg in _get_segs_for_clip(
            ev["event_mono"], clip_half_ns, seg_first_mono, seg_nums
        ):
            seg_to_events.setdefault(seg, []).append(ev)

    # ── stream through each needed segment once ───────────────────────────────
    for seg_num in sorted(seg_to_events):
        if seg_num not in log_files:
            continue

        ev_list = seg_to_events[seg_num]
        clip_ranges = [
            (ev["clip_id"], ev["event_mono"] - clip_half_ns, ev["event_mono"] + clip_half_ns)
            for ev in ev_list
        ]

        try:
            for msg in LogReader(
                str(log_files[seg_num]),
                only_union_types=True,
                sort_by_time=True,
            ):
                topic = msg.which()
                if topic not in TOPICS_SET:
                    continue
                mono = int(msg.logMonoTime)
                if route_first_mono is None:
                    route_first_mono = mono

                for clip_id, start_mono, end_mono in clip_ranges:
                    if not (start_mono <= mono <= end_mono):
                        continue
                    key = (clip_id, topic)
                    if key not in writers:
                        continue

                    payload = getattr(msg, topic).to_dict(verbose=True)
                    flat: Dict[str, Any] = {}
                    flatten_payload(payload, "", flat)

                    t_s = round((mono - (route_first_mono or mono)) / 1e9, 6)
                    row: Dict[str, Any] = {"logMonoTime": mono, "time_s": t_s}
                    for f in TOPIC_FIELDS[topic]:
                        row[f] = flat.get(f)

                    writers[key][1].writerow(row)

        except Exception as e:
            print(f"  [WARN] Pass-2 seg {seg_num}: {e}")
        finally:
            gc.collect()

    for fp, _ in writers.values():
        fp.close()


# ══════════════════════════════════════════════════════════════
#  CLEANUP HELPERS
# ══════════════════════════════════════════════════════════════

def _cleanup_empty_route_dir(out_dir: Path) -> None:
    """Remove route output dir if it contains no clip subdirectories."""
    if not out_dir.exists():
        return
    try:
        has_clips = any(d.is_dir() and d.name.isdigit() for d in out_dir.iterdir())
        if not has_clips:
            shutil.rmtree(out_dir, ignore_errors=True)
    except Exception:
        pass


def _prune_empty_parent_dirs(target_root: Path) -> None:
    """Remove empty dongle_id and car_model dirs under target_root."""
    if not target_root.exists():
        return
    for car_dir in sorted(target_root.iterdir()):
        if not car_dir.is_dir():
            continue
        for dongle_dir in sorted(car_dir.iterdir()):
            if not dongle_dir.is_dir():
                continue
            try:
                if not any(dongle_dir.iterdir()):
                    dongle_dir.rmdir()
                    print(f"[CLEANUP] removed empty dongle dir: {dongle_dir.name}")
            except Exception:
                pass
        try:
            if not any(car_dir.iterdir()):
                car_dir.rmdir()
                print(f"[CLEANUP] removed empty car dir: {car_dir.name}")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
#  PROCESS ONE ROUTE
# ══════════════════════════════════════════════════════════════

def process_one_route(
    route_dir: Path,
    target_root: Path,
    LogReader,
    camera_fps: int,
    overwrite: bool,
    min_on_s: float,
    min_off_s: float,
    gap_merge_s: float,
    segment_target_sec: float,
    keep_full_video: bool,
) -> int:
    """
    Full pipeline for a single route_dir.
    Returns number of takeover clips exported.
    """
    car_model, dongle_id, route_id = parse_route_ids(route_dir)
    out_dir = target_root / car_model / dongle_id / route_id

    # ── discover files ────────────────────────────────────────────────────────
    files = discover_route_files(route_dir)
    best  = choose_best_pair(files)
    if best is None:
        return 0
    log_kind, logs, vid_kind, vids, seg_nums = best
    if not seg_nums:
        return 0

    # Detect Hz from the actual first log filename (*--rlog* → 100, else → 10)
    first_log = logs[seg_nums[0]]
    log_hz = file_hz(first_log)

    # ── Clean up old clip directories when overwriting ──────────────────────
    if overwrite and out_dir.exists():
        for d in list(out_dir.iterdir()):
            if d.is_dir() and d.name.isdigit():
                shutil.rmtree(d, ignore_errors=True)

    # ── Step 1: build concatenated video.mp4 ─────────────────────────────────
    safe_mkdir(out_dir)
    video_path, video_dur, per_seg_dur = build_route_video_mp4(
        out_dir=out_dir,
        vid_kind=vid_kind,
        vids=vids,
        seg_nums=seg_nums,
        camera_fps=camera_fps,
        overwrite=overwrite,
        target_sec=segment_target_sec,
    )
    if video_path is None or not video_dur or video_dur < 1.0:
        _cleanup_empty_route_dir(out_dir)
        return 0

    # ── Step 2: Pass 1 – detect takeover events ───────────────────────────────
    active_logs: Dict[int, Path] = {n: logs[n] for n in seg_nums if n in logs}

    event_monos, seg_first_mono = _detect_adas_pass1(
        log_files=active_logs,
        seg_nums=seg_nums,
        LogReader=LogReader,
        min_on_s=min_on_s,
        min_off_s=min_off_s,
        gap_merge_s=gap_merge_s,
    )

    if not event_monos:
        if not keep_full_video:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass
        _cleanup_empty_route_dir(out_dir)
        return 0

    print(f"  → {len(event_monos)} takeover event(s) found in {route_id}")

    # ── Step 3: compute clip windows, cut video clips ─────────────────────────
    clip_half_ns = int(CLIP_HALF_WINDOW_S * 1e9)
    events_info: List[Dict] = []

    for clip_id, event_mono in enumerate(event_monos):
        video_t = mono_to_video_time(
            event_mono, seg_first_mono, per_seg_dur, seg_nums, segment_target_sec
        )
        if video_t is None:
            print(f"  [WARN] Cannot map mono→video for clip {clip_id}, skipping.")
            continue

        clip_start_s = max(0.0, video_t - CLIP_HALF_WINDOW_S)
        clip_dur_s   = min(2 * CLIP_HALF_WINDOW_S, video_dur - clip_start_s)
        if clip_dur_s < 3.0:
            continue

        clip_dir = out_dir / str(clip_id)
        safe_mkdir(clip_dir)
        out_video = clip_dir / "takeover.mp4"
        if not out_video.exists() or overwrite:
            try:
                cut_video_clip(video_path, out_video, clip_start_s, clip_dur_s)
            except Exception as ex:
                print(f"  [WARN] cut clip {clip_id} failed: {ex}")
                continue

        ev = {
            "clip_id":          clip_id,
            "clip_dir":         clip_dir,
            "event_mono":       event_mono,
            "clip_start_mono":  event_mono - clip_half_ns,
            "clip_end_mono":    event_mono + clip_half_ns,
            "video_time_s":     video_t,
            "clip_start_s":     clip_start_s,
            "clip_dur_s":       clip_dur_s,
        }
        events_info.append(ev)

        meta = {
            "car_model":    car_model,
            "dongle_id":    dongle_id,
            "route_id":     route_id,
            "log_kind":     log_kind,
            "log_hz":       log_hz,
            "vid_kind":     vid_kind,
            "camera_fps":   camera_fps,
            "clip_id":      clip_id,
            "event_mono":   event_mono,
            "video_time_s": video_t,
            "clip_start_s": clip_start_s,
            "clip_dur_s":   clip_dur_s,
            "seg_nums_used": seg_nums,
        }
        with (clip_dir / "meta.json").open("w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    if not events_info:
        if not keep_full_video:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass
        _cleanup_empty_route_dir(out_dir)
        return 0

    # ── Step 4: Pass 2 – export 8 topic CSVs per event ───────────────────────
    _export_topic_csvs(
        log_files=active_logs,
        events_info=events_info,
        out_dir=out_dir,
        seg_first_mono=seg_first_mono,
        seg_nums=seg_nums,
        LogReader=LogReader,
        overwrite=overwrite,
    )

    # ── Clean up full video if not needed ─────────────────────────────────────
    if not keep_full_video:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass

    gc.collect()
    return len(events_info)


# ══════════════════════════════════════════════════════════════
#  ITERATE ALL ROUTES
# ══════════════════════════════════════════════════════════════

def iter_routes(source_root: Path):
    """Yield every (car_model/dongle_id/route_id) directory."""
    skip_names = {"None", "mock", "MOCK", "EV6_platoon"}
    for car_dir in sorted(source_root.iterdir()):
        if not car_dir.is_dir() or car_dir.name in skip_names:
            continue
        for dongle_dir in sorted(car_dir.iterdir()):
            if not dongle_dir.is_dir():
                continue
            for route_dir in sorted(dongle_dir.iterdir()):
                if route_dir.is_dir():
                    yield route_dir


# ══════════════════════════════════════════════════════════════
#  MULTIPROCESSING HELPERS
# ══════════════════════════════════════════════════════════════

_worker_LogReader = None  # per-process global set by initializer


def _worker_init(openpilot_path: str) -> None:
    global _worker_LogReader
    _worker_LogReader = load_logreader(openpilot_path)


def _worker_process_route(kwargs: dict) -> Tuple[int, str]:
    kwargs = dict(kwargs)  # shallow copy to avoid mutating shared dict
    label = kwargs.pop("_label", str(kwargs.get("route_dir", "?")))
    try:
        n = process_one_route(LogReader=_worker_LogReader, **kwargs)
        return n, label
    except Exception as e:
        print(f"[WARN] worker failed {kwargs.get('route_dir')}: {e}")
        return 0, label


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build TakeOver dataset v2.\n"
            "ADAS detection: controlsState.enabled OR carState.cruiseState.enabled.\n"
            "Outputs: {clip_id}--takeover.mp4, {clip_id}--{topic}.csv (×8), {clip_id}--meta.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--openpilot-path", default=DEFAULT_OPENPILOT_PATH,
        help="Path to openpilot (or OP_CAN_DataProcessing) containing tools/lib/logreader.py",
    )
    ap.add_argument("--source-root",  default=DEFAULT_SOURCE_ROOT,  help="Root of source dataset")
    ap.add_argument("--target-root",  default=DEFAULT_TARGET_ROOT,  help="Root of output dataset")
    ap.add_argument(
        "--single-route", default=None,
        help="Process only this route directory (for testing)",
    )
    ap.add_argument("--camera-fps",    type=int,   default=DEFAULT_CAMERA_FPS)
    ap.add_argument("--segment-sec",   type=float, default=SEGMENT_TARGET_SEC,
                    help="Duration non-last video segments are normalised to")
    ap.add_argument("--overwrite",     action="store_true", help="Re-process even if output exists")
    ap.add_argument("--min-on-s",      type=float, default=DEFAULT_MIN_ON_S,
                    help="Minimum ADAS-ON duration to qualify as a real engagement")
    ap.add_argument("--min-off-s",     type=float, default=DEFAULT_MIN_OFF_S,
                    help="Minimum ADAS-OFF duration after event (debounce)")
    ap.add_argument("--gap-merge-s",   type=float, default=DEFAULT_GAP_MERGE_S,
                    help="Merge OFF gaps shorter than this inside an ON period")
    ap.add_argument("--keep-full-video", action="store_true",
                    help="Keep the full route video.mp4 in output dir after clipping")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel worker processes (default: 1)")
    ap.add_argument("--verbose",       action="store_true")
    args = ap.parse_args()

    # verify ffmpeg
    try:
        run_cmd(["ffmpeg",  "-version"], check=True)
        run_cmd(["ffprobe", "-version"], check=True)
    except Exception:
        print("[ERROR] ffmpeg/ffprobe not found in PATH")
        sys.exit(1)

    LogReader = load_logreader(args.openpilot_path)

    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()

    if args.single_route:
        route_dir = Path(args.single_route).expanduser().resolve()
        n = process_one_route(
            route_dir=route_dir,
            target_root=target_root,
            LogReader=LogReader,
            camera_fps=args.camera_fps,
            overwrite=args.overwrite,
            min_on_s=args.min_on_s,
            min_off_s=args.min_off_s,
            gap_merge_s=args.gap_merge_s,
            segment_target_sec=args.segment_sec,
            keep_full_video=args.keep_full_video,
        )
        print(f"[DONE] single route → {n} takeover clip(s) exported")
        _prune_empty_parent_dirs(target_root)
        return

    all_routes = list(iter_routes(source_root))
    common_kwargs = dict(
        target_root=target_root,
        camera_fps=args.camera_fps,
        overwrite=args.overwrite,
        min_on_s=args.min_on_s,
        min_off_s=args.min_off_s,
        gap_merge_s=args.gap_merge_s,
        segment_target_sec=args.segment_sec,
        keep_full_video=args.keep_full_video,
    )
    work_items = [{"route_dir": r, **common_kwargs} for r in all_routes]

    total_routes = total_clips = 0

    # Pre-compute labels for all routes
    route_labels = []
    for item in work_items:
        try:
            car, dongle, route = parse_route_ids(Path(item["route_dir"]))
            route_labels.append(f"{car}/{dongle}/{route}")
        except Exception:
            route_labels.append(str(item["route_dir"]))

    pbar = tqdm(total=len(work_items), desc="Routes",
                bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}] {postfix}")

    if args.workers > 1:
        # Inject label into each work item so workers can return it
        for item, label in zip(work_items, route_labels):
            item["_label"] = label

        with multiprocessing.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(args.openpilot_path,),
        ) as pool:
            for n, label in pool.imap_unordered(_worker_process_route, work_items):
                total_routes += 1
                total_clips  += n
                pbar.set_postfix_str(label, refresh=True)
                pbar.update(1)
    else:
        for item, label in zip(work_items, route_labels):
            pbar.set_postfix_str(label, refresh=True)
            try:
                n = process_one_route(LogReader=LogReader, **item)
                total_routes += 1
                total_clips  += n
            except Exception as e:
                print(f"[WARN] route failed {item['route_dir']}: {e}")
            finally:
                gc.collect()
            pbar.update(1)

    pbar.close()

    _prune_empty_parent_dirs(target_root)
    print(f"\n[DONE] routes processed: {total_routes}")
    print(f"[DONE] takeover clips exported: {total_clips}")
    print(f"[OUT]  dataset saved under: {target_root}")


if __name__ == "__main__":
    main()
