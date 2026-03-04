#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import gc
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# Your project paths
# =========================
sys.path.append('/home/henry/Dropbox/OP_CAN_DataProcessing')
sys.path.append('/home/henry/Dropbox/EV data 2')

from tools.lib.logreader import LogReader
from ReadLogOpAttr import read_route_log_into_df
from ReadRlogOpAttr_new import read_route_log_into_df_new

from CAN_decoder_functions import toyota_can_decoder as toyota_car_specific_can_decoder
from CAN_decoder_functions import TOYOTA_msg_dict as toyota_car_can_msg_dict

from CAN_decoder_functions import tesla_model3_can_decoder as tesla_car_specific_can_decoder
from CAN_decoder_functions import TESLA3_can_msg_dict as tesla_car_can_msg_dict

from CAN_decoder_functions import kia_ev6_can_decoder as ev6_car_specific_can_decoder
from CAN_decoder_functions import EV6_can_msg_dict as ev6_car_can_msg_dict

from CAN_decoder_functions import ioniq_can_decoder as ioniq_car_specific_can_decoder
from CAN_decoder_functions import IONIQ5_can_msg_dict as ioniq_car_can_msg_dict

from CAN_decoder_functions import mache_can_decoder as ford_car_specific_can_decoder
from CAN_decoder_functions import MACHE_msg_dict as ford_car_can_msg_dict

from CAN_decoder_functions import accord_can_decoder as honda_car_specific_can_decoder
from CAN_decoder_functions import ACCORD_can_msg_dict as honda_car_can_msg_dict

from CAN_decoder_functions import volkswagen_can_decoder as volkswagen_car_specific_can_decoder
from CAN_decoder_functions import volkswagen_can_msg_dict as volkswagen_car_can_msg_dict


# =========================
# Config
# =========================
DEFAULT_SOURCE_ROOT = "/home/henry/Desktop/Drive/Dataset"
DEFAULT_TARGET_ROOT = "/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver"

DEFAULT_CAMERA_FPS = 20

# 仅“非最后一段”强制归一化到 60s
SEGMENT_TARGET_SEC = 60.0

CLIP_HALF_WINDOW_S = 10.0  # takeover 前后各 10 秒

# 去抖参数（秒）
DEFAULT_MIN_ON_S = 2.0
DEFAULT_MIN_OFF_S = 2.0
DEFAULT_GAP_MERGE_S = 0.5  # 合并 ON 中短 OFF 缺口（秒）；0 关闭


# =========================
# Utils
# =========================
def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)

def ensure_ffmpeg() -> None:
    run_cmd(["ffmpeg", "-version"], check=True)
    run_cmd(["ffprobe", "-version"], check=True)

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def numeric_prefix(fname: str) -> Optional[int]:
    m = re.match(r"^(\d+)--", fname)
    return int(m.group(1)) if m else None

def ffprobe_duration(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        cp = run_cmd([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ], check=True)
        s = cp.stdout.strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def parse_route_ids(route_dir: Path) -> Tuple[str, str, str]:
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


# =========================
# CAN decoder selection
# =========================
def pick_can_decoder(path_hint: str):
    if "FORD" in path_hint or "MachE" in path_hint:
        return ford_car_specific_can_decoder, ford_car_can_msg_dict
    if "HONDA" in path_hint:
        return honda_car_specific_can_decoder, honda_car_can_msg_dict
    if "IONIQ" in path_hint or "HYUNDAI" in path_hint:
        return ioniq_car_specific_can_decoder, ioniq_car_can_msg_dict
    if "EV6" in path_hint or "KIA" in path_hint or "Kia" in path_hint:
        return ev6_car_specific_can_decoder, ev6_car_can_msg_dict
    if "TESLA" in path_hint or "RAVEN" in path_hint or "tesla" in path_hint or "MODEL_3" in path_hint:
        return tesla_car_specific_can_decoder, tesla_car_can_msg_dict
    if "VOLKSWAGEN" in path_hint:
        return volkswagen_car_specific_can_decoder, volkswagen_car_can_msg_dict
    return toyota_car_specific_can_decoder, toyota_car_can_msg_dict

from pathlib import Path
import pandas as pd
import traceback

def log2df(log_path: Path) -> pd.DataFrame:
    df = pd.DataFrame()  # ✅ 先定义，保证即使失败也能返回
    car_specific_can_decoder, car_can_msg_dict = pick_can_decoder(str(log_path))
    try:
        df, *_ = read_route_log_into_df(
            LogReader(str(log_path)),
            realign_index_name='vEgo',
            can_msg_extension=car_can_msg_dict,
            can_decoder_fn=car_specific_can_decoder,
        )
    except Exception as e:
        print("incorrect!", log_path, repr(e))
        traceback.print_exc()  # ✅ 把真实异常打印出来
    return df



# =========================
# Discover files
# =========================
def discover_route_files(route_dir: Path) -> Dict[str, Dict[int, Path]]:
    cand_dirs = [route_dir]
    for sub in ["video", "videos", "camera", "cameras"]:
        p = route_dir / sub
        if p.exists() and p.is_dir():
            cand_dirs.append(p)

    out = {"rlog": {}, "qlog": {}, "fcamera": {}, "qcamera": {}, "mp4": {}}

    for d in cand_dirs:
        try:
            for fname in os.listdir(d):
                fpath = d / fname
                if not fpath.is_file():
                    continue
                n = numeric_prefix(fname)
                if n is None:
                    continue

                if re.match(r"^\d+--rlog$", fname):
                    out["rlog"][n] = fpath
                elif re.match(r"^\d+--qlog$", fname):
                    out["qlog"][n] = fpath
                elif re.match(r"^\d+--fcamera\.hevc$", fname):
                    out["fcamera"][n] = fpath
                elif re.match(r"^\d+--qcamera\.ts$", fname):
                    out["qcamera"][n] = fpath
                elif re.match(r"^\d+--.*\.mp4$", fname):
                    out["mp4"][n] = fpath
        except FileNotFoundError:
            pass

    return out

def choose_best_pair(files: Dict[str, Dict[int, Path]]) -> Optional[Tuple[str, Dict[int, Path], str, Dict[int, Path], List[int]]]:
    log_candidates = []
    if files["rlog"]:
        log_candidates.append(("rlog", files["rlog"]))
    if files["qlog"]:
        log_candidates.append(("qlog", files["qlog"]))

    vid_candidates = []
    if files["fcamera"]:
        vid_candidates.append(("fcamera", files["fcamera"]))
    if files["qcamera"]:
        vid_candidates.append(("qcamera", files["qcamera"]))
    if files["mp4"]:
        vid_candidates.append(("mp4", files["mp4"]))

    if not log_candidates or not vid_candidates:
        return None

    log_bonus = {"rlog": 2, "qlog": 0}
    vid_bonus = {"fcamera": 2, "qcamera": 1, "mp4": 0}

    best = None
    best_score = -1

    for lk, logs in log_candidates:
        for vk, vids in vid_candidates:
            common = sorted(set(logs.keys()) & set(vids.keys()))
            if not common:
                continue
            run = longest_consecutive_run(common)
            if not run:
                continue
            score = len(run) * 100 + log_bonus.get(lk, 0) * 10 + vid_bonus.get(vk, 0) * 10
            if score > best_score:
                best_score = score
                best = (lk, logs, vk, vids, run)

    return best


# =========================
# Video: normalize all BUT last segment to 60s, last keeps real duration
# =========================
def transcode_keep_duration(in_path: Path, out_mp4: Path, camera_fps: int) -> None:
    """
    Convert to MP4 but keep original duration (no setpts stretch, no tpad-to-60).
    For raw .hevc use -r fps to make timestamps.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    suffix = in_path.suffix.lower()

    input_opts = []
    if suffix == ".hevc":
        input_opts = ["-r", str(camera_fps)]

    cmd = [
        "ffmpeg", "-y",
        *input_opts,
        "-i", str(in_path),
        "-an",
        "-vf", f"fps={camera_fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_mp4)
    ]
    run_cmd(cmd, check=True)

def transcode_normalize_to_60(in_path: Path, out_mp4: Path, camera_fps: int, target_sec: float) -> None:
    """
    Convert and force duration to target_sec by time-stretch (setpts) + -t target_sec.
    Used for all segments except the last.
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    suffix = in_path.suffix.lower()

    dur_in = ffprobe_duration(in_path)
    if dur_in is None or dur_in < 0.5:
        scale = 1.0
    else:
        scale = float(target_sec) / float(dur_in)

    input_opts = []
    if suffix == ".hevc":
        input_opts = ["-r", str(camera_fps)]

    vf = f"setpts=PTS*{scale:.12f},fps={camera_fps}"

    cmd = [
        "ffmpeg", "-y",
        *input_opts,
        "-i", str(in_path),
        "-an",
        "-vf", vf,
        "-t", f"{target_sec:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_mp4)
    ]
    run_cmd(cmd, check=True)

def concat_mp4_segments(mp4_paths: List[Path], out_path: Path) -> None:
    if not mp4_paths:
        raise ValueError("No mp4 segments to concatenate")

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
            str(out_path)
        ], check=True)
    finally:
        list_path.unlink(missing_ok=True)

def build_route_video_mp4(route_dir: Path, vid_kind: str, vids: Dict[int, Path], seg_nums: List[int],
                          camera_fps: int, overwrite: bool, target_sec: float) -> Tuple[Optional[Path], Optional[float], Dict[int, float]]:
    """
    Normalize all segments except last to 60s, last keeps its real duration.
    Return (video_path, video_dur, per_seg_out_dur)
    """
    out_video = route_dir / "video.mp4"
    if out_video.exists() and not overwrite:
        dur = ffprobe_duration(out_video)
        return out_video, dur, {}

    tmp_dir = route_dir / ".tmp_video_parts"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    last_seg = seg_nums[-1]
    mp4_parts: List[Path] = []
    per_seg_dur: Dict[int, float] = {}

    for n in tqdm(seg_nums, desc=f"[{route_dir.name}] build video parts", leave=False):
        if n not in vids:
            continue
        in_path = vids[n]
        out_part = tmp_dir / f"{n:06d}.mp4"

        try:
            if n == last_seg:
                # last: keep real duration
                transcode_keep_duration(in_path, out_part, camera_fps=camera_fps)
            else:
                # others: force 60s
                transcode_normalize_to_60(in_path, out_part, camera_fps=camera_fps, target_sec=target_sec)

            d = ffprobe_duration(out_part)
            if d is not None:
                per_seg_dur[n] = float(d)
            mp4_parts.append(out_part)

        except Exception as e:
            print(f"[WARN] video part failed: {in_path} -> {e}")

    if not mp4_parts:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None, {}

    mp4_parts = sorted(mp4_parts, key=lambda p: int(p.stem))
    try:
        concat_mp4_segments(mp4_parts, out_video)
        dur = ffprobe_duration(out_video)
        return out_video, dur, per_seg_dur
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()

def cut_video_clip(in_video: Path, out_video: Path, start_s: float, dur_s: float) -> None:
    """
    Always transcode clips for accuracy.
    """
    safe_mkdir(out_video.parent)
    start_s = max(0.0, float(start_s))
    dur_s = max(0.05, float(dur_s))

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.6f}",
        "-i", str(in_video),
        "-t", f"{dur_s:.6f}",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_video)
    ]
    run_cmd(cmd, check=True)


# =========================
# Enabled = op_enable OR acc_enable only
# =========================
def coerce_bool_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        s = x
    else:
        s = pd.Series(x)

    if s.dtype == bool:
        return s.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) != 0).to_numpy(dtype=bool)

    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["1", "true", "t", "yes", "y"]).to_numpy(dtype=bool)

def merge_short_gaps(enabled: np.ndarray, gap_merge_samples: int) -> np.ndarray:
    if gap_merge_samples <= 0 or enabled.size == 0:
        return enabled
    x = enabled.astype(np.uint8)
    n = len(x)
    i = 0
    while i < n:
        if x[i] == 0:
            j = i
            while j < n and x[j] == 0:
                j += 1
            gap_len = j - i
            left_on = (i - 1 >= 0 and x[i - 1] == 1)
            right_on = (j < n and x[j] == 1)
            if left_on and right_on and gap_len <= gap_merge_samples:
                x[i:j] = 1
            i = j
        else:
            i += 1
    return x.astype(bool)

def detect_takeover_events_streaming(enabled_chunk: np.ndarray, global_start_idx: int, state: Dict) -> None:
    if enabled_chunk.size == 0:
        return

    enabled = enabled_chunk
    gsm = state["gap_merge_samples"]
    if gsm > 0:
        enabled = merge_short_gaps(enabled, gsm)

    changes = np.flatnonzero(enabled[1:] != enabled[:-1]) + 1
    run_starts = np.r_[0, changes]
    run_ends = np.r_[changes, enabled.size]
    run_vals = enabled[run_starts]
    run_lens = run_ends - run_starts

    for rv, rs, rl in zip(run_vals, run_starts, run_lens):
        rv = bool(rv)
        rl = int(rl)

        if state["curr_state"] is None:
            state["curr_state"] = rv
            state["curr_run_len"] = 0

        if rv == state["curr_state"]:
            state["curr_run_len"] += rl
            continue

        transition_global_idx = global_start_idx + int(rs)
        prev_state = state["curr_state"]
        prev_len = state["curr_run_len"]

        # ON -> OFF
        if prev_state is True and rv is False:
            state["pending_event"] = {"event_idx": transition_global_idx, "on_len": prev_len}
        # OFF -> ON
        elif prev_state is False and rv is True:
            if state["pending_event"] is not None:
                off_len = prev_len
                if state["pending_event"]["on_len"] >= state["min_on_samples"] and off_len >= state["min_off_samples"]:
                    state["events"].append(int(state["pending_event"]["event_idx"]))
                state["pending_event"] = None

        state["curr_state"] = rv
        state["curr_run_len"] = rl

def finalize_streaming_events(state: Dict) -> List[int]:
    if state["curr_state"] is False and state["pending_event"] is not None:
        off_len = state["curr_run_len"]
        if state["pending_event"]["on_len"] >= state["min_on_samples"] and off_len >= state["min_off_samples"]:
            state["events"].append(int(state["pending_event"]["event_idx"]))
        state["pending_event"] = None

    events = sorted(state["events"])
    if not events:
        return []

    min_sep = int(round(1.0 * state["log_hz"]))
    dedup = []
    for e in events:
        if not dedup or (e - dedup[-1] >= min_sep):
            dedup.append(e)
    return dedup


# =========================
# Build route CSV with all columns (union schema)
# =========================
def add_time_columns(df: pd.DataFrame, start_idx: int, log_hz: int) -> pd.DataFrame:
    n = len(df)
    out = df.copy()
    out["Time"] = np.arange(start_idx, start_idx + n, dtype=np.int64)
    out["Time_seconds"] = out["Time"] / float(log_hz)
    return out

def build_route_csv_and_events_all_columns(
    route_dir: Path,
    log_kind: str,
    logs: Dict[int, Path],
    seg_nums: List[int],
    overwrite: bool,
    log_hz: int,
    target_samples: int,
    min_on_s: float,
    min_off_s: float,
    gap_merge_s: float
) -> Tuple[Path, int, List[int], List[str]]:
    route_csv = route_dir / "data.csv"
    if route_csv.exists() and overwrite:
        route_csv.unlink(missing_ok=True)

    tmp_seg_dir = route_dir / ".tmp_seg_csv"
    if tmp_seg_dir.exists():
        shutil.rmtree(tmp_seg_dir, ignore_errors=True)
    tmp_seg_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "log_hz": log_hz,
        "min_on_samples": int(round(min_on_s * log_hz)),
        "min_off_samples": int(round(min_off_s * log_hz)),
        "gap_merge_samples": int(round(gap_merge_s * log_hz)) if gap_merge_s > 0 else 0,
        "curr_state": None,
        "curr_run_len": 0,
        "pending_event": None,
        "events": [],
    }

    total = 0
    schema_cols: List[str] = []
    schema_set = set()
    seg_csv_paths: List[Path] = []

    for nseg in tqdm(seg_nums, desc=f"[{route_dir.name}] read {log_kind} (all cols)", leave=False):
        if nseg not in logs:
            continue
        if total >= target_samples:
            break

        try:
            df = log2df(logs[nseg])
        except Exception as e:
            print(f"[WARN] log read failed: {logs[nseg]} -> {e}")
            continue

        if df is None or len(df) == 0:
            del df
            gc.collect()
            continue

        remain = target_samples - total
        if remain <= 0:
            del df
            gc.collect()
            break
        if len(df) > remain:
            df = df.iloc[:remain].copy()

        op_en = coerce_bool_array(df["op_enable"]) if "op_enable" in df.columns else np.zeros(len(df), dtype=bool)
        acc_en = coerce_bool_array(df["acc_enable"]) if "acc_enable" in df.columns else np.zeros(len(df), dtype=bool)
        enabled = (op_en | acc_en)
        detect_takeover_events_streaming(enabled, global_start_idx=total, state=state)

        df_out = add_time_columns(df, start_idx=total, log_hz=log_hz)

        for c in df_out.columns:
            if c not in schema_set:
                schema_set.add(c)
                schema_cols.append(c)

        seg_csv = tmp_seg_dir / f"{nseg:06d}.csv"
        df_out.to_csv(seg_csv, index=False)
        seg_csv_paths.append(seg_csv)

        total += len(df)

        del df, df_out, enabled, op_en, acc_en
        gc.collect()

    events = finalize_streaming_events(state)

    header_written = False
    for seg_csv in tqdm(seg_csv_paths, desc=f"[{route_dir.name}] merge temp -> data.csv", leave=False):
        for chunk in pd.read_csv(seg_csv, chunksize=200000):
            for c in schema_cols:
                if c not in chunk.columns:
                    chunk[c] = np.nan
            chunk = chunk[schema_cols]
            chunk.to_csv(route_csv, mode=("a" if header_written else "w"), header=(not header_written), index=False)
            header_written = True
            del chunk
            gc.collect()

    shutil.rmtree(tmp_seg_dir, ignore_errors=True)
    gc.collect()

    return route_csv, total, events, schema_cols


# =========================
# Clip window by index
# =========================
def compute_clip_window_idx(event_idx: int, total_samples: int, half_window_samples: int) -> Tuple[int, int]:
    win = 2 * half_window_samples
    start = event_idx - half_window_samples
    end = event_idx + half_window_samples
    if start < 0:
        start = 0
        end = min(total_samples, win)
    if end > total_samples:
        end = total_samples
        start = max(0, total_samples - win)
    return int(start), int(end)

def export_clip_csvs_from_route_csv_by_idx(
    route_csv: Path,
    windows: List[Dict],
    out_dir: Path,
    out_log_kind: str,
    chunksize: int = 200000
) -> None:
    if not windows:
        return

    windows = sorted(windows, key=lambda x: x["start_idx"])
    next_i = 0
    active: List[Dict] = []
    header_done = set()

    for chunk in pd.read_csv(route_csv, chunksize=chunksize):
        if chunk.empty or "Time" not in chunk.columns:
            del chunk
            gc.collect()
            continue

        tmin = int(chunk["Time"].iloc[0])
        tmax = int(chunk["Time"].iloc[-1])

        while next_i < len(windows) and windows[next_i]["start_idx"] <= tmax:
            active.append(windows[next_i])
            next_i += 1

        active = [w for w in active if (w["end_idx_excl"] - 1) >= tmin]
        if not active:
            del chunk
            gc.collect()
            continue

        tt = chunk["Time"].to_numpy(dtype=np.int64)

        for w in active:
            mask = (tt >= w["start_idx"]) & (tt < w["end_idx_excl"])
            if not mask.any():
                continue
            sub = chunk.loc[mask]
            out_csv = out_dir / f'{w["clip_id"]}--{out_log_kind}.csv'
            if w["clip_id"] in header_done:
                sub.to_csv(out_csv, mode="a", header=False, index=False)
            else:
                sub.to_csv(out_csv, mode="w", header=True, index=False)
                header_done.add(w["clip_id"])

        del chunk
        gc.collect()


# =========================
# Process one route
# =========================
def process_one_route(
    route_dir: Path,
    target_root: Path,
    camera_fps: int,
    overwrite: bool,
    min_on_s: float,
    min_off_s: float,
    gap_merge_s: float,
    segment_target_sec: float
) -> int:
    car_model, dongle_id, route_id = parse_route_ids(route_dir)

    files = discover_route_files(route_dir)
    best = choose_best_pair(files)
    if best is None:
        return 0

    log_kind, logs, vid_kind, vids, seg_nums = best
    if not seg_nums:
        return 0

    log_hz = 100 if log_kind == "rlog" else 10

    # 1) video.mp4：除最后一段外强制 60s，最后一段保持原始长度
    video_path, video_dur, per_seg_dur = build_route_video_mp4(
        route_dir=route_dir,
        vid_kind=vid_kind,
        vids=vids,
        seg_nums=seg_nums,
        camera_fps=camera_fps,
        overwrite=overwrite,
        target_sec=segment_target_sec
    )
    if video_path is None or video_dur is None or video_dur <= 0.5:
        return 0

    # 2) 严格对齐：target_samples 基于最终拼接 video.mp4
    target_samples = int(round(video_dur * float(log_hz)))

    route_csv, total_samples, events, schema_cols = build_route_csv_and_events_all_columns(
        route_dir=route_dir,
        log_kind=log_kind,
        logs=logs,
        seg_nums=seg_nums,
        overwrite=overwrite,
        log_hz=log_hz,
        target_samples=target_samples,
        min_on_s=min_on_s,
        min_off_s=min_off_s,
        gap_merge_s=gap_merge_s
    )
    if total_samples <= 0 or not events:
        return 0

    # 3) 输出目录
    out_dir = Path(target_root) / car_model / dongle_id / route_id
    safe_mkdir(out_dir)

    # 4) clip window
    half_window_samples = int(round(CLIP_HALF_WINDOW_S * float(log_hz)))

    windows = []
    exported = 0
    for k, eidx in enumerate(events):
        eidx = int(eidx)
        if eidx < 0 or eidx >= total_samples:
            continue

        start_idx, end_idx_excl = compute_clip_window_idx(eidx, total_samples, half_window_samples)
        clip_start_s = start_idx / float(log_hz)
        clip_dur_s = (end_idx_excl - start_idx) / float(log_hz)

        out_video = out_dir / f"{k}--takeover.mp4"
        try:
            cut_video_clip(video_path, out_video, clip_start_s, clip_dur_s)
        except Exception as ex:
            print(f"[WARN] cut clip failed: {video_path} -> {out_video} : {ex}")
            continue

        windows.append({"clip_id": k, "start_idx": start_idx, "end_idx_excl": end_idx_excl})

        meta = {
            "car_model": car_model,
            "dongle_id": dongle_id,
            "route_id": route_id,
            "log_kind": log_kind,
            "vid_kind": vid_kind,
            "log_hz": log_hz,
            "camera_fps": camera_fps,
            "segment_target_sec": segment_target_sec,
            "seg_nums_used": seg_nums,
            "per_seg_duration_s": per_seg_dur,
            "video_duration_s": float(video_dur),
            "csv_total_samples": int(total_samples),
            "csv_duration_s": float(total_samples / float(log_hz)),
            "schema_num_cols": int(len(schema_cols)),
            "takeover_index": int(eidx),
            "clip_id": int(k),
            "clip_start_idx": int(start_idx),
            "clip_end_idx_excl": int(end_idx_excl),
            "clip_start_s": float(clip_start_s),
            "clip_dur_s": float(clip_dur_s),
            "source_route_dir": str(route_dir),
            "source_video_mp4": str(video_path),
            "source_route_csv": str(route_csv),
        }
        with open(out_dir / f"{k}--meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        exported += 1

    # 5) 导出 clip csv，命名包含 rlog/qlog
    if windows:
        export_clip_csvs_from_route_csv_by_idx(
            route_csv=route_csv,
            windows=windows,
            out_dir=out_dir,
            out_log_kind=log_kind,
            chunksize=200000
        )

    gc.collect()
    return exported


# =========================
# Iterate routes
# =========================
def iter_routes(source_root: Path):
    for car_model_dir in sorted(source_root.iterdir()):
        if not car_model_dir.is_dir():
            continue
        if car_model_dir.name in ["None", "mock", "MOCK"]:
            continue
        for dongle_dir in sorted(car_model_dir.iterdir()):
            if not dongle_dir.is_dir():
                continue
            for route_dir in sorted(dongle_dir.iterdir()):
                if route_dir.is_dir():
                    yield route_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build TakeOver dataset. enabled = op_enable OR acc_enable only. Normalize all BUT last video segment to 60s. Keep all columns."
    )
    parser.add_argument("--source_root", type=str, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target_root", type=str, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--single_route", type=str, default=None)
    parser.add_argument("--camera_fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--segment_sec", type=float, default=SEGMENT_TARGET_SEC)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min_on_s", type=float, default=DEFAULT_MIN_ON_S)
    parser.add_argument("--min_off_s", type=float, default=DEFAULT_MIN_OFF_S)
    parser.add_argument("--gap_merge_s", type=float, default=DEFAULT_GAP_MERGE_S)

    args = parser.parse_args()
    ensure_ffmpeg()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    if args.single_route:
        n = process_one_route(
            route_dir=Path(args.single_route),
            target_root=target_root,
            camera_fps=args.camera_fps,
            overwrite=args.overwrite,
            min_on_s=args.min_on_s,
            min_off_s=args.min_off_s,
            gap_merge_s=args.gap_merge_s,
            segment_target_sec=args.segment_sec
        )
        print(f"[DONE] single route exported takeover clips: {n}")
        return

    total_exported = 0
    total_routes = 0

    for route_dir in tqdm(iter_routes(source_root), desc="Processing routes"):
        try:
            n = process_one_route(
                route_dir=route_dir,
                target_root=target_root,
                camera_fps=args.camera_fps,
                overwrite=args.overwrite,
                min_on_s=args.min_on_s,
                min_off_s=args.min_off_s,
                gap_merge_s=args.gap_merge_s,
                segment_target_sec=args.segment_sec
            )
            total_routes += 1
            total_exported += n
        except Exception as e:
            print(f"[WARN] route failed: {route_dir} -> {e}")
        finally:
            gc.collect()

    print(f"[DONE] routes processed: {total_routes}, takeover clips exported: {total_exported}")
    print(f"[OUT] dataset saved under: {target_root}")


if __name__ == "__main__":
    main()
