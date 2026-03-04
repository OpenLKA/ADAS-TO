#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_scene_tagger.py
====================
Optional Stage 1 module: run YOLOv8 on keyframes around the takeover event
for clips flagged as anomalous or belonging to ambiguous scenarios.

This saves per-keyframe detections to ``stats_output/yolo_scene_tags.csv``
and per-clip aggregated presence flags to ``stats_output/yolo_clip_summary.csv``.

Prerequisites:
    pip install ultralytics opencv-python-headless

Usage:
    # Tag all clips flagged in anomaly_flags.csv
    python3 yolo_scene_tagger.py

    # Tag only specific clips (by clip_key)
    python3 yolo_scene_tagger.py --clips "ACURA_INTEGRA/3f8ae015ce70365f/00000000--5c5085329e/0"

    # Dry run (show which clips would be processed)
    python3 yolo_scene_tagger.py --dry

    # Limit to N clips
    python3 yolo_scene_tagger.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  Paths & Config
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
CODE = ROOT / "Code"
OUT = CODE / "stats_output"

CFG_PATH = CODE / "configs" / "analysis_thresholds.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

YOLO_CFG = CFG.get("yolo", {})
MODEL_NAME = YOLO_CFG.get("model", "yolov8n.pt")
CONF_THRESH = YOLO_CFG.get("confidence_threshold", 0.35)
KEYFRAME_OFFSETS = YOLO_CFG.get("keyframe_times_s", [-3, -2, -1, 0, 1])
TARGET_CLASSES = set(YOLO_CFG.get("target_classes", [
    "traffic light", "stop sign", "person", "bicycle",
    "car", "bus", "truck",
]))


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_video(clip_dir: Path) -> Path | None:
    """Locate the video file for a clip directory."""
    for name in ("takeover.mp4", "takeover.hevc"):
        p = clip_dir / name
        if p.exists():
            return p
    return None


def extract_keyframe(video_path: Path, time_s: float) -> np.ndarray | None:
    """Extract a single frame from a video at the given time (seconds)."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python-headless is required: pip install opencv-python-headless")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0  # default camera_fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(time_s * fps)

    # Clamp
    target_frame = max(0, min(target_frame, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None
    return frame


def load_meta(clip_dir: Path) -> dict | None:
    """Load meta.json for a clip."""
    meta_path = clip_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def get_clip_dir(clip_key: str) -> Path:
    """Convert clip_key (car_model/dongle/route/clip_id) to directory path."""
    return ROOT / clip_key


# ──────────────────────────────────────────────────────────────────────────────
#  YOLO detection
# ──────────────────────────────────────────────────────────────────────────────

def run_yolo_on_clip(clip_key: str, model) -> list[dict]:
    """
    Run YOLO detection on keyframes for a single clip.
    Returns a list of detection records.
    """
    clip_dir = get_clip_dir(clip_key)
    if not clip_dir.is_dir():
        return []

    meta = load_meta(clip_dir)
    if meta is None:
        return []

    video_path = find_video(clip_dir)
    if video_path is None:
        return []

    # video_time_s is the absolute time of the takeover event in the video
    event_video_time = meta.get("video_time_s", 10.0)
    clip_start_s = meta.get("clip_start_s", event_video_time - 10.0)
    # In the video file, t=0 corresponds to clip_start_s in absolute time.
    # The event is at (event_video_time - clip_start_s) seconds into the video.
    event_in_video = event_video_time - clip_start_s

    records = []

    for offset_s in KEYFRAME_OFFSETS:
        frame_time = event_in_video + offset_s
        if frame_time < 0:
            continue

        frame = extract_keyframe(video_path, frame_time)
        if frame is None:
            continue

        # Run YOLO
        results = model(frame, conf=CONF_THRESH, verbose=False)

        if len(results) == 0:
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            # No detections at this keyframe
            records.append({
                "clip_key": clip_key,
                "keyframe_offset_s": offset_s,
                "n_detections": 0,
                "class_name": "",
                "confidence": np.nan,
                "bbox_x1": np.nan, "bbox_y1": np.nan,
                "bbox_x2": np.nan, "bbox_y2": np.nan,
                "is_target_class": False,
            })
            continue

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            records.append({
                "clip_key": clip_key,
                "keyframe_offset_s": offset_s,
                "n_detections": len(boxes),
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_x1": round(x1, 1),
                "bbox_y1": round(y1, 1),
                "bbox_x2": round(x2, 1),
                "bbox_y2": round(y2, 1),
                "is_target_class": cls_name in TARGET_CLASSES,
            })

    return records


def summarize_clip_detections(records: list[dict], clip_key: str) -> dict:
    """Aggregate per-keyframe detections into a clip-level summary."""
    summary = {"clip_key": clip_key}

    if not records:
        summary["yolo_n_keyframes"] = 0
        summary["yolo_n_detections"] = 0
        for cls in TARGET_CLASSES:
            summary[f"yolo_{cls.replace(' ', '_')}_present"] = False
        return summary

    df = pd.DataFrame(records)
    target_df = df[df["is_target_class"]]

    summary["yolo_n_keyframes"] = df["keyframe_offset_s"].nunique()
    summary["yolo_n_detections"] = int(df[df["n_detections"] > 0].shape[0])

    # Per-class presence flags
    detected_classes = set(target_df["class_name"].unique()) if not target_df.empty else set()
    for cls in TARGET_CLASSES:
        col_name = f"yolo_{cls.replace(' ', '_')}_present"
        summary[col_name] = cls in detected_classes

    # Max confidence per target class
    if not target_df.empty:
        for cls in detected_classes:
            col_name = f"yolo_{cls.replace(' ', '_')}_max_conf"
            cls_conf = target_df[target_df["class_name"] == cls]["confidence"].max()
            summary[col_name] = round(cls_conf, 4)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
#  Clip selection
# ──────────────────────────────────────────────────────────────────────────────

def _make_clip_key(row) -> str:
    """Build clip_key from car_model/dongle_id/route_id/clip_id."""
    return f"{row['car_model']}/{row['dongle_id']}/{row['route_id']}/{int(row['clip_id'])}"


def select_clips_for_yolo() -> list[str]:
    """
    Select clips to run YOLO on.
    Criteria: anomaly_any=True OR scenario in {intersection_odd, uncertain_mixed}.
    """
    clip_keys = []

    # From anomaly flags
    anomaly_path = OUT / "anomaly_flags.csv"
    id_cols = ["car_model", "dongle_id", "route_id", "clip_id"]
    if anomaly_path.exists():
        af = pd.read_csv(anomaly_path)
        if "anomaly_any" in af.columns and all(c in af.columns for c in id_cols):
            anomalous = af[af["anomaly_any"] == True]
            clip_keys.extend(anomalous.apply(_make_clip_key, axis=1).tolist())

    # From scenario labels
    scenario_path = OUT / "scenario_labels.csv"
    if scenario_path.exists():
        sl = pd.read_csv(scenario_path)
        # Support both "scenario" and "scenario_primary" column names
        scen_col = "scenario" if "scenario" in sl.columns else "scenario_primary"
        if scen_col in sl.columns and all(c in sl.columns for c in id_cols):
            target_scenarios = {"intersection_odd", "uncertain_mixed"}
            ambiguous = sl[sl[scen_col].isin(target_scenarios)]
            clip_keys.extend(ambiguous.apply(_make_clip_key, axis=1).tolist())

    # Deduplicate
    return list(dict.fromkeys(clip_keys))


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO scene tagging for takeover clips")
    parser.add_argument("--clips", type=str, default=None,
                        help="Comma-separated clip_keys to process (overrides auto-selection)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of clips to process")
    parser.add_argument("--dry", action="store_true",
                        help="Show which clips would be processed without running YOLO")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"YOLO model name (default: {MODEL_NAME})")
    parser.add_argument("--conf", type=float, default=CONF_THRESH,
                        help=f"Confidence threshold (default: {CONF_THRESH})")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO SCENE TAGGER")
    print("=" * 60)

    # Select clips
    if args.clips:
        clip_keys = [c.strip() for c in args.clips.split(",")]
        print(f"  Manual selection: {len(clip_keys)} clips")
    else:
        clip_keys = select_clips_for_yolo()
        print(f"  Auto-selected {len(clip_keys)} clips (anomalous + ambiguous scenarios)")

    if args.limit:
        clip_keys = clip_keys[:args.limit]
        print(f"  Limited to {len(clip_keys)} clips")

    if not clip_keys:
        print("  No clips selected for YOLO tagging.")
        print("  Run compute_derived_signals_v3.py and label_scenarios.py first,")
        print("  or specify clips manually with --clips.")
        return

    if args.dry:
        print(f"\n  Dry run — {len(clip_keys)} clips would be processed:")
        for ck in clip_keys[:20]:
            print(f"    {ck}")
        if len(clip_keys) > 20:
            print(f"    ... and {len(clip_keys) - 20} more")
        return

    # Load YOLO model
    print(f"\n  Loading YOLO model: {args.model}")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: ultralytics not installed. Run: pip install ultralytics")
        return

    model = YOLO(args.model)
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Keyframe offsets: {KEYFRAME_OFFSETS}")
    print(f"  Target classes: {sorted(TARGET_CLASSES)}")
    print()

    # Process clips
    all_records = []
    clip_summaries = []
    n_ok = 0
    n_fail = 0

    for i, clip_key in enumerate(clip_keys, 1):
        if i % 50 == 0 or i == 1:
            print(f"  Processing clip {i}/{len(clip_keys)}: {clip_key}")

        try:
            records = run_yolo_on_clip(clip_key, model)
            all_records.extend(records)
            summary = summarize_clip_detections(records, clip_key)
            clip_summaries.append(summary)
            n_ok += 1
        except Exception as e:
            print(f"  WARNING: Failed on {clip_key}: {e}")
            n_fail += 1

    # Save outputs
    OUT.mkdir(parents=True, exist_ok=True)

    if all_records:
        det_df = pd.DataFrame(all_records)
        det_path = OUT / "yolo_scene_tags.csv"
        det_df.to_csv(det_path, index=False)
        print(f"\n  Per-keyframe detections: {det_path.name} ({len(det_df)} rows)")
    else:
        print("\n  No detections recorded.")

    if clip_summaries:
        sum_df = pd.DataFrame(clip_summaries)
        sum_path = OUT / "yolo_clip_summary.csv"
        sum_df.to_csv(sum_path, index=False)
        print(f"  Per-clip summary: {sum_path.name} ({len(sum_df)} clips)")

    print(f"\n  Processed: {n_ok} OK, {n_fail} failed out of {len(clip_keys)} total")
    print("  Done.")


if __name__ == "__main__":
    main()
