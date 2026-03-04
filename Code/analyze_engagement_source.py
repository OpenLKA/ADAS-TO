#!/usr/bin/env python3
"""
analyze_engagement_source.py
==============================
For every clip in the TakeOver dataset, determine whether the ADAS engagement
(in the pre-takeover window) was driven by:
  - controlsState.enabled  (openpilot longitudinal + lateral)
  - carState.cruiseState.enabled  (OEM ACC / cruise)
  - both simultaneously
  - neither (edge case)

Outputs:
  stats_output/engagement_source.csv   — per-clip classification
  stats_output/engagement_report.txt   — aggregate statistics
"""
import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
STATS = ROOT / "Code" / "stats_output"
PER_CLIP = STATS / "per_clip.csv"
OUT_CSV = STATS / "engagement_source.csv"
OUT_REPORT = STATS / "engagement_report.txt"

# Pre-event window: we look at [-10, 0] seconds relative to takeover
# The takeover event is at video_time_s in meta.json
# The clip spans [clip_start_s, clip_start_s + clip_dur_s]
# clip_start_s = video_time_s - 10  (event at center)
# So pre-event window in time_s coordinates: [clip_start_s, video_time_s]


def classify_clip(clip_dir: Path) -> dict:
    """Classify one clip's engagement source."""
    meta_path = clip_dir / "meta.json"
    ctrl_path = clip_dir / "controlsState.csv"
    car_path = clip_dir / "carState.csv"

    result = {
        "clip_dir": str(clip_dir),
        "car_model": "",
        "brand": "",
        "dongle_id": "",
        "route_id": "",
        "clip_id": "",
        "ctrl_enabled_any": False,
        "cruise_enabled_any": False,
        "ctrl_enabled_pct": 0.0,
        "cruise_enabled_pct": 0.0,
        "both_pct": 0.0,
        "source": "unknown",
        "n_ctrl_samples": 0,
        "n_car_samples": 0,
    }

    if not meta_path.exists() or not ctrl_path.exists() or not car_path.exists():
        result["source"] = "missing_files"
        return result

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        result["source"] = "bad_meta"
        return result

    result["car_model"] = meta.get("car_model", "")
    result["dongle_id"] = meta.get("dongle_id", "")
    result["route_id"] = meta.get("route_id", "")
    result["clip_id"] = str(meta.get("clip_id", ""))

    event_time = meta.get("video_time_s", 0)
    clip_start = meta.get("clip_start_s", 0)
    # Pre-event window: from clip_start to event_time
    t_lo = clip_start
    t_hi = event_time

    try:
        # Read controlsState
        ctrl_df = pd.read_csv(ctrl_path, usecols=["time_s", "enabled"])
        ctrl_pre = ctrl_df[(ctrl_df["time_s"] >= t_lo) & (ctrl_df["time_s"] <= t_hi)]
        n_ctrl = len(ctrl_pre)
        result["n_ctrl_samples"] = n_ctrl

        if n_ctrl > 0:
            ctrl_en = ctrl_pre["enabled"].astype(str).str.strip().str.lower() == "true"
            ctrl_any = ctrl_en.any()
            ctrl_pct = ctrl_en.sum() / n_ctrl * 100
        else:
            ctrl_any = False
            ctrl_pct = 0.0

        result["ctrl_enabled_any"] = bool(ctrl_any)
        result["ctrl_enabled_pct"] = round(ctrl_pct, 1)

        # Read carState
        car_df = pd.read_csv(car_path, usecols=["time_s", "cruiseState.enabled"])
        car_pre = car_df[(car_df["time_s"] >= t_lo) & (car_df["time_s"] <= t_hi)]
        n_car = len(car_pre)
        result["n_car_samples"] = n_car

        if n_car > 0:
            cruise_en = car_pre["cruiseState.enabled"].astype(str).str.strip().str.lower() == "true"
            cruise_any = cruise_en.any()
            cruise_pct = cruise_en.sum() / n_car * 100
        else:
            cruise_any = False
            cruise_pct = 0.0

        result["cruise_enabled_any"] = bool(cruise_any)
        result["cruise_enabled_pct"] = round(cruise_pct, 1)

        # Compute overlap: both enabled simultaneously
        # Merge on nearest time_s
        if n_ctrl > 0 and n_car > 0:
            ctrl_pre2 = ctrl_pre.copy()
            ctrl_pre2["ctrl_en"] = ctrl_en.values
            car_pre2 = car_pre.copy()
            car_pre2["cruise_en"] = cruise_en.values
            # Use merge_asof for time alignment
            ctrl_pre2 = ctrl_pre2.sort_values("time_s")
            car_pre2 = car_pre2.sort_values("time_s")
            merged = pd.merge_asof(
                ctrl_pre2[["time_s", "ctrl_en"]],
                car_pre2[["time_s", "cruise_en"]],
                on="time_s", direction="nearest", tolerance=0.15
            )
            both = (merged["ctrl_en"] & merged["cruise_en"].fillna(False))
            result["both_pct"] = round(both.sum() / len(merged) * 100, 1)

        # Classification
        if ctrl_any and cruise_any:
            # Both were enabled at some point
            if ctrl_pct > 50 and cruise_pct > 50:
                result["source"] = "both"
            elif ctrl_pct > cruise_pct:
                result["source"] = "openpilot_primary"
            else:
                result["source"] = "oem_primary"
        elif ctrl_any and not cruise_any:
            result["source"] = "openpilot_only"
        elif cruise_any and not ctrl_any:
            result["source"] = "oem_only"
        else:
            result["source"] = "neither"

    except Exception as e:
        result["source"] = f"error:{type(e).__name__}"

    return result


def find_all_clips() -> list[Path]:
    """Find all clip directories (those containing meta.json)."""
    clips = []
    for meta in ROOT.rglob("meta.json"):
        # Skip Code/ directory
        if "Code" in meta.parts:
            continue
        clips.append(meta.parent)
    return sorted(clips)


def main():
    print("Finding all clips...")
    clips = find_all_clips()
    print(f"Found {len(clips):,} clips")

    # Load per_clip.csv for brand info
    pc = pd.read_csv(PER_CLIP)
    brand_map = {}
    for _, row in pc.iterrows():
        key = (row["car_model"], row["dongle_id"], row["route_id"], str(row["clip_id"]))
        brand_map[key] = row.get("brand", "")

    print(f"Processing clips with {min(12, len(clips))} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(classify_clip, c): c for c in clips}
        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            # Fill brand from per_clip
            key = (r["car_model"], r["dongle_id"], r["route_id"], r["clip_id"])
            r["brand"] = brand_map.get(key, "")
            results.append(r)
            if done % 2000 == 0:
                print(f"  {done:,}/{len(clips):,} done")

    print(f"  {done:,}/{len(clips):,} done")

    # Save per-clip CSV
    fields = ["car_model", "brand", "dongle_id", "route_id", "clip_id",
              "source", "ctrl_enabled_any", "cruise_enabled_any",
              "ctrl_enabled_pct", "cruise_enabled_pct", "both_pct",
              "n_ctrl_samples", "n_car_samples"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved per-clip → {OUT_CSV}")

    # ── Aggregate statistics ──
    df = pd.DataFrame(results)
    N = len(df)

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 72)
    p("  ENGAGEMENT SOURCE ANALYSIS")
    p("=" * 72)

    p(f"\nTotal clips analyzed: {N:,}")

    # Overall source distribution
    p("\n" + "─" * 72)
    p("1. OVERALL SOURCE DISTRIBUTION")
    p("─" * 72)
    src_counts = df["source"].value_counts()
    for src, cnt in src_counts.items():
        p(f"  {src:30s}: {cnt:6,}  ({cnt/N*100:5.1f}%)")

    # Simplified 3-category
    p("\n  Simplified classification:")

    def simplify(src):
        if src in ("openpilot_only", "openpilot_primary"):
            return "openpilot_longitudinal"
        elif src in ("oem_only", "oem_primary"):
            return "oem_adas"
        elif src == "both":
            return "both_active"
        else:
            return "other"

    df["source_simple"] = df["source"].apply(simplify)
    for cat in ["openpilot_longitudinal", "oem_adas", "both_active", "other"]:
        cnt = (df["source_simple"] == cat).sum()
        p(f"    {cat:30s}: {cnt:6,}  ({cnt/N*100:5.1f}%)")

    # By brand
    p("\n" + "─" * 72)
    p("2. BY BRAND")
    p("─" * 72)
    brand_groups = df.groupby("brand")
    brand_summary = []
    for brand, bdf in sorted(brand_groups, key=lambda x: -len(x[1])):
        if not brand:
            continue
        n = len(bdf)
        op_n = (bdf["source_simple"] == "openpilot_longitudinal").sum()
        oem_n = (bdf["source_simple"] == "oem_adas").sum()
        both_n = (bdf["source_simple"] == "both_active").sum()
        other_n = (bdf["source_simple"] == "other").sum()
        brand_summary.append((brand, n, op_n, oem_n, both_n, other_n))

    p(f"  {'Brand':<15s} {'Total':>6s}  {'openpilot':>10s}  {'OEM ADAS':>10s}  {'Both':>8s}  {'Other':>6s}")
    p(f"  {'─'*15} {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*6}")
    for brand, n, op_n, oem_n, both_n, other_n in brand_summary:
        op_s = f"{op_n} ({op_n/n*100:.0f}%)" if op_n else "—"
        oem_s = f"{oem_n} ({oem_n/n*100:.0f}%)" if oem_n else "—"
        both_s = f"{both_n}" if both_n else "—"
        other_s = f"{other_n}" if other_n else "—"
        p(f"  {brand:<15s} {n:6,}  {op_s:>10s}  {oem_s:>10s}  {both_s:>8s}  {other_s:>6s}")

    # By car model (top 30)
    p("\n" + "─" * 72)
    p("3. BY CAR MODEL (top 30 by clip count)")
    p("─" * 72)
    model_groups = df.groupby("car_model")
    model_summary = []
    for model, mdf in sorted(model_groups, key=lambda x: -len(x[1])):
        if not model:
            continue
        n = len(mdf)
        op_n = (mdf["source_simple"] == "openpilot_longitudinal").sum()
        oem_n = (mdf["source_simple"] == "oem_adas").sum()
        both_n = (mdf["source_simple"] == "both_active").sum()
        model_summary.append((model, n, op_n, oem_n, both_n))

    p(f"  {'Car Model':<40s} {'Total':>6s} {'openpilot':>10s} {'OEM':>10s} {'Both':>6s}")
    p(f"  {'─'*40} {'─'*6} {'─'*10} {'─'*10} {'─'*6}")
    for model, n, op_n, oem_n, both_n in model_summary[:30]:
        op_s = f"{op_n} ({op_n/n*100:.0f}%)" if op_n else "—"
        oem_s = f"{oem_n} ({oem_n/n*100:.0f}%)" if oem_n else "—"
        p(f"  {model:<40s} {n:6,} {op_s:>10s} {oem_s:>10s} {both_n:>6}")

    # Engagement percentage stats
    p("\n" + "─" * 72)
    p("4. ENGAGEMENT PERCENTAGE STATISTICS (pre-event window)")
    p("─" * 72)
    p(f"  controlsState.enabled % (when any):")
    ctrl_any = df[df["ctrl_enabled_any"]]
    if len(ctrl_any):
        p(f"    n={len(ctrl_any):,}  mean={ctrl_any['ctrl_enabled_pct'].mean():.1f}%"
          f"  median={ctrl_any['ctrl_enabled_pct'].median():.1f}%")
    p(f"  cruiseState.enabled % (when any):")
    cruise_any = df[df["cruise_enabled_any"]]
    if len(cruise_any):
        p(f"    n={len(cruise_any):,}  mean={cruise_any['cruise_enabled_pct'].mean():.1f}%"
          f"  median={cruise_any['cruise_enabled_pct'].median():.1f}%")

    # Full model list with classification
    p("\n" + "─" * 72)
    p("5. COMPLETE MODEL LIST WITH CLASSIFICATION")
    p("─" * 72)
    p(f"  {'Car Model':<45s} {'N':>5s} {'openpilot':>5s} {'OEM':>5s} {'Both':>5s} {'Primary Source':<20s}")
    p(f"  {'─'*45} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*20}")
    for model, n, op_n, oem_n, both_n in model_summary:
        total_classified = op_n + oem_n + both_n
        if total_classified == 0:
            primary = "unclassified"
        elif both_n > op_n and both_n > oem_n:
            primary = "both"
        elif op_n >= oem_n:
            primary = "openpilot"
        else:
            primary = "OEM ADAS"
        p(f"  {model:<45s} {n:5,} {op_n:5,} {oem_n:5,} {both_n:5,} {primary:<20s}")

    # Save report
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved report → {OUT_REPORT}")


if __name__ == "__main__":
    main()
