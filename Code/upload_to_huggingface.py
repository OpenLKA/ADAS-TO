#!/usr/bin/env python3
"""
Upload HADAS-Takeover dataset to HuggingFace with:
  - Anonymized dongle_id / route_id
  - Rich progress bars (per-batch, per-file, overall)
  - Concurrent pre-upload hashing via HF's built-in multithreading
  - Automatic retry with exponential backoff
  - Checkpoint-based resume (persists completed batches to disk)

Usage:
  # Dry run — see what would be uploaded
  python upload_to_huggingface.py --dry-run

  # Full upload (auto-resumes from last checkpoint)
  python upload_to_huggingface.py

  # Custom batch size and max workers
  python upload_to_huggingface.py --batch-size 300 --workers 8

  # Force restart (ignore checkpoint)
  python upload_to_huggingface.py --restart
"""

import json
import os
import sys
import time
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from huggingface_hub import HfApi, CommitOperationAdd
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, SpinnerColumn, TaskProgressColumn,
    DownloadColumn, TransferSpeedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box

# ─── Configuration ────────────────────────────────────────────────────────────
REPO_ID = "HenryYHW/ADAS-TO"
BASE_DIR = Path(__file__).resolve().parent.parent          # TakeOver/
MAPPING_PATH = Path(__file__).resolve().parent / "anonymization_mapping.json"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "upload_checkpoint.json"
BATCH_SIZE = 500          # files per commit (HF recommended max)
MAX_RETRIES = 5           # retries per batch
RETRY_BASE_DELAY = 10     # seconds, doubles each retry
NUM_WORKERS = 8           # threads for pre-upload hashing/preparation
SKIP_DIRS = {"Code", ".git", ".github"}
CLIP_FILES = {
    "meta.json", "takeover.mp4",
    "carState.csv", "controlsState.csv", "carControl.csv", "carOutput.csv",
    "drivingModelData.csv", "radarState.csv", "accelerometer.csv",
    "longitudinalPlan.csv",
}

console = Console()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_mapping():
    with open(MAPPING_PATH) as f:
        return json.load(f)


def load_checkpoint() -> dict:
    """Load checkpoint. Returns {"completed_batches": [int], "batch_size": int}."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"completed_batches": [], "batch_size": BATCH_SIZE}


def save_checkpoint(ckpt: dict):
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f)
    tmp.replace(CHECKPOINT_PATH)   # atomic on POSIX


def anonymize_meta(meta_path: Path, dongle_map: dict, route_map: dict) -> bytes:
    with open(meta_path) as f:
        meta = json.load(f)
    meta["dongle_id"] = dongle_map[meta["dongle_id"]]
    meta["route_id"] = route_map[meta["route_id"]]
    return json.dumps(meta, indent=2).encode("utf-8")


def collect_all_clips(base_dir: Path):
    """Yield (car_model, dongle_id, route_id, clip_id, clip_dir) sorted deterministically."""
    for car_dir in sorted(base_dir.iterdir()):
        if not car_dir.is_dir() or car_dir.name in SKIP_DIRS:
            continue
        car_model = car_dir.name
        for dongle_dir in sorted(car_dir.iterdir()):
            if not dongle_dir.is_dir():
                continue
            for route_dir in sorted(dongle_dir.iterdir()):
                if not route_dir.is_dir():
                    continue
                for clip_dir in sorted(route_dir.iterdir()):
                    if not clip_dir.is_dir():
                        continue
                    yield car_model, dongle_dir.name, route_dir.name, clip_dir.name, clip_dir


def build_operations(clips, dongle_map, route_map, progress=None, task_id=None):
    """Build all CommitOperationAdd objects with anonymized paths."""
    operations = []
    total_bytes = 0
    for car_model, dongle_id, route_id, clip_id, clip_dir in clips:
        anon_dongle = dongle_map.get(dongle_id, dongle_id)
        anon_route = route_map.get(route_id, route_id)
        prefix = f"{car_model}/{anon_dongle}/{anon_route}/{clip_id}"

        for fname in sorted(os.listdir(clip_dir)):
            if fname not in CLIP_FILES:
                continue
            src = clip_dir / fname
            size = src.stat().st_size
            total_bytes += size

            if fname == "meta.json":
                content = anonymize_meta(src, dongle_map, route_map)
                op = CommitOperationAdd(path_in_repo=f"{prefix}/{fname}",
                                        path_or_fileobj=content)
            else:
                op = CommitOperationAdd(path_in_repo=f"{prefix}/{fname}",
                                        path_or_fileobj=str(src))
            operations.append((op, size))

        if progress and task_id is not None:
            progress.advance(task_id)

    return operations, total_bytes


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_duration(secs: float) -> str:
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"


# ─── Upload with retry ───────────────────────────────────────────────────────

def upload_batch_with_retry(api: HfApi, ops: list, batch_num: int,
                            total_batches: int, num_workers: int) -> bool:
    """Upload a single batch with exponential-backoff retry. Returns True on success."""
    msg = f"Batch {batch_num}/{total_batches} ({len(ops)} files)"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.create_commit(
                repo_id=REPO_ID,
                repo_type="dataset",
                operations=ops,
                commit_message=msg,
                num_threads=num_workers,
            )
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            console.print(
                f"  [yellow]Batch {batch_num} attempt {attempt}/{MAX_RETRIES} failed: "
                f"{type(e).__name__}: {e}[/yellow]"
            )
            if attempt < MAX_RETRIES:
                console.print(f"  [dim]Retrying in {delay}s...[/dim]")
                time.sleep(delay)
            else:
                console.print(f"  [red]Batch {batch_num} FAILED after {MAX_RETRIES} attempts.[/red]")
                return False
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload HADAS-TakeOver dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Files per commit (default: {BATCH_SIZE})")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Concurrent upload threads per batch (default: {NUM_WORKERS})")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore checkpoint and start from scratch")
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold cyan]HADAS-TakeOver → HuggingFace Uploader[/bold cyan]\n"
        f"Repo: [green]{REPO_ID}[/green]  |  Batch: {args.batch_size} files  |  "
        f"Workers: {args.workers}  |  Retries: {MAX_RETRIES}",
        border_style="cyan",
    ))

    # ── Load mapping ──────────────────────────────────────────────────────────
    console.print("\n[bold]1/4[/bold] Loading anonymization mapping...")
    mapping = load_mapping()
    dongle_map, route_map = mapping["dongle_id"], mapping["route_id"]
    console.print(f"     {len(dongle_map)} drivers, {len(route_map)} routes\n")

    # ── Collect clips ─────────────────────────────────────────────────────────
    console.print("[bold]2/4[/bold] Scanning clips...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        # First count clips for progress bar
        clips = list(collect_all_clips(BASE_DIR))
        task = progress.add_task("Building file operations...", total=len(clips))
        operations, total_bytes = build_operations(
            clips, dongle_map, route_map, progress, task
        )

    console.print(f"     {len(clips):,} clips  →  {len(operations):,} files  ({fmt_bytes(total_bytes)})\n")

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    batch_size = args.batch_size
    total_batches = (len(operations) + batch_size - 1) // batch_size

    if args.restart and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        console.print("     [yellow]Checkpoint cleared — starting from scratch[/yellow]")

    ckpt = load_checkpoint()
    # If batch size changed, invalidate checkpoint
    if ckpt.get("batch_size") != batch_size:
        ckpt = {"completed_batches": [], "batch_size": batch_size}
        save_checkpoint(ckpt)

    completed = set(ckpt["completed_batches"])
    remaining = [i for i in range(1, total_batches + 1) if i not in completed]

    console.print(f"[bold]3/4[/bold] Upload plan:")
    console.print(f"     Total batches: {total_batches}")
    console.print(f"     Already done:  {len(completed)}")
    console.print(f"     Remaining:     {len(remaining)}")

    if not remaining:
        console.print("\n[bold green]All batches already uploaded! Nothing to do.[/bold green]")
        return

    if args.dry_run:
        console.print("\n[bold yellow]── DRY RUN ──[/bold yellow]")
        for batch_num in remaining[:5]:
            start = (batch_num - 1) * batch_size
            end = min(start + batch_size, len(operations))
            ops = [op for op, _ in operations[start:end]]
            console.print(f"\n  Batch {batch_num}/{total_batches} ({len(ops)} files):")
            for op in ops[:3]:
                console.print(f"    {op.path_in_repo}")
            if len(ops) > 3:
                console.print(f"    ... +{len(ops)-3} more")
        if len(remaining) > 5:
            console.print(f"\n  ... +{len(remaining)-5} more batches")
        console.print(f"\n[dim]Remove --dry-run to upload.[/dim]")
        return

    # ── Upload ────────────────────────────────────────────────────────────────
    console.print(f"\n[bold]4/4[/bold] Uploading...\n")
    api = HfApi()

    failed_batches = []
    uploaded_files = sum(
        min(batch_size, len(operations) - (b - 1) * batch_size) for b in completed
    )
    uploaded_bytes = sum(
        sz for b in completed
        for _, sz in operations[(b-1)*batch_size : b*batch_size]
    )
    t_start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("•"),
        DownloadColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        overall = progress.add_task(
            "Overall", total=len(operations),
            completed=uploaded_files,
        )
        batch_task = progress.add_task(
            "Current batch", total=0, visible=True,
        )

        for idx, batch_num in enumerate(remaining):
            start = (batch_num - 1) * batch_size
            end = min(start + batch_size, len(operations))
            batch_ops = operations[start:end]
            ops = [op for op, _ in batch_ops]
            batch_bytes = sum(sz for _, sz in batch_ops)

            progress.update(batch_task,
                            description=f"Batch {batch_num}/{total_batches}",
                            completed=0, total=len(ops))

            ok = upload_batch_with_retry(api, ops, batch_num, total_batches, args.workers)

            if ok:
                # Update checkpoint
                completed.add(batch_num)
                ckpt["completed_batches"] = sorted(completed)
                save_checkpoint(ckpt)

                uploaded_files += len(ops)
                uploaded_bytes += batch_bytes
                progress.update(overall, completed=uploaded_files)
                progress.update(batch_task, completed=len(ops))

                # Stats line
                elapsed = time.time() - t_start
                speed = uploaded_bytes / elapsed if elapsed > 0 else 0
                eta = (total_bytes - uploaded_bytes) / speed if speed > 0 else 0
                progress.update(overall,
                                description=f"Overall [{fmt_bytes(uploaded_bytes)}/{fmt_bytes(total_bytes)}]")
            else:
                failed_batches.append(batch_num)
                progress.update(batch_task, completed=len(ops))
                progress.update(overall, advance=len(ops))

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    console.print()

    table = Table(title="Upload Summary", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Total files", f"{len(operations):,}")
    table.add_row("Uploaded", f"{uploaded_files:,}")
    table.add_row("Total data", fmt_bytes(total_bytes))
    table.add_row("Time elapsed", fmt_duration(elapsed))
    if elapsed > 0:
        table.add_row("Avg speed", f"{fmt_bytes(int(uploaded_bytes/elapsed))}/s")
    table.add_row("Failed batches", str(len(failed_batches)) if failed_batches else "0")
    console.print(table)

    if failed_batches:
        console.print(f"\n[red]Failed batches: {failed_batches}[/red]")
        console.print("[yellow]Re-run the script to retry failed batches (checkpoint saved).[/yellow]")
        sys.exit(1)
    else:
        console.print("\n[bold green]Upload complete![/bold green]")
        # Clean up checkpoint on full success
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            console.print("[dim]Checkpoint removed.[/dim]")


if __name__ == "__main__":
    main()
