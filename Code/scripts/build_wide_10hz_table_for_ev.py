#!/usr/bin/env python3
import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zstandard as zstd

BASE_COLS = ['dongle_id', 'route_id', 'segment', 'logMonoTime', 'time_s']
DISCRETE_KEYWORDS = {
  'status', 'state', 'personality', 'radartrackid', 'longcontrolstate',
  'fcw', 'shouldstop', 'cancel', 'override', 'resume', 'enabled', 'available',
  'standstill', 'charging', 'accfaulted', 'brakepressed', 'gaspressed',
  'stockaeb', 'stockfcw', 'stocklkas', 'nonadaptive', 'regenbraking',
  'radar', 'event', 'error', 'blinker', 'holdactive', 'fault',
}


def to_json_list(v: Any) -> str | None:
  if isinstance(v, list):
    return json.dumps(v, separators=(',', ':'))
  return None


def flatten_payload(obj: Any, prefix: str, out: dict[str, Any]) -> None:
  if isinstance(obj, dict):
    for k, v in obj.items():
      p = f'{prefix}.{k}' if prefix else k
      flatten_payload(v, p, out)
  elif isinstance(obj, list):
    out[prefix] = to_json_list(obj)
  else:
    out[prefix] = obj


def parse_route_segment(qlog_path: Path) -> tuple[str | None, str | None, str | None]:
  parts = qlog_path.parts
  if len(parts) < 4:
    return None, None, None
  segment = parts[-2]
  route_id = parts[-3]
  dongle = parts[-4]
  return dongle, route_id, segment


def iter_qlog_paths(
  base_dir: Path,
  max_files: int | None,
  random_sample: bool,
) -> list[Path]:
  """Return candidate log files, auto-detecting type from filename.

  rlog (*--rlog, *--rlog.bz2) is preferred over qlog for the same segment.
  Within the same type, raw beats bz2/zst so we never double-count a segment.
  Hz is detected per file by file_hz().
  """
  import re as _re

  def sort_key(p: Path) -> tuple[int, str]:
    name = p.name
    try:
      prefix = name.split('--', 1)[0]
      return (int(prefix), name)
    except (ValueError, IndexError):
      return (0, name)

  def _collect(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
      out.extend(base_dir.rglob(pat))
    return out

  rlog_files = _collect(['*--rlog', '*--rlog.bz2'])
  qlog_files = _collect(['*--qlog', '*--qlog.bz2', '*qlog*.zst'])

  def seg_key(p: Path) -> tuple[Path, int]:
    m = _re.match(r'^(\d+)--', p.name)
    return (p.parent, int(m.group(1)) if m else -1)

  # rlog takes priority; qlog only fills in segments without rlog
  rlog_segs = {seg_key(p) for p in rlog_files}
  qlog_fallback = [p for p in qlog_files if seg_key(p) not in rlog_segs]
  raw = rlog_files + qlog_fallback

  # De-duplicate by (parent, seg_num): raw > bz2 > zst
  def file_priority(p: Path) -> int:
    return 1 if (p.name.endswith('.bz2') or p.name.endswith('.zst')) else 0

  seen: dict[tuple[Path, int], Path] = {}
  for p in raw:
    m = _re.match(r'^(\d+)--', p.name)
    k = (p.parent, int(m.group(1)) if m else -1)
    if k not in seen or file_priority(p) < file_priority(seen[k]):
      seen[k] = p

  candidates = list(seen.values())
  if random_sample:
    random.shuffle(candidates)
  else:
    candidates = sorted(candidates, key=sort_key)
  if max_files is not None:
    candidates = candidates[:max_files]
  return candidates


def file_hz(log_path: Path) -> int:
  """Detect sampling rate from filename: *--rlog* → 100 Hz, else 10 Hz."""
  return 100 if '--rlog' in log_path.name else 10


def patch_logreader_zstd(logreader_module: Any) -> None:
  def _stream_decompress(dat: bytes) -> bytes:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(dat)) as reader:
      return reader.read()

  logreader_module.zstd.decompress = _stream_decompress


def load_fields_config(config_path: Path) -> dict[str, list[str]]:
  data = json.loads(config_path.read_text())
  topics_cfg = data.get('topics', data)
  if not isinstance(topics_cfg, dict):
    raise ValueError(f'Invalid config format: {config_path}')
  out: dict[str, list[str]] = {}
  for topic, fields in topics_cfg.items():
    if not isinstance(fields, list) or not all(isinstance(x, str) for x in fields):
      raise ValueError(f'Invalid field list for topic {topic} in {config_path}')
    out[topic] = fields
  return out


def is_numeric_value(v: Any) -> bool:
  return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def is_continuous_field(field: str, series: pd.Series) -> bool:
  field_l = field.lower()
  if any(k in field_l for k in DISCRETE_KEYWORDS):
    return False

  non_null = series.dropna()
  if non_null.empty:
    return False

  # JSON list strings and plain strings are discrete.
  sample = non_null.iloc[0]
  if isinstance(sample, str):
    return False
  if isinstance(sample, bool):
    return False

  # numeric columns default to continuous unless keyword says otherwise
  return non_null.map(is_numeric_value).all()


def linear_interp_with_gap(
  grid_mono: np.ndarray,
  sample_mono: np.ndarray,
  sample_val: np.ndarray,
  max_gap_ns: int,
) -> np.ndarray:
  out = np.full(len(grid_mono), np.nan, dtype=float)
  if len(sample_mono) == 0:
    return out

  left = pd.DataFrame({'g': grid_mono})
  right_prev = pd.DataFrame({'x_prev': sample_mono, 'v_prev': sample_val}).sort_values('x_prev')
  right_next = pd.DataFrame({'x_next': sample_mono, 'v_next': sample_val}).sort_values('x_next')

  prev = pd.merge_asof(left, right_prev, left_on='g', right_on='x_prev', direction='backward')
  nxt = pd.merge_asof(left, right_next, left_on='g', right_on='x_next', direction='forward')

  have_prev = prev['x_prev'].notna().to_numpy()
  have_next = nxt['x_next'].notna().to_numpy()
  valid = have_prev & have_next

  if not valid.any():
    return out

  g = grid_mono[valid].astype(np.float64)
  x0 = prev.loc[valid, 'x_prev'].to_numpy(dtype=np.float64)
  x1 = nxt.loc[valid, 'x_next'].to_numpy(dtype=np.float64)
  y0 = prev.loc[valid, 'v_prev'].to_numpy(dtype=np.float64)
  y1 = nxt.loc[valid, 'v_next'].to_numpy(dtype=np.float64)

  age_prev = g - x0
  age_next = x1 - g
  small_gap = (age_prev <= max_gap_ns) & (age_next <= max_gap_ns)

  idxs = np.where(valid)[0]
  denom = (x1 - x0)
  exact = denom == 0

  vals = np.empty_like(g)
  vals[exact] = y0[exact]
  vals[~exact] = y0[~exact] + (y1[~exact] - y0[~exact]) * ((g[~exact] - x0[~exact]) / denom[~exact])

  out[idxs[small_gap]] = vals[small_gap]
  return out


def ffill_with_gap(
  grid_mono: np.ndarray,
  sample_mono: np.ndarray,
  sample_val: np.ndarray,
  max_gap_ns: int,
) -> list[Any]:
  out: list[Any] = [None] * len(grid_mono)
  if len(sample_mono) == 0:
    return out

  left = pd.DataFrame({'g': grid_mono})
  right = pd.DataFrame({'x': sample_mono, 'v': sample_val}).sort_values('x')
  prev = pd.merge_asof(left, right, left_on='g', right_on='x', direction='backward')

  have_prev = prev['x'].notna().to_numpy()
  if not have_prev.any():
    return out

  age = (grid_mono[have_prev] - prev.loc[have_prev, 'x'].to_numpy(dtype=np.int64))
  keep = age <= max_gap_ns
  valid_idxs = np.where(have_prev)[0]
  prev_vals = prev.loc[have_prev, 'v'].tolist()
  for i, ok in enumerate(keep):
    if ok:
      out[valid_idxs[i]] = prev_vals[i]
  return out


def build_wide_for_qlog(
  qlog: Path,
  fields_by_topic: dict[str, list[str]],
  LogReader: Any,
  hz: int,
  duration_s: float,
  max_gap_cont_s: float,
  max_gap_disc_s: float,
) -> pd.DataFrame:
  topic_rows: dict[str, list[dict[str, Any]]] = {t: [] for t in fields_by_topic}
  all_mono: list[int] = []

  for msg in LogReader(qlog.as_posix(), only_union_types=True, sort_by_time=True):
    topic = msg.which()
    if topic not in fields_by_topic:
      continue

    mono = int(msg.logMonoTime)
    all_mono.append(mono)

    payload = getattr(msg, topic).to_dict(verbose=True)
    flat: dict[str, Any] = {}
    flatten_payload(payload, '', flat)

    row = {'logMonoTime': mono}
    for f in fields_by_topic[topic]:
      row[f] = flat.get(f)
    topic_rows[topic].append(row)

  if not all_mono:
    return pd.DataFrame()

  dongle_id, route_id, segment = parse_route_segment(qlog)
  start_mono = min(all_mono)

  n_steps = int(round(duration_s * hz))
  dt_ns = int(1e9 / hz)
  grid_mono = start_mono + np.arange(n_steps, dtype=np.int64) * dt_ns
  time_s = np.arange(n_steps, dtype=np.float64) / hz

  wide = pd.DataFrame({
    'dongle_id': dongle_id,
    'route_id': route_id,
    'segment': segment,
    'logMonoTime': grid_mono,
    'time_s': np.round(time_s, 6),
  })

  max_gap_cont_ns = int(max_gap_cont_s * 1e9)
  max_gap_disc_ns = int(max_gap_disc_s * 1e9)

  for topic, fields in fields_by_topic.items():
    rows = topic_rows.get(topic, [])
    if not rows:
      for f in fields:
        wide[f'{topic}.{f}'] = np.nan
      continue

    tdf = pd.DataFrame(rows).sort_values('logMonoTime').drop_duplicates('logMonoTime', keep='last')

    for f in fields:
      col = f'{topic}.{f}'
      if f not in tdf.columns:
        wide[col] = np.nan
        continue

      s = tdf[f]
      x = tdf['logMonoTime'].to_numpy(dtype=np.int64)

      if is_continuous_field(f, s):
        y = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
        valid = ~np.isnan(y)
        vals = linear_interp_with_gap(grid_mono, x[valid], y[valid], max_gap_cont_ns)
        wide[col] = vals
      else:
        valid = s.notna().to_numpy()
        vals = ffill_with_gap(grid_mono, x[valid], s.to_numpy(dtype=object)[valid], max_gap_disc_ns)
        wide[col] = vals

  return wide


def main() -> None:
  ap = argparse.ArgumentParser(
    description=(
      'Build a wide resampled table from selected qlog/rlog fields.\n'
      'Log type (rlog=100 Hz / qlog=10 Hz) is detected automatically from filename.'
    )
  )
  ap.add_argument('--openpilot-path', default='/home/henry/Dropbox/OP_CAN_DataProcessing',
                  help='Path to openpilot repo root (or OP_CAN_DataProcessing containing tools/)')
  ap.add_argument('--qlog-root', default='comma_downloads', help='Root directory containing log files')
  ap.add_argument('--fields-config', default='configs/qlog_field_catalog_EV_safety.json',
                  help='Field selection config JSON')
  ap.add_argument('--out-csv', default='analysis/wide_ev_safety.csv', help='Output combined CSV path')
  ap.add_argument('--max-files', type=int, default=1, help='Number of log files to process')
  ap.add_argument('--random-sample', action='store_true', help='Randomize log selection')
  ap.add_argument('--duration-s', type=float, default=60.0, help='Segment duration in seconds')
  ap.add_argument('--max-gap-cont-s', type=float, default=0.3,
                  help='Max prev/next gap for linear interpolation of continuous fields')
  ap.add_argument('--max-gap-disc-s', type=float, default=1.0,
                  help='Max age for forward-fill of discrete fields')
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args()

  openpilot_path = Path(args.openpilot_path).expanduser().resolve()
  qlog_root = Path(args.qlog_root).expanduser().resolve()
  fields_config = Path(args.fields_config).expanduser().resolve()
  out_csv = Path(args.out_csv).expanduser().resolve()

  if not openpilot_path.exists():
    raise FileNotFoundError(f'openpilot path not found: {openpilot_path}')
  if not qlog_root.exists():
    raise FileNotFoundError(f'qlog root not found: {qlog_root}')
  if not fields_config.exists():
    raise FileNotFoundError(f'fields config not found: {fields_config}')

  fields_by_topic = load_fields_config(fields_config)

  sys.path.insert(0, openpilot_path.as_posix())
  import tools.lib.logreader as logreader
  patch_logreader_zstd(logreader)
  LogReader = logreader.LogReader

  qlogs = iter_qlog_paths(qlog_root, args.max_files, args.random_sample)
  if not qlogs:
    raise RuntimeError(f'No log files found under: {qlog_root}')

  if args.verbose:
    print(f'files={len(qlogs)}  (hz detected per file from filename)')

  parts: list[pd.DataFrame] = []
  for i, qlog in enumerate(qlogs, 1):
    hz = file_hz(qlog)   # 100 for *--rlog*, 10 for *--qlog*
    if args.verbose:
      print(f'[{i}/{len(qlogs)}] hz={hz}  {qlog}')
    df = build_wide_for_qlog(
      qlog=qlog,
      fields_by_topic=fields_by_topic,
      LogReader=LogReader,
      hz=hz,
      duration_s=args.duration_s,
      max_gap_cont_s=args.max_gap_cont_s,
      max_gap_disc_s=args.max_gap_disc_s,
    )
    if not df.empty:
      parts.append(df)

  if not parts:
    raise RuntimeError('No selected topic messages found in the chosen qlogs')

  all_df = pd.concat(parts, ignore_index=True)
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  all_df.to_csv(out_csv, index=False)

  print('Done')
  print(f'files_processed={len(qlogs)}')
  print(f'rows={len(all_df)} cols={len(all_df.columns)}')
  print(f'output_csv={out_csv}')


if __name__ == '__main__':
  main()
