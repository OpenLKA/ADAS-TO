#!/usr/bin/env python3
import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

import zstandard as zstd

DEFAULT_TOPICS = [
  'accelerometer',
  'carControl',
  'carOutput',
  'carState',
  'controlsState',
  'radarState',
  'drivingModelData',
  'longitudinalPlan',
]

DEFAULT_TOPIC_FIELDS: dict[str, list[str]] = {
  'accelerometer': [
    'source', 'sensor', 'type', 'timestamp', 'acceleration.v', 'accel_status',
  ],
  'carControl': [
    'enabled', 'latActive', 'longActive', 'leftBlinker', 'rightBlinker',
    'actuators.accel', 'actuators.torque', 'actuators.speed', 'actuators.curvature',
    'hudControl.setSpeed', 'hudControl.leadVisible', 'angularVelocity', 'orientationNED',
  ],
  'carOutput': [
    'actuatorsOutput.accel', 'actuatorsOutput.brake', 'actuatorsOutput.gas',
    'actuatorsOutput.speed', 'actuatorsOutput.curvature', 'actuatorsOutput.steer',
    'actuatorsOutput.steerOutputCan', 'actuatorsOutput.steeringAngleDeg',
    'actuatorsOutput.longControlState',
  ],
  'carState': [
    'vEgo', 'aEgo', 'vEgoRaw', 'standstill', 'steeringAngleDeg', 'steeringTorque',
    'steeringPressed', 'gas', 'gasPressed', 'brake', 'brakePressed',
    'cruiseState.enabled', 'cruiseState.available', 'cruiseState.speed',
    'buttonEvents', 'canMonoTimesDEPRECATED', 'errorsDEPRECATED', 'events',
  ],
  'controlsState': [
    'enabled', 'active', 'curvature', 'desiredCurvature', 'vCruise', 'vCruiseCluster',
    'forceDecel', 'longControlState', 'alertText1', 'alertText2', 'canMonoTimesDEPRECATED',
  ],
  'radarState': [
    'cumLagMs', 'canMonoTimesDEPRECATED', 'radarErrors', 'warpMatrixDEPRECATED',
    'leadOne.status', 'leadOne.dRel', 'leadOne.vRel', 'leadOne.aRel', 'leadOne.yRel',
    'leadOne.vLead', 'leadOne.vLeadK', 'leadOne.aLeadK',
    'leadTwo.status', 'leadTwo.dRel', 'leadTwo.vRel', 'leadTwo.aRel', 'leadTwo.yRel',
    'leadTwo.vLead', 'leadTwo.vLeadK', 'leadTwo.aLeadK',
  ],
  'drivingModelData': [
    'frameId', 'frameIdExtra', 'frameDropPerc', 'modelExecutionTime',
    'action.desiredAcceleration', 'action.desiredCurvature', 'action.shouldStop',
    'path.xCoefficients', 'path.yCoefficients', 'path.zCoefficients',
    'laneLineMeta.leftProb', 'laneLineMeta.rightProb',
    'laneLineMeta.leftY', 'laneLineMeta.rightY',
    'meta.laneChangeState', 'meta.laneChangeDirection',
  ],
  'longitudinalPlan': [
    'aTarget', 'shouldStop', 'allowBrake', 'allowThrottle', 'hasLead', 'fcw',
    'longitudinalPlanSource', 'processingDelay', 'solverExecutionTime', 'modelMonoTime',
    'speeds', 'accels', 'jerks', 'dPolyDEPRECATED', 'eventsDEPRECATED',
    'gpsTrajectoryDEPRECATED.x', 'gpsTrajectoryDEPRECATED.y',
  ],
}


def parse_route_segment(qlog_path: Path) -> tuple[str | None, str | None, str | None]:
  parts = qlog_path.parts
  if len(parts) < 4:
    return None, None, None
  segment = parts[-2]
  route_id = parts[-3]
  dongle = parts[-4]
  return dongle, route_id, segment


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


class CsvTopicWriter:
  def __init__(self, out_path: Path, fieldnames: list[str]):
    self.out_path = out_path
    self.fieldnames = fieldnames
    self.out_path.parent.mkdir(parents=True, exist_ok=True)
    self.fp = self.out_path.open('w', newline='')
    self.writer = csv.DictWriter(self.fp, fieldnames=self.fieldnames, extrasaction='ignore')
    self.writer.writeheader()

  def write_row(self, row: dict[str, Any]) -> None:
    self.writer.writerow(row)

  def close(self) -> None:
    self.fp.close()


class ParquetTopicWriter:
  def __init__(self, out_path: Path, fieldnames: list[str]):
    try:
      import pyarrow as pa
      import pyarrow.parquet as pq
    except ImportError as e:
      raise RuntimeError('Parquet output requires pyarrow. Install with: pip install pyarrow') from e

    self.pa = pa
    self.pq = pq
    self.out_path = out_path
    self.fieldnames = fieldnames
    self.out_path.parent.mkdir(parents=True, exist_ok=True)
    self._writer = None

  def write_row(self, row: dict[str, Any]) -> None:
    normalized = {k: row.get(k) for k in self.fieldnames}
    table = self.pa.Table.from_pylist([normalized])
    if self._writer is None:
      self._writer = self.pq.ParquetWriter(self.out_path.as_posix(), table.schema)
    self._writer.write_table(table)

  def close(self) -> None:
    if self._writer is not None:
      self._writer.close()


def iter_qlog_paths(
  base_dir: Path,
  max_files: int | None,
  random_sample: bool,
) -> list[Path]:
  """Return candidate log files under base_dir.

  Log type is detected automatically from the filename:
    rlog:  *--rlog         (raw, 100 Hz)
           *--rlog.bz2     (bz2-compressed, 100 Hz)
    qlog:  *--qlog         (raw, 10 Hz)
           *--qlog.bz2     (bz2-compressed, 10 Hz)
           *qlog*.zst      (zstd-compressed, 10 Hz)

  When both rlog and qlog exist for the same segment, rlog wins (higher Hz).
  Within the same type, raw beats bz2/zst so we never double-count a segment.

  Within each log type, raw files take precedence over compressed duplicates
  for the same segment number so we never double-count a segment.

  Sorting: numeric prefix before the first '--' (0, 1, 2, … 10, 11, …).
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

  rlog_patterns = ['*--rlog', '*--rlog.bz2']
  qlog_patterns = ['*--qlog', '*--qlog.bz2', '*qlog*.zst']

  # Always auto-detect: rlog preferred per segment, qlog as fallback
  rlog_files = _collect(rlog_patterns)
  qlog_files = _collect(qlog_patterns)

  def seg_key(p: Path) -> tuple[Path, int]:
    m = _re.match(r'^(\d+)--', p.name)
    return (p.parent, int(m.group(1)) if m else -1)

  rlog_segs = {seg_key(p) for p in rlog_files}
  # Keep qlog files only for segments NOT already covered by rlog
  qlog_fallback = [p for p in qlog_files if seg_key(p) not in rlog_segs]
  raw = rlog_files + qlog_fallback

  # De-duplicate: for same (parent, seg_num), prefer raw > bz2 > zst
  def file_priority(p: Path) -> int:
    name = p.name
    if name.endswith('.bz2') or name.endswith('.zst'):
      return 1
    return 0  # raw wins (lower number = higher priority)

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


def patch_logreader_zstd(logreader_module: Any) -> None:
  def _stream_decompress(dat: bytes) -> bytes:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(dat)) as reader:
      return reader.read()

  logreader_module.zstd.decompress = _stream_decompress


def build_field_catalog(qlogs: list[Path], topics: list[str], LogReader: Any, max_files: int) -> dict[str, list[str]]:
  out: dict[str, set[str]] = {t: set() for t in topics}
  scan_qlogs = qlogs[:max_files]
  for i, qlog in enumerate(scan_qlogs, 1):
    if i % 10 == 0:
      print(f'catalog_progress={i}/{len(scan_qlogs)}')
    for msg in LogReader(qlog.as_posix(), only_union_types=True, sort_by_time=True):
      topic = msg.which()
      if topic not in out:
        continue
      payload = getattr(msg, topic).to_dict(verbose=True)
      flat: dict[str, Any] = {}
      flatten_payload(payload, '', flat)
      out[topic].update(flat.keys())
  return {k: sorted(v) for k, v in out.items()}


def load_fields_config(config_path: Path, requested_topics: list[str]) -> dict[str, list[str]]:
  if not config_path.exists():
    return {t: DEFAULT_TOPIC_FIELDS.get(t, []) for t in requested_topics}

  data = json.loads(config_path.read_text())
  topics_cfg = data.get('topics', data)
  if not isinstance(topics_cfg, dict):
    raise ValueError(f'Invalid config format in {config_path}: expected object with "topics"')

  fields_by_topic: dict[str, list[str]] = {}
  for topic in requested_topics:
    if topic not in topics_cfg:
      raise ValueError(f'Topic {topic} missing from fields config: {config_path}')
    fields = topics_cfg[topic]
    if not isinstance(fields, list) or not all(isinstance(x, str) for x in fields):
      raise ValueError(f'Invalid field list for topic {topic} in {config_path}')
    fields_by_topic[topic] = fields
  return fields_by_topic


def write_default_config(config_path: Path) -> None:
  config_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    'version': 1,
    'topics': DEFAULT_TOPIC_FIELDS,
  }
  config_path.write_text(json.dumps(payload, indent=2) + '\n')


def main() -> None:
  ap = argparse.ArgumentParser(description='Parse openpilot qlogs/rlogs with LogReader into topic tables.')
  ap.add_argument('--openpilot-path', default='/home/henry/Dropbox/OP_CAN_DataProcessing',
                  help='Path to openpilot repo root (or OP_CAN_DataProcessing containing tools/)')
  ap.add_argument('--qlog-root', default='comma_downloads', help='Root dir containing log files')
  ap.add_argument('--out-dir', default='analysis/qlog_tables', help='Output directory')
  ap.add_argument('--topics', default=','.join(DEFAULT_TOPICS), help='Comma-separated topics to extract')
  ap.add_argument('--fields-config', default='configs/qlog_field_config.json', help='JSON file specifying columns per topic')
  ap.add_argument('--write-default-config', action='store_true', help='Write default config to --fields-config and exit')
  ap.add_argument('--catalog-out', default=None, help='Write discovered flattened field catalog JSON and exit')
  ap.add_argument('--catalog-max-files', type=int, default=20, help='Number of qlogs to scan for --catalog-out')
  ap.add_argument('--format', choices=['csv', 'parquet'], default='csv', help='Output format')
  ap.add_argument('--max-files', type=int, default=None, help='Limit log files processed')
  ap.add_argument('--random-sample', action='store_true', help='Randomize log order before selecting --max-files')
  ap.add_argument('--include-raw', action='store_true', help='Include full topic JSON in raw_json column')
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args()

  openpilot_path = Path(args.openpilot_path).expanduser().resolve()
  qlog_root = Path(args.qlog_root).expanduser().resolve()
  out_dir = Path(args.out_dir).expanduser().resolve()
  fields_config = Path(args.fields_config).expanduser().resolve()

  if args.write_default_config:
    write_default_config(fields_config)
    print(f'wrote_default_config={fields_config}')
    return

  if not openpilot_path.exists():
    raise FileNotFoundError(f'openpilot path not found: {openpilot_path}')
  if not qlog_root.exists():
    raise FileNotFoundError(f'qlog root not found: {qlog_root}')

  requested_topics = [t.strip() for t in args.topics.split(',') if t.strip()]

  sys.path.insert(0, openpilot_path.as_posix())
  import tools.lib.logreader as logreader

  patch_logreader_zstd(logreader)
  LogReader = logreader.LogReader

  qlogs = iter_qlog_paths(qlog_root, args.max_files, args.random_sample)
  if not qlogs:
    raise RuntimeError(f'No log files found under: {qlog_root}')

  if args.catalog_out:
    catalog = build_field_catalog(qlogs, requested_topics, LogReader, args.catalog_max_files)
    catalog_path = Path(args.catalog_out).expanduser().resolve()
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps({'topics': catalog}, indent=2) + '\n')
    print(f'wrote_catalog={catalog_path}')
    return

  fields_by_topic = load_fields_config(fields_config, requested_topics)

  base_fields = ['dongle_id', 'route_id', 'segment', 'logMonoTime', 'time_s']
  writers = {}
  writer_cls = CsvTopicWriter if args.format == 'csv' else ParquetTopicWriter
  ext = 'csv' if args.format == 'csv' else 'parquet'

  for topic in requested_topics:
    fields = base_fields + fields_by_topic[topic]
    if args.include_raw:
      fields = fields + ['raw_json']
    writers[topic] = writer_cls(out_dir / f'{topic}.{ext}', fields)

  per_topic_count = {t: 0 for t in requested_topics}
  files_done = 0

  try:
    for qlog in qlogs:
      files_done += 1
      dongle_id, route_id, segment = parse_route_segment(qlog)
      if args.verbose:
        print(f'[{files_done}/{len(qlogs)}] {qlog}')

      base_mono = None
      for msg in LogReader(qlog.as_posix(), only_union_types=True, sort_by_time=True):
        topic = msg.which()
        if topic not in writers:
          continue

        mono = int(msg.logMonoTime)
        if base_mono is None:
          base_mono = mono
        t_rel_s = (mono - base_mono) / 1e9

        payload = getattr(msg, topic).to_dict(verbose=True)
        flat: dict[str, Any] = {}
        flatten_payload(payload, '', flat)

        row = {
          'dongle_id': dongle_id,
          'route_id': route_id,
          'segment': segment,
          'logMonoTime': mono,
          'time_s': round(t_rel_s, 6),
        }
        for field in fields_by_topic[topic]:
          row[field] = flat.get(field)

        if args.include_raw:
          row['raw_json'] = json.dumps(payload, separators=(',', ':'))

        writers[topic].write_row(row)
        per_topic_count[topic] += 1

    print('Done')
    print(f'files_processed={files_done}')
    print('rows_per_topic=')
    for topic, count in per_topic_count.items():
      print(f'  {topic}: {count}')
    print(f'output_dir={out_dir}')
  finally:
    for w in writers.values():
      w.close()


if __name__ == '__main__':
  main()
