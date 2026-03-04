#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_bool_series(series: pd.Series) -> pd.Series:
  if series.dtype == bool:
    return series
  s = series.astype(str).str.strip().str.lower()
  return s.isin(['true', '1', 'yes'])


def main() -> None:
  ap = argparse.ArgumentParser(description='Overlay ego speed and lead speed for a route table.')
  ap.add_argument('--input-csv', required=True, help='Route-level combined CSV path')
  ap.add_argument('--output-png', default=None, help='Output PNG path (default: next to input)')
  ap.add_argument('--title', default='Ego vs Lead Speed (lead absent -> 0)', help='Plot title')
  args = ap.parse_args()

  input_csv = Path(args.input_csv).expanduser().resolve()
  if not input_csv.exists():
    raise FileNotFoundError(f'input csv not found: {input_csv}')

  if args.output_png:
    output_png = Path(args.output_png).expanduser().resolve()
  else:
    output_png = input_csv.with_name(input_csv.stem + '--ego-vs-lead-speed.png')

  usecols = ['sec_in_route', 'carState.vEgo', 'radarState.leadOne.status', 'radarState.leadOne.vLead']
  df = pd.read_csv(input_csv, usecols=usecols).sort_values('sec_in_route')

  ego = pd.to_numeric(df['carState.vEgo'], errors='coerce')
  has_lead = parse_bool_series(df['radarState.leadOne.status'])
  lead_raw = pd.to_numeric(df['radarState.leadOne.vLead'], errors='coerce').fillna(0.0)
  lead = lead_raw.where(has_lead, 0.0)

  fig, ax = plt.subplots(figsize=(14, 5))
  ax.plot(df['sec_in_route'], ego, label='Ego Speed (vEgo)', linewidth=1.1, color='#1f77b4')
  ax.plot(df['sec_in_route'], lead, label='Lead Speed (vLead, no lead->0)', linewidth=1.1, color='#2ca02c')
  ax.set_title(args.title)
  ax.set_xlabel('Seconds in Route (s)')
  ax.set_ylabel('Speed (m/s)')
  ax.grid(True, alpha=0.3)
  ax.legend(loc='best')
  plt.tight_layout()

  output_png.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_png, dpi=150)
  print('saved', output_png)


if __name__ == '__main__':
  main()
