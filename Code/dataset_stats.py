#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TakeOver dataset statistics:
1) dongle_id / route_id distribution (HTML + CSV)
2) time distribution by YYYY-MM (PNG)
3) geographic distribution (Leaflet/Folium HTML) similar to your enhanced_global_map.html

Expected layout:
  <ROOT>/<CAR_MODEL>/<DONGLE_ID>/<ROUTE_ID>/...

Default output:
  <ROOT>/Code/output/
    - dataset_routes.csv
    - counts_by_car_model.csv
    - counts_by_dongle.csv
    - time_distribution_year_month.png
    - dataset_overview.html
    - geo_distribution_map.html
    - geo_points_sample.csv
    - geo_clusters.csv

Geo extraction heuristic:
  - Prefer route.coords-like JSON files inside each route folder (depth-limited search).
  - Fallback: parse JSON metadata for lat/lng fields (start/end).
  - Fallback: parse segment-like JSON arrays containing start_lat/start_lng/end_lat/end_lng.

Usage:
  python3 dataset_stats.py
  python3 dataset_stats.py --deep-scan --geo-scan-depth 3 --max-total-points 200000
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional deps
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

try:
    import folium  # type: ignore
    from folium import plugins  # type: ignore
except Exception:
    folium = None  # type: ignore
    plugins = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
except Exception:
    MiniBatchKMeans = None  # type: ignore


TS_RE = re.compile(r"(20\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})(?:--\d+)?")
TS_ISO_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})[T _](\d{2}):(\d{2}):(\d{2})")

JSON_TIME_KEYS = (
    "start_time_utc_millis",
    "start_time",
    "startTime",
    "start_time_utc",
    "timestamp",
    "time",
    "route_start_time_utc_millis",
)

JSON_LAT_KEYS = ("lat", "latitude", "start_lat", "end_lat")
JSON_LON_KEYS = ("lng", "lon", "longitude", "start_lng", "end_lng")


@dataclass(frozen=True)
class RouteRow:
    car_model: str
    dongle_id: str
    route_id: str
    route_datetime_utc: Optional[str]  # ISO8601
    year_month: Optional[str]          # YYYY-MM
    time_source: str
    route_center_lat: Optional[float]
    route_center_lng: Optional[float]
    geo_points_used: int
    geo_source: str
    path: str


def parse_ts_from_string(s: str) -> Optional[dt.datetime]:
    m = TS_RE.search(s)
    if m:
        try:
            return dt.datetime.strptime(m.group(1), "%Y-%m-%d--%H-%M-%S")
        except ValueError:
            pass
    m2 = TS_ISO_RE.search(s)
    if m2:
        try:
            ymd = m2.group(1)
            hh, mm, ss = int(m2.group(2)), int(m2.group(3)), int(m2.group(4))
            y, mo, d = map(int, ymd.split("-"))
            return dt.datetime(y, mo, d, hh, mm, ss)
        except Exception:
            pass
    return None


def safe_list_dirs(p: Path) -> Iterable[Path]:
    try:
        for x in p.iterdir():
            if x.is_dir():
                yield x
    except Exception:
        return


def parse_ts_from_json_obj(data) -> Optional[dt.datetime]:
    if isinstance(data, dict):
        for key in JSON_TIME_KEYS:
            if key in data:
                v = data[key]
                if isinstance(v, (int, float)):
                    try:
                        if v > 10_000_000_000:  # ms
                            return dt.datetime.utcfromtimestamp(v / 1000.0)
                        if v > 1_000_000_000:   # s
                            return dt.datetime.utcfromtimestamp(v)
                    except Exception:
                        pass
                if isinstance(v, str):
                    ts = parse_ts_from_string(v)
                    if ts:
                        return ts

        # last resort: stringify nested items
        for _, v in data.items():
            ts = parse_ts_from_string(str(v))
            if ts:
                return ts

    return None


def parse_ts_from_json_file(p: Path) -> Optional[dt.datetime]:
    try:
        if p.stat().st_size > 5_000_000:
            return None
        data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return parse_ts_from_json_obj(data)


def infer_route_datetime(route_dir: Path, route_id: str, deep_scan: bool = False) -> Tuple[Optional[dt.datetime], str]:
    for s, label in ((route_dir.name, "route_dir_name"), (route_id, "route_id")):
        ts = parse_ts_from_string(s)
        if ts:
            return ts, label

    for name in ("metadata.json", "meta.json", "route.json", "info.json", "route_info.json", "manifest.json"):
        p = route_dir / name
        if p.exists() and p.is_file():
            ts = parse_ts_from_json_file(p)
            if ts:
                return ts, f"json:{name}"

    try:
        for child in route_dir.iterdir():
            ts = parse_ts_from_string(child.name)
            if ts:
                return ts, "child_name"
    except Exception:
        pass

    if deep_scan:
        try:
            for child in route_dir.iterdir():
                if not child.is_dir():
                    continue
                for g in child.iterdir():
                    ts = parse_ts_from_string(g.name)
                    if ts:
                        return ts, "grandchild_name"
        except Exception:
            pass

    return None, "unknown"


def is_valid_latlon(lat: float, lon: float) -> bool:
    return (
        isinstance(lat, (int, float))
        and isinstance(lon, (int, float))
        and -90.0 <= float(lat) <= 90.0
        and -180.0 <= float(lon) <= 180.0
        and not (float(lat) == 0.0 and float(lon) == 0.0)
    )


def extract_latlon_from_obj(obj) -> List[Tuple[float, float]]:
    """
    Extract one or more lat/lon pairs from a JSON object.
    Supports:
      - [lat, lon] lists
      - {"lat":..., "lng":...} dicts
      - {"start_lat":..., "start_lng":..., "end_lat":..., "end_lng":...}
      - segment-like objects containing start/end lat/lng
    """
    out: List[Tuple[float, float]] = []

    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        try:
            lat, lon = float(obj[0]), float(obj[1])
            if is_valid_latlon(lat, lon):
                out.append((lat, lon))
        except Exception:
            pass
        return out

    if isinstance(obj, dict):
        # direct lat/lon
        lat = None
        lon = None
        for k in ("lat", "latitude"):
            if k in obj:
                lat = obj[k]
                break
        for k in ("lng", "lon", "longitude"):
            if k in obj:
                lon = obj[k]
                break
        if lat is not None and lon is not None:
            try:
                latf, lonf = float(lat), float(lon)
                if is_valid_latlon(latf, lonf):
                    out.append((latf, lonf))
            except Exception:
                pass

        # start/end
        if "start_lat" in obj and "start_lng" in obj:
            try:
                latf, lonf = float(obj["start_lat"]), float(obj["start_lng"])
                if is_valid_latlon(latf, lonf):
                    out.append((latf, lonf))
            except Exception:
                pass
        if "end_lat" in obj and "end_lng" in obj:
            try:
                latf, lonf = float(obj["end_lat"]), float(obj["end_lng"])
                if is_valid_latlon(latf, lonf):
                    out.append((latf, lonf))
            except Exception:
                pass

    return out


def limited_walk_files(root: Path, max_depth: int) -> Iterable[Path]:
    """
    Yield files under root with a maximum depth (root depth = 0).
    """
    root = root.resolve()
    base_parts = len(root.parts)
    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath)
        depth = len(dp.parts) - base_parts
        if depth > max_depth:
            # prune by modifying in-place is tricky with os.walk; use continue and rely on depth check for children
            continue
        for fn in filenames:
            yield dp / fn


def find_coords_like_files(route_dir: Path, scan_depth: int) -> List[Path]:
    """
    Find likely GPS coordinate files inside a route directory.
    """
    candidates = set()
    names = {
        "route.coords",
        "route.coords.json",
        "route_coords.json",
        "coords.json",
        "gps.json",
        "gps_coords.json",
        "gps_path.json",
        "route_gps.json",
    }
    for p in limited_walk_files(route_dir, max_depth=scan_depth):
        n = p.name.lower()
        if n in names or "route.coords" in n:
            candidates.add(p)
    return sorted(candidates)


def read_json_file(p: Path):
    try:
        if p.stat().st_size > 200_000_000:  # guard huge json
            return None
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def sample_points(points: Sequence[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    if max_points <= 0 or len(points) <= max_points:
        return list(points)
    step = max(1, math.ceil(len(points) / max_points))
    return [points[i] for i in range(0, len(points), step)][:max_points]


def infer_route_points(route_dir: Path, scan_depth: int, max_points_per_route: int) -> Tuple[List[Tuple[float, float]], str]:
    """
    Returns (points, source_label). Points represent the route path sample, or at least start/end.
    """
    # 1) route.coords-like file(s)
    files = find_coords_like_files(route_dir, scan_depth=scan_depth)
    for f in files:
        data = read_json_file(f)
        if data is None:
            continue
        pts: List[Tuple[float, float]] = []
        if isinstance(data, list):
            for item in data:
                pts.extend(extract_latlon_from_obj(item))
        elif isinstance(data, dict):
            # sometimes stored under key
            for key in ("coords", "points", "route", "path", "gps"):
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        pts.extend(extract_latlon_from_obj(item))
        pts = [p for p in pts if is_valid_latlon(p[0], p[1])]
        if pts:
            return sample_points(pts, max_points_per_route), f"coords:{f.name}"

    # 2) any small JSON metadata that includes lat/lon
    for name in ("metadata.json", "meta.json", "route.json", "info.json", "route_info.json", "manifest.json"):
        p = route_dir / name
        if p.exists() and p.is_file():
            data = read_json_file(p)
            if data is None:
                continue
            pts: List[Tuple[float, float]] = []
            if isinstance(data, dict):
                pts.extend(extract_latlon_from_obj(data))
            pts = [p for p in pts if is_valid_latlon(p[0], p[1])]
            if pts:
                return sample_points(pts, max_points_per_route), f"json:{name}"

    # 3) segment-like JSON arrays
    for p in limited_walk_files(route_dir, max_depth=min(scan_depth, 2)):
        if not p.name.lower().endswith(".json"):
            continue
        if p.stat().st_size > 20_000_000:
            continue
        data = read_json_file(p)
        if not isinstance(data, list):
            continue
        pts: List[Tuple[float, float]] = []
        # if list looks like segments
        for item in data[:50000]:
            pts.extend(extract_latlon_from_obj(item))
        pts = [p for p in pts if is_valid_latlon(p[0], p[1])]
        if len(pts) >= 2:
            return sample_points(pts, max_points_per_route), f"segments_json:{p.name}"

    return [], "unknown"


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    # Earth radius in meters
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def k_for_points(n: int, k_max: int = 30) -> int:
    # heuristic: similar cluster counts to your example
    return max(1, min(k_max, int(round(math.sqrt(n) / 2))))


def cluster_points(points: List[Tuple[float, float]], k_max: int = 30) -> List[Dict]:
    """
    Cluster points into K centers. Returns a list of dicts:
      {cluster_id, center_lat, center_lng, points, radius_m}
    Uses MiniBatchKMeans if available; otherwise uses a simple numpy kmeans fallback.
    """
    if not points:
        return []

    k = k_for_points(len(points), k_max=k_max)
    if k <= 1:
        lat = sum(p[0] for p in points) / len(points)
        lon = sum(p[1] for p in points) / len(points)
        r = max(haversine_m(lat, lon, p[0], p[1]) for p in points)
        return [dict(cluster_id=0, center_lat=lat, center_lng=lon, points=len(points), radius_m=r)]

    if np is None:
        # fallback: no numpy — single cluster
        lat = sum(p[0] for p in points) / len(points)
        lon = sum(p[1] for p in points) / len(points)
        r = max(haversine_m(lat, lon, p[0], p[1]) for p in points)
        return [dict(cluster_id=0, center_lat=lat, center_lng=lon, points=len(points), radius_m=r)]

    X = np.array(points, dtype=float)

    if MiniBatchKMeans is not None:
        model = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init="auto")
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
    else:
        # lightweight kmeans fallback
        rng = np.random.default_rng(0)
        # init centers by sampling points
        centers = X[rng.choice(len(X), size=k, replace=False)]
        for _ in range(12):
            # assign
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            # update
            new_centers = []
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centers.append(X[mask].mean(axis=0))
                else:
                    new_centers.append(X[rng.integers(0, len(X))])
            centers = np.vstack(new_centers)

    clusters: List[Dict] = []
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        c_lat, c_lon = float(centers[cid][0]), float(centers[cid][1])
        # radius = max distance to center
        r = 0.0
        for i in idx[:20000]:  # cap for speed
            r = max(r, haversine_m(c_lat, c_lon, float(X[i][0]), float(X[i][1])))
        clusters.append(dict(cluster_id=int(cid), center_lat=c_lat, center_lng=c_lon, points=int(idx.size), radius_m=float(r)))
    # sort by size desc
    clusters.sort(key=lambda d: (-d["points"], d["cluster_id"]))
    # reindex cluster ids to match sorted order
    for new_id, c in enumerate(clusters):
        c["cluster_id"] = new_id
    return clusters


# Region boxes (deg)
REGIONS = {
    "North America": dict(lat_min=15, lat_max=72, lon_min=-170, lon_max=-50),
    "Europe": dict(lat_min=35, lat_max=72, lon_min=-10, lon_max=45),
    "Asia": dict(lat_min=-10, lat_max=80, lon_min=45, lon_max=180),
    "Africa": dict(lat_min=-35, lat_max=37, lon_min=-20, lon_max=52),
}


def region_of(lat: float, lon: float) -> str:
    for name, b in REGIONS.items():
        if b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]:
            return name
    return "Other"


def region_stats(points: List[Tuple[float, float]]) -> List[Dict]:
    c = Counter(region_of(lat, lon) for lat, lon in points)
    stats = []
    for name, b in list(REGIONS.items()) + [("Other", None)]:
        pts = c.get(name, 0)
        if b is None:
            density = 0.0
        else:
            area_deg2 = (b["lat_max"] - b["lat_min"]) * (b["lon_max"] - b["lon_min"])
            density = pts / area_deg2 if area_deg2 > 0 else 0.0
        stats.append(dict(region=name, points=pts, density=density))
    return stats


def plot_time_distribution(rows: List[RouteRow], out_png: Path) -> Optional[str]:
    if plt is None:
        return None

    ym_counts = Counter(r.year_month for r in rows if r.year_month)
    if not ym_counts:
        return None

    keys = sorted(ym_counts.keys())
    values = [ym_counts[k] for k in keys]

    plt.figure(figsize=(max(10, min(24, len(keys) * 0.35)), 5))
    plt.bar(keys, values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Routes")
    plt.title("Route Count by Year-Month (UTC, inferred)")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

    try:
        return base64.b64encode(out_png.read_bytes()).decode("ascii")
    except Exception:
        return None


def write_csv(path: Path, header: List[str], rows: Iterable[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))


def write_routes_csv(rows: List[RouteRow], out_csv: Path) -> None:
    write_csv(
        out_csv,
        [
            "car_model",
            "dongle_id",
            "route_id",
            "route_datetime_utc",
            "year_month",
            "time_source",
            "route_center_lat",
            "route_center_lng",
            "geo_points_used",
            "geo_source",
            "path",
        ],
        [
            (
                r.car_model,
                r.dongle_id,
                r.route_id,
                r.route_datetime_utc or "",
                r.year_month or "",
                r.time_source,
                "" if r.route_center_lat is None else f"{r.route_center_lat:.8f}",
                "" if r.route_center_lng is None else f"{r.route_center_lng:.8f}",
                r.geo_points_used,
                r.geo_source,
                r.path,
            )
            for r in rows
        ],
    )


def counts_tables(rows: List[RouteRow]) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, str, int]]]:
    car_to_dongles: Dict[str, set] = defaultdict(set)
    car_to_routes: Counter[str] = Counter()
    dongle_routes: Counter[Tuple[str, str]] = Counter()

    for r in rows:
        car_to_dongles[r.car_model].add(r.dongle_id)
        car_to_routes[r.car_model] += 1
        dongle_routes[(r.car_model, r.dongle_id)] += 1

    car_model_counts = []
    for car_model, n_routes in car_to_routes.most_common():
        car_model_counts.append((car_model, len(car_to_dongles[car_model]), n_routes))

    dongle_counts = []
    for (car_model, dongle_id), n_routes in dongle_routes.most_common():
        dongle_counts.append((car_model, dongle_id, n_routes))

    return car_model_counts, dongle_counts


def build_rows_and_geo(
    root: Path,
    deep_scan_time: bool,
    geo_scan_depth: int,
    max_points_per_route: int,
    max_total_points: int,
    seed: int,
) -> Tuple[List[RouteRow], List[Tuple[float, float]]]:
    rng = random.Random(seed)

    rows: List[RouteRow] = []
    all_points: List[Tuple[float, float]] = []

    def maybe_add_points(new_pts: List[Tuple[float, float]]):
        nonlocal all_points
        if not new_pts:
            return
        # reservoir-ish sampling to enforce max_total_points
        for p in new_pts:
            if len(all_points) < max_total_points:
                all_points.append(p)
            else:
                j = rng.randrange(0, len(all_points) + 1)
                if j < len(all_points):
                    all_points[j] = p

    for car_dir in safe_list_dirs(root):
        car_model = car_dir.name
        if car_model.lower() in {"code", ".git", ".venv", "__pycache__", "output"}:
            continue

        for dongle_dir in safe_list_dirs(car_dir):
            dongle_id = dongle_dir.name

            for route_dir in safe_list_dirs(dongle_dir):
                route_id = route_dir.name

                ts, ts_src = infer_route_datetime(route_dir, route_id, deep_scan=deep_scan_time)
                if ts is not None:
                    iso = ts.replace(tzinfo=dt.timezone.utc).isoformat()
                    ym = f"{ts.year:04d}-{ts.month:02d}"
                else:
                    iso = None
                    ym = None

                pts, geo_src = infer_route_points(route_dir, scan_depth=geo_scan_depth, max_points_per_route=max_points_per_route)
                maybe_add_points(pts)

                if pts:
                    lat_c = sum(p[0] for p in pts) / len(pts)
                    lon_c = sum(p[1] for p in pts) / len(pts)
                else:
                    lat_c, lon_c = None, None

                rows.append(
                    RouteRow(
                        car_model=car_model,
                        dongle_id=dongle_id,
                        route_id=route_id,
                        route_datetime_utc=iso,
                        year_month=ym,
                        time_source=ts_src,
                        route_center_lat=lat_c,
                        route_center_lng=lon_c,
                        geo_points_used=len(pts),
                        geo_source=geo_src,
                        path=str(route_dir),
                    )
                )

    return rows, all_points


def add_fixed_overlay(m, html: str) -> None:
    if folium is None:
        return
    from folium import Element  # type: ignore
    m.get_root().html.add_child(Element(html))


def build_geo_map(
    points: List[Tuple[float, float]],
    clusters: List[Dict],
    reg_stats: List[Dict],
    out_html: Path,
    max_marker_points: int,
    seed: int,
) -> None:
    if folium is None or plugins is None:
        raise RuntimeError("folium is not installed; please `pip install folium`")

    # Base map: match your HTML defaults
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=3,
        control_scale=True,
        tiles=None,
        max_bounds=True,
        world_copy_jump=True,
        prefer_canvas=False,
    )

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="cartodbpositron",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
             '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Dark Mode",
    ).add_to(m)

    # Heatmap overlay (match gradient/radius/blur)
    heat = plugins.HeatMap(
        data=[[lat, lon, 1] for lat, lon in points],
        name="Heat Map",
        radius=25,
        blur=15,
        min_opacity=0.5,
        max_zoom=4,
        gradient={0.2: "blue", 0.4: "cyan", 0.6: "lime", 0.8: "yellow", 1.0: "red"},
    )
    heat.add_to(m)

    # Cluster layer (blue)
    cluster_fg = folium.FeatureGroup(name="Clusters", show=True)
    for c in clusters:
        lat, lon = c["center_lat"], c["center_lng"]
        npts = c["points"]
        radius = max(1.0, float(c["radius_m"]))
        cid = c["cluster_id"]

        folium.Circle(
            location=[lat, lon],
            radius=radius,
            color="#1f77b4",
            weight=3,
            opacity=0.4,
            fill=True,
            fill_opacity=0.2,
            fill_color="#1f77b4",
            popup=folium.Popup(f"Cluster {cid}<br>Points: {npts}", max_width=300),
        ).add_to(cluster_fg)

        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            color="#1f77b4",
            weight=2,
            opacity=1.0,
            fill=True,
            fill_opacity=0.2,
            fill_color="#1f77b4",
            popup=folium.Popup(f"Cluster Center {cid}<br>Points: {npts}", max_width=300),
        ).add_to(cluster_fg)

    cluster_fg.add_to(m)

    # Individual point layer (orange) — sampled for usability
    marker_fg = folium.FeatureGroup(name="Individual Locations (sample)", show=False)
    rng = random.Random(seed)
    if len(points) > max_marker_points:
        idx = rng.sample(range(len(points)), k=max_marker_points)
        pts_show = [points[i] for i in idx]
    else:
        pts_show = points

    for lat, lon in pts_show:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#ff7f0e",
            weight=1,
            opacity=0.7,
            fill=True,
            fill_opacity=0.4,
            fill_color="#ff7f0e",
        ).add_to(marker_fg)
    marker_fg.add_to(m)

    # Layer control (top-right, expanded)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    # Fixed overlays (match your sample)
    reg_rows = []
    for s in reg_stats:
        if s["region"] == "Other":
            continue  # keep the same 4-region box as your sample; "Other" is still computed elsewhere
        reg_rows.append(f"""
            <div style='border-bottom: 1px solid #eee; padding-bottom: 5px;'>
                <div style='font-weight: bold; color: #666;'>{s["region"]}</div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px;'>
                    <div>Points: {s["points"]}</div>
                    <div>Density: {s["density"]:.2e}</div>
                </div>
            </div>
        """.rstrip())

    regional_box = f"""
    <div style='
        position: fixed;
        bottom: 20px;
        left: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
        max-width: 300px;
        z-index: 1000;
    '>
        <h4 style='margin: 0 0 10px 0; color: #333;'>Regional Distribution</h4>
        <div style='display: grid; gap: 8px;'>
            {''.join(reg_rows)}
        </div>
    </div>
    """.rstrip()

    legend_box = """
    <div style='
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 12px;
        border-radius: 6px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
        z-index: 1000;
    '>
        <h4 style='margin: 0 0 8px 0; color: #333;'>Legend</h4>
        <div style='display: grid; gap: 6px;'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='width: 12px; height: 12px; background-color: #1f77b4; border-radius: 50%;'></div>
                <span>Cluster Center</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='width: 12px; height: 12px; background-color: #ff7f0e; border-radius: 50%;'></div>
                <span>Individual Location</span>
            </div>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div style='width: 20px; height: 8px; background: linear-gradient(to right, blue, red);'></div>
                <span>Density Distribution</span>
            </div>
        </div>
    </div>
    """.rstrip()

    add_fixed_overlay(m, regional_box)
    add_fixed_overlay(m, legend_box)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


def generate_overview_html(
    rows: List[RouteRow],
    car_model_counts: List[Tuple[str, int, int]],
    dongle_counts: List[Tuple[str, str, int]],
    time_plot_b64: Optional[str],
    geo_summary: Dict[str, str],
    out_html: Path,
    out_png: Path,
    geo_map_html: Path,
) -> None:
    total_routes = len(rows)
    total_dongles = len({(r.car_model, r.dongle_id) for r in rows})
    total_models = len({r.car_model for r in rows})
    known_time = sum(1 for r in rows if r.year_month)
    unknown_time = total_routes - known_time
    geo_known = sum(1 for r in rows if r.geo_points_used > 0)

    # Distribution tree
    tree: Dict[str, Dict[str, List[RouteRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tree[r.car_model][r.dongle_id].append(r)

    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    if time_plot_b64:
        time_plot_html = f"<img style='max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px;' src='data:image/png;base64,{time_plot_b64}' />"
        time_plot_note = f"<div class='text-muted' style='font-size: 0.9rem; margin-top: 6px;'>Also saved as: {esc(str(out_png))}</div>"
    else:
        time_plot_html = "<div class='alert alert-warning'>No time plot generated (either no timestamps found, or matplotlib not installed).</div>"
        time_plot_note = ""

    car_rows_html = "".join(
        f"<tr><td>{esc(cm)}</td><td>{d}</td><td>{r}</td></tr>"
        for cm, d, r in car_model_counts
    )
    dongle_rows_html = "".join(
        f"<tr><td>{esc(cm)}</td><td>{esc(did)}</td><td>{r}</td></tr>"
        for cm, did, r in dongle_counts[:2000]
    )

    parts = []
    for car_model, dongles in sorted(tree.items(), key=lambda kv: (-sum(len(v) for v in kv[1].values()), kv[0])):
        car_routes = sum(len(v) for v in dongles.values())
        parts.append(f"<details class='mb-2'><summary><b>{esc(car_model)}</b> — dongles: {len(dongles)}, routes: {car_routes}</summary>")
        parts.append("<div class='ms-3 mt-2'>")
        for dongle_id, rlist in sorted(dongles.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            parts.append(f"<details class='mb-2'><summary><code>{esc(dongle_id)}</code> — routes: {len(rlist)}</summary>")
            parts.append("<ul class='mt-2'>")
            for rr in sorted(rlist, key=lambda x: (x.route_datetime_utc or "", x.route_id))[:200]:
                ts = rr.route_datetime_utc or "unknown_time"
                geo = "geo" if rr.geo_points_used > 0 else "no_geo"
                parts.append(f"<li><code>{esc(rr.route_id)}</code> <span class='text-muted'>({esc(ts)}; {geo})</span></li>")
            if len(rlist) > 200:
                parts.append(f"<li class='text-muted'>… {len(rlist)-200} more (see CSV export)</li>")
            parts.append("</ul></details>")
        parts.append("</div></details>")
    distribution_html = "\n".join(parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>TakeOver Dataset Overview</title>
  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css"/>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js"></script>
  <style>
    body {{ padding: 18px; }}
    .fixed-box {{
      position: fixed;
      background-color: rgba(255, 255, 255, 0.92);
      padding: 14px;
      border-radius: 8px;
      box-shadow: 0 0 15px rgba(0,0,0,0.18);
      font-family: Arial, sans-serif;
      z-index: 1000;
      max-width: 360px;
    }}
    .box-bottom-left {{ bottom: 20px; left: 20px; }}
    .box-top-right {{ top: 20px; right: 20px; }}
    summary {{ cursor: pointer; }}
    code {{ font-size: 0.92em; }}
    table {{ font-size: 0.95rem; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
  </style>
</head>
<body>

  <div class="fixed-box box-bottom-left">
    <h5 style="margin:0 0 8px 0; color:#333;">Dataset Summary</h5>
    <div class="mono">Models: {total_models}</div>
    <div class="mono">Dongles: {total_dongles}</div>
    <div class="mono">Routes: {total_routes}</div>
    <div class="mono">Time-known: {known_time}</div>
    <div class="mono">Time-unknown: {unknown_time}</div>
    <div class="mono">Geo-known: {geo_known}</div>
    <hr style="margin: 10px 0;">
    <div class="mono">Geo points: {geo_summary.get("points_total","")}</div>
    <div class="mono">Clusters: {geo_summary.get("clusters","")}</div>
    <hr style="margin: 10px 0;">
    <div style="font-size: 0.9rem;" class="text-muted">
      Map: <span class="mono">{esc(str(geo_map_html))}</span><br/>
      Exports:<br/>
      <span class="mono">{esc(str(out_html.parent / "dataset_routes.csv"))}</span><br/>
      <span class="mono">{esc(str(out_html.parent / "counts_by_car_model.csv"))}</span><br/>
      <span class="mono">{esc(str(out_html.parent / "counts_by_dongle.csv"))}</span>
    </div>
  </div>

  <div class="fixed-box box-top-right">
    <h5 style="margin:0 0 8px 0; color:#333;">Legend</h5>
    <div style="display:grid; gap:6px;">
      <div><b>Distribution</b>: nested <span class="mono">&lt;details&gt;</span> tree</div>
      <div><b>Time source</b>: where timestamp was inferred</div>
      <div><b>Geo source</b>: where GPS was inferred</div>
      <div class="text-muted" style="font-size:0.9rem;">(Scroll main page; boxes stay fixed)</div>
    </div>
  </div>

  <div class="container-fluid" style="padding-right: 420px; padding-bottom: 260px;">
    <h2 class="mb-2">TakeOver Dataset Overview</h2>
    <div class="text-muted mb-3">
      Generated at: {dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()}
    </div>

    <div class="mb-4">
      <a class="btn btn-primary" href="{esc(geo_map_html.name)}" target="_blank">Open Geo Distribution Map</a>
      <span class="text-muted ms-2">({esc(str(geo_map_html))})</span>
    </div>

    <h4>1) Dongle/Route distribution</h4>
    <div class="mb-4">{distribution_html}</div>

    <h4>2) Counts by car model</h4>
    <div class="table-responsive mb-4">
      <table class="table table-sm table-striped">
        <thead><tr><th>car_model</th><th>dongles</th><th>routes</th></tr></thead>
        <tbody>{car_rows_html}</tbody>
      </table>
    </div>

    <h4>3) Counts by dongle (top 2000 shown)</h4>
    <div class="table-responsive mb-4">
      <table class="table table-sm table-striped">
        <thead><tr><th>car_model</th><th>dongle_id</th><th>routes</th></tr></thead>
        <tbody>{dongle_rows_html}</tbody>
      </table>
    </div>

    <h4>4) Time distribution (YYYY-MM)</h4>
    <div class="mb-2">{time_plot_html}{time_plot_note}</div>

  </div>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Traverse TakeOver dataset and generate distribution + time + geo map.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path.home() / "Zhouhaoseu Dropbox" / "Zhouhaoseu Team Folder" / "TakeOver"),
        help="Dataset root path (default matches your Dropbox TakeOver folder).",
    )
    parser.add_argument("--output", type=str, default="", help="Output directory (default: <root>/Code/output).")
    parser.add_argument("--deep-scan", action="store_true", help="Time: scan extra directory level for timestamps.")
    parser.add_argument("--geo-scan-depth", type=int, default=2, help="Geo: max directory depth under each route to search for coords-like files.")
    parser.add_argument("--max-points-per-route", type=int, default=1200, help="Geo: maximum points to take from one route.")
    parser.add_argument("--max-total-points", type=int, default=120000, help="Geo: cap total points used for heatmap/clustering (reservoir sampled).")
    parser.add_argument("--max-marker-points", type=int, default=4000, help="Geo: cap visible orange markers in map (sampled).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--k-max", type=int, default=30, help="Geo: max clusters to display.")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 2

    out_dir = Path(args.output).expanduser().resolve() if args.output else (root / "Code" / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] deep_scan_time: {args.deep_scan}")
    print(f"[INFO] geo_scan_depth: {args.geo_scan_depth}")
    print(f"[INFO] max_points_per_route: {args.max_points_per_route}")
    print(f"[INFO] max_total_points: {args.max_total_points}")

    rows, geo_points = build_rows_and_geo(
        root=root,
        deep_scan_time=args.deep_scan,
        geo_scan_depth=args.geo_scan_depth,
        max_points_per_route=args.max_points_per_route,
        max_total_points=args.max_total_points,
        seed=args.seed,
    )
    print(f"[INFO] Discovered routes: {len(rows)}")
    print(f"[INFO] Geo points retained (sampled cap): {len(geo_points)}")

    # CSVs
    routes_csv = out_dir / "dataset_routes.csv"
    write_routes_csv(rows, routes_csv)
    print(f"[OK] wrote {routes_csv}")

    car_model_counts, dongle_counts = counts_tables(rows)
    write_csv(out_dir / "counts_by_car_model.csv", ["car_model", "dongles", "routes"], car_model_counts)
    write_csv(out_dir / "counts_by_dongle.csv", ["car_model", "dongle_id", "routes"], dongle_counts)
    print("[OK] wrote counts CSVs")

    # Geo exports
    write_csv(out_dir / "geo_points_sample.csv", ["lat", "lng"], [(f"{lat:.8f}", f"{lon:.8f}") for lat, lon in geo_points])

    clusters = cluster_points(geo_points, k_max=args.k_max) if geo_points else []
    write_csv(
        out_dir / "geo_clusters.csv",
        ["cluster_id", "center_lat", "center_lng", "points", "radius_m"],
        [(c["cluster_id"], f"{c['center_lat']:.8f}", f"{c['center_lng']:.8f}", c["points"], f"{c['radius_m']:.2f}") for c in clusters],
    )

    reg = region_stats(geo_points)

    # Map HTML (folium)
    geo_map_html = out_dir / "geo_distribution_map.html"
    if geo_points and folium is not None:
        build_geo_map(
            points=geo_points,
            clusters=clusters,
            reg_stats=reg,
            out_html=geo_map_html,
            max_marker_points=args.max_marker_points,
            seed=args.seed,
        )
        print(f"[OK] wrote {geo_map_html}")
    else:
        print("[WARN] geo map not generated (no geo points found, or folium missing).")

    # Time plot
    plot_png = out_dir / "time_distribution_year_month.png"
    time_plot_b64 = plot_time_distribution(rows, plot_png)
    if time_plot_b64:
        print(f"[OK] wrote {plot_png}")
    else:
        print("[WARN] time plot not generated (no timestamps found or matplotlib missing).")

    # Overview HTML
    overview_html = out_dir / "dataset_overview.html"
    geo_summary = dict(points_total=str(len(geo_points)), clusters=str(len(clusters)))
    generate_overview_html(
        rows=rows,
        car_model_counts=car_model_counts,
        dongle_counts=dongle_counts,
        time_plot_b64=time_plot_b64,
        geo_summary=geo_summary,
        out_html=overview_html,
        out_png=plot_png,
        geo_map_html=geo_map_html,
    )
    print(f"[OK] wrote {overview_html}")

    # Coverage
    known_time = sum(1 for r in rows if r.year_month)
    known_geo = sum(1 for r in rows if r.geo_points_used > 0)
    if rows:
        print(f"[INFO] time-known routes: {known_time}/{len(rows)} ({known_time/len(rows):.1%})")
        print(f"[INFO] geo-known routes: {known_geo}/{len(rows)} ({known_geo/len(rows):.1%})")

    # Region breakdown quick print
    for s in reg:
        print(f"[INFO] region {s['region']}: points={s['points']} density={s['density']:.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
