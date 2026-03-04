#!/usr/bin/env python3
"""
fetch_gps_coords.py
====================
Query the comma API to get GPS coordinates for all TakeOver routes.
Uses GET /v1/route/:route_name/ endpoint.
Saves results to stats_output/route_gps.csv
"""
import csv
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
ROUTES_CSV = ROOT / "Code" / "stats_output" / "unique_routes.csv"
OUT_CSV = ROOT / "Code" / "stats_output" / "route_gps.csv"

JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3Nzk4MDk2NDcsIm5iZiI6MTc3MjAzMzY0NywiaWF0IjoxNzcyMDMzNjQ3LCJpZGVudGl0eSI6IjYyYzNlMjY1NGM3NWIyOTkifQ.sE4E6xzOF0mLdF63B2VjngdtqrfvU-sI8ad4ITgWvyk"
HEADERS = {"Authorization": f"JWT {JWT}"}
BASE_URL = "https://api.commadotai.com"


def fetch_route(dongle_id: str, route_id: str) -> dict | None:
    """Fetch route info from comma API."""
    route_name = f"{dongle_id}|{route_id}"
    url = f"{BASE_URL}/v1/route/{route_name}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            data = r.json()
            lat = data.get("start_lat") or data.get("startlat")
            lng = data.get("start_lng") or data.get("startlng")
            if lat and lng and abs(lat) > 0.01 and abs(lng) > 0.01:
                return {"dongle_id": dongle_id, "route_id": route_id,
                        "lat": lat, "lng": lng}
        return None
    except Exception:
        return None


def fetch_segments_for_device(dongle_id: str, route_ids: list) -> list:
    """Try fetching segments for a device to get GPS for hex-format routes."""
    url = f"{BASE_URL}/v1/devices/{dongle_id}/segments"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return []
        segments = r.json()
        # Build a map from route_id -> GPS
        route_gps = {}
        for seg in segments:
            # segment has route field like "dongle_id|route_id"
            route_name = seg.get("route", "")
            if "|" in route_name:
                rid = route_name.split("|", 1)[1]
                if rid in route_ids and rid not in route_gps:
                    lat = seg.get("start_lat") or seg.get("startlat")
                    lng = seg.get("start_lng") or seg.get("startlng")
                    if lat and lng and abs(lat) > 0.01 and abs(lng) > 0.01:
                        route_gps[rid] = (lat, lng)
        results = []
        for rid, (lat, lng) in route_gps.items():
            results.append({"dongle_id": dongle_id, "route_id": rid,
                            "lat": lat, "lng": lng})
        return results
    except Exception:
        return []


def main():
    # Load routes
    routes = []
    with open(ROUTES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = row["dongle_id"]
            rid = row["route_id"]
            if did == "0000000000000000":
                continue
            routes.append((did, rid))

    print(f"Total routes to query: {len(routes)}")

    # Separate date-format vs hex-format
    date_routes = [(d, r) for d, r in routes if "--" in r and r[4] == "-"]
    hex_routes = [(d, r) for d, r in routes if not ("--" in r and r[4] == "-")]
    print(f"  Date-format: {len(date_routes)}")
    print(f"  Hex-format:  {len(hex_routes)}")

    results = []

    # Phase 1: Query date-format routes via /v1/route/ (these are most reliable)
    print("\n[Phase 1] Querying date-format routes via /v1/route/...")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_route, d, r): (d, r)
                   for d, r in date_routes}
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res:
                results.append(res)
            if done % 50 == 0:
                print(f"  {done}/{len(date_routes)} done, {len(results)} with GPS")

    print(f"  Phase 1 result: {len(results)} routes with GPS")

    # Phase 2: Query hex-format routes via /v1/route/ (may work for some)
    print("\n[Phase 2] Querying hex-format routes via /v1/route/...")
    existing_count = len(results)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_route, d, r): (d, r)
                   for d, r in hex_routes}
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res:
                results.append(res)
            if done % 200 == 0:
                print(f"  {done}/{len(hex_routes)} done, "
                      f"{len(results) - existing_count} new with GPS")

    print(f"  Phase 2 result: {len(results) - existing_count} additional routes")

    # Phase 3: For remaining hex routes without GPS, try device segments endpoint
    found_rids = {(r["dongle_id"], r["route_id"]) for r in results}
    missing_hex = [(d, r) for d, r in hex_routes if (d, r) not in found_rids]

    if missing_hex:
        print(f"\n[Phase 3] Trying segments endpoint for {len(missing_hex)} "
              f"remaining hex routes...")
        # Group by dongle_id
        by_device = {}
        for d, r in missing_hex:
            by_device.setdefault(d, []).append(r)

        print(f"  {len(by_device)} unique devices to query")
        phase3_count = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(fetch_segments_for_device, did, rids): did
                       for did, rids in by_device.items()}
            done = 0
            for fut in as_completed(futures):
                done += 1
                res_list = fut.result()
                for r in res_list:
                    if (r["dongle_id"], r["route_id"]) not in found_rids:
                        results.append(r)
                        found_rids.add((r["dongle_id"], r["route_id"]))
                        phase3_count += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(by_device)} devices, "
                          f"{phase3_count} new GPS found")

        print(f"  Phase 3 result: {phase3_count} additional routes")

    # Save
    print(f"\nTotal routes with GPS: {len(results)} / {len(routes)}")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dongle_id", "route_id", "lat", "lng"])
        w.writeheader()
        w.writerows(results)
    print(f"Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
