#!/usr/bin/env python3
"""
make_takeover_map.py
=====================
Generate an interactive Leaflet HTML map for the TakeOver dataset,
styled like the OpenLKA enhanced_global_map.html reference.

Features:
  - Heatmap layer (leaflet-heat)
  - Cluster circles with popups
  - Individual orange markers
  - Regional distribution panel
  - Legend panel
  - Layer control (Light / Dark tile layers)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN

ROOT = Path("/home/henry/Zhouhaoseu Dropbox/Zhouhaoseu Team Folder/TakeOver")
GPS_CSV = ROOT / "Code" / "stats_output" / "route_gps.csv"
OUT_HTML = ROOT / "Code" / "paper_figs" / "takeover_global_map.html"

# ── Load data ──
df = pd.read_csv(GPS_CSV)
print(f"Loaded {len(df)} GPS points from {df.dongle_id.nunique()} devices")

# ── Region classification ──
regions = {
    "North America": lambda d: (d.lat > 24) & (d.lat < 50) & (d.lng > -130) & (d.lng < -60),
    "Europe":        lambda d: (d.lat > 35) & (d.lat < 72) & (d.lng > -12) & (d.lng < 40),
    "Asia":          lambda d: (d.lat > 0)  & (d.lat < 55) & (d.lng > 60)  & (d.lng < 145),
    "South America": lambda d: (d.lat > -56) & (d.lat < 15) & (d.lng > -82) & (d.lng < -34),
    "Oceania":       lambda d: (d.lat > -50) & (d.lat < -10) & (d.lng > 110) & (d.lng < 180),
    "Africa":        lambda d: (d.lat > -35) & (d.lat < 37) & (d.lng > -18) & (d.lng < 52),
}

region_counts = {}
classified = pd.Series(False, index=df.index)
for name, mask_fn in regions.items():
    m = mask_fn(df) & ~classified
    region_counts[name] = int(m.sum())
    classified |= m
region_counts["Other"] = int((~classified).sum())

# Area in million km² for density
region_areas = {
    "North America": 9.54, "Europe": 10.18, "Asia": 44.58,
    "South America": 17.84, "Oceania": 8.53, "Africa": 30.37, "Other": 1.0
}

print("\nRegional distribution:")
for r, c in region_counts.items():
    area = region_areas.get(r, 1.0)
    dens = c / area if area else 0
    print(f"  {r}: {c} ({c/len(df)*100:.1f}%), density={dens:.2e}")

# ── Clustering with DBSCAN ──
coords_rad = np.deg2rad(df[["lat", "lng"]].values)
clustering = DBSCAN(eps=0.05, min_samples=5, metric="haversine").fit(coords_rad)
df["cluster"] = clustering.labels_

clusters = []
for cid in sorted(df["cluster"].unique()):
    if cid == -1:
        continue
    c = df[df["cluster"] == cid]
    center_lat = float(c.lat.mean())
    center_lng = float(c.lng.mean())
    n_points = len(c)
    # Radius: approximate from std of distances
    from math import radians, cos
    lat_std = float(c.lat.std())
    lng_std = float(c.lng.std())
    radius_km = max(lat_std, lng_std) * 111.0 * 1.5  # rough conversion
    radius_m = max(radius_km * 1000, 50000)  # minimum 50km
    clusters.append({
        "lat": center_lat, "lng": center_lng,
        "n": n_points, "radius": radius_m
    })

print(f"\nFound {len(clusters)} clusters")

# ── Heatmap data: [lat, lng, intensity] ──
heat_data = [[round(r.lat, 4), round(r.lng, 4), 1] for _, r in df.iterrows()]

# ── Generate HTML ──
map_id = "map_takeover"

# Build cluster circles JS
cluster_js = ""
for i, cl in enumerate(clusters):
    circle_id = f"cluster_{i}"
    cluster_js += f"""
        var {circle_id} = L.circle(
            [{cl['lat']}, {cl['lng']}],
            {{bubblingMouseEvents: true, color: "#1f77b4", fill: true,
              fillColor: "#1f77b4", fillOpacity: 0.15, opacity: 0.4,
              radius: {cl['radius']:.0f}, weight: 2}}
        ).addTo({map_id});
        {circle_id}.bindPopup("<b>Cluster {i}</b><br>Routes: {cl['n']}");

        L.circleMarker(
            [{cl['lat']}, {cl['lng']}],
            {{color: "#1f77b4", fillColor: "#1f77b4", fillOpacity: 0.3,
              opacity: 1.0, radius: 8, weight: 2}}
        ).addTo({map_id}).bindPopup("<b>Cluster Center {i}</b><br>Routes: {cl['n']}");
    """

# Build individual point markers
marker_js = ""
for _, row in df.iterrows():
    marker_js += f"""
        L.circleMarker([{row.lat:.4f}, {row.lng:.4f}],
            {{radius: 3, color: "#ff7f0e", fillColor: "#ff7f0e",
              fillOpacity: 0.5, opacity: 0.7, weight: 1}})
        .addTo({map_id});
    """

# Regional distribution panel HTML
region_panel = ""
for r, c in region_counts.items():
    area = region_areas.get(r, 1.0)
    dens = c / area if area else 0
    region_panel += f"""
            <div style='border-bottom: 1px solid #eee; padding-bottom: 5px;'>
                <div style='font-weight: bold; color: #666;'>{r}</div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px;'>
                    <div>Points: {c}</div>
                    <div>Density: {dens:.2e}</div>
                </div>
            </div>
    """

html = f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
    <script>
        L_NO_TOUCH = false;
        L_DISABLE_3D = false;
    </script>
    <style>html, body {{width: 100%;height: 100%;margin: 0;padding: 0;}}</style>
    <style>#map {{position:absolute;top:0;bottom:0;right:0;left:0;}}</style>
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css"/>
    <link rel="stylesheet" href="https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css"/>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css"/>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <style>
        #{map_id} {{
            position: relative;
            width: 100.0%;
            height: 100.0%;
            left: 0.0%;
            top: 0.0%;
        }}
        .leaflet-container {{ font-size: 1rem; }}
    </style>
    <script src="https://cdn.jsdelivr.net/gh/python-visualization/folium@main/folium/templates/leaflet_heat.min.js"></script>
</head>
<body>
    <!-- Regional Distribution Panel -->
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
            {region_panel}
        </div>
        <div style='margin-top: 10px; font-size: 0.85em; color: #888;'>
            Total: {len(df):,} routes / {df.dongle_id.nunique()} devices
        </div>
    </div>

    <!-- Legend Panel -->
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

    <div class="folium-map" id="{map_id}"></div>
</body>
<script>
    var {map_id} = L.map(
        "{map_id}",
        {{
            center: [39.8283, -98.5795],
            crs: L.CRS.EPSG3857,
            maxBounds: [[-90, -180], [90, 180]],
            zoom: 3,
            zoomControl: true,
            preferCanvas: false,
            worldCopyJump: true,
        }}
    );
    L.control.scale().addTo({map_id});

    // Light tile layer
    var tile_light = L.tileLayer(
        "https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png",
        {{"attribution": "&copy; OpenStreetMap &copy; CARTO",
          "maxZoom": 20, "subdomains": "abcd"}}
    );

    // Dark tile layer
    var tile_dark = L.tileLayer(
        "https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png",
        {{"attribution": "&copy; OpenStreetMap &copy; CARTO",
          "maxZoom": 20, "subdomains": "abcd"}}
    );

    tile_light.addTo({map_id});

    // Cluster circles
    {cluster_js}

    // Individual markers
    {marker_js}

    // Heatmap layer
    var heat_layer = L.heatLayer(
        {json.dumps(heat_data)},
        {{"blur": 15, "maxZoom": 10, "minOpacity": 0.3, "radius": 20}}
    ).addTo({map_id});

    // Layer control
    var baseMaps = {{
        "CartoDB Positron": tile_light,
        "CartoDB Dark Matter": tile_dark
    }};
    var overlayMaps = {{
        "Heat Map": heat_layer
    }};
    L.control.layers(baseMaps, overlayMaps).addTo({map_id});
</script>
</html>
"""

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_HTML, "w") as f:
    f.write(html)

print(f"\nSaved → {OUT_HTML}")
print(f"File size: {OUT_HTML.stat().st_size / 1024:.0f} KB")
