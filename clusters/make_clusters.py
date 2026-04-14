import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_M = 6_371_000.0


def detect_project_root(script_path: Path) -> Path:
    """
    Heuristic: this file lives in `<project_root>/flico/clusters/make_clusters.py`.
    So `project_root` is three parents up from this file.
    """
    try:
        return script_path.resolve().parents[2]
    except Exception:
        # Fallback: current working directory
        return Path.cwd()


def load_all_metadata(
    metadata_dir: Path,
    useful_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read all CSVs in `metadata_dir` and return a single concatenated DataFrame.

    - Adds a `source_dataset` column with the CSV stem.
    - Keeps only `useful_columns` if provided and present in each CSV.
    """
    if useful_columns is None:
        useful_columns = [
            "id",
            "title",
            "description",
            "date_taken",
            "date_uploaded",
            "latitude",
            "longitude",
            "context",
            "image_url",
            "tags",
        ]

    csv_paths = sorted(metadata_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in metadata directory: {metadata_dir}")

    frames: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed to read {csv_path.name}: {exc}")
            continue

        if df.empty:
            continue

        df["source_dataset"] = csv_path.stem

        # Select only useful columns that exist in this CSV
        cols_to_keep = [c for c in useful_columns if c in df.columns]
        cols_to_keep.append("source_dataset")
        df = df[cols_to_keep]

        frames.append(df)

    if not frames:
        raise RuntimeError("All metadata CSVs were empty or unreadable.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded {len(combined):,} rows from {len(frames)} CSV files.")
    return combined


def filter_valid_geopoints(
    df: pd.DataFrame, allowed_contexts: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Keep only rows with valid latitude / longitude values.
    - Coerces latitude / longitude to numeric.
    - Filters out NaNs, zeros, and values out of [-90, 90] / [-180, 180].
    """
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            raise KeyError(f"Required column `{col}` not found in metadata.")

    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    mask_valid = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & (df["latitude"] != 0)
        & (df["longitude"] != 0)
        & (df["latitude"].between(-90, 90))
        & (df["longitude"].between(-180, 180))
    )

    # Optional Flickr geo context filter (e.g. {"2"} or {"1", "2"}).
    if "context" in df.columns and allowed_contexts is not None:
        context_values = df["context"].astype(str).str.strip()
        mask_valid = mask_valid & context_values.isin(allowed_contexts)
    elif "context" not in df.columns and allowed_contexts is not None:
        print("[WARN] Context filter requested, but no `context` column found. Ignoring it.")

    filtered = df.loc[mask_valid].reset_index(drop=True)
    print(f"[INFO] Kept {len(filtered):,} rows with valid geo points (from {len(df):,}).")
    return filtered


def parse_contexts_arg(raw_value: str) -> Optional[Set[str]]:
    """
    Parse context filter string.
    - "all", "*" or empty -> None (no context filter)
    - "2" or "1,2" etc -> set of context strings
    """
    raw = (raw_value or "").strip().lower()
    if raw in {"", "all", "*"}:
        return None

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    allowed = {"0", "1", "2"}
    invalid = [p for p in parts if p not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid context values: {invalid}. Allowed values are 0,1,2, or 'all'."
        )
    return set(parts)


def resolve_context_filter(
    args_contexts: Optional[str], has_context_column: bool
) -> Optional[Set[str]]:
    """
    Resolve context filter from CLI or interactive prompt.
    Returns None for no filter, or a set like {"2"}.
    """
    if not has_context_column:
        print("[INFO] No `context` column found in metadata. Skipping context filter.")
        return None

    if args_contexts is not None:
        selected = parse_contexts_arg(args_contexts)
        if selected is None:
            print("[INFO] Context filter: all contexts")
        else:
            print(f"[INFO] Context filter: {sorted(selected)}")
        return selected

    # Interactive mode: ask user when no CLI filter was provided.
    if sys.stdin.isatty():
        user_value = input(
            "Context filter (comma-separated 0/1/2, or 'all') [default: 2]: "
        ).strip()
        if user_value == "":
            user_value = "2"
        selected = parse_contexts_arg(user_value)
        if selected is None:
            print("[INFO] Context filter: all contexts")
        else:
            print(f"[INFO] Context filter: {sorted(selected)}")
        return selected

    # Non-interactive fallback keeps previous behavior.
    print("[INFO] Non-interactive run with no --contexts provided. Using default context=2.")
    return {"2"}


def run_dbscan_meters(
    df_geo: pd.DataFrame,
    eps_meters: float,
    min_samples: int,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    Run DBSCAN on latitude / longitude using haversine distance with `eps_meters`.

    Returns:
      - DataFrame with an extra `cluster_id` column (-1 = noise)
      - A cluster summary dict keyed by cluster_id
    """
    if df_geo.empty:
        raise ValueError("No geo points to cluster.")

    coords_rad = np.radians(df_geo[["latitude", "longitude"]].to_numpy())
    eps_radians = eps_meters / EARTH_RADIUS_M

    print(
        f"[INFO] Running DBSCAN with eps={eps_meters} m "
        f"({eps_radians:.6f} rad), min_samples={min_samples}..."
    )

    db = DBSCAN(
        eps=eps_radians,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    )
    labels = db.fit_predict(coords_rad)

    df_geo = df_geo.copy()
    df_geo["cluster_id"] = labels.astype(int)

    # Build a basic cluster summary (excluding noise)
    cluster_summary: Dict[int, Dict[str, float]] = {}
    mask_cluster = df_geo["cluster_id"] >= 0
    grouped = df_geo.loc[mask_cluster].groupby("cluster_id")

    for cid, group in grouped:
        lat_mean = group["latitude"].mean()
        lon_mean = group["longitude"].mean()
        cluster_summary[int(cid)] = {
            "cluster_id": int(cid),
            "count": int(len(group)),
            "latitude_mean": float(lat_mean),
            "longitude_mean": float(lon_mean),
            "latitude_min": float(group["latitude"].min()),
            "latitude_max": float(group["latitude"].max()),
            "longitude_min": float(group["longitude"].min()),
            "longitude_max": float(group["longitude"].max()),
        }

    n_clusters = len(cluster_summary)
    n_noise = int((labels == -1).sum())
    print(f"[INFO] Found {n_clusters} clusters and {n_noise} noise points.")

    return df_geo, cluster_summary


def export_cluster_csvs(
    df_with_clusters: pd.DataFrame,
    cluster_summary: Dict[int, Dict[str, float]],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """
    Write:
      - `clusters.csv`: one row per image with `cluster_id`
      - `cluster_summary.csv`: one row per cluster with stats
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters_csv = out_dir / "clusters.csv"
    summary_csv = out_dir / "cluster_summary.csv"

    df_with_clusters.to_csv(clusters_csv, index=False)
    pd.DataFrame.from_dict(cluster_summary, orient="index").sort_values(
        "count", ascending=False
    ).to_csv(summary_csv, index=False)

    print(f"[INFO] Wrote full clusters to {clusters_csv}")
    print(f"[INFO] Wrote cluster summary to {summary_csv}")

    return clusters_csv, summary_csv


def build_html_report(
    df_with_clusters: pd.DataFrame,
    cluster_summary: Dict[int, Dict[str, float]],
    out_dir: Path,
    max_clusters_in_report: int = 20,
    max_points_per_cluster: int = 0,
) -> Path:
    """
    Generate a small interactive HTML report using Leaflet to visually inspect
    the top clusters by size.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "cluster_report.html"

    if not cluster_summary:
        html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Image Clusters Report</title>
</head>
<body>
  <h1>Image Clusters Report</h1>
  <p>No clusters found (all points were noise or no valid geo data).</p>
</body>
</html>
"""
        report_path.write_text(html, encoding="utf-8")
        print(f"[INFO] Wrote HTML report (no clusters) to {report_path}")
        return report_path

    # Sort clusters by descending size and take top N clusters into the report
    summary_sorted = sorted(
        cluster_summary.values(), key=lambda d: d["count"], reverse=True
    )
    top_clusters = summary_sorted[:max_clusters_in_report]
    top_ids = {c["cluster_id"] for c in top_clusters}

    # Prepare data structure for the front-end
    # These fields are exposed in the marker popups if available.
    fields_for_popup = ["id", "title", "description", "image_url", "source_dataset"]
    cluster_data: List[Dict] = []

    for cluster_info in top_clusters:
        cid = cluster_info["cluster_id"]
        subset = df_with_clusters[df_with_clusters["cluster_id"] == cid].copy()

        # Limit number of points per cluster in the HTML only if
        # `max_points_per_cluster` is a positive value. A value of 0 or less
        # means "no limit".
        if max_points_per_cluster and len(subset) > max_points_per_cluster:
            subset = subset.iloc[:max_points_per_cluster]

        points: List[Dict[str, object]] = []
        for _, row in subset.iterrows():
            point = {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
            for field in fields_for_popup:
                if field in row and not (isinstance(row[field], float) and math.isnan(row[field])):
                    point[field] = str(row[field])
            points.append(point)

        cluster_entry: Dict[str, object] = {
            "cluster_id": cid,
            "count": int(cluster_info["count"]),
            "latitude_mean": cluster_info["latitude_mean"],
            "longitude_mean": cluster_info["longitude_mean"],
            "points": points,
        }
        cluster_data.append(cluster_entry)

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Embed data directly into HTML
    cluster_json = json.dumps(cluster_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Image Clusters Report</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #layout {{
      display: grid;
      grid-template-columns: 280px 1fr;
      grid-template-rows: auto 1fr;
      grid-template-areas:
        "header header"
        "sidebar map";
      height: 100vh;
    }}
    header {{
      grid-area: header;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid #ddd;
      background: #fafafa;
    }}
    #sidebar {{
      grid-area: sidebar;
      border-right: 1px solid #ddd;
      padding: 0.75rem;
      overflow-y: auto;
      font-size: 0.9rem;
    }}
    #map {{
      grid-area: map;
      height: 100%;
      width: 100%;
    }}
    .cluster-item {{
      padding: 0.25rem 0.4rem;
      cursor: pointer;
      border-radius: 4px;
    }}
    .cluster-item:hover {{
      background: #f0f0f0;
    }}
    .cluster-item.active {{
      background: #e0f2ff;
    }}
    .cluster-meta {{
      color: #666;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>
  <div id="layout">
    <header>
      <strong>Image Clusters Report</strong>
      <span style="color:#666; font-size:0.85rem;">&nbsp;&mdash; generated {generated_at}</span>
      <button id="export-errors-btn" style="margin-left:1rem;padding:0.2rem 0.6rem;font-size:0.85rem;cursor:pointer;">
        Export erroneous as CSV
      </button>
    </header>
    <aside id="sidebar">
      <p><strong>Top clusters</strong> (sorted by size). Click one to zoom.</p>
      <div id="cluster-list"></div>
    </aside>
    <div id="map"></div>
  </div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin="">
  </script>

  <script>
    const clusterData = {cluster_json};

    const map = L.map('map', {{
      maxZoom: 22,
    }});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 22,
      maxNativeZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}).addTo(map);

    // clusterId -> {{
    //   mainGroup, errGroup, pointsById, mainItem, errItem
    // }}
    const markersByCluster = {{}};
    let activeClusterId = null;
    let activeSubset = 'main'; // 'main' or 'err'
    let currentPopupContext = null; // {{ clusterId, imageId }}

    function initClusters() {{
      const listEl = document.getElementById('cluster-list');

      if (!clusterData.length) {{
        listEl.innerHTML = '<p>No clusters to display.</p>';
        map.setView([0, 0], 2);
        return;
      }}

      clusterData.forEach((cluster) => {{
        // Sidebar items: main cluster and erroneous subcluster
        const mainItem = document.createElement('div');
        mainItem.className = 'cluster-item';
        mainItem.dataset.clusterId = cluster.cluster_id;
        mainItem.dataset.subset = 'main';
        mainItem.innerHTML = `
          <div><strong>Cluster #${{cluster.cluster_id}}</strong></div>
          <div class="cluster-meta">
            ${{
              cluster.count === 1
                ? '1 image'
                : cluster.count + ' images'
            }}
          </div>
        `;
        mainItem.addEventListener('click', () => {{
          setActiveCluster(cluster.cluster_id, 'main');
        }});

        const errItem = document.createElement('div');
        errItem.className = 'cluster-item';
        errItem.dataset.clusterId = cluster.cluster_id;
        errItem.dataset.subset = 'err';
        errItem.innerHTML = `
          <div><strong>Erroneous in #${{cluster.cluster_id}}</strong></div>
          <div class="cluster-meta">0 images</div>
        `;
        errItem.addEventListener('click', () => {{
          setActiveCluster(cluster.cluster_id, 'err');
        }});

        listEl.appendChild(mainItem);
        listEl.appendChild(errItem);

        const mainGroup = L.featureGroup();
        const errGroup = L.featureGroup();

        // Group points with same coordinates and spiderfy them around the
        // true location with short stems (flower pattern).
        const buckets = {{}};
        const keyPrecision = 6; // ~0.1 m precision

        cluster.points.forEach((p) => {{
          const key =
            p.latitude.toFixed(keyPrecision) +
            ',' +
            p.longitude.toFixed(keyPrecision);
          if (!buckets[key]) {{
            buckets[key] = {{
              centerLat: p.latitude,
              centerLon: p.longitude,
              items: [],
            }};
          }}
          buckets[key].items.push(p);
        }});

        const pointsById = {{}};
        let autoIdCounter = 0;

        Object.values(buckets).forEach((bucket) => {{
          const centerLat = bucket.centerLat;
          const centerLon = bucket.centerLon;
          const items = bucket.items;

          if (items.length === 1) {{
            const p = items[0];
            const imageId = String(p.id || `auto_${{autoIdCounter++}}`);
            const feature = createImageMarker(
              p,
              centerLat,
              centerLon,
              centerLat,
              centerLon,
              cluster.cluster_id,
              imageId
            );
            feature.marker.addTo(mainGroup);
            if (feature.stem) feature.stem.addTo(mainGroup);
            pointsById[imageId] = feature;
            return;
          }}

          const count = items.length;
          const angleStep = (2 * Math.PI) / count;
          const baseLatRad = (centerLat * Math.PI) / 180.0;
          const radiusDegLat = 0.00015; // ~16m
          const radiusDegLon = radiusDegLat / Math.max(Math.cos(baseLatRad), 0.1);

          items.forEach((p, index) => {{
            const angle = index * angleStep;
            const markerLat = centerLat + radiusDegLat * Math.sin(angle);
            const markerLon = centerLon + radiusDegLon * Math.cos(angle);
            const imageId = String(p.id || `auto_${{autoIdCounter++}}`);
            const feature = createImageMarker(
              p,
              centerLat,
              centerLon,
              markerLat,
              markerLon,
              cluster.cluster_id,
              imageId
            );
            feature.marker.addTo(mainGroup);
            if (feature.stem) feature.stem.addTo(mainGroup);
            pointsById[imageId] = feature;
          }});
        }});

        markersByCluster[cluster.cluster_id] = {{
          mainGroup,
          errGroup,
          pointsById,
          mainItem,
          errItem,
        }};
      }});

      // Activate the first cluster (main subset)
      setActiveCluster(clusterData[0].cluster_id, 'main', true);
    }}

    function setActiveCluster(clusterId, subset = 'main', initial = false) {{
      if (activeClusterId === clusterId && activeSubset === subset && !initial) {{
        return;
      }}

      activeClusterId = clusterId;
      activeSubset = subset;

      // Toggle sidebar active state
      document
        .querySelectorAll('.cluster-item')
        .forEach((el) => {{
          if (
            String(el.dataset.clusterId) === String(clusterId) &&
            el.dataset.subset === subset
          ) {{
            el.classList.add('active');
          }} else {{
            el.classList.remove('active');
          }}
        }});

      // Clear previous markers and add the selected cluster subset
      Object.values(markersByCluster).forEach((entry) => {{
        map.removeLayer(entry.mainGroup);
        map.removeLayer(entry.errGroup);
      }});

      const entry = markersByCluster[clusterId];
      if (!entry) {{
        map.setView([0, 0], 2);
        return;
      }}

      const layer = subset === 'err' ? entry.errGroup : entry.mainGroup;
      if (layer) {{
        layer.addTo(map);
        const bounds = layer.getBounds();
        if (bounds && bounds.isValid && bounds.isValid()) {{
          map.fitBounds(bounds.pad(0.2));
        }} else {{
          map.setView([0, 0], 2);
        }}
      }}
    }}

    function createImageMarker(
      p,
      centerLat,
      centerLon,
      markerLat,
      markerLon,
      clusterId,
      imageId
    ) {{
      const title = p.title || '';
      const id = p.id || '';
      const src = p.source_dataset || '';
      const desc = p.description || '';
      const imageUrl = p.image_url || '';

      let html = '';
      if (title) html += `<strong>${{title}}</strong><br/>`;
      if (id) html += `ID: ${{id}}<br/>`;
      if (src) html += `<span style="color:#666;">Source: ${{src}}</span><br/>`;
      if (desc)
        html += `<div style="margin-top:4px;font-size:0.85rem;color:#444;max-width:260px;white-space:normal;word-wrap:break-word;">${{desc}}</div>`;
      if (imageUrl) {{
        const safeUrl = imageUrl.replace(/"/g, '&quot;');
        html += `<div style="margin-top:4px;"><img src="${{safeUrl}}" alt="" style="max-width:260px;max-height:180px;object-fit:contain;"/></div>`;
      }}

      html += `
        <div style="margin-top:6px;display:flex;gap:6px;">
          <button
            type="button"
            class="nav-btn"
            data-nav="prev"
            data-cluster-id="${{clusterId}}"
            data-image-id="${{imageId}}"
          >
            Prev
          </button>
          <button
            type="button"
            class="nav-btn"
            data-nav="next"
            data-cluster-id="${{clusterId}}"
            data-image-id="${{imageId}}"
          >
            Next
          </button>
        </div>
        <div style="margin-top:6px;">
          <button
            type="button"
            class="err-btn"
            data-cluster-id="${{clusterId}}"
            data-image-id="${{imageId}}"
            data-mode="error"
          >
            Mark as erroneous
          </button>
        </div>
      `;

      // Small blue pin style for dense spiderfy views.
      const pinIcon = L.divIcon({{
        className: 'pin-marker',
        html: '<div style="background: #1f77ff; width: 10px; height: 10px; border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%; border: 1.5px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.25);"></div>',
        iconSize: [14, 14],
        iconAnchor: [7, 14],
      }});

      const marker = L.marker([markerLat, markerLon], {{ icon: pinIcon }});
      marker.bindPopup(html);

      let stem = null;
      if (centerLat !== markerLat || centerLon !== markerLon) {{
        stem = L.polyline(
          [
            [centerLat, centerLon],
            [markerLat, markerLon],
          ],
          {{
            color: '#555555',
            weight: 1,
            opacity: 0.7,
          }}
        );
      }}

      return {{ point: p, marker, stem }};
    }}

    function updateErroneousCount(entry) {{
      const errCount = Object.values(entry.pointsById).filter((r) =>
        entry.errGroup.hasLayer(r.marker)
      ).length;
      entry.errItem.querySelector('.cluster-meta').textContent =
        errCount === 1 ? '1 image' : `${{errCount}} images`;
    }}

    function moveToErroneous(clusterId, imageId) {{
      const entry = markersByCluster[clusterId];
      if (!entry) return;
      const rec = entry.pointsById[imageId];
      if (!rec || !rec.marker) return;
      if (entry.errGroup.hasLayer(rec.marker)) return;

      entry.mainGroup.removeLayer(rec.marker);
      if (rec.stem) entry.mainGroup.removeLayer(rec.stem);
      entry.errGroup.addLayer(rec.marker);
      if (rec.stem) entry.errGroup.addLayer(rec.stem);

      updateErroneousCount(entry);

      if (activeClusterId === clusterId) {{
        setActiveCluster(clusterId, activeSubset, true);
      }}
    }}

    function moveToMain(clusterId, imageId) {{
      const entry = markersByCluster[clusterId];
      if (!entry) return;
      const rec = entry.pointsById[imageId];
      if (!rec || !rec.marker) return;
      if (!entry.errGroup.hasLayer(rec.marker)) return;

      entry.errGroup.removeLayer(rec.marker);
      if (rec.stem) entry.errGroup.removeLayer(rec.stem);
      entry.mainGroup.addLayer(rec.marker);
      if (rec.stem) entry.mainGroup.addLayer(rec.stem);

      updateErroneousCount(entry);

      if (activeClusterId === clusterId) {{
        setActiveCluster(clusterId, activeSubset, true);
      }}
    }}

    function getVisibleImageIds(clusterId) {{
      const entry = markersByCluster[clusterId];
      if (!entry) return [];
      const layer = activeSubset === 'err' ? entry.errGroup : entry.mainGroup;
      return Object.keys(entry.pointsById).filter((id) => {{
        const rec = entry.pointsById[id];
        return rec && rec.marker && layer.hasLayer(rec.marker);
      }});
    }}

    function navigateImage(clusterId, imageId, direction) {{
      const entry = markersByCluster[clusterId];
      if (!entry) return;

      const ids = getVisibleImageIds(clusterId);
      if (!ids.length) return;

      const idx = ids.indexOf(String(imageId));
      if (idx === -1) return;

      const nextIdx = (idx + direction + ids.length) % ids.length;
      const targetId = ids[nextIdx];
      const targetRec = entry.pointsById[targetId];
      if (!targetRec || !targetRec.marker) return;

      targetRec.marker.openPopup();
      const latLng = targetRec.marker.getLatLng();
      if (latLng) {{
        map.panTo(latLng, {{ animate: true }});
      }}
    }}

    // Handle popup buttons (navigation + erroneous toggle).
    map.on('popupopen', (e) => {{
      const root = e.popup.getElement();
      if (!root) return;
      const btn = root.querySelector('.err-btn');
      if (!btn) return;

      const clusterId = Number(btn.dataset.clusterId);
      const imageId = btn.dataset.imageId;
      const entry = markersByCluster[clusterId];
      const rec = entry ? entry.pointsById[imageId] : null;
      const isErroneous = Boolean(entry && rec && entry.errGroup.hasLayer(rec.marker));
      currentPopupContext = {{ clusterId, imageId: String(imageId) }};

      root.querySelectorAll('.nav-btn').forEach((navBtn) => {{
        navBtn.addEventListener(
          'click',
          () => {{
            const navDir = navBtn.dataset.nav === 'prev' ? -1 : 1;
            navigateImage(clusterId, imageId, navDir);
          }},
          {{ once: true }}
        );
      }});

      if (isErroneous) {{
        btn.dataset.mode = 'restore';
        btn.textContent = 'Mark as non-erroneous';
      }} else {{
        btn.dataset.mode = 'error';
        btn.textContent = 'Mark as erroneous';
      }}

      btn.addEventListener(
        'click',
        () => {{
          const mode = btn.dataset.mode || 'error';
          if (mode === 'error') {{
            moveToErroneous(clusterId, imageId);
          }} else if (mode === 'restore') {{
            moveToMain(clusterId, imageId);
          }}
          map.closePopup(e.popup);
        }},
        {{ once: true }}
      );
    }});

    map.on('popupclose', () => {{
      currentPopupContext = null;
    }});

    document.addEventListener('keydown', (ev) => {{
      if (!currentPopupContext) return;
      if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;

      ev.preventDefault();
      const dir = ev.key === 'ArrowLeft' ? -1 : 1;
      navigateImage(currentPopupContext.clusterId, currentPopupContext.imageId, dir);
    }});

    // Export current erroneous selections as a CSV download.
    function exportErroneousCsv() {{
      const rows = [];
      rows.push([
        'cluster_id',
        'image_id',
        'source_dataset',
        'title',
        'description',
        'image_url',
      ]);

      Object.keys(markersByCluster).forEach((cid) => {{
        const entry = markersByCluster[cid];
        if (!entry) return;
        Object.entries(entry.pointsById).forEach(([imageId, rec]) => {{
          if (!entry.errGroup.hasLayer(rec.marker)) return;
          const p = rec.point || {{}};
          rows.push([
            cid,
            imageId,
            p.source_dataset || '',
            (p.title || '').split('\\n').join(' ').split('\\r').join(' '),
            (p.description || '').split('\\n').join(' ').split('\\r').join(' '),
            p.image_url || '',
          ]);
        }});
      }});

      if (rows.length === 1) {{
        alert('No erroneous images selected yet.');
        return;
      }}

      const csvText = rows
        .map((cols) =>
          cols
            .map((v) => {{
              const s = String(v ?? '');
              // Quote if the value contains a comma, quote, or newline.
              if (s.indexOf(',') !== -1 || s.indexOf('"') !== -1 || s.indexOf('\\n') !== -1) {{
                return '"' + s.replace(/"/g, '""') + '"';
              }}
              return s;
            }})
            .join(',')
        )
        .join('\\n');

      const blob = new Blob([csvText], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      a.href = url;
      a.download = `erroneous_images_${{timestamp}}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    document
      .getElementById('export-errors-btn')
      .addEventListener('click', exportErroneousCsv);

    initClusters();
  </script>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    print(f"[INFO] Wrote HTML report to {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__)
    project_root = detect_project_root(script_path)
    default_metadata_dir = project_root / "metadata"
    default_out_dir = script_path.resolve().parent / "outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Cluster Flickr metadata by geolocation using DBSCAN and "
            "generate CSV outputs + an HTML report."
        )
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=default_metadata_dir,
        help=f"Directory containing metadata CSV files (default: {default_metadata_dir})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out_dir,
        help=f"Directory to write cluster outputs (default: {default_out_dir})",
    )
    parser.add_argument(
        "--eps-m",
        type=float,
        default=50.0,
        help="DBSCAN epsilon radius in meters (default: 50 m).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter (default: 5).",
    )
    parser.add_argument(
        "--max-clusters-report",
        type=int,
        default=20,
        help="Maximum number of clusters to visualize in the HTML report (default: 20).",
    )
    parser.add_argument(
        "--max-points-per-cluster",
        type=int,
        default=0,
        help=(
            "Maximum number of points per cluster to embed in the HTML report. "
            "Use 0 to disable the limit (default: 0, no limit)."
        ),
    )
    parser.add_argument(
        "--contexts",
        type=str,
        default=None,
        help=(
            "Context values to include, e.g. '2' or '1,2'. "
            "Use 'all' for no context filter. "
            "If omitted, script prompts interactively."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Using metadata directory: {args.metadata_dir}")
    print(f"[INFO] Output will be written to: {args.output_dir}")

    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory does not exist: {args.metadata_dir}")

    df_all = load_all_metadata(args.metadata_dir)
    context_filter = resolve_context_filter(
        args_contexts=args.contexts,
        has_context_column=("context" in df_all.columns),
    )
    df_geo = filter_valid_geopoints(df_all, allowed_contexts=context_filter)

    if df_geo.empty:
        print("[WARN] No valid geo points after filtering; nothing to cluster.")
        return

    df_clustered, cluster_summary = run_dbscan_meters(
        df_geo, eps_meters=args.eps_m, min_samples=args.min_samples
    )

    export_cluster_csvs(df_clustered, cluster_summary, args.output_dir)
    build_html_report(
        df_clustered,
        cluster_summary,
        args.output_dir,
        max_clusters_in_report=args.max_clusters_report,
        max_points_per_cluster=args.max_points_per_cluster,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

