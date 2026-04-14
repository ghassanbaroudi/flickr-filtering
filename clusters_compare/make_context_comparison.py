import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_M = 6_371_000.0

# Default keyword list for “building / architecture” style images (substring match, case-insensitive).
# Override with --keywords or --keywords-file.
DEFAULT_BUILDING_KEYWORDS: List[str] = [
    "building",
    "architecture",
    "architectural",
    "facade",
    "cathedral",
    "church",
    "chapel",
    "tower",
    "castle",
    "palace",
    "museum",
    "monument",
    "dome",
    "city hall",
    "town hall",
    "theatre",
    "theater",
    "college",
    "landmark",
    "skyscraper",
    "structure",
]


def detect_project_root(script_path: Path) -> Path:
    # This file is expected at <project_root>/flico/clusters_compare/...
    try:
        return script_path.resolve().parents[2]
    except Exception:
        return Path.cwd()


def parse_contexts(raw: str) -> Set[str]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    allowed = {"0", "1", "2"}
    invalid = [p for p in parts if p not in allowed]
    if invalid:
        raise ValueError(f"Invalid contexts: {invalid}. Allowed: 0,1,2")
    if not parts:
        raise ValueError("At least one context value is required.")
    return set(parts)


def parse_cluster_contexts(raw: str) -> Optional[Set[str]]:
    """None = do not filter by Flickr context (all contexts)."""
    s = (raw or "").strip().lower()
    if s in ("", "all", "*"):
        return None
    return parse_contexts(raw)


def parse_overlay_contexts(raw: str) -> Optional[Set[str]]:
    """None = no overlay layer."""
    s = (raw or "").strip().lower()
    if s in ("", "none", "off"):
        return None
    return parse_contexts(s)


def load_keywords_list(keywords_arg: Optional[str], keywords_file: Optional[Path]) -> List[str]:
    """Load keywords from --keywords-file (wins) or --keywords or defaults."""
    if keywords_file is not None:
        if not keywords_file.exists():
            raise FileNotFoundError(f"Keywords file not found: {keywords_file}")
        out: List[str] = []
        for line in keywords_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
        if not out:
            raise ValueError(f"No keywords found in {keywords_file}")
        return out

    if keywords_arg is not None and str(keywords_arg).strip():
        parts = [p.strip() for p in str(keywords_arg).split(",") if p.strip()]
        if not parts:
            raise ValueError("--keywords is empty")
        return parts

    return list(DEFAULT_BUILDING_KEYWORDS)


def row_text_for_keywords(row: pd.Series) -> str:
    parts = []
    for col in ("title", "description", "tags"):
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    return " ".join(parts).lower()


def filter_by_keywords(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """
    Keep rows where at least one keyword appears as a substring in the
    combined title + description + tags (case-insensitive).
    """
    if not keywords:
        raise ValueError("Keyword list is empty.")
    kws = [k.lower().strip() for k in keywords if k.strip()]
    if not kws:
        raise ValueError("No non-empty keywords after trimming.")

    mask = []
    for _, row in df.iterrows():
        text = row_text_for_keywords(row)
        mask.append(any(kw in text for kw in kws))
    out = df.loc[mask].reset_index(drop=True)
    print(f"[INFO] Keyword filter ({len(kws)} terms, match ANY) -> {len(out):,} / {len(df):,} rows.")
    return out


def load_all_metadata(metadata_dir: Path) -> pd.DataFrame:
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
        raise FileNotFoundError(f"No CSV files found in: {metadata_dir}")

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
        cols = [c for c in useful_columns if c in df.columns] + ["source_dataset"]
        frames.append(df[cols])

    if not frames:
        raise RuntimeError("All CSV files are empty or unreadable.")

    out = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Loaded {len(out):,} rows from {len(frames)} CSV files.")
    return out


def keep_valid_geo(df: pd.DataFrame) -> pd.DataFrame:
    required = {"latitude", "longitude"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["context"] = out.get("context", "").astype(str).str.strip()

    mask = (
        out["latitude"].notna()
        & out["longitude"].notna()
        & (out["latitude"] != 0)
        & (out["longitude"] != 0)
        & out["latitude"].between(-90, 90)
        & out["longitude"].between(-180, 180)
    )
    out = out.loc[mask].reset_index(drop=True)
    print(f"[INFO] Kept {len(out):,} rows with valid non-zero geo.")
    return out


def filter_contexts(df: pd.DataFrame, contexts: Set[str]) -> pd.DataFrame:
    if "context" not in df.columns:
        raise KeyError("`context` column is required for context comparison.")
    out = df[df["context"].astype(str).str.strip().isin(contexts)].reset_index(drop=True)
    print(f"[INFO] Context filter {sorted(contexts)} -> {len(out):,} rows.")
    return out


def run_dbscan(df: pd.DataFrame, eps_meters: float, min_samples: int) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    if df.empty:
        raise ValueError("No rows to cluster.")

    coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_meters / EARTH_RADIUS_M

    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    )
    labels = model.fit_predict(coords_rad).astype(int)
    out = df.copy()
    out["cluster_id"] = labels

    summary: Dict[int, Dict[str, float]] = {}
    grouped = out[out["cluster_id"] >= 0].groupby("cluster_id")
    for cid, g in grouped:
        summary[int(cid)] = {
            "cluster_id": int(cid),
            "count": int(len(g)),
            "latitude_mean": float(g["latitude"].mean()),
            "longitude_mean": float(g["longitude"].mean()),
            "latitude_min": float(g["latitude"].min()),
            "latitude_max": float(g["latitude"].max()),
            "longitude_min": float(g["longitude"].min()),
            "longitude_max": float(g["longitude"].max()),
        }

    print(
        f"[INFO] DBSCAN found {len(summary)} clusters and {(labels == -1).sum()} noise points."
    )
    return out, summary


def export_csvs(clustered_df: pd.DataFrame, summary: Dict[int, Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(out_dir / "clusters.csv", index=False)
    # Empty summary when DBSCAN labels every point as noise (-1): no "count" column → sort_values fails.
    if summary:
        summary_df = pd.DataFrame.from_dict(summary, orient="index").sort_values("count", ascending=False)
    else:
        summary_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "count",
                "latitude_mean",
                "longitude_mean",
                "latitude_min",
                "latitude_max",
                "longitude_min",
                "longitude_max",
            ]
        )
    summary_df.to_csv(out_dir / "cluster_summary.csv", index=False)
    print(f"[INFO] Wrote CSV outputs to {out_dir}")


def dataframe_row_to_str_dict(row: pd.Series) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in row.index:
        v = row[name]
        out[str(name)] = "" if pd.isna(v) else str(v)
    return out


def json_for_html_embed(obj: object) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</script>", "<\\/script>")


def build_report(
    clustered_df: pd.DataFrame,
    summary: Dict[int, Dict[str, float]],
    overlay_df: pd.DataFrame,
    out_dir: Path,
    max_clusters: int,
    max_points_per_cluster: int,
    max_overlay_points: int,
    cluster_contexts_label: str,
    overlay_contexts_label: str,
    keywords_label: str,
    has_overlay: bool,
    *,
    report_filename: str = "context_comparison_report.html",
    page_title: str = "Keyword cluster comparison",
    header_heading: str = "Keyword-based cluster report",
    sidebar_cluster_caption: str = "Only keyword-matching images are clustered.",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / report_filename

    all_clusters_sorted = sorted(summary.values(), key=lambda x: x["count"], reverse=True)
    top_clusters = (
        all_clusters_sorted[:max_clusters] if max_clusters > 0 else all_clusters_sorted
    )

    csv_columns = [str(c) for c in clustered_df.columns]

    clusters_payload: List[Dict] = []
    for info in top_clusters:
        cid = info["cluster_id"]
        subset = clustered_df[clustered_df["cluster_id"] == cid]
        if max_points_per_cluster > 0 and len(subset) > max_points_per_cluster:
            subset = subset.iloc[:max_points_per_cluster]

        points = []
        for _, row in subset.iterrows():
            points.append({"row": dataframe_row_to_str_dict(row)})
        clusters_payload.append({"cluster_id": cid, "count": int(info["count"]), "points": points})

    overlay_subset = overlay_df
    if max_overlay_points > 0 and len(overlay_subset) > max_overlay_points:
        overlay_subset = overlay_subset.iloc[:max_overlay_points]

    overlay_payload: List[Dict] = []
    for _, row in overlay_subset.iterrows():
        overlay_payload.append({"row": dataframe_row_to_str_dict(row)})

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    clusters_json = json_for_html_embed(clusters_payload)
    overlay_json = json_for_html_embed(overlay_payload)
    csv_columns_json = json_for_html_embed(csv_columns)

    script_path = Path(__file__).resolve().parent / "context_report_interactive.js"
    script_js = script_path.read_text(encoding="utf-8")

    overlay_toggle_html = ""
    if has_overlay and len(overlay_payload) > 0:
        overlay_toggle_html = f"""
      <div class="control-row">
        <label><input type="checkbox" id="toggle-overlay" checked /> Show overlay points (same keywords)</label>
      </div>
      <div class="meta">Overlay points: {len(overlay_payload):,}</div>"""

    head_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{page_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #layout {{
      display: grid;
      grid-template-columns: 320px 1fr;
      grid-template-rows: auto 1fr;
      grid-template-areas: "header header" "sidebar map";
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
    #map {{ grid-area: map; width: 100%; height: 100%; }}
    .cluster-item {{ padding: 0.25rem 0.4rem; cursor: pointer; border-radius: 4px; }}
    .cluster-item:hover {{ background: #f2f2f2; }}
    .cluster-item.active {{ background: #e7f2ff; }}
    .cluster-meta {{ color: #666; font-size: 0.8rem; }}
    .meta {{ color: #666; font-size: 0.8rem; }}
    .control-row {{ margin: 0.5rem 0; }}
    .pin-marker {{ background: transparent; border: none; }}
  </style>
</head>
<body>
  <div id="layout">
    <header>
      <strong>{header_heading}</strong>
      <span style="color:#666;font-size:0.85rem;">&nbsp;&mdash; generated {generated_at}</span>
      <span style="margin-left:0.75rem;display:inline-flex;gap:0.35rem;flex-wrap:wrap;align-items:center;font-size:0.85rem;">
        <button type="button" id="export-err-btn" style="padding:0.2rem 0.55rem;cursor:pointer;">Export erroneous CSV</button>
        <button type="button" id="export-sanitized-btn" style="padding:0.2rem 0.55rem;cursor:pointer;">Export sanitized CSV</button>
        <button type="button" id="import-err-btn" style="padding:0.2rem 0.55rem;cursor:pointer;">Import erroneous CSV…</button>
        <input type="file" id="import-err-file" accept=".csv,text/csv" style="display:none" />
      </span>
    </header>
    <aside id="sidebar">
      <div><strong>Keywords (match ANY in title / description / tags):</strong></div>
      <div class="meta" style="white-space:pre-wrap;word-break:break-word;">{keywords_label}</div>
      <hr />
      <div><strong>Cluster pre-filter (context):</strong> {cluster_contexts_label}</div>
      <div><strong>Overlay pre-filter (context):</strong> {overlay_contexts_label}</div>
      {overlay_toggle_html}
      <hr />
      <p><strong>Clusters</strong> (by size). {sidebar_cluster_caption}</p>
      <p class="meta">Each cluster has a main list and an <strong>Erroneous in #…</strong> sub-list. Mark images in the popup; export CSVs or import a prior erroneous export.</p>
      <div id="cluster-list"></div>
    </aside>
    <div id="map"></div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
"""
    html = (
        head_html
        + "<script>\nconst clustersData = "
        + clusters_json
        + ";\nconst overlayData = "
        + overlay_json
        + ";\nconst csvColumns = "
        + csv_columns_json
        + ";\n</script>\n<script>\n"
        + script_js
        + "\n</script>\n</body>\n</html>\n"
    )

    report.write_text(html, encoding="utf-8")
    print(f"[INFO] Wrote comparison report: {report}")
    return report


def parse_args() -> argparse.Namespace:
    script = Path(__file__)
    root = detect_project_root(script)
    default_metadata = root / "metadata"
    default_output = script.resolve().parent / "outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Cluster geo-tagged photos that match keywords (title/description/tags). "
            "Optional Flickr context pre-filters + optional overlay for comparison."
        )
    )
    parser.add_argument("--metadata-dir", type=Path, default=default_metadata)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--cluster-contexts",
        type=str,
        default="all",
        help="Flickr geo context(s) before keyword filter: 0,1,2 or comma list, or 'all'.",
    )
    parser.add_argument(
        "--overlay-contexts",
        type=str,
        default="none",
        help="Optional second layer: same keywords, these contexts, unclustered. "
        "Use 'none' to disable overlay.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="",
        help="Comma-separated keywords (match ANY as substring). Default: built-in building list.",
    )
    parser.add_argument(
        "--keywords-file",
        type=Path,
        default=None,
        help="Path to a text file: one keyword per line (# comments allowed). Overrides --keywords.",
    )
    parser.add_argument("--eps-m", type=float, default=50.0)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument(
        "--max-clusters-report",
        type=int,
        default=0,
        help="Max clusters in report. Use 0 for all clusters.",
    )
    parser.add_argument("--max-points-per-cluster", type=int, default=0)
    parser.add_argument(
        "--max-overlay-points",
        type=int,
        default=0,
        help="Max overlay points in report. Use 0 for all points.",
    )
    return parser.parse_args()


def _format_context_label(ctx: Optional[Set[str]]) -> str:
    if ctx is None:
        return "all (no context filter)"
    return ",".join(sorted(ctx))


def main() -> None:
    args = parse_args()
    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory does not exist: {args.metadata_dir}")

    cluster_ctx = parse_cluster_contexts(args.cluster_contexts)
    overlay_ctx = parse_overlay_contexts(args.overlay_contexts)
    print(f"[INFO] Cluster context pre-filter: {_format_context_label(cluster_ctx)}")
    print(f"[INFO] Overlay context pre-filter: {_format_context_label(overlay_ctx)}")

    keywords = load_keywords_list(
        keywords_arg=args.keywords if args.keywords.strip() else None,
        keywords_file=args.keywords_file,
    )
    print(f"[INFO] Using {len(keywords)} keyword(s).")
    keywords_label = ", ".join(keywords[:40]) + (" …" if len(keywords) > 40 else "")

    df_all = load_all_metadata(args.metadata_dir)
    df_geo = keep_valid_geo(df_all)
    if "context" not in df_geo.columns:
        raise KeyError("`context` column not found. Regenerate metadata with geo extras.")

    df_for_cluster = df_geo if cluster_ctx is None else filter_contexts(df_geo, cluster_ctx)
    clustered_input = filter_by_keywords(df_for_cluster, keywords)
    if clustered_input.empty:
        raise ValueError(
            "No rows left after keyword filter for clustering. "
            "Try more keywords, or use --cluster-contexts all."
        )

    overlay_input = pd.DataFrame()
    if overlay_ctx is not None:
        overlay_input = filter_by_keywords(filter_contexts(df_geo, overlay_ctx), keywords)
        print(f"[INFO] Overlay rows (keyword-matched): {len(overlay_input):,}")

    clustered_df, summary = run_dbscan(clustered_input, eps_meters=args.eps_m, min_samples=args.min_samples)
    export_csvs(clustered_df, summary, args.output_dir)
    build_report(
        clustered_df=clustered_df,
        summary=summary,
        overlay_df=overlay_input,
        out_dir=args.output_dir,
        max_clusters=args.max_clusters_report,
        max_points_per_cluster=args.max_points_per_cluster,
        max_overlay_points=args.max_overlay_points,
        cluster_contexts_label=_format_context_label(cluster_ctx),
        overlay_contexts_label=_format_context_label(overlay_ctx),
        keywords_label=keywords_label,
        has_overlay=overlay_ctx is not None and not overlay_input.empty,
    )
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

