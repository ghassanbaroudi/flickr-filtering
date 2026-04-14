#!/usr/bin/env python3
"""
DBSCAN (haversine) + Leaflet report for vision-filtered CSVs
(e.g. keyword_then_vision_keep.csv from filter_keyword_vision.py).

Writes clusters.csv, cluster_summary.csv, vision_cluster_report.html next to the input
or under --output-dir.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
import pandas as pd

# Parent flico/ on path → import keyword_vision.*
_FLICO = Path(__file__).resolve().parent.parent
if str(_FLICO) not in sys.path:
    sys.path.insert(0, str(_FLICO))


def detect_project_root(script_path: Path) -> Path:
    try:
        return script_path.resolve().parents[2]
    except Exception:
        return Path.cwd()


def _empty_input_help(path: Path) -> str:
    return (
        f"No usable rows from: {path}\n\n"
        "• Run the vision step first so you have scored rows:\n"
        "  python3 flico/keyword_vision/filter_keyword_vision.py --output-dir flico/keyword_vision/outputs\n\n"
        "• Or point --input-csv to keyword_then_vision.csv (must include vision_label; YES rows are used).\n\n"
        "• If keyword_then_vision_keep.csv is empty, every image was vision-rejected (NO) or skipped — "
        "check keyword_then_vision.csv and vision_summary.json."
    )


def load_vision_input_dataframe(input_csv: Path) -> pd.DataFrame:
    """
    Read CSV. If the file is empty and the path is keyword_then_vision_keep.csv,
    fall back to keyword_then_vision.csv in the same folder (then filter YES in caller).
    """
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Create it by running: python3 flico/keyword_vision/filter_keyword_vision.py --output-dir …"
        )
    df = pd.read_csv(input_csv)
    if df.empty and input_csv.name == "keyword_then_vision_keep.csv":
        alt = input_csv.parent / "keyword_then_vision.csv"
        if alt.exists():
            df_alt = pd.read_csv(alt)
            if not df_alt.empty:
                print(
                    f"[INFO] {input_csv.name} is empty; using {alt.name} "
                    f"({len(df_alt):,} rows) and will keep vision_label=YES only."
                )
                return df_alt
    if df.empty:
        raise ValueError(_empty_input_help(input_csv))
    return df


def load_context_compare_module():
    """Load clusters_compare/make_context_comparison.py as a module."""
    mod_path = _FLICO / "clusters_compare" / "make_context_comparison.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Expected {mod_path}")
    spec = importlib.util.spec_from_file_location("make_context_comparison_kv", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cluster_and_report(
    input_csv: Path,
    output_dir: Path,
    *,
    eps_m: float = 50.0,
    min_samples: int = 5,
    max_clusters_report: int = 0,
    max_points_per_cluster: int = 0,
    keywords_label: str = "Keyword list + vision YES (see filter_keyword_vision.py).",
) -> Path:
    """
    Load CSV, optional vision_label==YES filter, valid geo, DBSCAN, export CSVs + HTML map.
    Returns path to vision_cluster_report.html.
    """
    cc = load_context_compare_module()

    df = load_vision_input_dataframe(input_csv)

    if "vision_label" in df.columns:
        df = df[df["vision_label"].astype(str).str.upper().str.strip() == "YES"].copy()
        print(f"[INFO] Rows with vision_label=YES: {len(df):,}")
    if df.empty:
        raise ValueError(
            "No rows with vision_label=YES.\n"
            "All images were scored NO/SKIP/ERROR, or the CSV has no YES rows. "
            "See keyword_then_vision.csv and vision_summary.json."
        )

    df_geo = cc.keep_valid_geo(df)
    if df_geo.empty:
        raise ValueError("No rows with valid coordinates after keep_valid_geo.")

    clustered_df, summary = cc.run_dbscan(df_geo, eps_meters=eps_m, min_samples=min_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    cc.export_csvs(clustered_df, summary, output_dir)

    overlay_empty = pd.DataFrame(columns=clustered_df.columns)
    report_path = cc.build_report(
        clustered_df=clustered_df,
        summary=summary,
        overlay_df=overlay_empty,
        out_dir=output_dir,
        max_clusters=max_clusters_report,
        max_points_per_cluster=max_points_per_cluster,
        max_overlay_points=0,
        cluster_contexts_label="from vision CSV (pre-filtered)",
        overlay_contexts_label="none",
        keywords_label=keywords_label,
        has_overlay=False,
        report_filename="vision_cluster_report.html",
        page_title="Keyword + vision — clusters",
        header_heading="Keyword + vision cluster map",
        sidebar_cluster_caption=(
            "Images passed text keywords and the vision model (YES), then clustered with DBSCAN (haversine)."
        ),
    )
    print(f"[INFO] Leaflet report: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    script = Path(__file__)
    root = detect_project_root(script)
    default_in = script.parent / "outputs" / "keyword_then_vision_keep.csv"
    default_out = script.parent / "outputs"

    p = argparse.ArgumentParser(
        description="Cluster vision-approved rows (DBSCAN) and build Leaflet report with images."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        default=default_in,
        help="Usually keyword_then_vision_keep.csv from filter_keyword_vision.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write clusters.csv, cluster_summary.csv, vision_cluster_report.html (default: same folder as input CSV).",
    )
    p.add_argument("--eps-m", type=float, default=50.0)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument(
        "--max-clusters-report",
        type=int,
        default=0,
        help="Max clusters in the HTML report (0 = all).",
    )
    p.add_argument(
        "--max-points-per-cluster",
        type=int,
        default=0,
        help="Max points per cluster in the report (0 = all).",
    )
    p.add_argument(
        "--keywords-label",
        type=str,
        default="",
        help="Short text shown in sidebar (default: generic vision description).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    out_dir = args.output_dir if args.output_dir is not None else args.input_csv.parent
    kw = (
        args.keywords_label.strip()
        if args.keywords_label.strip()
        else "Keyword list + vision YES (see filter_keyword_vision.py)."
    )
    run_cluster_and_report(
        args.input_csv,
        out_dir,
        eps_m=args.eps_m,
        min_samples=args.min_samples,
        max_clusters_report=args.max_clusters_report,
        max_points_per_cluster=args.max_points_per_cluster,
        keywords_label=kw,
    )
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
