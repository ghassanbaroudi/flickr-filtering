"""
Apply a browser-exported erroneous review CSV to the original clusters.csv.

Writes:
  - clusters_sanitized.csv — rows whose `id` was not listed as erroneous
  - clusters_erroneous.csv — rows marked erroneous (full columns from clusters.csv)
  - clusters_erroneous_by_cluster.csv — same as erroneous, sorted by cluster_id then id

The erroneous export from context_comparison_report.html includes columns:
  cluster_id, image_id, then all original columns. This script matches on `id` ==
  `image_id` (or falls back to `id` in the erroneous file if named `id`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "clusters_csv",
        type=Path,
        help="Original clusters.csv from make_context_comparison.py",
    )
    ap.add_argument(
        "erroneous_csv",
        type=Path,
        help="Erroneous export from the HTML report (or compatible CSV)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as clusters_csv)",
    )
    args = ap.parse_args()
    out_dir = args.output_dir or args.clusters_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clusters_csv)
    err = pd.read_csv(args.erroneous_csv)

    if "id" not in df.columns:
        raise SystemExit("clusters_csv must contain an `id` column.")

    err_key = "image_id" if "image_id" in err.columns else "id"
    if err_key not in err.columns:
        raise SystemExit("erroneous_csv must contain `image_id` or `id`.")

    bad = set(err[err_key].astype(str).str.strip())
    mask = df["id"].astype(str).str.strip().isin(bad)
    erroneous = df.loc[mask].copy()
    sanitized = df.loc[~mask].copy()

    sanitized_path = out_dir / "clusters_sanitized.csv"
    erroneous_path = out_dir / "clusters_erroneous.csv"
    by_cluster_path = out_dir / "clusters_erroneous_by_cluster.csv"

    sanitized.to_csv(sanitized_path, index=False)
    erroneous.to_csv(erroneous_path, index=False)
    if not erroneous.empty and "cluster_id" in erroneous.columns:
        erroneous.sort_values(["cluster_id", "id"]).to_csv(by_cluster_path, index=False)
    else:
        erroneous.to_csv(by_cluster_path, index=False)

    print(f"[INFO] Wrote {sanitized_path} ({len(sanitized):,} rows)")
    print(f"[INFO] Wrote {erroneous_path} ({len(erroneous):,} rows)")
    print(f"[INFO] Wrote {by_cluster_path}")


if __name__ == "__main__":
    main()
