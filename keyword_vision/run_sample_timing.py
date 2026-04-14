#!/usr/bin/env python3
"""
Quick sample run (default: 10 rows) with a **separate output directory** so you can
execute it in parallel with a long `filter_keyword_vision.py` job without touching
its CSVs or vision_cache.jsonl.

Example (from project root):

  python flico/keyword_vision/run_sample_timing.py
  python flico/keyword_vision/run_sample_timing.py --limit 20
  python flico/keyword_vision/run_sample_timing.py --backend ollama

Any extra arguments are forwarded to filter_keyword_vision.py (see its --help).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    filter_script = here / "filter_keyword_vision.py"
    default_out = here / "outputs_sample_10"

    parser = argparse.ArgumentParser(
        description=(
            "Run filter_keyword_vision on a small row limit with a dedicated output dir "
            "(safe beside a full run)."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max rows after keyword filter (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Separate output directory (default: {default_out}).",
    )
    args, forwarded = parser.parse_known_args()

    if args.limit < 1:
        print("[FATAL] --limit must be >= 1", file=sys.stderr)
        sys.exit(2)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(filter_script),
        "--limit",
        str(args.limit),
        "--output-dir",
        str(out),
    ] + forwarded

    print("[INFO] Sample timing run (parallel-safe output dir).")
    print("[INFO] Command:", " ".join(cmd))
    print()
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0 and args.limit > 0:
        per_row = elapsed / args.limit
        print()
        print(f"[INFO] Wall time: {elapsed:.1f}s for {args.limit} row(s) (~{per_row:.2f}s/row).")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
