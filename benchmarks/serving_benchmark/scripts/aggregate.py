#!/usr/bin/env python3
"""Aggregate raw logs into metrics with bootstrap CIs.

Usage:
    python -m benchmarks.serving_benchmark.scripts.aggregate \
        --input output/<experiment_id>/raw_logs \
        --output output/<experiment_id>/agg_metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate raw logs")
    parser.add_argument("--input", required=True, help="Path to raw_logs/ directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: sibling agg_metrics/)",
    )
    parser.add_argument("--bootstrap-B", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir.parent / "agg_metrics"

    from benchmarks.serving_benchmark.analysis.aggregator import aggregate_experiment

    print(f"Aggregating: {input_dir}")
    agg_df = aggregate_experiment(input_dir, output_dir)
    print(f"Results: {output_dir / 'agg_metrics.csv'}")
    print(f"Rows: {len(agg_df)}")
    print()
    print(agg_df.to_string())


if __name__ == "__main__":
    main()
