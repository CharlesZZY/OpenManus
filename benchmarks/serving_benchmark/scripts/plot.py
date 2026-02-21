#!/usr/bin/env python3
"""Generate all charts from raw logs.

Usage:
    python -m benchmarks.serving_benchmark.scripts.plot \
        --input output/<experiment_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument("--input", required=True, help="Experiment output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.input)
    raw_logs_dir = exp_dir / "raw_logs"

    from benchmarks.serving_benchmark.analysis.aggregator import (
        load_gpu_logs,
        load_request_logs,
    )
    from benchmarks.serving_benchmark.analysis.visualizer import generate_all_plots

    print(f"Loading logs from: {raw_logs_dir}")
    request_df = load_request_logs(raw_logs_dir)
    gpu_df = load_gpu_logs(raw_logs_dir)

    print(f"Generating plots for {len(request_df)} requests...")
    generate_all_plots(request_df, gpu_df, exp_dir)

    print(f"Plots saved to: {exp_dir / 'plots'}")
    print(f"Mermaid saved to: {exp_dir / 'mermaid'}")


if __name__ == "__main__":
    main()
