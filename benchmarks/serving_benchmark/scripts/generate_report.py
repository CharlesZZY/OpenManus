#!/usr/bin/env python3
"""Generate the full Markdown experiment report.

Usage:
    python -m benchmarks.serving_benchmark.scripts.generate_report \
        --input output/<experiment_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--input", required=True, help="Experiment output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.input)

    from benchmarks.serving_benchmark.analysis.report_gen import generate_report

    agg_path = exp_dir / "agg_metrics" / "agg_metrics.csv"
    configs_dir = exp_dir / "configs"
    plots_dir = exp_dir / "plots"
    mermaid_dir = exp_dir / "mermaid"
    output_path = exp_dir / "reports" / "experiment_report.md"

    experiment_id = exp_dir.name

    print(f"Generating report for: {experiment_id}")
    generate_report(
        agg_metrics_path=agg_path,
        configs_dir=configs_dir,
        plots_dir=plots_dir,
        mermaid_dir=mermaid_dir,
        output_path=output_path,
        experiment_id=experiment_id,
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
