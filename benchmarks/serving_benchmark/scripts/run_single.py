#!/usr/bin/env python3
"""Run a single experiment cell.

Usage:
    python -m benchmarks.serving_benchmark.scripts.run_single \
        --baseline fifo \
        --workload poisson \
        --mode single \
        --seeds 42 123 2024
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Run single experiment")
    parser.add_argument("--baseline", required=True, help="Baseline name (e.g. fifo)")
    parser.add_argument(
        "--workload", required=True, help="Workload pattern (e.g. poisson)"
    )
    parser.add_argument("--mode", default="single", choices=["single", "workflow"])
    parser.add_argument("--suite", default="mixed", help="Suite: S, R, L, or mixed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--output", default="output")
    parser.add_argument("--max-samples", type=int, default=50)
    return parser.parse_args()


async def main():
    args = parse_args()

    baseline_cfg = CONFIGS / "baselines" / f"{args.baseline}.yaml"
    workload_cfg = CONFIGS / "workloads" / f"{args.workload}.yaml"

    if not baseline_cfg.exists():
        print(f"ERROR: {baseline_cfg} not found")
        return
    if not workload_cfg.exists():
        print(f"ERROR: {workload_cfg} not found")
        return

    from benchmarks.serving_benchmark.runners.experiment_runner import ExperimentRunner

    runner = ExperimentRunner(
        baseline_cfg_path=str(baseline_cfg),
        workload_cfg_path=str(workload_cfg),
        mode=args.mode,
        suite=args.suite,
        seeds=args.seeds,
        output_root=args.output,
        max_samples_per_dataset=args.max_samples,
    )

    print(f"Running: {args.baseline} × {args.workload} × {args.mode}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output}/{runner.experiment_id}")

    results = await runner.run_all()
    print(f"\nDone. Experiment ID: {runner.experiment_id}")
    print(f"Logs: {args.output}/{runner.experiment_id}/raw_logs/")


if __name__ == "__main__":
    asyncio.run(main())
