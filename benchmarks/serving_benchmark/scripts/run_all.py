#!/usr/bin/env python3
"""Run the full experiment matrix.

Iterates over all combinations of baseline × workload × mode defined in
``configs/experiment_matrix.yaml`` and runs each with 3 seeds.

Usage:
    python -m benchmarks.serving_benchmark.scripts.run_all [--output OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Run full experiment matrix")
    parser.add_argument("--output", default="output", help="Output root directory")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples per dataset (for quick testing)",
    )
    parser.add_argument(
        "--matrix",
        default=str(CONFIGS / "experiment_matrix.yaml"),
        help="Path to experiment matrix YAML",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)

    baselines = matrix.get("baselines", ["fifo"])
    patterns = matrix.get("patterns", ["poisson"])
    modes = matrix.get("modes", ["single"])
    defaults = matrix.get("defaults", {})
    seeds = defaults.get("seeds", [42, 123, 2024])

    from benchmarks.serving_benchmark.runners.experiment_runner import ExperimentRunner

    total = len(baselines) * len(patterns) * len(modes)
    print(
        f"Experiment matrix: {len(baselines)} baselines × {len(patterns)} patterns × {len(modes)} modes = {total} cells"
    )
    print(f"Seeds: {seeds}, Repeats per cell: {len(seeds)}")
    print(f"Output: {args.output}")
    print()

    completed = 0
    for baseline, pattern, mode in itertools.product(baselines, patterns, modes):
        baseline_cfg = CONFIGS / "baselines" / f"{baseline}.yaml"
        workload_cfg = CONFIGS / "workloads" / f"{pattern}.yaml"

        if not baseline_cfg.exists():
            print(f"SKIP: baseline config not found: {baseline_cfg}")
            continue
        if not workload_cfg.exists():
            print(f"SKIP: workload config not found: {workload_cfg}")
            continue

        print(f"[{completed + 1}/{total}] {baseline} × {pattern} × {mode}")

        runner = ExperimentRunner(
            baseline_cfg_path=str(baseline_cfg),
            workload_cfg_path=str(workload_cfg),
            mode=mode,
            seeds=seeds,
            output_root=args.output,
            max_samples_per_dataset=args.max_samples,
        )

        try:
            results = await runner.run_all()
            print(f"  -> {runner.experiment_id} done")
        except Exception as e:
            print(f"  -> FAILED: {e}")

        completed += 1

    # Run stress tests
    stress_tests = matrix.get("stress_tests", [])
    for stress in stress_tests:
        print(f"\n[STRESS] {stress.get('name', 'unnamed')}")
        # Stress tests would use custom workload configs — placeholder
        print("  -> stress test placeholder (custom arrival params)")

    print(f"\nAll experiments completed: {completed}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
