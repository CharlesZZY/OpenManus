#!/usr/bin/env python3
"""Run the full experiment matrix.

Iterates over all combinations of baseline × workload × mode defined in
``configs/experiment_matrix.yaml`` and runs each with 3 seeds.

Usage:
    python -m benchmarks.serving_benchmark.scripts.run_all \
        [--output OUTPUT_DIR] [--model MODEL_ALIAS]
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
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
    parser.add_argument(
        "--model",
        default=None,
        help="Model alias from configs/models.yaml (default: first model)",
    )
    parser.add_argument(
        "--models-config",
        default=str(CONFIGS / "models.yaml"),
        help="Path to models.yaml",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    with open(args.matrix) as f:
        matrix = yaml.safe_load(f)

    baselines = matrix.get("baselines", ["fifo"])
    patterns = matrix.get("patterns", ["poisson"])
    modes = matrix.get("modes", ["single"])
    suites = matrix.get("suites", ["mixed"])
    defaults = matrix.get("defaults", {})
    seeds = defaults.get("seeds", [42, 123, 2024])

    from benchmarks.serving_benchmark.core.vllm_client import ModelRegistry
    from benchmarks.serving_benchmark.runners.experiment_runner import ExperimentRunner

    registry = ModelRegistry(args.models_config)
    model_alias = args.model or registry.first_model

    total = len(baselines) * len(patterns) * len(modes) * len(suites)
    print(
        f"Experiment matrix: {len(baselines)} baselines × {len(patterns)} patterns "
        f"× {len(modes)} modes × {len(suites)} suites = {total} cells"
    )
    print(f"Model: {model_alias}")
    print(f"Seeds: {seeds}, Repeats per cell: {len(seeds)}")
    print(f"Output: {args.output}")
    print()

    completed = 0
    for baseline, pattern, mode, suite in itertools.product(
        baselines, patterns, modes, suites
    ):
        baseline_cfg = CONFIGS / "baselines" / f"{baseline}.yaml"
        workload_cfg = CONFIGS / "workloads" / f"{pattern}.yaml"

        if not baseline_cfg.exists():
            print(f"SKIP: baseline config not found: {baseline_cfg}")
            continue
        if not workload_cfg.exists():
            print(f"SKIP: workload config not found: {workload_cfg}")
            continue

        print(f"[{completed + 1}/{total}] {baseline} × {pattern} × {mode} × {suite}")

        runner = ExperimentRunner(
            baseline_cfg_path=str(baseline_cfg),
            workload_cfg_path=str(workload_cfg),
            mode=mode,
            suite=suite,
            seeds=seeds,
            output_root=args.output,
            max_samples_per_dataset=args.max_samples,
            model_alias=model_alias,
            registry=registry,
        )

        try:
            results = await runner.run_all()
            print(f"  -> {runner.experiment_id} done")
        except Exception as e:
            print(f"  -> FAILED: {e}")

        completed += 1

    # Stress tests
    stress_tests = matrix.get("stress_tests", [])
    for stress in stress_tests:
        name = stress.get("name", "unnamed")
        print(f"\n[STRESS] {name}")
        stress_pattern = stress.get("pattern", "poisson")
        stress_workload_cfg = CONFIGS / "workloads" / f"{stress_pattern}.yaml"
        if stress_workload_cfg.exists():
            for bl in stress.get("baselines", baselines):
                bl_cfg = CONFIGS / "baselines" / f"{bl}.yaml"
                if not bl_cfg.exists():
                    continue
                for m in stress.get("modes", ["single"]):
                    runner = ExperimentRunner(
                        baseline_cfg_path=str(bl_cfg),
                        workload_cfg_path=str(stress_workload_cfg),
                        mode=m,
                        seeds=seeds,
                        output_root=args.output,
                        max_samples_per_dataset=args.max_samples,
                        model_alias=model_alias,
                        registry=registry,
                    )
                    try:
                        await runner.run_all()
                        print(f"  -> {runner.experiment_id} done ({bl} × {m})")
                    except Exception as e:
                        print(f"  -> FAILED: {bl} × {m}: {e}")
        else:
            print(f"  -> stress test workload not found: {stress_workload_cfg}")

    print(f"\nAll experiments completed: {completed}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
