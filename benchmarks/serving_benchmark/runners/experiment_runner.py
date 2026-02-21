"""Experiment runner — orchestrates workload generation, scheduling,
request execution, logging, and repetition.

Flow:
  1. Load config → init scheduler + workload generator
  2. Start GPU logger (background)
  3. Warm-up phase
  4. Sampling phase
  5. Repeat × 3 with different seeds
  6. Save config snapshot + raw logs
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from benchmarks.serving_benchmark.core.schema import (
    BenchmarkRequest,
    BenchmarkSample,
    ExperimentConfig,
    RequestLog,
    RequestMode,
    Suite,
)
from benchmarks.serving_benchmark.datasets.loader import load_dataset, load_suite
from benchmarks.serving_benchmark.schedulers import create_scheduler
from benchmarks.serving_benchmark.workload.generator import (
    WorkloadConfig,
    WorkloadGenerator,
)
from benchmarks.serving_benchmark.workload.mixing import WorkloadItem


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


class ExperimentRunner:
    """Runs a single experiment configuration (one cell in the matrix)."""

    def __init__(
        self,
        baseline_cfg_path: str,
        workload_cfg_path: str,
        mode: str = "single",
        suite: str = "mixed",
        seeds: Optional[List[int]] = None,
        output_root: str = "output",
        max_samples_per_dataset: int = 100,
    ):
        self.mode = mode
        self.suite = suite
        self.seeds = seeds or [42, 123, 2024]
        self.output_root = Path(output_root)
        self.max_samples = max_samples_per_dataset

        # Load baseline config
        with open(baseline_cfg_path, "r") as f:
            self.baseline_cfg = yaml.safe_load(f)

        # Load workload config
        with open(workload_cfg_path, "r") as f:
            self.workload_cfg = yaml.safe_load(f)

        self.experiment_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    def _make_output_dir(self) -> Path:
        out = self.output_root / self.experiment_id
        for sub in (
            "raw_logs",
            "agg_metrics",
            "plots",
            "configs",
            "reports",
            "mermaid",
        ):
            (out / sub).mkdir(parents=True, exist_ok=True)
        return out

    def _load_sample_pools(self) -> Dict[str, List[BenchmarkSample]]:
        """Load samples grouped by suite."""
        pools: Dict[str, List[BenchmarkSample]] = {}
        for s in (Suite.S, Suite.R, Suite.L):
            try:
                pools[s.value] = load_suite(s, max_samples_per_dataset=self.max_samples)
            except Exception:
                pools[s.value] = []
        return pools

    async def _run_single_request(self, request: BenchmarkRequest) -> List[RequestLog]:
        if self.mode == "single":
            from benchmarks.serving_benchmark.runners.single_runner import SingleRunner

            runner = SingleRunner()
            log = await runner.run(request)
            return [log]
        else:
            from benchmarks.serving_benchmark.runners.workflow_runner import (
                WorkflowRunner,
            )

            runner = WorkflowRunner()
            return await runner.run(request)

    async def _execute_phase(
        self,
        items: List[WorkloadItem],
        scheduler,
        config_id: str,
        seed: int,
        is_warmup: bool = False,
    ) -> List[RequestLog]:
        """Execute a list of workload items through the scheduler."""
        all_logs: List[RequestLog] = []
        base_time = time.time()

        # Create BenchmarkRequests
        requests: List[BenchmarkRequest] = []
        for item in items:
            req = BenchmarkRequest(
                sample=item.sample,
                mode=self.mode,
                config_id=config_id,
                seed=seed,
                arrive_time=base_time + item.arrive_time,
                predicted_in_tokens=len(
                    (item.sample.raw_prompt if item.sample else "").split()
                ),
            )
            requests.append(req)

        # Enqueue all
        for req in requests:
            await scheduler.enqueue(req)

        # Process through scheduler
        max_empty_polls = 10
        empty_count = 0
        while empty_count < max_empty_polls:
            batch = await scheduler.schedule()
            if not batch:
                empty_count += 1
                await asyncio.sleep(0.05)  # allow flush timers to expire
                continue
            empty_count = 0
            for req in batch:
                now = time.time()
                if req.arrive_time > now:
                    await asyncio.sleep(req.arrive_time - now)

                logs = await self._run_single_request(req)
                all_logs.extend(logs)

                await scheduler.on_complete(req.req_id, req)

        # Final drain
        remaining = await scheduler.drain()
        for req in remaining:
            logs = await self._run_single_request(req)
            all_logs.extend(logs)

        return all_logs

    async def run_once(self, seed: int) -> tuple[ExperimentConfig, List[RequestLog]]:
        """Run one repetition of the experiment."""
        sched_cfg = self.baseline_cfg.get("scheduler", {})
        sched_name = sched_cfg.get("name", "fifo")
        sched_params = sched_cfg.get("params", {})
        scheduler = create_scheduler(sched_name, **sched_params)

        wl_cfg = self.workload_cfg.get("workload", {})
        wl_config = WorkloadConfig(
            pattern=wl_cfg.get("pattern", "poisson"),
            arrival_kwargs=wl_cfg.get("arrival_kwargs", {}),
            suite_ratio=wl_cfg.get("suite_ratio"),
            seed=seed,
            warmup_duration_s=wl_cfg.get("warmup_duration_s", 300),
            warmup_max_requests=wl_cfg.get("warmup_max_requests", 200),
            run_duration_s=wl_cfg.get("run_duration_s", 1800),
            run_max_requests=wl_cfg.get("run_max_requests", 10000),
        )

        pools = self._load_sample_pools()
        gen = WorkloadGenerator(wl_config, pools)
        warmup_items, run_items = gen.generate()

        config_id = (
            f"{sched_name}_{wl_cfg.get('pattern', 'poisson')}_{self.mode}_{seed}"
        )

        exp_config = ExperimentConfig(
            config_id=config_id,
            baseline=sched_name,
            pattern=wl_cfg.get("pattern", "poisson"),
            suite=self.suite,
            mode=self.mode,
            seed=seed,
            git_commit=_git_commit_hash(),
        )

        # Warm-up
        await self._execute_phase(
            warmup_items, scheduler, config_id, seed, is_warmup=True
        )

        # Sampling
        scheduler = create_scheduler(sched_name, **sched_params)
        logs = await self._execute_phase(run_items, scheduler, config_id, seed)

        return exp_config, logs

    async def run_all(self) -> Dict[str, Any]:
        """Run all repetitions and save outputs."""
        out_dir = self._make_output_dir()
        all_results: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "configs": [],
            "logs_by_seed": {},
        }

        for repeat_idx, seed in enumerate(self.seeds):
            exp_config, logs = await self.run_once(seed)
            exp_config.repeat = repeat_idx
            all_results["configs"].append(asdict(exp_config))
            all_results["logs_by_seed"][seed] = [log.to_dict() for log in logs]

            # Save config
            cfg_path = out_dir / "configs" / f"config_{seed}.json"
            with open(cfg_path, "w") as f:
                json.dump(asdict(exp_config), f, indent=2)

        # Save raw logs
        self._save_logs(out_dir, all_results["logs_by_seed"])

        return all_results

    def _save_logs(self, out_dir: Path, logs_by_seed: Dict[int, list]):
        """Save raw logs as CSV and Parquet."""
        try:
            import pandas as pd

            all_rows = []
            for seed, rows in logs_by_seed.items():
                all_rows.extend(rows)

            if not all_rows:
                return

            df = pd.DataFrame(all_rows)
            csv_path = out_dir / "raw_logs" / "request_logs.csv"
            df.to_csv(csv_path, index=False)

            try:
                parquet_path = out_dir / "raw_logs" / "request_logs.parquet"
                df.to_parquet(parquet_path, index=False)
            except Exception:
                pass  # pyarrow not available

        except ImportError:
            # pandas not available — save as JSON
            json_path = out_dir / "raw_logs" / "request_logs.json"
            all_rows = []
            for seed, rows in logs_by_seed.items():
                all_rows.extend(rows)
            with open(json_path, "w") as f:
                json.dump(all_rows, f, indent=2)
