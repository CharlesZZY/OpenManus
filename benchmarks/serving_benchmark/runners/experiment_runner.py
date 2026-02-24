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
import logging
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("benchmark")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

from benchmarks.serving_benchmark.core.schema import (
    BenchmarkRequest,
    BenchmarkSample,
    ExperimentConfig,
    RequestLog,
    RequestMode,
    Suite,
)
from benchmarks.serving_benchmark.core.vllm_client import ModelRegistry
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
        model_alias: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.mode = mode
        self.suite = suite
        self.seeds = seeds or [42, 123, 2024]
        self.output_root = Path(output_root)
        self.max_samples = max_samples_per_dataset

        with open(baseline_cfg_path, "r") as f:
            self.baseline_cfg = yaml.safe_load(f)

        with open(workload_cfg_path, "r") as f:
            self.workload_cfg = yaml.safe_load(f)

        self.registry = registry or ModelRegistry()
        self.model_alias = model_alias or self.registry.first_model

        self.experiment_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    def _make_output_dir(self) -> Path:
        out = self.output_root / self.experiment_id
        for sub in (
            "raw_logs",
            "agg_metrics",
            "plots",
            "plot_data",
            "configs",
            "reports",
            "mermaid",
        ):
            (out / sub).mkdir(parents=True, exist_ok=True)
        return out

    def _load_sample_pools(self) -> Dict[str, List[BenchmarkSample]]:
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

            runner = SingleRunner(
                registry=self.registry,
                model_alias=self.model_alias,
            )
            log = await runner.run(request)
            return [log]
        else:
            from benchmarks.serving_benchmark.runners.workflow_runner import (
                WorkflowRunner,
            )

            runner = WorkflowRunner(
                registry=self.registry,
                default_model=self.model_alias,
            )
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
        phase = "warmup" if is_warmup else "run"
        total_items = len(items)
        completed_count = 0

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

        for req in requests:
            await scheduler.enqueue(req)

        max_empty_polls = 10
        empty_count = 0
        while empty_count < max_empty_polls:
            batch = await scheduler.schedule()
            if not batch:
                empty_count += 1
                await asyncio.sleep(0.05)
                continue
            empty_count = 0
            for req in batch:
                now = time.time()
                if req.arrive_time > now:
                    await asyncio.sleep(req.arrive_time - now)

                req_t0 = time.time()
                logs = await self._run_single_request(req)
                all_logs.extend(logs)
                completed_count += 1

                req_elapsed = time.time() - req_t0
                sample = req.sample
                dataset = sample.dataset if sample else "?"
                status = logs[0].status if logs else "?"
                logger.info(
                    "[%s] %d/%d  dataset=%-16s status=%-9s  %.2fs",
                    phase, completed_count, total_items, dataset, status, req_elapsed,
                )

                await scheduler.on_complete(req.req_id, req)

        remaining = await scheduler.drain()
        if remaining:
            logger.info("[%s] Draining %d remaining requests ...", phase, len(remaining))
        for req in remaining:
            req_t0 = time.time()
            logs = await self._run_single_request(req)
            all_logs.extend(logs)
            completed_count += 1

            req_elapsed = time.time() - req_t0
            sample = req.sample
            dataset = sample.dataset if sample else "?"
            status = logs[0].status if logs else "?"
            logger.info(
                "[%s] %d/%d  dataset=%-16s status=%-9s  %.2fs",
                phase, completed_count, total_items, dataset, status, req_elapsed,
            )

        return all_logs

    async def run_once(self, seed: int):
        """Run one repetition of the experiment."""
        sched_cfg = self.baseline_cfg.get("scheduler", {})
        sched_name = sched_cfg.get("name", "fifo")
        sched_params = sched_cfg.get("params", {})
        scheduler = create_scheduler(sched_name, **sched_params)
        logger.info("Scheduler: %s  params=%s", sched_name, sched_params)

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

        logger.info("Loading sample pools ...")
        pools = self._load_sample_pools()
        pool_sizes = {k: len(v) for k, v in pools.items()}
        logger.info("Sample pools loaded: %s", pool_sizes)

        gen = WorkloadGenerator(wl_config, pools)
        warmup_items, run_items = gen.generate()
        logger.info(
            "Workload generated — warmup: %d items, run: %d items",
            len(warmup_items), len(run_items),
        )

        config_id = (
            f"{sched_name}_{wl_cfg.get('pattern', 'poisson')}"
            f"_{self.mode}_{self.model_alias}_{seed}"
        )

        model_cfg = self.registry.get(self.model_alias)
        exp_config = ExperimentConfig(
            config_id=config_id,
            baseline=sched_name,
            pattern=wl_cfg.get("pattern", "poisson"),
            suite=self.suite,
            mode=self.mode,
            model_size=model_cfg.size_tier,
            seed=seed,
            git_commit=_git_commit_hash(),
        )

        # GPU logger
        gpu_logger = None
        try:
            from benchmarks.serving_benchmark.core.gpu_logger import GPULogger

            out_dir = (
                self._make_output_dir()
                if not hasattr(self, "_out_dir")
                else self._out_dir
            )
            gpu_logger = GPULogger(
                output_dir=out_dir / "raw_logs",
                prefix=f"gpu_samples_{seed}",
            )
            gpu_logger.start()
            logger.info("GPU logger started")
        except Exception:
            logger.warning("GPU logger unavailable, skipping GPU sampling")

        # Warm-up
        logger.info("--- Warm-up phase (%d requests) ---", len(warmup_items))
        warmup_t0 = time.time()
        await self._execute_phase(
            warmup_items, scheduler, config_id, seed, is_warmup=True
        )
        logger.info("Warm-up done in %.1fs", time.time() - warmup_t0)

        # Fresh scheduler for sampling phase
        scheduler = create_scheduler(sched_name, **sched_params)
        logger.info("--- Sampling phase (%d requests) ---", len(run_items))
        run_t0 = time.time()
        logs = await self._execute_phase(run_items, scheduler, config_id, seed)
        logger.info("Sampling done in %.1fs — %d logs collected", time.time() - run_t0, len(logs))

        if gpu_logger is not None:
            try:
                gpu_logger.stop()
                logger.info("GPU logger stopped")
            except Exception:
                pass

        return exp_config, logs, gpu_logger

    async def run_all(self) -> Dict[str, Any]:
        """Run all repetitions and save outputs."""
        self._out_dir = self._make_output_dir()
        out_dir = self._out_dir
        all_results: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "configs": [],
            "logs_by_seed": {},
        }

        total_seeds = len(self.seeds)
        logger.info(
            "Experiment %s started — %d seed(s), output: %s",
            self.experiment_id, total_seeds, out_dir,
        )
        exp_t0 = time.time()

        for repeat_idx, seed in enumerate(self.seeds):
            seed_t0 = time.time()
            logger.info(
                "=== Repeat %d/%d  seed=%d ===", repeat_idx + 1, total_seeds, seed
            )

            exp_config, logs, gpu_logger = await self.run_once(seed)

            seed_elapsed = time.time() - seed_t0
            ok = sum(1 for l in logs if l.status == "completed")
            logger.info(
                "Repeat %d/%d done — %d logs (%d ok) in %.1fs",
                repeat_idx + 1, total_seeds, len(logs), ok, seed_elapsed,
            )

            exp_config.repeat = repeat_idx
            all_results["configs"].append(asdict(exp_config))
            all_results["logs_by_seed"][seed] = [log.to_dict() for log in logs]

            cfg_path = out_dir / "configs" / f"config_{seed}.json"
            with open(cfg_path, "w") as f:
                json.dump(asdict(exp_config), f, indent=2)

            if gpu_logger is not None:
                try:
                    gpu_df = gpu_logger.get_dataframe()
                    if gpu_df is not None and not gpu_df.empty:
                        gpu_df.to_csv(
                            out_dir / "raw_logs" / f"gpu_samples_{seed}.csv",
                            index=False,
                        )
                        try:
                            gpu_df.to_parquet(
                                out_dir / "raw_logs" / f"gpu_samples_{seed}.parquet",
                                index=False,
                            )
                        except Exception:
                            pass
                except Exception:
                    pass

        self._save_logs(out_dir, all_results["logs_by_seed"])

        total_elapsed = time.time() - exp_t0
        total_logs = sum(len(v) for v in all_results["logs_by_seed"].values())
        logger.info(
            "Experiment %s finished — %d total logs, %.1fs elapsed",
            self.experiment_id, total_logs, total_elapsed,
        )

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
                pass

        except ImportError:
            json_path = out_dir / "raw_logs" / "request_logs.json"
            all_rows = []
            for seed, rows in logs_by_seed.items():
                all_rows.extend(rows)
            with open(json_path, "w") as f:
                json.dump(all_rows, f, indent=2)
