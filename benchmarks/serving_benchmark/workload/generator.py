"""Workload generator — combines arrival process + suite mixing.

Supports warm-up phase (5 min or 200 requests, whichever comes first)
and sampling phase (30 min or 10k requests, whichever comes first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkSample, Suite
from benchmarks.serving_benchmark.workload.arrival import (
    ARRIVAL_REGISTRY,
    ArrivalProcess,
)
from benchmarks.serving_benchmark.workload.mixing import (
    DEFAULT_RATIO,
    SuiteMixer,
    WorkloadItem,
)


@dataclass
class WorkloadConfig:
    """Parameters for one workload generation."""

    pattern: str = "poisson"  # poisson | onoff | diurnal | longtail
    suite_ratio: Dict[str, float] = None  # type: ignore[assignment]
    seed: int = 42

    # Arrival params (passed to constructor)
    arrival_kwargs: Dict = None  # type: ignore[assignment]

    # Warm-up
    warmup_duration_s: float = 300.0  # 5 minutes
    warmup_max_requests: int = 200

    # Sampling
    run_duration_s: float = 1800.0  # 30 minutes
    run_max_requests: int = 10_000

    def __post_init__(self):
        if self.suite_ratio is None:
            self.suite_ratio = dict(DEFAULT_RATIO)
        if self.arrival_kwargs is None:
            self.arrival_kwargs = {}


class WorkloadGenerator:
    """Generates a full workload (warm-up + run) from config."""

    def __init__(
        self,
        config: WorkloadConfig,
        pools: Dict[str, List[BenchmarkSample]],
    ):
        self.config = config

        arrival_cls = ARRIVAL_REGISTRY.get(config.pattern)
        if arrival_cls is None:
            raise ValueError(
                f"Unknown arrival pattern: {config.pattern}. "
                f"Available: {list(ARRIVAL_REGISTRY)}"
            )
        self.arrival: ArrivalProcess = arrival_cls(**config.arrival_kwargs)
        self.mixer = SuiteMixer(pools, config.suite_ratio)

    def generate_warmup(self) -> List[WorkloadItem]:
        """Generate warm-up traffic."""
        cfg = self.config
        all_arrivals = self.arrival.generate(cfg.warmup_duration_s, seed=cfg.seed)
        all_arrivals = all_arrivals[: cfg.warmup_max_requests]
        return self.mixer.mix(all_arrivals, seed=cfg.seed)

    def generate_run(self) -> List[WorkloadItem]:
        """Generate the main sampling traffic."""
        cfg = self.config
        # Use a different seed offset so run != warmup
        run_seed = cfg.seed + 1000
        all_arrivals = self.arrival.generate(cfg.run_duration_s, seed=run_seed)
        all_arrivals = all_arrivals[: cfg.run_max_requests]
        return self.mixer.mix(all_arrivals, seed=run_seed)

    def generate(self) -> tuple[List[WorkloadItem], List[WorkloadItem]]:
        """Return (warmup_items, run_items)."""
        return self.generate_warmup(), self.generate_run()
