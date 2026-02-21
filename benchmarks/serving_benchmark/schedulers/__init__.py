"""Scheduler registry."""

from benchmarks.serving_benchmark.schedulers.base import BaseScheduler
from benchmarks.serving_benchmark.schedulers.continuous_batch import (
    ContinuousBatchScheduler,
)
from benchmarks.serving_benchmark.schedulers.fifo import FIFOScheduler
from benchmarks.serving_benchmark.schedulers.latency_aware import LatencyAwareScheduler
from benchmarks.serving_benchmark.schedulers.orca import ORCAScheduler
from benchmarks.serving_benchmark.schedulers.parrot import ParrotScheduler
from benchmarks.serving_benchmark.schedulers.srf import SRFScheduler
from benchmarks.serving_benchmark.schedulers.static_batch import StaticBatchScheduler

SCHEDULER_REGISTRY = {
    "fifo": FIFOScheduler,
    "srf": SRFScheduler,
    "static_batch": StaticBatchScheduler,
    "continuous_batch": ContinuousBatchScheduler,
    "orca": ORCAScheduler,
    "latency_aware": LatencyAwareScheduler,
    "parrot": ParrotScheduler,
}


def create_scheduler(name: str, **kwargs) -> BaseScheduler:
    cls = SCHEDULER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: {list(SCHEDULER_REGISTRY)}"
        )
    return cls(**kwargs)
