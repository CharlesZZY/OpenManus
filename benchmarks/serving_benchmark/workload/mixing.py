"""Suite mixing — assigns each arrival to a suite and dataset sample.

Default ratio: S:R:L = 4:3:3.
Within each suite, samples are drawn uniformly at random from the datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from benchmarks.serving_benchmark.core.schema import (
    BenchmarkRequest,
    BenchmarkSample,
    Suite,
)

DEFAULT_RATIO = {"S": 0.4, "R": 0.3, "L": 0.3}


@dataclass
class WorkloadItem:
    """A scheduled arrival with assigned sample."""

    arrive_time: float = 0.0
    sample: Optional[BenchmarkSample] = None
    suite: str = ""


class SuiteMixer:
    """Assigns each arrival timestamp to a suite and sample."""

    def __init__(
        self,
        pools: Dict[str, List[BenchmarkSample]],
        ratio: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        pools : mapping Suite.value -> list of BenchmarkSample
        ratio : mixing weights (normalised internally)
        """
        self.pools = pools
        raw = ratio or DEFAULT_RATIO
        total = sum(raw.values())
        self.ratio = {k: v / total for k, v in raw.items()}

    def mix(
        self,
        arrivals: List[float],
        seed: int = 42,
    ) -> List[WorkloadItem]:
        """Assign suites and samples to arrival times."""
        rng = np.random.default_rng(seed)
        suites = list(self.ratio.keys())
        probs = [self.ratio[s] for s in suites]

        items: List[WorkloadItem] = []
        for t in arrivals:
            suite_key = rng.choice(suites, p=probs)
            pool = self.pools.get(suite_key, [])
            if pool:
                sample = pool[rng.integers(0, len(pool))]
            else:
                sample = BenchmarkSample(dataset="unknown", suite=suite_key)
            items.append(WorkloadItem(arrive_time=t, sample=sample, suite=suite_key))

        return items
