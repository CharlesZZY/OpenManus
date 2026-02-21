"""Latency-aware Earliest-Deadline-First (EDF) scheduler.

Sorts requests by remaining slack to their SLO deadline.

SLO thresholds 未指定, defaults:
  TTFT <= 0.5s, TPOT <= 0.1s  (configurable, flagged in report).
"""

from __future__ import annotations

import heapq
import time
from typing import Any, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest
from benchmarks.serving_benchmark.schedulers.base import BaseScheduler


class LatencyAwareScheduler(BaseScheduler):
    name = "latency_aware"

    def __init__(
        self,
        ttft_slo: float = 0.5,
        tpot_slo: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self._heap: List[tuple[float, int, BenchmarkRequest]] = []
        self._counter = 0

    def _compute_deadline(self, request: BenchmarkRequest) -> float:
        """Absolute deadline = arrive_time + TTFT_SLO + predicted_out * TPOT_SLO."""
        pred_out = request.predicted_out_tokens or self.predicted_out_tokens
        return request.arrive_time + self.ttft_slo + pred_out * self.tpot_slo

    async def enqueue(self, request: BenchmarkRequest) -> None:
        deadline = self._compute_deadline(request)
        request.deadline = deadline
        heapq.heappush(self._heap, (deadline, self._counter, request))
        self._counter += 1

    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        if not self._heap:
            return None
        _, _, req = heapq.heappop(self._heap)
        return [req]
