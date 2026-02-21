"""Size-based / Shortest-Remaining-First scheduler.

Priority = in_tokens + predicted_out_tokens (ascending).
predicted_out_tokens: 未指定方法, using historical average / upper-bound fallback.
"""

from __future__ import annotations

import heapq
from typing import Any, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest
from benchmarks.serving_benchmark.schedulers.base import BaseScheduler


class SRFScheduler(BaseScheduler):
    name = "srf"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._heap: List[tuple[int, int, BenchmarkRequest]] = []
        self._counter = 0

    async def enqueue(self, request: BenchmarkRequest) -> None:
        size = request.predicted_in_tokens + (
            request.predicted_out_tokens or self.predicted_out_tokens
        )
        heapq.heappush(self._heap, (size, self._counter, request))
        self._counter += 1

    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        if not self._heap:
            return None
        _, _, req = heapq.heappop(self._heap)
        return [req]
