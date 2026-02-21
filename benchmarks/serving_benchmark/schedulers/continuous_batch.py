"""vLLM-style continuous batching scheduler.

Simulates iteration-level insertion: new requests may join the running batch
at any iteration boundary as long as *max_num_seqs* is not exceeded.

max_num_seqs: 未指定, default 64.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest
from benchmarks.serving_benchmark.schedulers.base import BaseScheduler


class ContinuousBatchScheduler(BaseScheduler):
    name = "continuous_batch"

    def __init__(self, max_num_seqs: int = 64, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_num_seqs = max_num_seqs
        self._waiting: deque[BenchmarkRequest] = deque()
        self._running_count = 0

    async def enqueue(self, request: BenchmarkRequest) -> None:
        self._waiting.append(request)

    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        available_slots = self.max_num_seqs - self._running_count
        if available_slots <= 0 or not self._waiting:
            return None

        batch: List[BenchmarkRequest] = []
        while self._waiting and len(batch) < available_slots:
            batch.append(self._waiting.popleft())

        self._running_count += len(batch)
        return batch

    async def on_complete(self, req_id: str, result: Any) -> None:
        await super().on_complete(req_id, result)
        self._running_count = max(0, self._running_count - 1)
