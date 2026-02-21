"""Static batching scheduler.

Collects up to *batch_size* requests or flushes after *flush_interval_ms*.
Defaults: batch_size=16, flush_interval=50ms.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest
from benchmarks.serving_benchmark.schedulers.base import BaseScheduler


class StaticBatchScheduler(BaseScheduler):
    name = "static_batch"

    def __init__(
        self,
        batch_size: int = 16,
        flush_interval_ms: float = 50.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_ms / 1000.0
        self._buffer: List[BenchmarkRequest] = []
        self._last_flush: float = time.monotonic()

    async def enqueue(self, request: BenchmarkRequest) -> None:
        self._buffer.append(request)

    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        now = time.monotonic()
        elapsed = now - self._last_flush

        if len(self._buffer) >= self.batch_size or (
            self._buffer and elapsed >= self.flush_interval_s
        ):
            batch = self._buffer[: self.batch_size]
            self._buffer = self._buffer[self.batch_size :]
            self._last_flush = now
            return batch

        return None

    async def drain(self) -> List[BenchmarkRequest]:
        """Force-flush remaining buffer."""
        remaining = list(self._buffer)
        self._buffer.clear()
        return remaining
