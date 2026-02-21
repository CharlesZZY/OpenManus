"""FIFO scheduler — first-in, first-out. No parameters."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest
from benchmarks.serving_benchmark.schedulers.base import BaseScheduler


class FIFOScheduler(BaseScheduler):
    name = "fifo"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._queue: asyncio.Queue[BenchmarkRequest] = asyncio.Queue()

    async def enqueue(self, request: BenchmarkRequest) -> None:
        await self._queue.put(request)

    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        if self._queue.empty():
            return None
        req = self._queue.get_nowait()
        return [req]
