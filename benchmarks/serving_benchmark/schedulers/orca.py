"""ORCA-like iteration-level scheduling.

At each scheduling iteration, re-selects active requests based on their
progress and priority.  Selective batching = ON by default.

iter_quantum=1, selective_batching=on.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest
from benchmarks.serving_benchmark.schedulers.base import BaseScheduler


class ORCAScheduler(BaseScheduler):
    name = "orca"

    def __init__(
        self,
        iter_quantum: int = 1,
        selective_batching: bool = True,
        max_batch: int = 64,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.iter_quantum = iter_quantum
        self.selective_batching = selective_batching
        self.max_batch = max_batch

        self._waiting: deque[BenchmarkRequest] = deque()
        self._active: Dict[str, BenchmarkRequest] = {}

    async def enqueue(self, request: BenchmarkRequest) -> None:
        self._waiting.append(request)

    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        # Fill active set up to max_batch
        while self._waiting and len(self._active) < self.max_batch:
            req = self._waiting.popleft()
            self._active[req.req_id] = req

        if not self._active:
            return None

        if self.selective_batching:
            # Prefer shorter requests (by predicted size)
            sorted_reqs = sorted(
                self._active.values(),
                key=lambda r: r.predicted_in_tokens + (r.predicted_out_tokens or 256),
            )
            batch = sorted_reqs[: self.max_batch]
        else:
            batch = list(self._active.values())[: self.max_batch]

        # Remove scheduled from active (they'll be re-added if preempted)
        for req in batch:
            self._active.pop(req.req_id, None)

        return batch

    async def on_complete(self, req_id: str, result: Any) -> None:
        await super().on_complete(req_id, result)
        self._active.pop(req_id, None)
