"""Base scheduler interface.

All schedulers operate at the **application layer** between the workload
generator and the LLM API.  They do NOT modify the underlying inference engine.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest


class BaseScheduler(ABC):
    """Abstract scheduler interface."""

    name: str = "base"

    # Running statistics (updated by on_complete)
    _history_out_tokens: List[int] = []
    _completed_count: int = 0

    def __init__(self, **kwargs: Any):
        self._history_out_tokens = []
        self._completed_count = 0
        self._extra = kwargs

    @abstractmethod
    async def enqueue(self, request: BenchmarkRequest) -> None:
        """Accept a new request into the scheduling queue."""

    @abstractmethod
    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        """Return the next batch of requests to execute, or None if empty."""

    async def on_complete(self, req_id: str, result: Any) -> None:
        """Callback when a request finishes (for bookkeeping / adaptive strategies)."""
        self._completed_count += 1
        if hasattr(result, "out_tokens"):
            self._history_out_tokens.append(result.out_tokens)

    @property
    def predicted_out_tokens(self) -> int:
        """Predicted output tokens based on history (cold start = max_tokens/2)."""
        if not self._history_out_tokens:
            return 256  # conservative default (未指定)
        return int(sum(self._history_out_tokens) / len(self._history_out_tokens))

    async def drain(self) -> List[BenchmarkRequest]:
        """Drain all remaining requests (used at shutdown)."""
        all_reqs: List[BenchmarkRequest] = []
        while True:
            batch = await self.schedule()
            if not batch:
                break
            all_reqs.extend(batch)
        return all_reqs
