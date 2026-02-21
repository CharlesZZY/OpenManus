"""Parrot-style workflow-aware scheduler.

Features:
- Groups requests by workflow_id and respects step dependencies.
- Semantic cache (prompt hash) with configurable TTL.
- Reports cache hit rate and reuse gain.

Cache TTL: 300s. Hash method: SHA256(prompt[:512]) — 未指定, using default.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.serving_benchmark.core.schema import BenchmarkRequest
from benchmarks.serving_benchmark.schedulers.base import BaseScheduler


class _CacheEntry:
    __slots__ = ("response", "created_at")

    def __init__(self, response: str, created_at: float):
        self.response = response
        self.created_at = created_at


class ParrotScheduler(BaseScheduler):
    name = "parrot"

    def __init__(self, cache_ttl: float = 300.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.cache_ttl = cache_ttl

        self._waiting: deque[BenchmarkRequest] = deque()
        self._workflow_groups: Dict[str, List[BenchmarkRequest]] = defaultdict(list)
        self._cache: Dict[str, _CacheEntry] = {}

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.reuse_gain_tokens = 0

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt[:512].encode("utf-8")).hexdigest()

    def _cache_lookup(self, prompt: str) -> Optional[str]:
        key = self._prompt_hash(prompt)
        entry = self._cache.get(key)
        if entry is None:
            self.cache_misses += 1
            return None
        if time.time() - entry.created_at > self.cache_ttl:
            del self._cache[key]
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        return entry.response

    def cache_store(self, prompt: str, response: str) -> None:
        key = self._prompt_hash(prompt)
        self._cache[key] = _CacheEntry(response, time.time())

    # ------------------------------------------------------------------
    # Scheduler interface
    # ------------------------------------------------------------------

    async def enqueue(self, request: BenchmarkRequest) -> None:
        if request.workflow_id:
            self._workflow_groups[request.workflow_id].append(request)
        else:
            self._waiting.append(request)

    async def schedule(self) -> Optional[List[BenchmarkRequest]]:
        # 1. Prioritise workflow groups with satisfied dependencies
        for wf_id, reqs in list(self._workflow_groups.items()):
            if not reqs:
                del self._workflow_groups[wf_id]
                continue
            # Sort by step_id to respect ordering
            reqs.sort(key=lambda r: r.step_id)
            next_req = reqs[0]

            # Check cache first
            prompt = next_req.sample.raw_prompt if next_req.sample else ""
            cached = self._cache_lookup(prompt)
            if cached is not None:
                next_req.response_text = cached
                self.reuse_gain_tokens += len(cached.split())
                reqs.pop(0)
                return [next_req]

            reqs.pop(0)
            return [next_req]

        # 2. Fall back to non-workflow waiting queue
        if self._waiting:
            req = self._waiting.popleft()
            prompt = req.sample.raw_prompt if req.sample else ""
            cached = self._cache_lookup(prompt)
            if cached is not None:
                req.response_text = cached
                self.reuse_gain_tokens += len(cached.split())
            return [req]

        return None

    async def on_complete(self, req_id: str, result: Any) -> None:
        await super().on_complete(req_id, result)

    def get_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(total, 1),
            "reuse_gain_tokens": self.reuse_gain_tokens,
        }
