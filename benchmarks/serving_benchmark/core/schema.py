"""Unified log field definitions and metric specifications.

Defines the canonical schema for request-level logs and GPU sampling logs.
All downstream modules (loggers, metrics, aggregators) import from here.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RequestMode(str, Enum):
    SINGLE = "single"
    WORKFLOW = "workflow"


class RequestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Suite(str, Enum):
    S = "S"  # Short: MMLU + TruthfulQA
    R = "R"  # Reasoning: GSM8K + HumanEval
    L = "L"  # Long: LongBench v2 + Selective_Context


SUITE_DATASETS = {
    Suite.S: ["mmlu", "truthfulqa"],
    Suite.R: ["gsm8k", "humaneval"],
    Suite.L: ["longbench_v2", "selective_context"],
}


# ---------------------------------------------------------------------------
# Request-level log record
# ---------------------------------------------------------------------------


@dataclass
class RequestLog:
    """One row per LLM request (or per workflow step)."""

    req_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    dataset: str = ""
    suite: str = ""
    mode: str = RequestMode.SINGLE.value  # single | workflow
    workflow_id: str = ""
    step_id: str = ""

    # Timestamps (epoch seconds, float for sub-ms precision)
    t_arrive: float = 0.0
    t_enqueue: float = 0.0
    t_schedule: float = 0.0
    t_first_token: float = 0.0
    t_last_token: float = 0.0
    t_finish: float = 0.0

    # Token counts
    in_tokens: int = 0
    out_tokens: int = 0

    # Status & quality
    status: str = RequestStatus.PENDING.value
    quality_ok: bool = False

    # Experiment identifiers
    config_id: str = ""
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# GPU sampling log record
# ---------------------------------------------------------------------------


@dataclass
class GPULog:
    """One row per GPU sample."""

    ts: float = 0.0  # epoch seconds
    gpu_util: float = 0.0  # 0-100 percent
    gpu_mem_used: float = 0.0  # MiB

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def columns(cls) -> List[str]:
        return [f.name for f in fields(cls)]


# ---------------------------------------------------------------------------
# Benchmark sample (unified across datasets)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSample:
    """A single evaluation sample from any dataset."""

    id: str = ""
    dataset: str = ""
    suite: str = ""
    raw_prompt: str = ""  # Original prompt — NEVER modified
    reference: Any = None  # Ground truth (type varies by dataset)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark request (enriched sample ready for scheduling)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRequest:
    """A request ready to be enqueued into a scheduler."""

    req_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    sample: Optional[BenchmarkSample] = None
    mode: str = RequestMode.SINGLE.value
    workflow_id: str = ""
    step_id: str = ""
    config_id: str = ""
    seed: int = 0

    # Scheduling metadata
    arrive_time: float = 0.0
    predicted_in_tokens: int = 0
    predicted_out_tokens: int = 0
    slo_ttft: float = 0.5  # seconds — default, configurable
    slo_tpot: float = 0.1  # seconds — default, configurable
    deadline: float = 0.0  # absolute epoch deadline (computed)

    # Result (filled after completion)
    log: Optional[RequestLog] = None
    response_text: str = ""


# ---------------------------------------------------------------------------
# SLO configuration
# ---------------------------------------------------------------------------


@dataclass
class SLOConfig:
    """Service Level Objective thresholds (defaults are placeholders — 未指定)."""

    ttft_threshold: float = 0.5  # seconds
    tpot_threshold: float = 0.1  # seconds
    e2e_threshold: float = 30.0  # seconds
    qoe_ttft_good: float = 0.3  # below this → QoE=1.0
    qoe_ttft_bad: float = 2.0  # above this → QoE=0.0
    qoe_e2e_good: float = 5.0
    qoe_e2e_bad: float = 60.0


# ---------------------------------------------------------------------------
# Experiment configuration (serialisable snapshot)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Immutable snapshot of one experiment run."""

    config_id: str = ""
    baseline: str = ""
    pattern: str = ""  # poisson | onoff | diurnal | longtail
    suite: str = ""  # S | R | L | mixed
    mode: str = ""  # single | workflow
    model_size: str = ""  # small | medium | large (未指定阈值)
    deployment: str = "single_node"  # single_node | multi_node (未指定)
    seed: int = 42
    repeat: int = 0  # which repeat (0-indexed)
    warmup_duration_s: float = 300.0
    warmup_max_requests: int = 200
    run_duration_s: float = 1800.0
    run_max_requests: int = 10000
    git_commit: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
