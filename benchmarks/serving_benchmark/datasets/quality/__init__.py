"""Quality judges for each dataset.

Usage::

    from benchmarks.serving_benchmark.datasets.quality import judge
    ok = judge("gsm8k", model_output, reference)
"""

from __future__ import annotations

from typing import Any

from benchmarks.serving_benchmark.datasets.quality.gsm8k import judge_gsm8k
from benchmarks.serving_benchmark.datasets.quality.humaneval import judge_humaneval
from benchmarks.serving_benchmark.datasets.quality.longbench import judge_longbench
from benchmarks.serving_benchmark.datasets.quality.mmlu import judge_mmlu
from benchmarks.serving_benchmark.datasets.quality.selective_context import (
    judge_selective_context,
)
from benchmarks.serving_benchmark.datasets.quality.truthfulqa import judge_truthfulqa

_JUDGES = {
    "humaneval": judge_humaneval,
    "gsm8k": judge_gsm8k,
    "mmlu": judge_mmlu,
    "truthfulqa": judge_truthfulqa,
    "longbench_v2": judge_longbench,
    "selective_context": judge_selective_context,
}


def judge(dataset: str, model_output: str, reference: Any) -> bool:
    """Dispatch to the appropriate quality judge."""
    fn = _JUDGES.get(dataset)
    if fn is None:
        return False
    return fn(model_output, reference)
