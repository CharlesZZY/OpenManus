"""GSM8K quality judge — numerical accuracy.

Parsing rule (稳健解析):
1. First try to find ``#### <number>`` pattern (GSM8K convention).
2. Fall back to the last number in the text.
3. Compare as floats with tolerance 1e-3.

数值解析规则: 未指定, using robust regex approach.
"""

from __future__ import annotations

import re
from typing import Any

_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_LAST_NUMBER = re.compile(r"([-+]?\d[\d,]*\.?\d*)")


def _extract_number(text: str) -> float | None:
    """Extract the answer number from text."""
    m = _ANSWER_PATTERN.search(text)
    if m:
        return float(m.group(1).replace(",", ""))

    numbers = _LAST_NUMBER.findall(text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None


def judge_gsm8k(model_output: str, reference: Any) -> bool:
    """Return True if the model's final answer matches the reference numerically."""
    pred = _extract_number(model_output)
    if pred is None:
        return False

    ref_num = _extract_number(str(reference))
    if ref_num is None:
        return False

    return abs(pred - ref_num) < 1e-3
