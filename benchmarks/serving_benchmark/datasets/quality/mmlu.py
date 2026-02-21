"""MMLU quality judge — multiple-choice accuracy.

Parsing: extract A/B/C/D (or 0/1/2/3) from model output.
未指定 exact format — we try several heuristics.
"""

from __future__ import annotations

import re
from typing import Any

_LETTER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
_LETTER_PATTERN = re.compile(r"\b([A-D])\b")
_DIGIT_PATTERN = re.compile(r"\b([0-3])\b")


def _extract_choice(text: str) -> int | None:
    """Extract the chosen option index (0-3) from model output."""
    text_upper = text.strip().upper()

    # 1. Single letter answer (e.g., "A" or "The answer is B")
    if len(text_upper) == 1 and text_upper in _LETTER_MAP:
        return _LETTER_MAP[text_upper]

    # 2. Pattern like "answer is X" or "Answer: X"
    answer_match = re.search(
        r"(?:answer|choice|option)\s*(?:is|:)\s*([A-D])", text_upper
    )
    if answer_match:
        return _LETTER_MAP[answer_match.group(1)]

    # 3. First standalone letter A-D
    m = _LETTER_PATTERN.search(text_upper)
    if m:
        return _LETTER_MAP[m.group(1)]

    # 4. Digit 0-3
    m = _DIGIT_PATTERN.search(text.strip())
    if m:
        return int(m.group(1))

    return None


def judge_mmlu(model_output: str, reference: Any) -> bool:
    """Return True if extracted choice matches reference (0-3)."""
    pred = _extract_choice(model_output)
    if pred is None:
        return False
    try:
        ref = int(reference)
    except (ValueError, TypeError):
        return False
    return pred == ref
