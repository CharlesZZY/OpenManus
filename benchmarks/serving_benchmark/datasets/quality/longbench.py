"""LongBench v2 quality judge — multiple-choice accuracy.

Extracts a letter option from the model output and compares with the reference.
"""

from __future__ import annotations

import re
from typing import Any

_LETTER_PATTERN = re.compile(r"\b([A-F])\b")


def _extract_option(text: str) -> str | None:
    text_upper = text.strip().upper()

    # "answer is X"
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:)\s*([A-F])", text_upper)
    if m:
        return m.group(1)

    # Single character
    if len(text_upper) == 1 and text_upper in "ABCDEF":
        return text_upper

    # First standalone letter
    m = _LETTER_PATTERN.search(text_upper)
    if m:
        return m.group(1)

    return None


def judge_longbench(model_output: str, reference: Any) -> bool:
    """Return True if extracted option matches reference answer."""
    pred = _extract_option(model_output)
    if pred is None:
        return False

    ref_str = str(reference).strip().upper()

    # Direct letter comparison
    if ref_str in "ABCDEF" and pred == ref_str:
        return True

    # Reference might be the full text — check first letter
    if ref_str and pred == ref_str[0]:
        return True

    return False
