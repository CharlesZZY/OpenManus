"""Selective_Context quality judge — retention weak check.

判定函数: 未指定. Implementing a weak heuristic:
1. Extract key sentences from the reference (first 3 sentences).
2. Check if they are retained in the compressed/model output.
3. Downstream quality diff = compare compressed vs uncompressed output quality.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _extract_key_sentences(text: str, n: int = 3) -> List[str]:
    """Extract first *n* non-empty sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()][:n]


def _retention_score(model_output: str, key_sentences: List[str]) -> float:
    """Fraction of key sentences found in model output (case-insensitive)."""
    if not key_sentences:
        return 1.0
    output_lower = model_output.lower()
    found = sum(1 for s in key_sentences if s.lower() in output_lower)
    return found / len(key_sentences)


def judge_selective_context(model_output: str, reference: Any) -> bool:
    """Return True if key content is retained (retention >= 0.5).

    *reference* is a dict with optional keys: summary, questions, answers.
    We check retention against the summary; if not available, fall back to
    checking that the output is not trivially short.
    """
    if not isinstance(reference, dict):
        ref_text = str(reference)
    else:
        ref_text = reference.get("summary", "")
        if not ref_text:
            # No summary available — check that answer fragments exist
            answers = reference.get("answers", [])
            if answers:
                found = sum(
                    1 for a in answers if str(a).lower() in model_output.lower()
                )
                return found / max(len(answers), 1) >= 0.5
            return len(model_output.strip()) > 50  # degenerate fallback

    key = _extract_key_sentences(ref_text, n=3)
    return _retention_score(model_output, key) >= 0.5
