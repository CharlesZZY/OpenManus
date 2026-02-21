"""HumanEval quality judge — pass@k (k=1 default).

Executes generated code in a sandboxed exec() with the dataset's test harness.
k>1 fan-out: 未指定, not enabled.
"""

from __future__ import annotations

import contextlib
import io
import signal
import traceback
from typing import Any, Dict

EXEC_TIMEOUT = 10  # seconds


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def judge_humaneval(model_output: str, reference: Any) -> bool:
    """Return True if generated code passes the test cases.

    ``reference`` is a dict with keys: canonical_solution, test, entry_point, task_id.
    """
    if not isinstance(reference, dict):
        return False

    test_code = reference.get("test", "")
    entry_point = reference.get("entry_point", "")

    if not test_code or not entry_point:
        return False

    full_code = model_output + "\n" + test_code + f"\ncheck({entry_point})\n"

    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EXEC_TIMEOUT)
        try:
            exec_globals: Dict[str, Any] = {}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                exec(full_code, exec_globals)  # noqa: S102
            return True
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except Exception:
        return False
