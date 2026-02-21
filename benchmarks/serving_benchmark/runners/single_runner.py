"""Single-model runner — direct LLM.ask() without OpenManus workflow.

System prompt contains ONLY output-format constraints; user content is the
dataset's original prompt verbatim (never rewritten).
"""

from __future__ import annotations

import time
from typing import Optional

from benchmarks.serving_benchmark.core.schema import (
    BenchmarkRequest,
    BenchmarkSample,
    RequestLog,
    RequestMode,
    RequestStatus,
)
from benchmarks.serving_benchmark.datasets.quality import judge

# Format-constraint system prompts per dataset (NOT altering the user prompt)
SYSTEM_PROMPTS = {
    "mmlu": "You are answering a multiple-choice question. Output ONLY the letter of your answer (A, B, C, or D).",
    "truthfulqa": "You are answering a multiple-choice question. Output ONLY the letter of your answer.",
    "gsm8k": "Solve the math problem step by step. End your answer with '#### <number>' where <number> is the final numerical answer.",
    "humaneval": "Complete the Python function. Output ONLY the code, no explanations.",
    "longbench_v2": "Answer the question based on the context. Output ONLY the letter of your answer.",
    "selective_context": "Summarise or answer based on the provided text. Be concise and accurate.",
}

DEFAULT_SYSTEM = "Answer the question accurately and concisely."


class SingleRunner:
    """Runs a single request directly against the LLM API."""

    def __init__(self, llm=None):
        """
        Parameters
        ----------
        llm : app.llm.LLM instance (lazy-imported to avoid config dependency at import time)
        """
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from app.llm import LLM

            self._llm = LLM()
        return self._llm

    async def run(self, request: BenchmarkRequest) -> RequestLog:
        """Execute a single LLM request and return a populated RequestLog."""
        llm = self._get_llm()
        sample: BenchmarkSample = request.sample or BenchmarkSample()
        log = RequestLog(
            req_id=request.req_id,
            dataset=sample.dataset,
            suite=sample.suite,
            mode=RequestMode.SINGLE.value,
            workflow_id="",
            step_id="",
            config_id=request.config_id,
            seed=request.seed,
        )

        log.t_arrive = request.arrive_time or time.time()
        log.t_enqueue = time.time()

        sys_prompt = SYSTEM_PROMPTS.get(sample.dataset, DEFAULT_SYSTEM)

        from app.schema import Message

        system_msgs = [Message.system_message(sys_prompt)]
        user_msgs = [Message.user_message(sample.raw_prompt)]

        log.t_schedule = time.time()

        try:
            # Count input tokens
            formatted = llm.format_messages(user_msgs)
            log.in_tokens = llm.count_message_tokens(
                llm.format_messages(system_msgs) + formatted
            )

            # Stream to capture TTFT
            first_token_seen = False
            collected: list[str] = []

            params = {
                "model": llm.model,
                "messages": llm.format_messages(system_msgs) + formatted,
                "stream": True,
            }
            if hasattr(llm, "max_tokens"):
                params["max_tokens"] = llm.max_tokens
            if hasattr(llm, "temperature"):
                params["temperature"] = llm.temperature

            response = await llm.client.chat.completions.create(**params)

            async for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta and not first_token_seen:
                    log.t_first_token = time.time()
                    first_token_seen = True
                collected.append(delta)

            log.t_last_token = time.time()
            full_response = "".join(collected).strip()
            request.response_text = full_response

            log.out_tokens = llm.count_tokens(full_response)
            log.t_finish = time.time()
            log.status = RequestStatus.COMPLETED.value

            # Quality judgement
            log.quality_ok = judge(sample.dataset, full_response, sample.reference)

        except Exception as e:
            log.t_finish = time.time()
            log.status = RequestStatus.FAILED.value
            log.quality_ok = False

        request.log = log
        return log
