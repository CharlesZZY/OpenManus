"""Single-model runner — direct vLLM call without OpenManus workflow.

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
from benchmarks.serving_benchmark.core.vllm_client import ModelRegistry, VLLMClient
from benchmarks.serving_benchmark.datasets.quality import judge

SYSTEM_PROMPTS = {
    "mmlu": (
        "You are answering a multiple-choice question. "
        "Output ONLY the letter of your answer (A, B, C, or D)."
    ),
    "truthfulqa": (
        "You are answering a multiple-choice question. "
        "Output ONLY the letter of your answer."
    ),
    "gsm8k": (
        "Solve the math problem step by step. "
        "End your answer with '#### <number>' where <number> is the final numerical answer."
    ),
    "humaneval": "Complete the Python function. Output ONLY the code, no explanations.",
    "longbench_v2": (
        "Answer the question based on the context. "
        "Output ONLY the letter of your answer."
    ),
    "selective_context": (
        "Summarise or answer based on the provided text. " "Be concise and accurate."
    ),
}

DEFAULT_SYSTEM = "Answer the question accurately and concisely."


class SingleRunner:
    """Runs a single request directly against a vLLM server."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        model_alias: Optional[str] = None,
    ):
        self._registry = registry
        self._model_alias = model_alias
        self._client: Optional[VLLMClient] = None

    def _ensure_client(self) -> VLLMClient:
        if self._client is not None:
            return self._client
        if self._registry is None:
            self._registry = ModelRegistry()
        alias = self._model_alias or self._registry.first_model
        self._model_alias = alias
        cfg = self._registry.get(alias)
        self._client = VLLMClient(cfg)
        return self._client

    async def run(self, request: BenchmarkRequest) -> RequestLog:
        """Execute a single LLM request and return a populated RequestLog."""
        client = self._ensure_client()
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
            model_id=self._model_alias or "",
        )

        log.t_arrive = request.arrive_time or time.time()
        log.t_enqueue = time.time()

        sys_prompt = SYSTEM_PROMPTS.get(sample.dataset, DEFAULT_SYSTEM)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": sample.raw_prompt},
        ]

        log.t_schedule = time.time()
        log.in_tokens = client.count_messages_tokens(messages)

        try:
            first_token_seen = False
            collected: list[str] = []

            async for delta in client.chat_stream(messages):
                if not first_token_seen:
                    log.t_first_token = time.time()
                    first_token_seen = True
                collected.append(delta)

            log.t_last_token = time.time()
            full_response = "".join(collected).strip()
            request.response_text = full_response

            log.out_tokens = client.count_tokens(full_response)
            log.t_finish = time.time()
            log.status = RequestStatus.COMPLETED.value

            log.quality_ok = judge(sample.dataset, full_response, sample.reference)

        except Exception:
            log.t_finish = time.time()
            log.status = RequestStatus.FAILED.value
            log.quality_ok = False

        request.log = log
        return log
