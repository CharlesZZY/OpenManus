"""Workflow runner — uses OpenManus PlanningFlow for 3-step workflow.

Steps:
  1. Analyse the problem (analysis agent)
  2. Reason / retrieve (reasoning agent)
  3. Generate final answer (output agent)

Each step logs a separate RequestLog with step_id, sharing the same workflow_id.
The original prompt is embedded in each step but NEVER rewritten.
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

from benchmarks.serving_benchmark.core.schema import (
    BenchmarkRequest,
    BenchmarkSample,
    RequestLog,
    RequestMode,
    RequestStatus,
)
from benchmarks.serving_benchmark.datasets.quality import judge

WORKFLOW_STEPS = [
    {
        "step_id": "step_1_analyse",
        "system": (
            "You are an analysis assistant. Read the following problem carefully "
            "and identify the key requirements. Output a structured analysis."
        ),
        "user_template": "Analyse the following problem:\n\n{prompt}",
    },
    {
        "step_id": "step_2_reason",
        "system": (
            "You are a reasoning assistant. Based on the analysis, perform "
            "step-by-step reasoning to solve the problem."
        ),
        "user_template": (
            "Previous analysis:\n{prev}\n\n"
            "Now reason through the problem:\n\n{prompt}"
        ),
    },
    {
        "step_id": "step_3_answer",
        "system": (
            "You are an answer assistant. Based on the reasoning, provide "
            "the final concise answer."
        ),
        "user_template": (
            "Previous reasoning:\n{prev}\n\n"
            "Original problem:\n{prompt}\n\n"
            "Provide the final answer."
        ),
    },
]


class WorkflowRunner:
    """Runs a 3-step workflow via sequential LLM calls (PlanningFlow-style)."""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from app.llm import LLM

            self._llm = LLM()
        return self._llm

    async def run(self, request: BenchmarkRequest) -> List[RequestLog]:
        """Execute 3-step workflow, return list of RequestLog (one per step)."""
        llm = self._get_llm()
        sample: BenchmarkSample = request.sample or BenchmarkSample()
        workflow_id = request.workflow_id or uuid.uuid4().hex[:12]

        logs: List[RequestLog] = []
        prev_output = ""

        for step_def in WORKFLOW_STEPS:
            log = RequestLog(
                req_id=uuid.uuid4().hex[:16],
                dataset=sample.dataset,
                suite=sample.suite,
                mode=RequestMode.WORKFLOW.value,
                workflow_id=workflow_id,
                step_id=step_def["step_id"],
                config_id=request.config_id,
                seed=request.seed,
            )

            log.t_arrive = request.arrive_time or time.time()
            log.t_enqueue = time.time()

            from app.schema import Message

            sys_msg = Message.system_message(step_def["system"])
            user_text = step_def["user_template"].format(
                prompt=sample.raw_prompt,
                prev=prev_output[:2000],
            )
            user_msg = Message.user_message(user_text)

            log.t_schedule = time.time()

            try:
                formatted = llm.format_messages([user_msg])
                sys_formatted = llm.format_messages([sys_msg])
                log.in_tokens = llm.count_message_tokens(sys_formatted + formatted)

                first_token_seen = False
                collected: list[str] = []

                params = {
                    "model": llm.model,
                    "messages": sys_formatted + formatted,
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
                prev_output = full_response

                log.out_tokens = llm.count_tokens(full_response)
                log.t_finish = time.time()
                log.status = RequestStatus.COMPLETED.value

            except Exception:
                log.t_finish = time.time()
                log.status = RequestStatus.FAILED.value

            logs.append(log)

        # Quality check on the final step output
        if logs and logs[-1].status == RequestStatus.COMPLETED.value:
            logs[-1].quality_ok = judge(sample.dataset, prev_output, sample.reference)
            request.response_text = prev_output

        request.log = logs[-1] if logs else None
        return logs
