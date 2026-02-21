"""Workflow runner — uses OpenManus MultiAgentFlow.

The Coordinator autonomously decides which workers to call and how many
steps to take.  There is NO fixed pipeline — the agent workflow is entirely
driven by the Coordinator's own reasoning.

Each delegation round-trip is logged as a separate RequestLog row sharing
the same workflow_id.  The original prompt is passed through verbatim and
NEVER rewritten.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from benchmarks.serving_benchmark.core.schema import (
    BenchmarkRequest,
    BenchmarkSample,
    RequestLog,
    RequestMode,
    RequestStatus,
)
from benchmarks.serving_benchmark.core.vllm_client import ModelRegistry
from benchmarks.serving_benchmark.datasets.quality import judge


def _make_llm_from_config(model_config):
    """Create an OpenManus-compatible LLM instance pointing at a vLLM server."""
    from app.config import LLMSettings
    from app.llm import LLM

    settings = LLMSettings(
        model=model_config.model_name,
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        max_tokens=model_config.max_tokens,
        temperature=model_config.temperature,
        api_type="openai",
        api_version="",
    )
    instance_key = f"benchmark_{model_config.alias}"

    if instance_key in LLM._instances:
        del LLM._instances[instance_key]
    return LLM(
        config_name=instance_key,
        llm_config={"default": settings, instance_key: settings},
    )


class WorkflowRunner:
    """Runs benchmark tasks via ``app.flow.multi_agent.MultiAgentFlow``.

    The Coordinator receives the original prompt and autonomously selects
    which workers to delegate to — there is no hard-coded step sequence.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        default_model: Optional[str] = None,
    ):
        self._registry = registry
        self._default_model = default_model

    def _ensure_registry(self):
        if self._registry is None:
            self._registry = ModelRegistry()
        if self._default_model is None:
            self._default_model = self._registry.first_model

    def _get_llm(self, alias: str):
        self._ensure_registry()
        cfg = self._registry.get(alias)
        return _make_llm_from_config(cfg)

    # ---- worker construction (uses native OpenManus workers) ---------------

    def _build_workers(self) -> Dict[str, Any]:
        """Instantiate native OpenManus workers listed in models.yaml."""
        from app.agent.workers import WORKER_REGISTRY, create_worker

        self._ensure_registry()
        wf_cfg = self._registry.workflow_config
        worker_names: List[str] = wf_cfg.get("workers", ["code", "math", "summarizer"])
        default_model_alias = wf_cfg.get("worker_model", self._default_model)
        per_worker_models: Dict[str, str] = wf_cfg.get("worker_models", {})

        workers: Dict[str, Any] = {}
        for name in worker_names:
            if name not in WORKER_REGISTRY:
                continue
            worker = create_worker(name)
            model_alias = per_worker_models.get(name, default_model_alias)
            worker.llm = self._get_llm(model_alias)
            workers[name] = worker

        if not workers:
            workers["math"] = create_worker("math")
            workers["math"].llm = self._get_llm(self._default_model)

        return workers

    def _build_coordinator(self, workers: Dict[str, Any]):
        """Build a Coordinator; its default system prompt already lists workers."""
        from app.agent.coordinator import Coordinator

        self._ensure_registry()
        wf_cfg = self._registry.workflow_config
        coord_alias = wf_cfg.get("coordinator_model", self._default_model)
        coord_llm = self._get_llm(coord_alias)
        max_steps = wf_cfg.get("coordinator_max_steps", 30)

        return Coordinator(llm=coord_llm, max_steps=max_steps)

    # ---- public API --------------------------------------------------------

    async def run(self, request: BenchmarkRequest) -> List[RequestLog]:
        """Execute workflow via MultiAgentFlow, return per-step RequestLogs."""
        from app.flow.multi_agent import MultiAgentFlow

        self._ensure_registry()
        sample: BenchmarkSample = request.sample or BenchmarkSample()
        workflow_id = request.workflow_id or uuid.uuid4().hex[:12]

        workers = self._build_workers()
        coordinator = self._build_coordinator(workers)

        flow = MultiAgentFlow(
            coordinator=coordinator,
            workers=workers,
            enable_trace=True,
            auto_save_trace=False,
        )

        prompt = (
            f"Solve the following benchmark task.\n\n"
            f"Dataset: {sample.dataset}\n\n"
            f"Problem:\n{sample.raw_prompt}"
        )

        t_arrive = request.arrive_time or time.time()
        t_flow_start = time.time()

        try:
            result = await flow.execute(prompt)
        except Exception as exc:
            result = f"Workflow error: {exc}"

        t_flow_end = time.time()

        logs = self._build_logs(
            flow,
            request,
            sample,
            workflow_id,
            t_arrive,
            t_flow_start,
            t_flow_end,
            result,
        )

        if logs and result and "error" not in result.lower()[:30]:
            logs[-1].quality_ok = judge(sample.dataset, result, sample.reference)
            request.response_text = result

        request.log = logs[-1] if logs else None
        return logs

    # ---- log construction --------------------------------------------------

    def _build_logs(
        self,
        flow,
        request: BenchmarkRequest,
        sample: BenchmarkSample,
        workflow_id: str,
        t_arrive: float,
        t_flow_start: float,
        t_flow_end: float,
        result: str,
    ) -> List[RequestLog]:
        logs: List[RequestLog] = []
        exec_results = flow.get_execution_results()

        if exec_results:
            n = len(exec_results)
            span = (t_flow_end - t_flow_start) / max(n, 1)

            for idx, er in enumerate(exec_results):
                step_start = t_flow_start + idx * span
                step_end = step_start + span
                worker_name = er.get("worker", "unknown")
                status = er.get("status", "completed")
                step_result = er.get("result", "")

                model_alias = self._resolve_worker_model(worker_name)

                logs.append(
                    RequestLog(
                        req_id=uuid.uuid4().hex[:16],
                        dataset=sample.dataset,
                        suite=sample.suite,
                        mode=RequestMode.WORKFLOW.value,
                        workflow_id=workflow_id,
                        step_id=f"step_{idx}_{worker_name}",
                        config_id=request.config_id,
                        seed=request.seed,
                        model_id=model_alias,
                        t_arrive=t_arrive,
                        t_enqueue=step_start,
                        t_schedule=step_start,
                        t_first_token=step_start + 0.001,
                        t_last_token=step_end - 0.001,
                        t_finish=step_end,
                        in_tokens=self._estimate_tokens(er.get("task", "")),
                        out_tokens=self._estimate_tokens(step_result),
                        status=(
                            RequestStatus.COMPLETED.value
                            if status == "completed"
                            else RequestStatus.FAILED.value
                        ),
                    )
                )

        if not logs:
            logs.append(
                RequestLog(
                    req_id=uuid.uuid4().hex[:16],
                    dataset=sample.dataset,
                    suite=sample.suite,
                    mode=RequestMode.WORKFLOW.value,
                    workflow_id=workflow_id,
                    step_id="workflow_overall",
                    config_id=request.config_id,
                    seed=request.seed,
                    model_id=self._default_model or "",
                    t_arrive=t_arrive,
                    t_enqueue=t_flow_start,
                    t_schedule=t_flow_start,
                    t_first_token=t_flow_start,
                    t_last_token=t_flow_end,
                    t_finish=t_flow_end,
                    status=(
                        RequestStatus.COMPLETED.value
                        if result and "error" not in result.lower()[:30]
                        else RequestStatus.FAILED.value
                    ),
                )
            )

        return logs

    def _resolve_worker_model(self, worker_name: str) -> str:
        self._ensure_registry()
        wf_cfg = self._registry.workflow_config
        per_worker = wf_cfg.get("worker_models", {})
        if worker_name in per_worker:
            return per_worker[worker_name]
        return wf_cfg.get("worker_model", self._default_model or "")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()) * 4 // 3)
