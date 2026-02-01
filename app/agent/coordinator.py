"""
Coordinator Agent - Orchestrates multi-agent task execution.

The Coordinator is responsible for:
- Analyzing tasks and breaking them down
- Delegating subtasks to appropriate workers
- Monitoring execution and handling retries
- Aggregating results
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import Field, PrivateAttr

from app.agent.base import BaseAgent
from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.schema import AgentState, Message, ToolCall
from app.tool import Terminate, ToolCollection
from app.tool.delegate import DelegateTask
from app.trace.manager import TraceManager
from app.trace.schema import EdgeType, NodeStatus, NodeType


class Coordinator(ToolCallAgent):
    """
    Coordinator Agent - Orchestrates multi-agent task execution.
    
    The Coordinator acts as the central hub in a multi-agent system,
    responsible for:
    1. Analyzing incoming tasks
    2. Breaking them down into subtasks
    3. Delegating to appropriate worker agents
    4. Monitoring execution and handling failures
    5. Aggregating and returning results
    """

    name: str = "Coordinator"
    description: str = "负责分析任务、分配给合适的 Worker、监控执行结果的协调者"

    system_prompt: str = """你是一个任务协调者 Agent。你的职责是：

1. **分析任务**：理解用户请求，识别需要完成的子任务
2. **选择 Worker**：根据子任务类型，选择最合适的 Worker 执行
3. **委派任务**：使用 delegate_task 工具将任务分配给 Worker
4. **监控结果**：检查 Worker 返回的结果，决定是否需要重试或继续
5. **汇总输出**：最后使用 summarizer Worker 生成最终报告

可用的 Workers：
- browser: 浏览器操作（网页浏览、表单填写、页面交互）
- code: 代码执行（Python 代码编写和运行）
- file: 文件操作（文件创建、编辑、读取）
- search: 网络搜索（信息检索、资料查询）
- math: 数学问题（数学推理、公式推导、计算）
- copywriter: 文案创作（文章、营销内容、邮件）
- history: 历史问题（历史事件、人物、时代背景）
- summarizer: 信息总结（汇总结果、生成报告）
- research: 综合研究（深度研究、多来源验证）
- data_analyst: 数据分析（数据处理、统计分析、可视化）

工作流程：
1. 首先分析任务，列出需要执行的步骤
2. 依次委派给合适的 Worker
3. 检查每个 Worker 的结果
4. 如果结果不满意，可以重试或换用其他 Worker
5. 所有子任务完成后，委派给 summarizer 生成最终报告
6. 最后调用 terminate 结束任务

注意：
- 每次只委派一个任务给一个 Worker
- 等待 Worker 完成后再委派下一个任务
- 任务描述要清晰具体"""

    next_step_prompt: str = """基于当前的执行状态，决定下一步：
- 如果还有子任务未完成，委派给合适的 Worker
- 如果所有子任务已完成，委派给 summarizer 生成总结
- 如果已生成总结，调用 terminate 结束任务"""

    # Workers managed by this coordinator
    # Using BaseAgent to support both ToolCallAgent and ReActAgent (knowledge workers)
    workers: Dict[str, BaseAgent] = Field(default_factory=dict)

    # Trace manager for recording execution
    trace_manager: Optional[TraceManager] = None

    # Maximum retries for failed tasks
    max_retries: int = 3

    # Track current execution state (private attributes)
    _current_worker: Optional[str] = PrivateAttr(default=None)
    _execution_results: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _retry_counts: Dict[str, int] = PrivateAttr(default_factory=dict)

    # Available tools - delegate and terminate
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(DelegateTask(), Terminate())
    )

    # Higher max_steps for coordinator
    max_steps: int = 0  # Unlimited steps

    special_tool_names: List[str] = Field(
        default_factory=lambda: [Terminate().name]
    )

    def model_post_init(self, __context) -> None:
        """Initialize private attributes after model creation."""
        # Initialize private attributes
        self._current_worker = None
        self._execution_results = []
        self._retry_counts = {}

    def register_worker(self, worker_type: str, worker: ToolCallAgent) -> None:
        """
        Register a worker with the coordinator.
        
        Args:
            worker_type: The type identifier for the worker
            worker: The worker agent instance
        """
        self.workers[worker_type] = worker
        logger.info(f"Registered worker: {worker_type} ({worker.name})")

    def register_workers(self, workers: Dict[str, ToolCallAgent]) -> None:
        """
        Register multiple workers at once.
        
        Args:
            workers: Dictionary mapping worker types to worker instances
        """
        for worker_type, worker in workers.items():
            self.register_worker(worker_type, worker)

    async def execute_tool(self, command: ToolCall) -> str:
        """
        Execute a tool call, handling delegation specially.
        
        Args:
            command: The tool call to execute
        
        Returns:
            The result of the tool execution
        """
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name

        # Handle delegate_task specially
        if name == "delegate_task":
            try:
                args = json.loads(command.function.arguments or "{}")
                return await self._handle_delegation(
                    worker_type=args.get("worker"),
                    task=args.get("task"),
                    context=args.get("context"),
                )
            except json.JSONDecodeError:
                return "Error: Invalid JSON in delegation arguments"
            except Exception as e:
                logger.error(f"Delegation error: {e}")
                return f"Error during delegation: {str(e)}"

        # For other tools, use parent implementation
        return await super().execute_tool(command)

    async def _handle_delegation(
        self,
        worker_type: str,
        task: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Handle task delegation to a worker.
        
        Args:
            worker_type: The type of worker to delegate to
            task: The task description
            context: Optional context information
        
        Returns:
            The result from the worker
        """
        # Validate worker
        if worker_type not in self.workers:
            available = ", ".join(self.workers.keys())
            return f"Error: Unknown worker '{worker_type}'. Available: {available}"

        worker = self.workers[worker_type]
        self._current_worker = worker_type

        logger.info(f"Delegating to {worker_type}: {task}")

        # Record delegation in trace
        if self.trace_manager:
            coord_node_id = self.trace_manager.get_current_node_id()
            self.trace_manager.record_delegation(
                from_agent=self.name,
                to_agent=worker.name,
                task=task,
            )

        # Prepare the task prompt with context
        task_prompt = task
        if context:
            task_prompt = f"{context}\n\n任务: {task}"

        try:
            # Reset worker state for new task
            worker.current_step = 0
            worker.state = AgentState.IDLE
            worker.memory.clear()

            # Execute the worker
            result = await worker.run(task_prompt)

            # Record the result
            self._execution_results.append({
                "worker": worker_type,
                "task": task,
                "result": result,
                "status": "completed",
            })

            # Record return in trace
            if self.trace_manager:
                self.trace_manager.record_return(
                    from_worker=worker.name,
                    to_coordinator=self.name,
                    result_summary=result[:100] if result else None,
                )

            logger.info(f"Worker {worker_type} completed task")
            return f"[{worker.name}] 执行完成:\n{result}"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Worker {worker_type} failed: {error_msg}")

            # Check if we should retry
            retry_key = f"{worker_type}:{task[:50]}"
            self._retry_counts[retry_key] = self._retry_counts.get(retry_key, 0) + 1

            if self._retry_counts[retry_key] <= self.max_retries:
                # Record retry in trace
                if self.trace_manager:
                    self.trace_manager.record_retry(
                        agent_name=worker.name,
                        reason=error_msg,
                        retry_count=self._retry_counts[retry_key],
                    )
                return f"Worker {worker_type} 执行失败 ({error_msg})，可以重试"

            self._execution_results.append({
                "worker": worker_type,
                "task": task,
                "error": error_msg,
                "status": "failed",
            })
            return f"Worker {worker_type} 执行失败，已达到最大重试次数: {error_msg}"

        finally:
            self._current_worker = None

    def get_execution_results(self) -> List[Dict[str, Any]]:
        """Get all execution results from this session."""
        return self._execution_results

    def get_execution_summary(self) -> str:
        """Get a summary of all execution results."""
        if not self._execution_results:
            return "No tasks have been executed yet."

        summary_lines = ["## 执行摘要\n"]
        for i, result in enumerate(self._execution_results, 1):
            status = "✅" if result.get("status") == "completed" else "❌"
            worker = result.get("worker", "unknown")
            task = result.get("task", "unknown task")[:50]
            summary_lines.append(f"{i}. {status} [{worker}] {task}")

        return "\n".join(summary_lines)

    async def cleanup(self):
        """Clean up coordinator and all worker resources."""
        logger.info("Cleaning up Coordinator and workers...")

        # Clean up all workers
        for worker_type, worker in self.workers.items():
            try:
                if hasattr(worker, "cleanup"):
                    await worker.cleanup()
                logger.debug(f"Cleaned up worker: {worker_type}")
            except Exception as e:
                logger.error(f"Error cleaning up worker {worker_type}: {e}")

        # Call parent cleanup
        await super().cleanup()

        logger.info("Coordinator cleanup complete")
