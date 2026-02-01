"""
Multi-Agent Flow - Orchestrates multi-agent execution with tracing support.

This flow manages the execution of a Coordinator and multiple Worker agents,
providing:
- Task delegation from Coordinator to Workers
- Execution tracing and graph generation
- Result aggregation and reporting
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.agent.coordinator import Coordinator
from app.agent.toolcall import ToolCallAgent
from app.flow.base import BaseFlow
from app.logger import logger
from app.schema import AgentState
from app.trace.manager import TraceManager
from app.trace.schema import NodeStatus, NodeType


class MultiAgentFlow(BaseFlow):
    """
    Multi-Agent execution flow with tracing support.

    This flow orchestrates a Coordinator agent that delegates tasks to
    specialized Worker agents. It provides:

    - Automatic worker registration with the Coordinator
    - Execution tracing (optional)
    - Multiple output formats (JSON, Mermaid, DOT)
    - Result aggregation

    Example:
        ```python
        from app.agent.coordinator import Coordinator
        from app.agent.workers import BrowserWorker, CodeWorker
        from app.flow.multi_agent import MultiAgentFlow

        workers = {
            "browser": BrowserWorker(),
            "code": CodeWorker(),
        }

        coordinator = Coordinator()
        flow = MultiAgentFlow(
            coordinator=coordinator,
            workers=workers,
            enable_trace=True,
        )

        result = await flow.execute("Search for weather and process the data")
        print(flow.get_trace_mermaid())
        ```
    """

    # The coordinator agent
    coordinator: Coordinator = Field(default_factory=Coordinator)

    # Worker agents (will be registered with coordinator)
    # Using BaseAgent to support both ToolCallAgent and ReActAgent (knowledge workers)
    workers: Dict[str, BaseAgent] = Field(default_factory=dict)

    # Trace manager for recording execution
    trace_manager: TraceManager = Field(default_factory=TraceManager)

    # Whether to enable tracing
    enable_trace: bool = Field(default=True)

    # Directory for saving trace files
    trace_output_dir: str = Field(default="./traces")

    # Whether to auto-save traces on completion
    auto_save_trace: bool = Field(default=True)

    def __init__(
        self,
        coordinator: Optional[Coordinator] = None,
        workers: Optional[Dict[str, ToolCallAgent]] = None,
        enable_trace: bool = True,
        trace_output_dir: str = "./traces",
        auto_save_trace: bool = True,
        **kwargs,
    ):
        """
        Initialize the multi-agent flow.

        Args:
            coordinator: The Coordinator agent (created if not provided)
            workers: Dictionary of worker agents
            enable_trace: Whether to enable execution tracing
            trace_output_dir: Directory for saving trace files
            auto_save_trace: Whether to auto-save traces on completion
            **kwargs: Additional arguments for BaseFlow
        """
        # Create coordinator if not provided
        if coordinator is None:
            coordinator = Coordinator()

        # Initialize workers dict
        workers = workers or {}

        # Create trace manager
        trace_manager = TraceManager(
            auto_save=auto_save_trace,
            save_dir=trace_output_dir,
        )

        # Build agents dict for BaseFlow
        agents = {"coordinator": coordinator}
        agents.update(workers)

        super().__init__(
            agents=agents,
            coordinator=coordinator,
            workers=workers,
            trace_manager=trace_manager,
            enable_trace=enable_trace,
            trace_output_dir=trace_output_dir,
            auto_save_trace=auto_save_trace,
            **kwargs,
        )

        # Register workers with coordinator
        self.coordinator.register_workers(self.workers)

        # Set up trace manager in coordinator if tracing enabled
        if self.enable_trace:
            self.coordinator.trace_manager = self.trace_manager
            # Also set in workers for nested tracing
            for worker in self.workers.values():
                worker.trace_manager = self.trace_manager

    async def execute(self, input_text: str) -> str:
        """
        Execute the multi-agent flow.

        Args:
            input_text: The user's request/task

        Returns:
            The final result from the Coordinator
        """
        logger.info(f"Starting MultiAgentFlow execution: {input_text[:100]}...")

        try:
            # Start tracing if enabled
            if self.enable_trace:
                self.trace_manager.start_trace(
                    request=input_text,
                    metadata={
                        "flow_type": "multi_agent",
                        "workers": list(self.workers.keys()),
                    },
                )

                # Create initial coordinator node
                self.trace_manager.start_node(
                    agent_name=self.coordinator.name,
                    step_name="分析任务",
                    node_type=NodeType.COORDINATOR,
                )

            # Run the coordinator
            result = await self.coordinator.run(input_text)

            # End tracing
            if self.enable_trace:
                self.trace_manager.end_node(NodeStatus.COMPLETED)
                self.trace_manager.end_trace(NodeStatus.COMPLETED)

                # Auto-save if enabled
                if self.auto_save_trace:
                    self._save_trace()

            logger.info("MultiAgentFlow execution completed successfully")
            return result

        except Exception as e:
            logger.error(f"MultiAgentFlow execution failed: {e}")

            # Record failure in trace
            if self.enable_trace and self.trace_manager.graph:
                self.trace_manager.end_node(NodeStatus.FAILED, str(e))
                self.trace_manager.end_trace(NodeStatus.FAILED)

                if self.auto_save_trace:
                    self._save_trace()

            raise

        finally:
            # Clean up coordinator (which cleans up workers)
            await self.coordinator.cleanup()

    def _save_trace(self) -> Dict[str, str]:
        """Save the trace to files."""
        return self.trace_manager.save_to_file()

    def get_trace_json(self) -> str:
        """Get the trace as JSON string."""
        if not self.enable_trace or not self.trace_manager.graph:
            return "{}"
        return self.trace_manager.to_json()

    def get_trace_mermaid(self, **kwargs) -> str:
        """Get the trace as Mermaid flowchart."""
        if not self.enable_trace or not self.trace_manager.graph:
            return "flowchart TD\n    NoTrace[No trace available]"
        return self.trace_manager.to_mermaid(**kwargs)

    def get_trace_dot(self, **kwargs) -> str:
        """Get the trace as DOT graph."""
        if not self.enable_trace or not self.trace_manager.graph:
            return 'digraph { NoTrace [label="No trace available"] }'
        return self.trace_manager.to_dot(**kwargs)

    def get_trace_summary(self) -> Dict:
        """Get a summary of the execution trace."""
        if not self.enable_trace or not self.trace_manager.graph:
            return {"error": "No trace available"}
        return self.trace_manager.get_summary()

    def save_trace(
        self,
        path: Optional[Union[str, Path]] = None,
        formats: Optional[list] = None,
    ) -> Dict[str, str]:
        """
        Save the trace to file(s).

        Args:
            path: Base path for saving
            formats: List of formats to save (json, mermaid, dot)

        Returns:
            Dictionary mapping format names to file paths
        """
        if not self.enable_trace or not self.trace_manager.graph:
            logger.warning("No trace to save")
            return {}
        return self.trace_manager.save_to_file(path, formats)

    def add_worker(self, worker_type: str, worker: ToolCallAgent) -> None:
        """
        Add a worker to the flow dynamically.

        Args:
            worker_type: The type identifier for the worker
            worker: The worker agent instance
        """
        self.workers[worker_type] = worker
        self.agents[worker_type] = worker
        self.coordinator.register_worker(worker_type, worker)

        # Set trace manager if tracing enabled
        if self.enable_trace:
            worker.trace_manager = self.trace_manager

    def remove_worker(self, worker_type: str) -> Optional[ToolCallAgent]:
        """
        Remove a worker from the flow.

        Args:
            worker_type: The type identifier of the worker to remove

        Returns:
            The removed worker, or None if not found
        """
        worker = self.workers.pop(worker_type, None)
        self.agents.pop(worker_type, None)

        if worker and worker_type in self.coordinator.workers:
            del self.coordinator.workers[worker_type]

        return worker

    def get_execution_results(self):
        """Get all execution results from the coordinator."""
        return self.coordinator.get_execution_results()

    def get_execution_summary(self) -> str:
        """Get a text summary of execution results."""
        return self.coordinator.get_execution_summary()
