"""
Trace manager module - Manages execution tracing lifecycle.

Provides a centralized interface for recording execution steps,
tool calls, and agent interactions during multi-agent workflows.
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.trace.exporter import TraceExporter
from app.trace.schema import (
    EdgeType,
    ExecutionGraph,
    NodeStatus,
    NodeType,
    ToolCallRecord,
    TraceEdge,
    TraceNode,
)


class TraceManager:
    """
    Manages execution tracing for multi-agent systems.
    
    This class provides methods to:
    - Start and end execution traces
    - Record nodes (steps) and edges (transitions)
    - Track tool calls within steps
    - Export traces to various formats
    """

    def __init__(self, auto_save: bool = False, save_dir: Optional[str] = None):
        """
        Initialize the trace manager.
        
        Args:
            auto_save: Whether to automatically save traces when completed
            save_dir: Directory for saving traces (defaults to ./traces)
        """
        self.auto_save = auto_save
        self.save_dir = Path(save_dir) if save_dir else Path("./traces")
        self.graph: Optional[ExecutionGraph] = None
        self._current_node: Optional[TraceNode] = None
        self._last_node_id: Optional[str] = None
        self._node_stack: list = []  # Stack for nested node tracking

    def start_trace(self, request: str, metadata: Optional[Dict[str, Any]] = None) -> ExecutionGraph:
        """
        Start a new execution trace.
        
        Args:
            request: The original user request
            metadata: Optional metadata to attach to the trace
        
        Returns:
            The newly created ExecutionGraph
        """
        graph_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.graph = ExecutionGraph(
            graph_id=graph_id,
            request=request,
            start_time=datetime.now(),
            metadata=metadata or {},
        )
        self._current_node = None
        self._last_node_id = None
        self._node_stack = []
        return self.graph

    def end_trace(self, status: NodeStatus = NodeStatus.COMPLETED) -> ExecutionGraph:
        """
        End the current execution trace.
        
        Args:
            status: Final status of the execution
        
        Returns:
            The completed ExecutionGraph
        """
        if not self.graph:
            raise RuntimeError("No active trace to end")

        # Complete any pending node
        if self._current_node and self._current_node.status == NodeStatus.RUNNING:
            self._current_node.complete(status)

        self.graph.complete(status)

        if self.auto_save:
            self.save_to_file()

        return self.graph

    def start_node(
        self,
        agent_name: str,
        step_name: str,
        node_type: NodeType = NodeType.WORKER,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceNode:
        """
        Start a new node (step) in the trace.
        
        Args:
            agent_name: Name of the agent executing this step
            step_name: Name/description of the step
            node_type: Type of node (COORDINATOR, WORKER, etc.)
            metadata: Optional metadata for the node
        
        Returns:
            The newly created TraceNode
        """
        if not self.graph:
            raise RuntimeError("No active trace. Call start_trace() first.")

        # Complete previous node if still running
        if self._current_node and self._current_node.status == NodeStatus.RUNNING:
            self._current_node.complete()

        node_id = self.graph.generate_node_id()
        node = TraceNode(
            node_id=node_id,
            node_type=node_type,
            agent_name=agent_name,
            step_name=step_name,
            start_time=datetime.now(),
            metadata=metadata or {},
        )

        self.graph.add_node(node)

        # Add edge from previous node
        if self._last_node_id:
            self.add_edge(
                source_node_id=self._last_node_id,
                target_node_id=node_id,
                edge_type=EdgeType.NORMAL,
            )

        self._current_node = node
        self._last_node_id = node_id

        return node

    def end_node(
        self,
        status: NodeStatus = NodeStatus.COMPLETED,
        error: Optional[str] = None,
    ) -> Optional[TraceNode]:
        """
        End the current node.
        
        Args:
            status: Final status of the node
            error: Optional error message if failed
        
        Returns:
            The completed TraceNode, or None if no current node
        """
        if not self._current_node:
            return None

        self._current_node.complete(status, error)
        return self._current_node

    def add_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: EdgeType = EdgeType.NORMAL,
        label: Optional[str] = None,
    ) -> TraceEdge:
        """
        Add an edge between two nodes.
        
        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            edge_type: Type of edge (DELEGATE, RETURN, RETRY, etc.)
            label: Optional label for the edge
        
        Returns:
            The created TraceEdge
        """
        if not self.graph:
            raise RuntimeError("No active trace")

        edge_id = self.graph.generate_edge_id()
        edge = TraceEdge(
            edge_id=edge_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            label=label,
            timestamp=datetime.now(),
        )

        self.graph.add_edge(edge)
        return edge

    def record_delegation(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
    ) -> tuple:
        """
        Record a task delegation from one agent to another.
        
        This creates:
        1. A new node for the delegating agent's decision
        2. A DELEGATE edge to the target agent
        
        Args:
            from_agent: Name of the delegating agent (usually Coordinator)
            to_agent: Name of the target worker agent
            task: Description of the delegated task
        
        Returns:
            Tuple of (coordinator_node, delegate_edge)
        """
        # Save current state for later return edge
        source_node_id = self._last_node_id

        # Start the target agent's node
        target_node = self.start_node(
            agent_name=to_agent,
            step_name=task,
            node_type=NodeType.WORKER,
        )

        # Change the edge type to DELEGATE
        if self.graph and self.graph.edges:
            # Modify the last added edge to be a DELEGATE edge
            last_edge = self.graph.edges[-1]
            if last_edge.source_node_id == source_node_id:
                last_edge.edge_type = EdgeType.DELEGATE

        return target_node

    def record_return(
        self,
        from_worker: str,
        to_coordinator: str,
        result_summary: Optional[str] = None,
    ) -> TraceEdge:
        """
        Record a return from a worker to the coordinator.
        
        Args:
            from_worker: Name of the returning worker
            to_coordinator: Name of the coordinator
            result_summary: Optional summary of the result
        
        Returns:
            The RETURN edge
        """
        if not self._current_node:
            raise RuntimeError("No current node to return from")

        # End the current worker node
        self.end_node()
        worker_node_id = self._last_node_id

        # Start coordinator's receiving node
        coord_node = self.start_node(
            agent_name=to_coordinator,
            step_name=f"接收 {from_worker} 结果",
            node_type=NodeType.COORDINATOR,
        )

        # Change the edge type to RETURN
        if self.graph and self.graph.edges:
            last_edge = self.graph.edges[-1]
            if last_edge.source_node_id == worker_node_id:
                last_edge.edge_type = EdgeType.RETURN
                last_edge.label = result_summary

        return self.graph.edges[-1] if self.graph else None

    def record_retry(
        self,
        agent_name: str,
        reason: str,
        retry_count: int = 1,
    ) -> TraceEdge:
        """
        Record a retry attempt (creates a cycle in the graph).
        
        Args:
            agent_name: Name of the agent retrying
            reason: Reason for the retry
            retry_count: Which retry attempt this is
        
        Returns:
            The RETRY edge
        """
        if not self._current_node:
            raise RuntimeError("No current node for retry")

        current_node_id = self._current_node.node_id

        # Create a new node for the retry attempt
        retry_node = self.start_node(
            agent_name=agent_name,
            step_name=f"重试 #{retry_count}",
            node_type=self._current_node.node_type,
            metadata={"retry_reason": reason, "retry_count": retry_count},
        )

        # Change the edge type to RETRY
        if self.graph and self.graph.edges:
            last_edge = self.graph.edges[-1]
            last_edge.edge_type = EdgeType.RETRY
            last_edge.label = f"#{retry_count}: {reason}"

        return self.graph.edges[-1] if self.graph else None

    def start_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> ToolCallRecord:
        """
        Start recording a tool call within the current node.
        
        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
            tool_call_id: Optional ID for the tool call
        
        Returns:
            The ToolCallRecord
        """
        if not self._current_node:
            raise RuntimeError("No current node for tool call")

        tool_call = ToolCallRecord(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id or f"tc_{uuid.uuid4().hex[:8]}",
            start_time=datetime.now(),
        )

        self._current_node.add_tool_call(tool_call)
        return tool_call

    def end_tool_call(
        self,
        tool_call: ToolCallRecord,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> ToolCallRecord:
        """
        End a tool call recording.
        
        Args:
            tool_call: The ToolCallRecord to complete
            result: The result of the tool call
            error: Optional error message
        
        Returns:
            The completed ToolCallRecord
        """
        tool_call.complete(result, error)
        return tool_call

    def set_llm_thoughts(self, thoughts: str) -> None:
        """
        Set the LLM thoughts for the current node.
        
        Args:
            thoughts: The LLM's reasoning/thoughts
        """
        if self._current_node:
            self._current_node.llm_thoughts = thoughts

    def get_current_node(self) -> Optional[TraceNode]:
        """Get the current active node."""
        return self._current_node

    def get_current_node_id(self) -> Optional[str]:
        """Get the ID of the current active node."""
        return self._current_node.node_id if self._current_node else None

    def save_to_file(
        self,
        path: Optional[Union[str, Path]] = None,
        formats: Optional[list] = None,
    ) -> Dict[str, str]:
        """
        Save the trace to file(s).
        
        Args:
            path: Base path for saving (defaults to save_dir/graph_id)
            formats: List of formats to save (defaults to ["json", "markdown"])
        
        Returns:
            Dictionary mapping format names to file paths
        """
        if not self.graph:
            raise RuntimeError("No trace to save")

        if path:
            base_path = Path(path)
        else:
            base_path = self.save_dir

        base_path.mkdir(parents=True, exist_ok=True)

        exporter = TraceExporter(self.graph)

        if formats is None:
            return exporter.save_all(base_path, self.graph.graph_id)

        paths = {}
        if "json" in formats:
            json_path = base_path / f"{self.graph.graph_id}.json"
            exporter.save_json(json_path)
            paths["json"] = str(json_path)
        if "markdown" in formats or "md" in formats:
            md_path = base_path / f"{self.graph.graph_id}_report.md"
            exporter.save_markdown(md_path)
            paths["markdown"] = str(md_path)

        return paths

    def to_json(self) -> str:
        """Export the current trace to JSON string."""
        if not self.graph:
            raise RuntimeError("No trace to export")
        return TraceExporter(self.graph).to_json()

    def to_mermaid(self) -> str:
        """Export the current trace to Mermaid format."""
        if not self.graph:
            raise RuntimeError("No trace to export")
        return TraceExporter(self.graph).to_mermaid()

    def to_markdown(self) -> str:
        """Export the current trace to Markdown report format."""
        if not self.graph:
            raise RuntimeError("No trace to export")
        return TraceExporter(self.graph).to_markdown()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current trace."""
        if not self.graph:
            raise RuntimeError("No trace to summarize")
        return self.graph.get_execution_summary()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.graph and self.graph.status == NodeStatus.RUNNING:
            status = NodeStatus.FAILED if exc_type else NodeStatus.COMPLETED
            self.end_trace(status)
        return False
