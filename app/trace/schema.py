"""
Trace schema module - Data models for execution tracing with cyclic graph support.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EdgeType(str, Enum):
    """Edge types for distinguishing different flow relationships."""

    NORMAL = "normal"  # Normal flow transition
    DELEGATE = "delegate"  # Coordinator delegates to Worker
    RETURN = "return"  # Worker returns result to Coordinator
    RETRY = "retry"  # Retry (forms a cycle)
    LOOP = "loop"  # Loop iteration (forms a cycle)
    CONDITION = "condition"  # Conditional branch


class NodeType(str, Enum):
    """Node types in the execution graph."""

    START = "start"
    END = "end"
    COORDINATOR = "coordinator"
    WORKER = "worker"
    TOOL_CALL = "tool_call"


class NodeStatus(str, Enum):
    """Node execution status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ToolCallRecord:
    """Record of a single tool call within a step."""

    tool_name: str
    tool_args: Dict[str, Any]
    tool_call_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None

    def complete(self, result: Optional[str] = None, error: Optional[str] = None) -> None:
        """Mark the tool call as completed."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.result = result
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_call_id": self.tool_call_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class TraceNode:
    """A node in the execution graph representing a step."""

    node_id: str
    node_type: NodeType
    agent_name: str
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: NodeStatus = NodeStatus.RUNNING
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    llm_thoughts: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, status: NodeStatus = NodeStatus.COMPLETED, error: Optional[str] = None) -> None:
        """Mark the node as completed."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        if error:
            self.error = error

    def add_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Add a tool call record to this node."""
        self.tool_calls.append(tool_call)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "agent_name": self.agent_name,
            "step_name": self.step_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "llm_thoughts": self.llm_thoughts,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class TraceEdge:
    """An edge in the execution graph representing a transition."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    label: Optional[str] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type.value,
            "label": self.label,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class ExecutionGraph:
    """
    Execution graph that supports cycles.
    
    This graph represents the execution flow of a multi-agent system,
    where nodes are execution steps and edges are transitions between them.
    """

    graph_id: str
    request: str
    start_time: datetime
    end_time: Optional[datetime] = None
    nodes: Dict[str, TraceNode] = field(default_factory=dict)
    edges: List[TraceEdge] = field(default_factory=list)
    status: NodeStatus = NodeStatus.RUNNING
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal counters for generating unique IDs
    _node_counter: int = field(default=0, repr=False)
    _edge_counter: int = field(default=0, repr=False)

    def add_node(self, node: TraceNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: TraceEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[TraceNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def generate_node_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def generate_edge_id(self) -> str:
        """Generate a unique edge ID."""
        self._edge_counter += 1
        return f"edge_{self._edge_counter}"

    def complete(self, status: NodeStatus = NodeStatus.COMPLETED) -> None:
        """Mark the execution as completed."""
        self.end_time = datetime.now()
        self.status = status

    def get_duration_ms(self) -> Optional[float]:
        """Get the total execution duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def get_nodes_by_agent(self, agent_name: str) -> List[TraceNode]:
        """Get all nodes executed by a specific agent."""
        return [node for node in self.nodes.values() if node.agent_name == agent_name]

    def get_nodes_by_type(self, node_type: NodeType) -> List[TraceNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def get_edges_by_type(self, edge_type: EdgeType) -> List[TraceEdge]:
        """Get all edges of a specific type."""
        return [edge for edge in self.edges if edge.edge_type == edge_type]

    def get_outgoing_edges(self, node_id: str) -> List[TraceEdge]:
        """Get all outgoing edges from a node."""
        return [edge for edge in self.edges if edge.source_node_id == node_id]

    def get_incoming_edges(self, node_id: str) -> List[TraceEdge]:
        """Get all incoming edges to a node."""
        return [edge for edge in self.edges if edge.target_node_id == node_id]

    def has_cycles(self) -> bool:
        """Check if the graph contains any cycles."""
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for edge in self.get_outgoing_edges(node_id):
                target = edge.target_node_id
                if target not in visited:
                    if dfs(target):
                        return True
                elif target in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "graph_id": self.graph_id,
            "request": self.request,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.get_duration_ms(),
            "status": self.status.value,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
            "has_cycles": self.has_cycles(),
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        agent_stats = {}
        for node in self.nodes.values():
            if node.agent_name not in agent_stats:
                agent_stats[node.agent_name] = {
                    "node_count": 0,
                    "total_duration_ms": 0,
                    "completed": 0,
                    "failed": 0,
                }
            stats = agent_stats[node.agent_name]
            stats["node_count"] += 1
            if node.duration_ms:
                stats["total_duration_ms"] += node.duration_ms
            if node.status == NodeStatus.COMPLETED:
                stats["completed"] += 1
            elif node.status == NodeStatus.FAILED:
                stats["failed"] += 1

        return {
            "graph_id": self.graph_id,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "duration_ms": self.get_duration_ms(),
            "status": self.status.value,
            "has_cycles": self.has_cycles(),
            "agent_stats": agent_stats,
            "edge_type_counts": {
                edge_type.value: len(self.get_edges_by_type(edge_type))
                for edge_type in EdgeType
            },
        }
