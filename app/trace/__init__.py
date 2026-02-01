"""
Trace module - Execution tracing for multi-agent systems.

This module provides comprehensive execution tracing capabilities:
- Data models for representing execution graphs (with cycle support)
- Export to JSON and Markdown formats for research analysis
- TraceManager for recording and managing traces

Designed for research on multi-agent scheduling optimization,
focusing on execution flow, timing, and potential optimization points.

Example:
    ```python
    from app.trace import TraceManager, NodeType, EdgeType
    
    # Create a trace manager
    manager = TraceManager()
    
    # Start a trace
    manager.start_trace("User request")
    
    # Record execution steps
    manager.start_node("Coordinator", "分析任务", NodeType.COORDINATOR)
    manager.end_node()
    
    # Export results
    print(manager.to_markdown())
    manager.save_to_file("./traces")
    ```
"""

from app.trace.schema import (
    # Enums
    EdgeType,
    NodeStatus,
    NodeType,
    # Data classes
    ExecutionGraph,
    ToolCallRecord,
    TraceEdge,
    TraceNode,
)

from app.trace.manager import TraceManager

from app.trace.exporter import (
    TraceExporter,
    export_to_json,
    export_to_markdown,
)

__all__ = [
    # Enums
    "EdgeType",
    "NodeStatus",
    "NodeType",
    # Data classes
    "ExecutionGraph",
    "ToolCallRecord",
    "TraceEdge",
    "TraceNode",
    # Manager
    "TraceManager",
    # Exporter
    "TraceExporter",
    "export_to_json",
    "export_to_markdown",
]
