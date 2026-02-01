"""
Trace exporter module - Export execution graphs for analysis.

Supports:
- JSON: Structured data for programmatic access
- Markdown: Human-readable report with embedded Mermaid flowchart

Designed for research on multi-agent scheduling optimization,
focusing on execution flow, timing, and potential optimization points.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from app.trace.schema import EdgeType, ExecutionGraph, NodeStatus, NodeType, TraceNode


class TraceExporter:
    """Export execution graphs for research analysis."""

    def __init__(self, graph: ExecutionGraph):
        self.graph = graph

    def to_json(self, indent: int = 2) -> str:
        """Export the graph to JSON format."""
        return json.dumps(self.graph.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save the graph to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def to_mermaid(self) -> str:
        """
        Export the graph to simplified Mermaid flowchart format.

        Only shows execution flow, timing, and cycles - no content details.
        Suitable for visualizing agent scheduling and identifying bottlenecks.
        """
        lines = ["```mermaid", "flowchart TD"]

        # Add start node
        lines.append('    Start(["开始"])')

        # Add nodes with timing info only
        for node_id, node in self.graph.nodes.items():
            label = self._format_node_label(node)
            # Different shapes for different node types
            if node.node_type == NodeType.COORDINATOR:
                lines.append(f'    {node_id}{{"{label}"}}')  # Diamond shape
            elif node.node_type == NodeType.WORKER:
                lines.append(f'    {node_id}["{label}"]')  # Rectangle
            elif node.node_type == NodeType.TOOL_CALL:
                lines.append(f'    {node_id}("{label}")')  # Rounded rectangle
            else:
                lines.append(f'    {node_id}["{label}"]')

        # Add end node
        lines.append('    End(["结束"])')

        # Connect Start to first node
        if self.graph.nodes:
            first_node_id = list(self.graph.nodes.keys())[0]
            lines.append(f"    Start --> {first_node_id}")

        # Add edges with type labels
        for edge in self.graph.edges:
            edge_label = self._format_edge_label(edge)
            if edge_label:
                lines.append(
                    f"    {edge.source_node_id} -->|{edge_label}| {edge.target_node_id}"
                )
            else:
                lines.append(f"    {edge.source_node_id} --> {edge.target_node_id}")

        # Connect last nodes to End
        if self.graph.nodes:
            nodes_with_outgoing = {edge.source_node_id for edge in self.graph.edges}
            last_nodes = [
                node_id
                for node_id in self.graph.nodes
                if node_id not in nodes_with_outgoing
            ]
            if last_nodes:
                for last_node_id in last_nodes:
                    lines.append(f"    {last_node_id} --> End")
            else:
                last_node_id = list(self.graph.nodes.keys())[-1]
                lines.append(f"    {last_node_id} --> End")

        # Add styling for different node types
        lines.append("")
        lines.append("    %% 样式定义")
        lines.append("    style Start fill:#90EE90")
        lines.append("    style End fill:#FFB6C1")
        for node_id, node in self.graph.nodes.items():
            if node.node_type == NodeType.COORDINATOR:
                lines.append(f"    style {node_id} fill:#87CEEB")
            elif node.status == NodeStatus.FAILED:
                lines.append(f"    style {node_id} fill:#FF6B6B")

        lines.append("```")
        return "\n".join(lines)

    def _format_node_label(self, node: TraceNode) -> str:
        """Format node label showing only agent name and duration."""
        duration_str = f"{node.duration_ms:.0f}ms" if node.duration_ms else "运行中"
        return f"{node.agent_name}<br/>{duration_str}"

    def _format_edge_label(self, edge) -> Optional[str]:
        """Format edge label."""
        if edge.edge_type == EdgeType.DELEGATE:
            return "委派"
        elif edge.edge_type == EdgeType.RETURN:
            return "返回"
        elif edge.edge_type == EdgeType.RETRY:
            return "重试"
        elif edge.edge_type == EdgeType.LOOP:
            return "循环"
        elif edge.edge_type == EdgeType.CONDITION:
            return "条件"
        return None

    def to_markdown(self) -> str:
        """
        Generate a comprehensive Markdown report for research analysis.

        Includes:
        - Execution overview (total duration, node count, cycle detection)
        - Simplified flowchart (Mermaid)
        - Detailed timeline with timing data
        - Agent statistics for optimization analysis
        """
        lines = []

        # Title
        lines.append("# 多Agent执行追踪报告")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Overview section
        lines.append("## 1. 执行概览")
        lines.append("")

        total_duration = self.graph.get_duration_ms()
        duration_str = (
            f"{total_duration:.2f}ms ({total_duration/1000:.2f}s)"
            if total_duration
            else "未完成"
        )

        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 追踪ID | `{self.graph.graph_id}` |")
        lines.append(
            f"| 用户请求 | {self.graph.request[:50]}{'...' if len(self.graph.request) > 50 else ''} |"
        )
        lines.append(
            f"| 开始时间 | {self.graph.start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} |"
        )
        lines.append(
            f"| 结束时间 | {self.graph.end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if self.graph.end_time else '未完成'} |"
        )
        lines.append(f"| **总耗时** | **{duration_str}** |")
        lines.append(f"| 执行状态 | {self._status_to_chinese(self.graph.status)} |")
        lines.append(f"| 节点总数 | {len(self.graph.nodes)} |")
        lines.append(f"| 边总数 | {len(self.graph.edges)} |")
        lines.append(
            f"| **存在环路** | **{'是 ⚠️' if self.graph.has_cycles() else '否'}** |"
        )
        lines.append("")

        # Flowchart section
        lines.append("## 2. 执行流程图")
        lines.append("")
        lines.append(self.to_mermaid())
        lines.append("")

        # Timeline section
        lines.append("## 3. 执行时间线")
        lines.append("")
        lines.append("按执行顺序排列，展示每个Agent的运行时序：")
        lines.append("")
        lines.append(
            "| 序号 | 节点ID | Agent名称 | 节点类型 | 开始时间 | 结束时间 | 耗时(ms) | 状态 |"
        )
        lines.append(
            "|------|--------|-----------|----------|----------|----------|----------|------|"
        )

        # Sort nodes by start time
        sorted_nodes = sorted(self.graph.nodes.values(), key=lambda n: n.start_time)

        for idx, node in enumerate(sorted_nodes, 1):
            start_time = node.start_time.strftime("%H:%M:%S.%f")[:-3]
            end_time = (
                node.end_time.strftime("%H:%M:%S.%f")[:-3] if node.end_time else "-"
            )
            duration = f"{node.duration_ms:.2f}" if node.duration_ms else "-"
            status = self._status_to_chinese(node.status)
            node_type = self._node_type_to_chinese(node.node_type)

            lines.append(
                f"| {idx} | `{node.node_id}` | {node.agent_name} | {node_type} | "
                f"{start_time} | {end_time} | {duration} | {status} |"
            )

        lines.append("")

        # Agent statistics section
        lines.append("## 4. Agent统计分析")
        lines.append("")
        lines.append("各Agent的执行统计，用于识别性能瓶颈：")
        lines.append("")

        agent_stats = self._calculate_agent_stats()

        lines.append(
            "| Agent名称 | 调用次数 | 总耗时(ms) | 平均耗时(ms) | 最大耗时(ms) | 最小耗时(ms) | 成功率 |"
        )
        lines.append(
            "|-----------|----------|------------|--------------|--------------|--------------|--------|"
        )

        for agent_name, stats in sorted(
            agent_stats.items(), key=lambda x: -x[1]["total_duration"]
        ):
            avg_duration = (
                stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
            )
            success_rate = (
                stats["completed"] / stats["count"] * 100 if stats["count"] > 0 else 0
            )

            lines.append(
                f"| {agent_name} | {stats['count']} | {stats['total_duration']:.2f} | "
                f"{avg_duration:.2f} | {stats['max_duration']:.2f} | {stats['min_duration']:.2f} | "
                f"{success_rate:.1f}% |"
            )

        lines.append("")

        # Edge statistics section
        lines.append("## 5. 边类型统计")
        lines.append("")
        lines.append("展示Agent之间的交互模式：")
        lines.append("")

        edge_stats = self._calculate_edge_stats()

        lines.append("| 边类型 | 数量 | 说明 |")
        lines.append("|--------|------|------|")

        edge_descriptions = {
            EdgeType.NORMAL: "普通顺序执行",
            EdgeType.DELEGATE: "任务委派（Coordinator→Worker）",
            EdgeType.RETURN: "结果返回（Worker→Coordinator）",
            EdgeType.RETRY: "重试操作（形成环路）",
            EdgeType.LOOP: "循环迭代（形成环路）",
            EdgeType.CONDITION: "条件分支",
        }

        for edge_type, count in edge_stats.items():
            desc = edge_descriptions.get(edge_type, "")
            lines.append(
                f"| {self._edge_type_to_chinese(edge_type)} | {count} | {desc} |"
            )

        lines.append("")

        # Cycle detection section
        if self.graph.has_cycles():
            lines.append("## 6. 环路分析 ⚠️")
            lines.append("")
            lines.append("检测到执行过程中存在环路，可能由以下原因造成：")
            lines.append("")

            retry_edges = [e for e in self.graph.edges if e.edge_type == EdgeType.RETRY]
            loop_edges = [e for e in self.graph.edges if e.edge_type == EdgeType.LOOP]

            if retry_edges:
                lines.append("### 重试环路")
                lines.append("")
                for edge in retry_edges:
                    lines.append(
                        f"- `{edge.source_node_id}` → `{edge.target_node_id}`: {edge.label or '重试'}"
                    )
                lines.append("")

            if loop_edges:
                lines.append("### 循环迭代")
                lines.append("")
                for edge in loop_edges:
                    lines.append(
                        f"- `{edge.source_node_id}` → `{edge.target_node_id}`: {edge.label or '循环'}"
                    )
                lines.append("")

        # Tool call analysis
        tool_calls = self._get_all_tool_calls()
        if tool_calls:
            lines.append("## 7. 工具调用分析")
            lines.append("")
            lines.append(
                "| 工具名称 | 调用次数 | 总耗时(ms) | 平均耗时(ms) | 失败次数 |"
            )
            lines.append(
                "|----------|----------|------------|--------------|----------|"
            )

            tool_stats = {}
            for tc in tool_calls:
                if tc.tool_name not in tool_stats:
                    tool_stats[tc.tool_name] = {
                        "count": 0,
                        "total_duration": 0,
                        "failed": 0,
                    }
                stats = tool_stats[tc.tool_name]
                stats["count"] += 1
                if tc.duration_ms:
                    stats["total_duration"] += tc.duration_ms
                if tc.error:
                    stats["failed"] += 1

            for tool_name, stats in sorted(
                tool_stats.items(), key=lambda x: -x[1]["total_duration"]
            ):
                avg = (
                    stats["total_duration"] / stats["count"]
                    if stats["count"] > 0
                    else 0
                )
                lines.append(
                    f"| {tool_name} | {stats['count']} | {stats['total_duration']:.2f} | "
                    f"{avg:.2f} | {stats['failed']} |"
                )
            lines.append("")

        return "\n".join(lines)

    def _calculate_agent_stats(self) -> Dict[str, Dict]:
        """Calculate statistics for each agent."""
        stats = {}
        for node in self.graph.nodes.values():
            if node.agent_name not in stats:
                stats[node.agent_name] = {
                    "count": 0,
                    "total_duration": 0,
                    "max_duration": 0,
                    "min_duration": float("inf"),
                    "completed": 0,
                    "failed": 0,
                }
            s = stats[node.agent_name]
            s["count"] += 1
            if node.duration_ms:
                s["total_duration"] += node.duration_ms
                s["max_duration"] = max(s["max_duration"], node.duration_ms)
                s["min_duration"] = min(s["min_duration"], node.duration_ms)
            if node.status == NodeStatus.COMPLETED:
                s["completed"] += 1
            elif node.status == NodeStatus.FAILED:
                s["failed"] += 1

        # Fix min_duration for agents with no duration data
        for s in stats.values():
            if s["min_duration"] == float("inf"):
                s["min_duration"] = 0

        return stats

    def _calculate_edge_stats(self) -> Dict[EdgeType, int]:
        """Calculate edge type statistics."""
        stats = {edge_type: 0 for edge_type in EdgeType}
        for edge in self.graph.edges:
            stats[edge.edge_type] += 1
        return stats

    def _get_all_tool_calls(self) -> List:
        """Get all tool calls from all nodes."""
        tool_calls = []
        for node in self.graph.nodes.values():
            tool_calls.extend(node.tool_calls)
        return tool_calls

    def _status_to_chinese(self, status: NodeStatus) -> str:
        """Convert status to Chinese."""
        mapping = {
            NodeStatus.RUNNING: "运行中 🔄",
            NodeStatus.COMPLETED: "已完成 ✅",
            NodeStatus.FAILED: "失败 ❌",
        }
        return mapping.get(status, str(status))

    def _node_type_to_chinese(self, node_type: NodeType) -> str:
        """Convert node type to Chinese."""
        mapping = {
            NodeType.START: "开始",
            NodeType.END: "结束",
            NodeType.COORDINATOR: "协调者",
            NodeType.WORKER: "工作者",
            NodeType.TOOL_CALL: "工具调用",
        }
        return mapping.get(node_type, str(node_type))

    def _edge_type_to_chinese(self, edge_type: EdgeType) -> str:
        """Convert edge type to Chinese."""
        mapping = {
            EdgeType.NORMAL: "顺序执行",
            EdgeType.DELEGATE: "任务委派",
            EdgeType.RETURN: "结果返回",
            EdgeType.RETRY: "重试",
            EdgeType.LOOP: "循环",
            EdgeType.CONDITION: "条件分支",
        }
        return mapping.get(edge_type, str(edge_type))

    def save_markdown(self, path: Union[str, Path]) -> None:
        """Save the graph to a Markdown file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

    def save_all(
        self,
        base_path: Union[str, Path],
        prefix: Optional[str] = None,
    ) -> dict:
        """
        Save the graph to supported formats.

        Args:
            base_path: Base directory for saving files
            prefix: Optional prefix for file names

        Returns:
            Dictionary with paths to saved files
        """
        base_path = Path(base_path)
        prefix = prefix or self.graph.graph_id

        paths = {
            "json": base_path / f"{prefix}.json",
            "markdown": base_path / f"{prefix}_report.md",
        }

        self.save_json(paths["json"])
        self.save_markdown(paths["markdown"])

        return {k: str(v) for k, v in paths.items()}


def export_to_json(
    graph: ExecutionGraph, path: Optional[Union[str, Path]] = None
) -> str:
    """Convenience function to export a graph to JSON."""
    exporter = TraceExporter(graph)
    if path:
        exporter.save_json(path)
    return exporter.to_json()


def export_to_markdown(
    graph: ExecutionGraph, path: Optional[Union[str, Path]] = None
) -> str:
    """Convenience function to export a graph to Markdown."""
    exporter = TraceExporter(graph)
    if path:
        exporter.save_markdown(path)
    return exporter.to_markdown()
