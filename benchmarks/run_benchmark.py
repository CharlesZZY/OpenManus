#!/usr/bin/env python3
"""
Benchmark 运行脚本

针对多Agent调度优化研究，收集执行trace数据：
- 并行执行多个数据集的测试
- 为每个query生成独立trace
- 对每个数据集生成综合分析
- 对所有数据集生成总体分析

目录结构：
traces/
├── benchmark_{timestamp}/
│   ├── math/
│   │   ├── gsm8k/
│   │   │   ├── trace_001.json
│   │   │   ├── trace_001_report.md
│   │   │   └── ...
│   │   ├── gsm8k_analysis.md
│   │   └── ...
│   ├── history/
│   ├── qa/
│   ├── code/
│   ├── dataset_summary.md
│   └── overall_analysis.md
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.trace import TraceManager, NodeType, NodeStatus
from app.logger import logger


# 配置
MAX_CONCURRENT_TASKS = 3  # 最大并发数
TIMEOUT_PER_QUERY = 120   # 每个query的超时时间（秒）


class BenchmarkRunner:
    """Benchmark 运行器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or Path("traces") / f"benchmark_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计数据
        self.stats = {
            "total_queries": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "total_duration_ms": 0,
            "datasets": {}
        }
    
    def get_dataset_dir(self, category: str, dataset_name: str) -> Path:
        """获取数据集目录"""
        path = self.output_dir / category / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    async def run_single_query(
        self,
        query_id: str,
        query: str,
        category: str,
        dataset_name: str,
        agent_type: str = "coordinator"
    ) -> Dict[str, Any]:
        """
        运行单个query并生成trace
        
        Args:
            query_id: 查询ID
            query: 查询内容
            category: 数据集类别
            dataset_name: 数据集名称
            agent_type: 使用的agent类型
        
        Returns:
            执行结果统计
        """
        trace_manager = TraceManager()
        result = {
            "query_id": query_id,
            "status": "pending",
            "duration_ms": 0,
            "node_count": 0,
            "edge_count": 0,
            "has_cycles": False,
            "error": None
        }
        
        try:
            # 开始trace
            trace_manager.start_trace(
                request=query[:200],  # 截断过长的query
                metadata={
                    "query_id": query_id,
                    "category": category,
                    "dataset": dataset_name
                }
            )
            
            # 根据类别选择合适的agent组合
            await self._execute_with_agents(
                query=query,
                category=category,
                trace_manager=trace_manager
            )
            
            # 结束trace
            trace_manager.end_trace(NodeStatus.COMPLETED)
            result["status"] = "completed"
            
        except asyncio.TimeoutError:
            trace_manager.end_trace(NodeStatus.FAILED)
            result["status"] = "timeout"
            result["error"] = "Query execution timeout"
            
        except Exception as e:
            trace_manager.end_trace(NodeStatus.FAILED)
            result["status"] = "failed"
            result["error"] = str(e)
        
        # 收集统计数据
        if trace_manager.graph:
            result["duration_ms"] = trace_manager.graph.get_duration_ms() or 0
            result["node_count"] = len(trace_manager.graph.nodes)
            result["edge_count"] = len(trace_manager.graph.edges)
            result["has_cycles"] = trace_manager.graph.has_cycles()
            
            # 保存trace文件
            save_dir = self.get_dataset_dir(category, dataset_name)
            trace_manager.save_to_file(save_dir)
        
        return result
    
    async def _execute_with_agents(
        self,
        query: str,
        category: str,
        trace_manager: TraceManager
    ):
        """
        使用多Agent执行查询
        
        根据不同类别使用不同的Agent组合来模拟真实的多Agent协作场景
        """
        # 模拟Coordinator分析任务
        trace_manager.start_node(
            agent_name="Coordinator",
            step_name="分析任务",
            node_type=NodeType.COORDINATOR
        )
        await asyncio.sleep(0.1)  # 模拟LLM思考时间
        trace_manager.end_node()
        
        if category == "math":
            # 数学问题：先搜索，再计算
            await self._simulate_worker(trace_manager, "SearchWorker", "搜索相关公式", 0.2)
            await self._simulate_worker(trace_manager, "MathWorker", "数学推理", 0.5)
            
        elif category == "history":
            # 历史问题：搜索+分析
            await self._simulate_worker(trace_manager, "SearchWorker", "搜索历史资料", 0.3)
            await self._simulate_worker(trace_manager, "HistoryWorker", "历史分析", 0.4)
            
        elif category == "code":
            # 代码问题：分析+编码+测试
            await self._simulate_worker(trace_manager, "CodeWorker", "代码分析", 0.2)
            await self._simulate_worker(trace_manager, "CodeWorker", "代码生成", 0.6)
            await self._simulate_worker(trace_manager, "CodeWorker", "代码测试", 0.3)
            
        elif category in ["qa", "truthful", "social"]:
            # 常识问答：搜索+推理
            await self._simulate_worker(trace_manager, "SearchWorker", "信息检索", 0.3)
            await self._simulate_worker(trace_manager, "SummarizerWorker", "信息整合", 0.2)
        
        else:
            # 默认流程
            await self._simulate_worker(trace_manager, "SearchWorker", "信息搜索", 0.3)
        
        # Coordinator 汇总结果
        trace_manager.start_node(
            agent_name="Coordinator",
            step_name="汇总结果",
            node_type=NodeType.COORDINATOR
        )
        await asyncio.sleep(0.1)
        trace_manager.end_node()
        
        # SummarizerWorker 生成最终输出
        await self._simulate_worker(trace_manager, "SummarizerWorker", "生成回答", 0.2)
    
    async def _simulate_worker(
        self,
        trace_manager: TraceManager,
        worker_name: str,
        task_name: str,
        duration: float
    ):
        """模拟Worker执行"""
        # 委派任务
        trace_manager.start_node(
            agent_name="Coordinator",
            step_name=f"委派给{worker_name}",
            node_type=NodeType.COORDINATOR
        )
        await asyncio.sleep(0.05)
        trace_manager.end_node()
        
        # Worker执行
        trace_manager.start_node(
            agent_name=worker_name,
            step_name=task_name,
            node_type=NodeType.WORKER
        )
        
        # 模拟工具调用
        tool_call = trace_manager.start_tool_call(
            tool_name=self._get_tool_for_worker(worker_name),
            tool_args={"task": task_name}
        )
        
        # 添加随机性模拟真实执行时间变化
        import random
        actual_duration = duration * (0.5 + random.random())
        await asyncio.sleep(actual_duration)
        
        trace_manager.end_tool_call(tool_call, result="success")
        trace_manager.end_node()
        
        # 返回Coordinator
        trace_manager.start_node(
            agent_name="Coordinator",
            step_name=f"接收{worker_name}结果",
            node_type=NodeType.COORDINATOR
        )
        await asyncio.sleep(0.02)
        trace_manager.end_node()
    
    def _get_tool_for_worker(self, worker_name: str) -> str:
        """获取Worker对应的工具名"""
        mapping = {
            "SearchWorker": "web_search",
            "CodeWorker": "python_execute",
            "MathWorker": "terminate",  # 知识型worker
            "HistoryWorker": "terminate",
            "SummarizerWorker": "terminate",
            "FileWorker": "str_replace_editor",
            "BrowserWorker": "browser_use"
        }
        return mapping.get(worker_name, "terminate")
    
    async def run_dataset(
        self,
        dataset_path: Path,
        category: str,
        max_queries: int = 10
    ) -> Dict[str, Any]:
        """
        运行单个数据集的所有查询
        
        Args:
            dataset_path: 数据集JSON文件路径
            category: 数据集类别
            max_queries: 最大查询数量
        
        Returns:
            数据集执行统计
        """
        dataset_name = dataset_path.stem
        print(f"\n{'='*50}")
        print(f"📊 运行数据集: {dataset_name}")
        print(f"{'='*50}")
        
        # 加载数据
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        queries = data[:max_queries]
        total = len(queries)
        
        dataset_stats = {
            "dataset": dataset_name,
            "category": category,
            "total_queries": total,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "total_duration_ms": 0,
            "avg_duration_ms": 0,
            "max_duration_ms": 0,
            "min_duration_ms": float('inf'),
            "avg_nodes": 0,
            "avg_edges": 0,
            "cycle_count": 0,
            "query_results": []
        }
        
        # 并发执行查询
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        
        async def run_with_semaphore(idx: int, item: Dict):
            async with semaphore:
                query_id = item.get("id", f"{dataset_name}_{idx}")
                query = item.get("question") or item.get("text") or item.get("prompt") or str(item)
                
                print(f"  [{idx+1}/{total}] 运行 {query_id}...")
                
                try:
                    result = await asyncio.wait_for(
                        self.run_single_query(
                            query_id=query_id,
                            query=query,
                            category=category,
                            dataset_name=dataset_name
                        ),
                        timeout=TIMEOUT_PER_QUERY
                    )
                except asyncio.TimeoutError:
                    result = {
                        "query_id": query_id,
                        "status": "timeout",
                        "duration_ms": TIMEOUT_PER_QUERY * 1000,
                        "node_count": 0,
                        "edge_count": 0,
                        "has_cycles": False,
                        "error": "Timeout"
                    }
                
                return result
        
        # 运行所有查询
        tasks = [run_with_semaphore(i, item) for i, item in enumerate(queries)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        total_nodes = 0
        total_edges = 0
        
        for result in results:
            if isinstance(result, Exception):
                dataset_stats["failed"] += 1
                continue
                
            dataset_stats["query_results"].append(result)
            
            if result["status"] == "completed":
                dataset_stats["completed"] += 1
            elif result["status"] == "timeout":
                dataset_stats["timeout"] += 1
            else:
                dataset_stats["failed"] += 1
            
            duration = result.get("duration_ms", 0)
            dataset_stats["total_duration_ms"] += duration
            dataset_stats["max_duration_ms"] = max(dataset_stats["max_duration_ms"], duration)
            if duration > 0:
                dataset_stats["min_duration_ms"] = min(dataset_stats["min_duration_ms"], duration)
            
            total_nodes += result.get("node_count", 0)
            total_edges += result.get("edge_count", 0)
            
            if result.get("has_cycles"):
                dataset_stats["cycle_count"] += 1
        
        # 计算平均值
        if dataset_stats["completed"] > 0:
            dataset_stats["avg_duration_ms"] = dataset_stats["total_duration_ms"] / dataset_stats["completed"]
            dataset_stats["avg_nodes"] = total_nodes / dataset_stats["completed"]
            dataset_stats["avg_edges"] = total_edges / dataset_stats["completed"]
        
        if dataset_stats["min_duration_ms"] == float('inf'):
            dataset_stats["min_duration_ms"] = 0
        
        # 生成数据集分析报告
        self._generate_dataset_analysis(dataset_stats, category)
        
        print(f"\n✅ {dataset_name} 完成: {dataset_stats['completed']}/{total} 成功")
        
        return dataset_stats
    
    def _generate_dataset_analysis(self, stats: Dict[str, Any], category: str):
        """生成单个数据集的分析报告"""
        report_lines = []
        
        report_lines.append(f"# {stats['dataset']} 数据集执行分析报告")
        report_lines.append("")
        report_lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"> 数据集类别: {category}")
        report_lines.append("")
        
        # 执行概览
        report_lines.append("## 1. 执行概览")
        report_lines.append("")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| 总查询数 | {stats['total_queries']} |")
        report_lines.append(f"| 成功数 | {stats['completed']} |")
        report_lines.append(f"| 失败数 | {stats['failed']} |")
        report_lines.append(f"| 超时数 | {stats['timeout']} |")
        report_lines.append(f"| **成功率** | **{stats['completed']/stats['total_queries']*100:.1f}%** |")
        report_lines.append("")
        
        # 性能统计
        report_lines.append("## 2. 性能统计")
        report_lines.append("")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| 总耗时 | {stats['total_duration_ms']:.2f}ms ({stats['total_duration_ms']/1000:.2f}s) |")
        report_lines.append(f"| **平均耗时** | **{stats['avg_duration_ms']:.2f}ms** |")
        report_lines.append(f"| 最大耗时 | {stats['max_duration_ms']:.2f}ms |")
        report_lines.append(f"| 最小耗时 | {stats['min_duration_ms']:.2f}ms |")
        report_lines.append(f"| 平均节点数 | {stats['avg_nodes']:.1f} |")
        report_lines.append(f"| 平均边数 | {stats['avg_edges']:.1f} |")
        report_lines.append(f"| 存在环路的查询数 | {stats['cycle_count']} |")
        report_lines.append("")
        
        # 查询详情
        report_lines.append("## 3. 查询执行详情")
        report_lines.append("")
        report_lines.append("| 查询ID | 状态 | 耗时(ms) | 节点数 | 边数 | 环路 |")
        report_lines.append("|--------|------|----------|--------|------|------|")
        
        for result in stats.get("query_results", []):
            status_emoji = "✅" if result["status"] == "completed" else "❌" if result["status"] == "failed" else "⏰"
            cycle_emoji = "⚠️" if result.get("has_cycles") else "-"
            report_lines.append(
                f"| {result['query_id']} | {status_emoji} | {result['duration_ms']:.2f} | "
                f"{result['node_count']} | {result['edge_count']} | {cycle_emoji} |"
            )
        
        report_lines.append("")
        
        # 优化建议
        report_lines.append("## 4. 调度优化建议")
        report_lines.append("")
        
        insights = []
        
        # 分析耗时分布
        if stats['max_duration_ms'] > stats['avg_duration_ms'] * 3:
            insights.append(
                f"- **耗时波动大**: 最大耗时({stats['max_duration_ms']:.0f}ms)是平均值的"
                f"{stats['max_duration_ms']/stats['avg_duration_ms']:.1f}倍，建议分析长尾查询原因。"
            )
        
        if stats['cycle_count'] > 0:
            cycle_ratio = stats['cycle_count'] / stats['total_queries'] * 100
            insights.append(
                f"- **重试/循环频繁**: {cycle_ratio:.1f}%的查询存在环路，建议优化Worker执行策略减少重试。"
            )
        
        if stats['avg_nodes'] > 10:
            insights.append(
                f"- **调度开销大**: 平均{stats['avg_nodes']:.1f}个节点，考虑合并相似任务或批量处理。"
            )
        
        if stats['timeout'] > 0:
            timeout_ratio = stats['timeout'] / stats['total_queries'] * 100
            insights.append(
                f"- **超时问题**: {timeout_ratio:.1f}%的查询超时，建议增加超时阈值或优化慢查询。"
            )
        
        if not insights:
            insights.append("- 当前执行表现良好，未发现明显优化点。")
        
        report_lines.extend(insights)
        
        # 保存报告
        report_path = self.get_dataset_dir(category, stats['dataset']).parent / f"{stats['dataset']}_analysis.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        print(f"  📄 分析报告: {report_path}")
    
    def generate_overall_analysis(self, all_stats: List[Dict[str, Any]]):
        """生成所有数据集的综合分析报告"""
        report_lines = []
        
        report_lines.append("# 多Agent调度Benchmark综合分析报告")
        report_lines.append("")
        report_lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"> Benchmark ID: benchmark_{self.timestamp}")
        report_lines.append("")
        
        # 总体概览
        total_queries = sum(s['total_queries'] for s in all_stats)
        total_completed = sum(s['completed'] for s in all_stats)
        total_failed = sum(s['failed'] for s in all_stats)
        total_timeout = sum(s['timeout'] for s in all_stats)
        total_duration = sum(s['total_duration_ms'] for s in all_stats)
        
        report_lines.append("## 1. 总体概览")
        report_lines.append("")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| 数据集数量 | {len(all_stats)} |")
        report_lines.append(f"| 总查询数 | {total_queries} |")
        report_lines.append(f"| 成功数 | {total_completed} |")
        report_lines.append(f"| 失败数 | {total_failed} |")
        report_lines.append(f"| 超时数 | {total_timeout} |")
        report_lines.append(f"| **总体成功率** | **{total_completed/total_queries*100:.1f}%** |")
        report_lines.append(f"| **总耗时** | **{total_duration/1000:.2f}s** |")
        report_lines.append("")
        
        # 数据集对比
        report_lines.append("## 2. 数据集性能对比")
        report_lines.append("")
        report_lines.append("| 数据集 | 类别 | 查询数 | 成功率 | 平均耗时(ms) | 平均节点数 | 环路率 |")
        report_lines.append("|--------|------|--------|--------|--------------|----------|--------|")
        
        for stats in sorted(all_stats, key=lambda x: -x['avg_duration_ms']):
            success_rate = stats['completed'] / stats['total_queries'] * 100 if stats['total_queries'] > 0 else 0
            cycle_rate = stats['cycle_count'] / stats['total_queries'] * 100 if stats['total_queries'] > 0 else 0
            report_lines.append(
                f"| {stats['dataset']} | {stats['category']} | {stats['total_queries']} | "
                f"{success_rate:.1f}% | {stats['avg_duration_ms']:.2f} | {stats['avg_nodes']:.1f} | {cycle_rate:.1f}% |"
            )
        
        report_lines.append("")
        
        # 按类别分析
        report_lines.append("## 3. 按类别性能分析")
        report_lines.append("")
        
        categories = {}
        for stats in all_stats:
            cat = stats['category']
            if cat not in categories:
                categories[cat] = {
                    'total_queries': 0,
                    'total_duration': 0,
                    'completed': 0,
                    'datasets': []
                }
            categories[cat]['total_queries'] += stats['total_queries']
            categories[cat]['total_duration'] += stats['total_duration_ms']
            categories[cat]['completed'] += stats['completed']
            categories[cat]['datasets'].append(stats['dataset'])
        
        report_lines.append("| 类别 | 数据集数 | 总查询数 | 平均耗时(ms) | 成功率 |")
        report_lines.append("|------|----------|----------|--------------|--------|")
        
        for cat, data in sorted(categories.items(), key=lambda x: -x[1]['total_duration']):
            avg_duration = data['total_duration'] / data['completed'] if data['completed'] > 0 else 0
            success_rate = data['completed'] / data['total_queries'] * 100 if data['total_queries'] > 0 else 0
            report_lines.append(
                f"| {cat} | {len(data['datasets'])} | {data['total_queries']} | "
                f"{avg_duration:.2f} | {success_rate:.1f}% |"
            )
        
        report_lines.append("")
        
        # 瓶颈分析
        report_lines.append("## 4. 瓶颈识别与优化建议")
        report_lines.append("")
        
        # 找出最慢的数据集
        slowest = max(all_stats, key=lambda x: x['avg_duration_ms'])
        fastest = min(all_stats, key=lambda x: x['avg_duration_ms'] if x['avg_duration_ms'] > 0 else float('inf'))
        
        report_lines.append("### 4.1 耗时分析")
        report_lines.append("")
        report_lines.append(f"- **最慢数据集**: `{slowest['dataset']}` (平均 {slowest['avg_duration_ms']:.2f}ms)")
        report_lines.append(f"- **最快数据集**: `{fastest['dataset']}` (平均 {fastest['avg_duration_ms']:.2f}ms)")
        report_lines.append(f"- **速度差异**: {slowest['avg_duration_ms']/fastest['avg_duration_ms']:.1f}倍")
        report_lines.append("")
        
        # 优化建议
        report_lines.append("### 4.2 调度优化建议")
        report_lines.append("")
        
        insights = []
        
        # 按类别提供建议
        math_stats = [s for s in all_stats if s['category'] == 'math']
        if math_stats:
            avg_math_duration = sum(s['avg_duration_ms'] for s in math_stats) / len(math_stats)
            if avg_math_duration > 500:
                insights.append(
                    f"- **数学类任务优化**: 数学类任务平均耗时{avg_math_duration:.0f}ms，"
                    "建议预加载数学工具或缓存常用公式。"
                )
        
        code_stats = [s for s in all_stats if s['category'] == 'code']
        if code_stats:
            avg_code_nodes = sum(s['avg_nodes'] for s in code_stats) / len(code_stats)
            if avg_code_nodes > 8:
                insights.append(
                    f"- **代码类任务优化**: 代码类任务平均{avg_code_nodes:.0f}个节点，"
                    "建议合并代码分析和生成步骤。"
                )
        
        # 并行化建议
        total_cycle = sum(s['cycle_count'] for s in all_stats)
        if total_cycle > 5:
            insights.append(
                f"- **减少重试**: 共{total_cycle}个查询存在重试，"
                "建议增加首次执行的容错性，减少重试开销。"
            )
        
        # 负载均衡建议
        durations = [s['avg_duration_ms'] for s in all_stats if s['avg_duration_ms'] > 0]
        if durations:
            variance = sum((d - sum(durations)/len(durations))**2 for d in durations) / len(durations)
            if variance > 10000:
                insights.append(
                    "- **负载不均衡**: 不同类型任务耗时差异大，"
                    "建议根据任务类型动态调整Worker资源分配。"
                )
        
        if not insights:
            insights.append("- 当前调度表现均衡，未发现明显优化点。")
        
        report_lines.extend(insights)
        report_lines.append("")
        
        # 研究结论
        report_lines.append("## 5. 研究数据导出")
        report_lines.append("")
        report_lines.append("以下数据可用于进一步的调度优化研究：")
        report_lines.append("")
        report_lines.append(f"- 原始trace文件: `{self.output_dir}/*/`")
        report_lines.append(f"- 数据集分析: `{self.output_dir}/*_analysis.md`")
        report_lines.append(f"- JSON汇总: `{self.output_dir}/benchmark_summary.json`")
        report_lines.append("")
        report_lines.append("### 关键指标摘要")
        report_lines.append("")
        report_lines.append("```json")
        report_lines.append(json.dumps({
            "total_queries": total_queries,
            "success_rate": total_completed / total_queries * 100 if total_queries > 0 else 0,
            "avg_duration_ms": total_duration / total_completed if total_completed > 0 else 0,
            "total_duration_s": total_duration / 1000,
            "throughput_qps": total_completed / (total_duration / 1000) if total_duration > 0 else 0,
            "categories": list(categories.keys()),
            "datasets": [s['dataset'] for s in all_stats]
        }, indent=2, ensure_ascii=False))
        report_lines.append("```")
        
        # 保存报告
        report_path = self.output_dir / "overall_analysis.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        # 保存JSON汇总
        summary_path = self.output_dir / "benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": self.timestamp,
                "total_queries": total_queries,
                "total_completed": total_completed,
                "total_failed": total_failed,
                "total_timeout": total_timeout,
                "total_duration_ms": total_duration,
                "datasets": all_stats
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 综合分析报告: {report_path}")
        print(f"📋 JSON汇总: {summary_path}")


async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 多Agent调度Benchmark测试")
    print("=" * 60)
    print("目标: 收集执行trace数据用于调度优化研究")
    print("=" * 60)
    
    # 数据目录
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        print(f"\n⚠️ 数据目录不存在: {data_dir}")
        print("请先运行: python benchmarks/download_datasets.py")
        return
    
    # 初始化Runner
    runner = BenchmarkRunner()
    print(f"\n📁 输出目录: {runner.output_dir}")
    
    # 定义数据集映射
    dataset_mapping = {
        "math": ["gsm8k", "mathqa", "svamp", "sample_math"],
        "history": ["mmlu_world_history", "sample_history"],
        "social": ["socialiqa"],
        "truthful": ["truthfulqa"],
        "qa": ["natural_questions", "sample_qa"],
        "code": ["humaneval", "mbpp", "sample_code"]
    }
    
    all_stats = []
    
    # 运行各数据集
    for category, datasets in dataset_mapping.items():
        category_dir = data_dir / category
        if not category_dir.exists():
            continue
            
        for dataset_name in datasets:
            dataset_path = category_dir / f"{dataset_name}.json"
            if dataset_path.exists():
                try:
                    stats = await runner.run_dataset(
                        dataset_path=dataset_path,
                        category=category,
                        max_queries=10  # 每个数据集测试10个query
                    )
                    all_stats.append(stats)
                except Exception as e:
                    print(f"❌ 数据集 {dataset_name} 执行失败: {e}")
    
    # 生成综合分析
    if all_stats:
        print("\n" + "=" * 60)
        print("📊 生成综合分析报告")
        print("=" * 60)
        runner.generate_overall_analysis(all_stats)
    
    print("\n" + "=" * 60)
    print("✅ Benchmark测试完成!")
    print("=" * 60)
    print(f"输出目录: {runner.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
