#!/usr/bin/env python3
"""
使用真实Multi-Agent系统运行Benchmark

与 run_benchmark.py 的区别：
- 使用真实的 MultiAgentFlow 执行查询
- 收集真实的LLM调用和工具执行trace
- 用于研究真实场景下的多Agent调度性能

注意：运行此脚本会消耗API额度，建议先用模拟版本测试
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.coordinator import Coordinator
from app.agent.workers import (
    SearchWorker, CodeWorker, MathWorker, 
    CopywriterWorker, HistoryWorker, SummarizerWorker
)
from app.flow.multi_agent import MultiAgentFlow
from app.trace import TraceManager, NodeType, NodeStatus
from app.logger import logger


# 配置
MAX_CONCURRENT_TASKS = 2  # 真实执行时并发数应较低
TIMEOUT_PER_QUERY = 300   # 真实执行超时时间（秒）
MAX_QUERIES_PER_DATASET = 5  # 每个数据集最多运行的查询数


class RealBenchmarkRunner:
    """使用真实Agent的Benchmark运行器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or Path("traces") / f"real_benchmark_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total_queries": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "total_duration_ms": 0,
            "datasets": {}
        }
    
    def _create_flow_for_category(self, category: str) -> MultiAgentFlow:
        """根据数据集类别创建适合的MultiAgentFlow"""
        
        # 创建Coordinator
        coordinator = Coordinator()
        
        # 根据类别选择Worker组合
        workers = {}
        
        if category == "math":
            workers = {
                "search": SearchWorker(),
                "math": MathWorker(),
                "summarizer": SummarizerWorker()
            }
        elif category == "history":
            workers = {
                "search": SearchWorker(),
                "history": HistoryWorker(),
                "summarizer": SummarizerWorker()
            }
        elif category == "code":
            workers = {
                "code": CodeWorker(),
                "summarizer": SummarizerWorker()
            }
        else:  # qa, truthful, social
            workers = {
                "search": SearchWorker(),
                "copywriter": CopywriterWorker(),
                "summarizer": SummarizerWorker()
            }
        
        # 创建带trace的flow
        flow = MultiAgentFlow(
            coordinator=coordinator,
            workers=workers,
            enable_trace=True
        )
        
        return flow
    
    async def run_single_query(
        self,
        query_id: str,
        query: str,
        category: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """运行单个查询"""
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
            # 创建Flow
            flow = self._create_flow_for_category(category)
            
            # 执行查询
            start_time = datetime.now()
            await asyncio.wait_for(
                flow.execute(query),
                timeout=TIMEOUT_PER_QUERY
            )
            end_time = datetime.now()
            
            result["status"] = "completed"
            result["duration_ms"] = (end_time - start_time).total_seconds() * 1000
            
            # 如果flow有trace，保存它
            if flow.trace_manager and flow.trace_manager.graph:
                graph = flow.trace_manager.graph
                result["node_count"] = len(graph.nodes)
                result["edge_count"] = len(graph.edges)
                result["has_cycles"] = graph.has_cycles()
                
                # 保存trace
                save_dir = self.output_dir / category / dataset_name
                save_dir.mkdir(parents=True, exist_ok=True)
                flow.trace_manager.save_to_file(save_dir)
            
        except asyncio.TimeoutError:
            result["status"] = "timeout"
            result["error"] = f"Timeout after {TIMEOUT_PER_QUERY}s"
            result["duration_ms"] = TIMEOUT_PER_QUERY * 1000
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Query {query_id} failed: {e}")
        
        return result
    
    async def run_dataset(
        self,
        dataset_path: Path,
        category: str,
        max_queries: int = MAX_QUERIES_PER_DATASET
    ) -> Dict[str, Any]:
        """运行单个数据集"""
        dataset_name = dataset_path.stem
        print(f"\n{'='*50}")
        print(f"📊 运行数据集: {dataset_name} (真实执行)")
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
        
        # 串行执行（真实执行时避免过多并发）
        for idx, item in enumerate(queries):
            query_id = item.get("id", f"{dataset_name}_{idx}")
            query = item.get("question") or item.get("text") or item.get("prompt") or str(item)
            
            print(f"  [{idx+1}/{total}] 运行 {query_id}...")
            
            result = await self.run_single_query(
                query_id=query_id,
                query=query,
                category=category,
                dataset_name=dataset_name
            )
            
            dataset_stats["query_results"].append(result)
            
            # 更新统计
            if result["status"] == "completed":
                dataset_stats["completed"] += 1
                duration = result["duration_ms"]
                dataset_stats["total_duration_ms"] += duration
                dataset_stats["max_duration_ms"] = max(dataset_stats["max_duration_ms"], duration)
                if duration > 0:
                    dataset_stats["min_duration_ms"] = min(dataset_stats["min_duration_ms"], duration)
            elif result["status"] == "timeout":
                dataset_stats["timeout"] += 1
            else:
                dataset_stats["failed"] += 1
            
            if result.get("has_cycles"):
                dataset_stats["cycle_count"] += 1
            
            print(f"      状态: {result['status']}, 耗时: {result['duration_ms']:.0f}ms")
        
        # 计算平均值
        if dataset_stats["completed"] > 0:
            dataset_stats["avg_duration_ms"] = dataset_stats["total_duration_ms"] / dataset_stats["completed"]
            total_nodes = sum(r.get("node_count", 0) for r in dataset_stats["query_results"])
            total_edges = sum(r.get("edge_count", 0) for r in dataset_stats["query_results"])
            dataset_stats["avg_nodes"] = total_nodes / dataset_stats["completed"]
            dataset_stats["avg_edges"] = total_edges / dataset_stats["completed"]
        
        if dataset_stats["min_duration_ms"] == float('inf'):
            dataset_stats["min_duration_ms"] = 0
        
        # 生成分析报告
        self._generate_dataset_analysis(dataset_stats, category)
        
        print(f"\n✅ {dataset_name} 完成: {dataset_stats['completed']}/{total} 成功")
        return dataset_stats
    
    def _generate_dataset_analysis(self, stats: Dict[str, Any], category: str):
        """生成数据集分析报告"""
        report_lines = []
        
        report_lines.append(f"# {stats['dataset']} 真实执行分析报告")
        report_lines.append("")
        report_lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"> 执行模式: 真实Multi-Agent执行")
        report_lines.append(f"> 数据集类别: {category}")
        report_lines.append("")
        
        # 执行概览
        report_lines.append("## 执行概览")
        report_lines.append("")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| 总查询数 | {stats['total_queries']} |")
        report_lines.append(f"| 成功数 | {stats['completed']} |")
        report_lines.append(f"| 失败数 | {stats['failed']} |")
        report_lines.append(f"| 超时数 | {stats['timeout']} |")
        success_rate = stats['completed']/stats['total_queries']*100 if stats['total_queries'] > 0 else 0
        report_lines.append(f"| **成功率** | **{success_rate:.1f}%** |")
        report_lines.append("")
        
        # 性能统计
        report_lines.append("## 性能统计")
        report_lines.append("")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| 总耗时 | {stats['total_duration_ms']/1000:.2f}s |")
        report_lines.append(f"| **平均耗时** | **{stats['avg_duration_ms']:.2f}ms** |")
        report_lines.append(f"| 最大耗时 | {stats['max_duration_ms']:.2f}ms |")
        report_lines.append(f"| 最小耗时 | {stats['min_duration_ms']:.2f}ms |")
        report_lines.append(f"| 平均节点数 | {stats['avg_nodes']:.1f} |")
        report_lines.append(f"| 平均边数 | {stats['avg_edges']:.1f} |")
        report_lines.append(f"| 环路查询数 | {stats['cycle_count']} |")
        report_lines.append("")
        
        # 查询详情
        report_lines.append("## 查询详情")
        report_lines.append("")
        report_lines.append("| 查询ID | 状态 | 耗时(ms) | 节点数 | 边数 | 环路 | 错误 |")
        report_lines.append("|--------|------|----------|--------|------|------|------|")
        
        for result in stats.get("query_results", []):
            status_emoji = "✅" if result["status"] == "completed" else "❌" if result["status"] == "failed" else "⏰"
            cycle_emoji = "⚠️" if result.get("has_cycles") else "-"
            error = result.get("error", "-")[:20] if result.get("error") else "-"
            report_lines.append(
                f"| {result['query_id']} | {status_emoji} | {result['duration_ms']:.0f} | "
                f"{result['node_count']} | {result['edge_count']} | {cycle_emoji} | {error} |"
            )
        
        # 保存报告
        report_dir = self.output_dir / category
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{stats['dataset']}_analysis.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        print(f"  📄 分析报告: {report_path}")
    
    def generate_overall_analysis(self, all_stats: List[Dict[str, Any]]):
        """生成综合分析报告"""
        report_lines = []
        
        report_lines.append("# 真实Multi-Agent执行综合分析报告")
        report_lines.append("")
        report_lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"> Benchmark ID: real_benchmark_{self.timestamp}")
        report_lines.append("")
        
        total_queries = sum(s['total_queries'] for s in all_stats)
        total_completed = sum(s['completed'] for s in all_stats)
        total_duration = sum(s['total_duration_ms'] for s in all_stats)
        
        report_lines.append("## 总体概览")
        report_lines.append("")
        report_lines.append("| 指标 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| 数据集数量 | {len(all_stats)} |")
        report_lines.append(f"| 总查询数 | {total_queries} |")
        report_lines.append(f"| 成功数 | {total_completed} |")
        success_rate = total_completed/total_queries*100 if total_queries > 0 else 0
        report_lines.append(f"| **成功率** | **{success_rate:.1f}%** |")
        report_lines.append(f"| **总耗时** | **{total_duration/1000:.2f}s** |")
        avg_duration = total_duration/total_completed if total_completed > 0 else 0
        report_lines.append(f"| 平均耗时 | {avg_duration:.2f}ms |")
        throughput = total_completed/(total_duration/1000) if total_duration > 0 else 0
        report_lines.append(f"| **吞吐量** | **{throughput:.2f} QPS** |")
        report_lines.append("")
        
        # 数据集对比
        report_lines.append("## 数据集性能对比")
        report_lines.append("")
        report_lines.append("| 数据集 | 类别 | 成功/总数 | 平均耗时(ms) | 节点数 | 边数 |")
        report_lines.append("|--------|------|-----------|--------------|--------|------|")
        
        for stats in sorted(all_stats, key=lambda x: -x['avg_duration_ms']):
            report_lines.append(
                f"| {stats['dataset']} | {stats['category']} | "
                f"{stats['completed']}/{stats['total_queries']} | "
                f"{stats['avg_duration_ms']:.0f} | {stats['avg_nodes']:.1f} | {stats['avg_edges']:.1f} |"
            )
        
        report_lines.append("")
        
        # 调度优化建议
        report_lines.append("## 调度优化建议")
        report_lines.append("")
        
        if total_completed > 0:
            # 找出瓶颈
            slowest = max(all_stats, key=lambda x: x['avg_duration_ms'])
            report_lines.append(f"- **最慢类型**: `{slowest['category']}` 类任务平均耗时 {slowest['avg_duration_ms']:.0f}ms")
            
            # 分析节点开销
            avg_nodes = sum(s['avg_nodes'] for s in all_stats) / len(all_stats)
            if avg_nodes > 10:
                report_lines.append(f"- **调度开销大**: 平均 {avg_nodes:.0f} 个节点/查询，建议合并相似操作")
            
            # 分析并行潜力
            report_lines.append("- **并行化建议**: 当前为串行执行，可根据Worker依赖关系实现并行调度")
        
        # 保存报告
        report_path = self.output_dir / "overall_analysis.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        # 保存JSON
        summary_path = self.output_dir / "benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": self.timestamp,
                "mode": "real_execution",
                "total_queries": total_queries,
                "total_completed": total_completed,
                "total_duration_ms": total_duration,
                "throughput_qps": throughput,
                "datasets": all_stats
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 综合分析: {report_path}")
        print(f"📋 JSON汇总: {summary_path}")


async def main():
    """主函数"""
    print("=" * 60)
    print("🚀 真实Multi-Agent Benchmark测试")
    print("=" * 60)
    print("⚠️ 注意: 此脚本会调用真实LLM API，消耗API额度")
    print("=" * 60)
    
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        print(f"\n⚠️ 数据目录不存在: {data_dir}")
        print("请先运行: python benchmarks/download_datasets.py")
        return
    
    runner = RealBenchmarkRunner()
    print(f"\n📁 输出目录: {runner.output_dir}")
    
    # 使用示例数据集进行测试（节省API消耗）
    test_datasets = [
        ("math", "sample_math"),
        ("history", "sample_history"),
        ("qa", "sample_qa"),
        ("code", "sample_code")
    ]
    
    all_stats = []
    
    for category, dataset_name in test_datasets:
        dataset_path = data_dir / category / f"{dataset_name}.json"
        if dataset_path.exists():
            try:
                stats = await runner.run_dataset(
                    dataset_path=dataset_path,
                    category=category,
                    max_queries=3  # 每个数据集只测试3个query
                )
                all_stats.append(stats)
            except Exception as e:
                print(f"❌ 数据集 {dataset_name} 执行失败: {e}")
    
    if all_stats:
        runner.generate_overall_analysis(all_stats)
    
    print("\n" + "=" * 60)
    print("✅ 真实Benchmark测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
