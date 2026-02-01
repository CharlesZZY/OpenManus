# 多Agent调度Benchmark测试框架

本框架用于收集多Agent系统的执行trace数据，支持调度优化研究。

## 目标

- 收集不同类型任务的执行流程数据
- 分析端到端延迟的组成
- 识别调度瓶颈和优化点
- 为提升吞吐量提供数据支撑

## 快速开始

### 1. 下载数据集

```bash
cd /path/to/OpenManus
python benchmarks/download_datasets.py
```

支持的数据集：
- **数学推理**: GSM8K, MathQA, SVAMP
- **历史/社会**: MMLU (world_history), SocialIQA
- **真实性/常识**: TruthfulQA, Natural Questions
- **代码生成**: HumanEval, MBPP

### 2. 运行Benchmark

#### 模拟执行（推荐先用这个测试）

```bash
python benchmarks/run_benchmark.py
```

特点：
- 不消耗API额度
- 模拟多Agent执行流程
- 快速收集trace结构数据

#### 真实执行

```bash
python benchmarks/run_benchmark_real.py
```

特点：
- 使用真实LLM和工具
- 消耗API额度
- 收集真实的执行性能数据

## 输出说明

### 目录结构

```
traces/
├── benchmark_{timestamp}/
│   ├── math/
│   │   ├── gsm8k/
│   │   │   ├── trace_20260202_xxx.json      # 原始trace数据
│   │   │   └── trace_20260202_xxx_report.md # 单次执行报告
│   │   ├── mathqa/
│   │   └── gsm8k_analysis.md                # 数据集分析报告
│   ├── history/
│   ├── qa/
│   ├── code/
│   ├── benchmark_summary.json               # JSON格式汇总
│   └── overall_analysis.md                  # 综合分析报告
```

### 报告内容

每个分析报告包含：

1. **执行概览**: 总查询数、成功率、失败率、超时率
2. **性能统计**: 总耗时、平均耗时、最大/最小耗时
3. **执行详情**: 每个查询的详细timing数据
4. **调度分析**: 节点数、边数、环路检测
5. **优化建议**: 基于数据的调度优化建议

## 研究数据

### JSON Schema

```json
{
  "graph_id": "trace_xxx",
  "request": "查询内容",
  "start_time": "2026-02-02T00:00:00.000000",
  "end_time": "2026-02-02T00:00:01.000000",
  "duration_ms": 1000.0,
  "status": "completed",
  "nodes": {
    "node_1": {
      "node_id": "node_1",
      "node_type": "coordinator",
      "agent_name": "Coordinator",
      "step_name": "分析任务",
      "start_time": "...",
      "end_time": "...",
      "duration_ms": 100.0,
      "status": "completed",
      "tool_calls": [...]
    }
  },
  "edges": [
    {
      "source_node_id": "node_1",
      "target_node_id": "node_2",
      "edge_type": "delegate"
    }
  ],
  "has_cycles": false
}
```

### 关键指标

| 指标 | 说明 | 优化意义 |
|------|------|----------|
| duration_ms | 端到端延迟 | 主要优化目标 |
| node_count | 执行步骤数 | 调度开销 |
| edge_count | 状态转换数 | 通信开销 |
| has_cycles | 是否有重试/循环 | 执行效率 |
| tool_call_duration | 工具调用耗时 | IO瓶颈 |

## 配置参数

在脚本中可调整的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MAX_CONCURRENT_TASKS | 3 | 最大并发数 |
| TIMEOUT_PER_QUERY | 120 | 查询超时(秒) |
| MAX_QUERIES_PER_DATASET | 10 | 每数据集最大查询数 |

## 扩展

### 添加新数据集

1. 在 `download_datasets.py` 中添加下载函数
2. 在 `run_benchmark.py` 的 `dataset_mapping` 中添加映射
3. 如需特殊处理，在 `_execute_with_agents` 中添加分支

### 自定义Agent组合

修改 `_create_flow_for_category` 方法来定义不同类型任务使用的Worker组合。

## 参考

- [OpenManus项目](https://github.com/xxx/OpenManus)
- [Trace模块文档](../app/trace/README.md)
