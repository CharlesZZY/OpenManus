# LLM 在线服务 Benchmark + Baseline 对比实验

基于 OpenManus 的 LLM serving benchmark 系统，支持 6 个数据集、4 种负载模式、7 种调度策略和完整的实验矩阵。

## 快速开始

### 1. 安装依赖

```bash
pip install -r benchmarks/serving_benchmark/requirements.txt
```

### 2. 下载数据集

```bash
python benchmarks/download_datasets.py
```

### 3. 运行单组实验

```bash
python -m benchmarks.serving_benchmark.scripts.run_single \
    --baseline fifo \
    --workload poisson \
    --mode single \
    --max-samples 20
```

### 4. 运行全部实验矩阵

```bash
python -m benchmarks.serving_benchmark.scripts.run_all --max-samples 20
```

### 5. 聚合指标

```bash
python -m benchmarks.serving_benchmark.scripts.aggregate \
    --input output/<experiment_id>/raw_logs
```

### 6. 生成图表

```bash
python -m benchmarks.serving_benchmark.scripts.plot \
    --input output/<experiment_id>
```

### 7. 生成报告

```bash
python -m benchmarks.serving_benchmark.scripts.generate_report \
    --input output/<experiment_id>
```

## 目录结构

```
benchmarks/serving_benchmark/
├── core/                  # 日志字段、指标计算、Bootstrap CI
├── datasets/              # 数据集加载器 + 质量判定器
├── workload/              # 负载生成器 (Poisson/OnOff/Diurnal/LongTail)
├── schedulers/            # 7 个调度策略 (FIFO/SRF/StaticBatch/...)
├── runners/               # 单模型/Workflow 执行器 + 实验编排器
├── analysis/              # 聚合、可视化、报告生成
├── configs/               # YAML 配置文件
└── scripts/               # 入口脚本
```

## 输出目录

```
output/<experiment_id>/
├── raw_logs/              # Parquet + CSV 原始日志
├── agg_metrics/           # 聚合指标 CSV
├── plots/                 # PNG 图表 + data/ 子目录 (绘图原始数据)
├── configs/               # 实验配置快照 (JSON)
├── reports/               # Markdown 实验报告
└── mermaid/               # Mermaid 时序图 + 流程图
```

## 数据集与套件

| 套件 | 数据集 | 质量判定 |
|------|--------|---------|
| S (Short) | MMLU | 多选 accuracy (A/B/C/D) |
| S (Short) | TruthfulQA | MC1/MC2 |
| R (Reasoning) | GSM8K | 数值精确匹配 |
| R (Reasoning) | HumanEval | pass@k (k=1) |
| L (Long) | LongBench v2 | 多选 accuracy |
| L (Long) | Selective_Context | 关键片段 retention |

## 调度策略

| # | 策略 | 关键参数 |
|---|------|---------|
| 1 | FIFO | 无 |
| 2 | SRF (Shortest-Remaining-First) | size=in+predicted_out |
| 3 | Static Batching | batch_size=16, flush=50ms |
| 4 | Continuous Batching (vLLM-style) | max_num_seqs=64 |
| 5 | ORCA-like | iter_quantum=1, selective_batching=on |
| 6 | Latency-aware EDF | TTFT_SLO=0.5s, TPOT_SLO=0.1s |
| 7 | Parrot (workflow-aware) | cache_TTL=300s |

## 指标覆盖

E2E latency (mean/P50/P95/P99), TTFT, TPOT/ITL, latency breakdown,
throughput (QPS/tokens/s), goodput, cost (GPU-seconds/resp), SLO violation rate,
QoE, GPU utilization, fairness (Jain), stability (滑窗P99方差), scalability (预留).

## 未指定项

所有未指定的参数在报告中集中列出，附有当前默认值，便于后续调整。
详见 `analysis/report_gen.py` 中的 `UNSPECIFIED_ITEMS` 列表。
