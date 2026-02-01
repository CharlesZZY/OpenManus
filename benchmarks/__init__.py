"""
Benchmark 模块 - 多Agent调度性能测试框架

用于收集多Agent系统执行trace数据，支持调度优化研究。

主要脚本：
- download_datasets.py: 下载benchmark数据集
- run_benchmark.py: 模拟执行benchmark（快速测试）
- run_benchmark_real.py: 真实Agent执行benchmark（消耗API）

输出目录结构：
traces/
├── benchmark_{timestamp}/
│   ├── math/
│   │   ├── gsm8k/
│   │   │   ├── trace_xxx.json
│   │   │   └── trace_xxx_report.md
│   │   └── gsm8k_analysis.md
│   ├── history/
│   ├── qa/
│   ├── code/
│   └── overall_analysis.md
"""
