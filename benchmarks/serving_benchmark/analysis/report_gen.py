"""Markdown report generator.

Produces ``reports/experiment_report.md`` containing:
  - Experiment overview + config table
  - Aggregated metrics table (with 95% CI)
  - CI algorithm description
  - Chart list with embedded images
  - Consolidated "unspecified items" list
  - Script manifest
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from benchmarks.serving_benchmark.core.bootstrap import CI_METHOD_DESCRIPTION

# -----------------------------------------------------------------------
# "未指定" items — centralised list
# -----------------------------------------------------------------------

UNSPECIFIED_ITEMS: List[Dict[str, str]] = [
    {
        "item": "模型规模阈值",
        "default": "small(<10B), medium(10B-70B), large(>70B)",
        "note": "按参数量分档",
    },
    {
        "item": "QoE 阈值",
        "default": "分段线性映射 (TTFT good=0.3s bad=2.0s; E2E good=5s bad=60s)",
        "note": "",
    },
    {"item": "SLO 阈值", "default": "TTFT<=0.5s, TPOT<=0.1s", "note": "可在报告中调整"},
    {"item": "vLLM max_num_seqs", "default": "64", "note": ""},
    {
        "item": "SRF predicted_out_tokens",
        "default": "历史均值 / max_tokens÷2 (冷启动)",
        "note": "",
    },
    {"item": "HumanEval pass@k, k>1 fan-out", "default": "未启用 (k=1)", "note": ""},
    {
        "item": "GSM8K 数值解析规则",
        "default": "#### (\\d+) 正则 + 末尾数字 fallback",
        "note": "",
    },
    {"item": "TruthfulQA 评估选型", "default": "MC1/MC2 (非 Truth*Info)", "note": ""},
    {
        "item": "Selective_Context 判定函数",
        "default": "弱判定: 关键句保留检查 (retention>=0.5)",
        "note": "",
    },
    {"item": "Long-tail L 套件提升幅度", "default": "+20%", "note": ""},
    {"item": "多机部署", "default": "接口预留, 未实现", "note": ""},
    {"item": "Scalability 指标", "default": "占位 (placeholder)", "note": ""},
    {"item": "GPU 采样周期", "default": "100ms", "note": ""},
    {"item": "输出目录", "default": "output/{experiment_id}/", "note": ""},
    {"item": "滑窗大小 (stability)", "default": "60s", "note": ""},
    {"item": "语义缓存 hash", "default": "SHA256(prompt[:512])", "note": ""},
]


def generate_report(
    agg_metrics_path: Path,
    configs_dir: Path,
    plots_dir: Path,
    mermaid_dir: Path,
    output_path: Path,
    experiment_id: str = "",
):
    """Generate the full experiment report in Markdown."""
    lines: List[str] = []

    # Title
    lines.append(f"# LLM 在线服务 Benchmark 实验报告")
    lines.append("")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 实验 ID: {experiment_id}")
    lines.append("")

    # 1. Experiment overview
    lines.append("## 1. 实验概览")
    lines.append("")
    _add_config_table(lines, configs_dir)

    # 2. Aggregated metrics
    lines.append("## 2. 聚合指标表 (含 95% CI)")
    lines.append("")
    _add_metrics_table(lines, agg_metrics_path)

    # 3. CI algorithm
    lines.append("## 3. 置信区间计算方法")
    lines.append("")
    lines.append(CI_METHOD_DESCRIPTION)
    lines.append("")

    # 4. Charts
    lines.append("## 4. 图表清单")
    lines.append("")
    _add_chart_list(lines, plots_dir, mermaid_dir)

    # 5. Unspecified items
    lines.append("## 5. 未指定项与默认值 (需后续确认)")
    lines.append("")
    lines.append("| # | 项目 | 默认值 | 备注 |")
    lines.append("|---|------|--------|------|")
    for i, item in enumerate(UNSPECIFIED_ITEMS, 1):
        lines.append(f"| {i} | {item['item']} | {item['default']} | {item['note']} |")
    lines.append("")

    # 6. Script manifest
    lines.append("## 6. 脚本清单")
    lines.append("")
    lines.append("| 脚本 | 用途 | 运行命令 |")
    lines.append("|------|------|---------|")
    lines.append(
        "| `scripts/run_all.py` | 一键运行全部实验 | `python -m benchmarks.serving_benchmark.scripts.run_all` |"
    )
    lines.append(
        "| `scripts/run_single.py` | 运行单组实验 | `python -m benchmarks.serving_benchmark.scripts.run_single --baseline fifo --workload poisson` |"
    )
    lines.append(
        "| `scripts/aggregate.py` | 聚合原始日志 | `python -m benchmarks.serving_benchmark.scripts.aggregate --input output/<id>/raw_logs` |"
    )
    lines.append(
        "| `scripts/plot.py` | 生成图表 | `python -m benchmarks.serving_benchmark.scripts.plot --input output/<id>` |"
    )
    lines.append(
        "| `scripts/generate_report.py` | 生成报告 | `python -m benchmarks.serving_benchmark.scripts.generate_report --input output/<id>` |"
    )
    lines.append("")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _add_config_table(lines: List[str], configs_dir: Path):
    """Append config summary table."""
    configs = list(configs_dir.glob("config_*.json"))
    if not configs:
        lines.append("*未找到配置文件*\n")
        return

    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")

    first_cfg = json.loads(configs[0].read_text())
    for k, v in first_cfg.items():
        if k != "extra":
            lines.append(f"| {k} | `{v}` |")
    lines.append(f"| 重复次数 | {len(configs)} |")
    lines.append("")


def _add_metrics_table(lines: List[str], agg_path: Path):
    """Append metrics from agg_metrics.csv."""
    if not agg_path.exists():
        lines.append("*聚合指标文件未找到*\n")
        return

    df = pd.read_csv(agg_path)

    # Select key columns to display
    display_cols = [
        c
        for c in df.columns
        if not c.startswith("_") and c != "scalability_placeholder"
    ]

    lines.append(df[display_cols].to_markdown(index=False))
    lines.append("")


def _add_chart_list(lines: List[str], plots_dir: Path, mermaid_dir: Path):
    """List generated chart files."""
    lines.append("### 图片文件")
    lines.append("")
    for img in sorted(plots_dir.glob("*.png")):
        rel = img.relative_to(plots_dir.parent.parent)
        lines.append(f"- `{rel}` — ![{img.stem}]({rel})")
    lines.append("")

    lines.append("### Mermaid 图")
    lines.append("")
    for md in sorted(mermaid_dir.glob("*.md")):
        content = md.read_text()
        lines.append(f"**{md.stem}**:")
        lines.append("")
        lines.append(content)
        lines.append("")

    lines.append("### 绘图原始数据")
    lines.append("")
    data_dir = plots_dir / "data"
    if data_dir.exists():
        for csv_file in sorted(data_dir.glob("*.csv")):
            lines.append(f"- `{csv_file.relative_to(plots_dir.parent.parent)}`")
    lines.append("")
