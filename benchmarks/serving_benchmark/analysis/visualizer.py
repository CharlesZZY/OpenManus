"""Visualization module — generates all required charts + raw data exports.

Charts:
  - CDF: E2E / TTFT (single vs workflow)
  - Latency breakdown: stacked bar (queueing / scheduling / inference / post)
  - Time-series: QPS, P99, GPU util
  - Heatmap: in_tokens x out_tokens → E2E latency
  - Mermaid: single-request sequence diagram + workflow DAG
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _ensure_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib and seaborn are required for plotting")


# -----------------------------------------------------------------------
# 1. CDF plots
# -----------------------------------------------------------------------


def plot_cdf(
    df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str = "",
    group_col: str = "mode",
):
    """Plot CDF of a metric, optionally grouped by mode (single/workflow)."""
    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, group in df.groupby(group_col):
        values = group[metric_col].dropna().sort_values()
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, label=str(name), linewidth=1.5)

    ax.set_xlabel(metric_col)
    ax.set_ylabel("CDF")
    ax.set_title(title or f"CDF of {metric_col}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # Save raw data
    _save_plot_data(df[[group_col, metric_col]].dropna(), output_path)


def plot_e2e_cdf(df: pd.DataFrame, plots_dir: Path):
    completed = df[df["status"] == "completed"].copy()
    completed["e2e_latency"] = completed["t_finish"] - completed["t_arrive"]
    for mode in completed["mode"].unique():
        sub = completed[completed["mode"] == mode]
        plot_cdf(
            sub,
            "e2e_latency",
            plots_dir / f"cdf_e2e_{mode}.png",
            title=f"E2E Latency CDF ({mode})",
            group_col="dataset",
        )


def plot_ttft_cdf(df: pd.DataFrame, plots_dir: Path):
    completed = df[df["status"] == "completed"].copy()
    completed["ttft"] = completed["t_first_token"] - completed["t_arrive"]
    for mode in completed["mode"].unique():
        sub = completed[completed["mode"] == mode]
        plot_cdf(
            sub,
            "ttft",
            plots_dir / f"cdf_ttft_{mode}.png",
            title=f"TTFT CDF ({mode})",
            group_col="dataset",
        )


# -----------------------------------------------------------------------
# 2. Latency breakdown stacked bar
# -----------------------------------------------------------------------


def plot_latency_breakdown(df: pd.DataFrame, plots_dir: Path):
    _ensure_mpl()
    completed = df[df["status"] == "completed"].copy()
    completed["queue"] = completed["t_enqueue"] - completed["t_arrive"]
    completed["sched"] = completed["t_schedule"] - completed["t_enqueue"]
    completed["infer"] = completed["t_last_token"] - completed["t_schedule"]
    completed["post"] = completed["t_finish"] - completed["t_last_token"]

    if "config_id" not in completed.columns:
        completed["config_id"] = "default"

    means = completed.groupby("config_id")[["queue", "sched", "infer", "post"]].mean()

    fig, ax = plt.subplots(figsize=(max(10, len(means) * 1.5), 6))
    means.plot.bar(
        stacked=True, ax=ax, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    )
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Breakdown by Baseline")
    ax.legend(["Queueing", "Scheduling", "Inference", "Post-processing"])
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "breakdown_all.png", dpi=150)
    plt.close(fig)

    means.to_csv(plots_dir / "data" / "breakdown_data.csv")


# -----------------------------------------------------------------------
# 3. Time-series plots
# -----------------------------------------------------------------------


def _rolling_metric(
    df: pd.DataFrame, time_col: str, value_col: str, window_s: float = 10.0
):
    """Compute rolling aggregation over time windows."""
    sorted_df = df.sort_values(time_col)
    times = sorted_df[time_col].values
    values = sorted_df[value_col].values

    if len(times) == 0:
        return np.array([]), np.array([])

    t_start = times[0]
    bins = []
    bin_vals = []
    current_bin_start = t_start

    while current_bin_start < times[-1]:
        mask = (times >= current_bin_start) & (times < current_bin_start + window_s)
        if mask.any():
            bins.append(current_bin_start - t_start)
            bin_vals.append(np.mean(values[mask]))
        current_bin_start += window_s

    return np.array(bins), np.array(bin_vals)


def plot_timeseries_qps(df: pd.DataFrame, plots_dir: Path, window_s: float = 10.0):
    _ensure_mpl()
    completed = df[df["status"] == "completed"].copy()
    completed["e2e"] = completed["t_finish"] - completed["t_arrive"]

    fig, ax = plt.subplots(figsize=(12, 5))

    t_ref = completed["t_finish"].min()
    completed["rel_time"] = completed["t_finish"] - t_ref

    # Count completions per window
    bins = np.arange(0, completed["rel_time"].max() + window_s, window_s)
    counts, edges = np.histogram(completed["rel_time"].values, bins=bins)
    qps = counts / window_s
    centers = (edges[:-1] + edges[1:]) / 2

    ax.plot(centers, qps, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("QPS")
    ax.set_title("Throughput Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "ts_qps.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"time_s": centers, "qps": qps}).to_csv(
        plots_dir / "data" / "ts_qps_data.csv", index=False
    )


def plot_timeseries_p99(df: pd.DataFrame, plots_dir: Path, window_s: float = 30.0):
    _ensure_mpl()
    completed = df[df["status"] == "completed"].copy()
    completed["e2e"] = completed["t_finish"] - completed["t_arrive"]

    t_ref = completed["t_finish"].min()
    completed["rel_time"] = completed["t_finish"] - t_ref

    times_list, p99_list = [], []
    t = 0.0
    max_t = completed["rel_time"].max()
    while t + window_s <= max_t:
        mask = (completed["rel_time"] >= t) & (completed["rel_time"] < t + window_s)
        if mask.any():
            times_list.append(t + window_s / 2)
            p99_list.append(float(completed.loc[mask, "e2e"].quantile(0.99)))
        t += window_s / 2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times_list, p99_list, linewidth=1.2, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("P99 Latency (s)")
    ax.set_title("P99 Latency Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "ts_p99.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"time_s": times_list, "p99_latency": p99_list}).to_csv(
        plots_dir / "data" / "ts_p99_data.csv", index=False
    )


def plot_timeseries_gpu(gpu_df: pd.DataFrame, plots_dir: Path):
    _ensure_mpl()
    if gpu_df is None or gpu_df.empty:
        return

    t_ref = gpu_df["ts"].min()
    rel_time = gpu_df["ts"] - t_ref

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(rel_time, gpu_df["gpu_util"], linewidth=0.5, alpha=0.7)
    ax1.set_ylabel("GPU Utilization (%)")
    ax1.set_title("GPU Utilization Over Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rel_time, gpu_df["gpu_mem_used"], linewidth=0.5, alpha=0.7, color="orange")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("GPU Memory Used (MiB)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "ts_gpu.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# 4. Latency heatmap
# -----------------------------------------------------------------------


def plot_latency_heatmap(df: pd.DataFrame, plots_dir: Path, bins: int = 20):
    _ensure_mpl()
    completed = df[df["status"] == "completed"].copy()
    completed["e2e"] = completed["t_finish"] - completed["t_arrive"]

    if completed.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Bin in_tokens and out_tokens
    in_bins = pd.cut(completed["in_tokens"], bins=bins)
    out_bins = pd.cut(completed["out_tokens"], bins=bins)
    pivot = completed.groupby([in_bins, out_bins])["e2e"].mean().unstack()

    if pivot.empty:
        plt.close(fig)
        return

    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=False)
    ax.set_xlabel("out_tokens")
    ax.set_ylabel("in_tokens")
    ax.set_title("E2E Latency Heatmap (in_tokens × out_tokens)")
    fig.tight_layout()
    fig.savefig(plots_dir / "heatmap_latency.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# 5. Mermaid diagrams
# -----------------------------------------------------------------------


def generate_mermaid_single_request(mermaid_dir: Path):
    """Generate Mermaid sequence diagram for a single request flow."""
    mermaid = """\
sequenceDiagram
    participant C as Client
    participant S as Scheduler
    participant L as LLM API
    participant G as GPU

    C->>S: enqueue(request)
    Note over S: t_enqueue
    S->>S: schedule()
    Note over S: t_schedule
    S->>L: send request
    L->>G: inference start
    G-->>L: first token
    Note over L: t_first_token
    loop token generation
        G-->>L: next token
    end
    G-->>L: last token
    Note over L: t_last_token
    L-->>S: response
    S-->>C: result
    Note over C: t_finish
"""
    path = mermaid_dir / "seq_single.md"
    path.write_text(f"```mermaid\n{mermaid}```\n")


def generate_mermaid_workflow_dag(mermaid_dir: Path):
    """Generate Mermaid DAG for the 3-step workflow."""
    mermaid = """\
flowchart TD
    A[Client Request] --> B[Step 1: Analyse Problem]
    B --> C[Step 2: Reason / Retrieve]
    C --> D[Step 3: Generate Answer]
    D --> E[Quality Check]
    E --> F[Return Result]

    B -.-> |workflow_id| C
    C -.-> |workflow_id| D
"""
    path = mermaid_dir / "dag_workflow.md"
    path.write_text(f"```mermaid\n{mermaid}```\n")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _save_plot_data(df: pd.DataFrame, plot_path: Path):
    """Save raw data alongside the plot image."""
    data_dir = plot_path.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / (plot_path.stem + "_data.csv")
    df.to_csv(csv_path, index=False)


# -----------------------------------------------------------------------
# Master function
# -----------------------------------------------------------------------


def generate_all_plots(
    request_df: pd.DataFrame,
    gpu_df: Optional[pd.DataFrame],
    output_dir: Path,
):
    """Generate every chart defined in the spec."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "data").mkdir(exist_ok=True)
    mermaid_dir = output_dir / "mermaid"
    mermaid_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_MPL:
        print("WARNING: matplotlib not available — skipping image plots")
    else:
        try:
            plot_e2e_cdf(request_df, plots_dir)
        except Exception as e:
            print(f"WARNING: CDF E2E plot failed: {e}")

        try:
            plot_ttft_cdf(request_df, plots_dir)
        except Exception as e:
            print(f"WARNING: CDF TTFT plot failed: {e}")

        try:
            plot_latency_breakdown(request_df, plots_dir)
        except Exception as e:
            print(f"WARNING: Breakdown plot failed: {e}")

        try:
            plot_timeseries_qps(request_df, plots_dir)
        except Exception as e:
            print(f"WARNING: QPS timeseries failed: {e}")

        try:
            plot_timeseries_p99(request_df, plots_dir)
        except Exception as e:
            print(f"WARNING: P99 timeseries failed: {e}")

        try:
            if gpu_df is not None:
                plot_timeseries_gpu(gpu_df, plots_dir)
        except Exception as e:
            print(f"WARNING: GPU timeseries failed: {e}")

        try:
            plot_latency_heatmap(request_df, plots_dir)
        except Exception as e:
            print(f"WARNING: Heatmap failed: {e}")

    generate_mermaid_single_request(mermaid_dir)
    generate_mermaid_workflow_dag(mermaid_dir)
