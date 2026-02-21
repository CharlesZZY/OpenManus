"""Aggregator — reads raw Parquet/CSV logs, computes metrics + bootstrap CIs.

Output: agg_metrics.csv with one row per config_id, all metric columns +
lower/upper CI bounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from benchmarks.serving_benchmark.core.bootstrap import bootstrap_all_stats
from benchmarks.serving_benchmark.core.metrics import compute_all_metrics
from benchmarks.serving_benchmark.core.schema import SLOConfig


def load_request_logs(log_dir: Path) -> pd.DataFrame:
    """Load request logs (Parquet preferred, CSV fallback)."""
    parquet = log_dir / "request_logs.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv_path = log_dir / "request_logs.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No request logs found in {log_dir}")


def load_gpu_logs(log_dir: Path) -> Optional[pd.DataFrame]:
    """Load GPU logs if available (handles seed-suffixed filenames)."""
    # Try single files first
    for name in ("gpu_samples.parquet", "gpu_samples.csv"):
        p = log_dir / name
        if p.exists():
            return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)

    # Try seed-suffixed files and merge
    frames: list[pd.DataFrame] = []
    for p in sorted(log_dir.glob("gpu_samples_*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        for p in sorted(log_dir.glob("gpu_samples_*.csv")):
            frames.append(pd.read_csv(p))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def aggregate_one_config(
    df: pd.DataFrame,
    gpu_df: Optional[pd.DataFrame] = None,
    slo: Optional[SLOConfig] = None,
) -> Dict[str, Any]:
    """Compute all metrics for a single config_id group."""
    return compute_all_metrics(df, gpu_df, slo)


def aggregate_with_ci(
    df: pd.DataFrame,
    gpu_df: Optional[pd.DataFrame] = None,
    slo: Optional[SLOConfig] = None,
    bootstrap_B: int = 10_000,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    """Compute metrics + bootstrap 95% CIs for key latency statistics."""
    base = aggregate_one_config(df, gpu_df, slo)

    # Add CIs for latency metrics
    completed = df[df["status"] == "completed"]
    if not completed.empty:
        e2e = (completed["t_finish"] - completed["t_arrive"]).values
        ci_e2e = bootstrap_all_stats(e2e, B=bootstrap_B, seed=bootstrap_seed)
        for stat_name, (point, lo, hi) in ci_e2e.items():
            base[f"e2e_{stat_name}_ci_lo"] = lo
            base[f"e2e_{stat_name}_ci_hi"] = hi

        ttft = (completed["t_first_token"] - completed["t_arrive"]).values
        ci_ttft = bootstrap_all_stats(ttft, B=bootstrap_B, seed=bootstrap_seed)
        for stat_name, (point, lo, hi) in ci_ttft.items():
            base[f"ttft_{stat_name}_ci_lo"] = lo
            base[f"ttft_{stat_name}_ci_hi"] = hi

    return base


def aggregate_experiment(
    raw_logs_dir: Path,
    output_dir: Path,
    slo: Optional[SLOConfig] = None,
) -> pd.DataFrame:
    """Aggregate all configs in an experiment directory.

    Returns a DataFrame with one row per config_id.
    """
    df = load_request_logs(raw_logs_dir)
    gpu_df = load_gpu_logs(raw_logs_dir)

    if "config_id" not in df.columns:
        df["config_id"] = "default"

    results: List[Dict[str, Any]] = []
    for config_id, group in df.groupby("config_id"):
        row = aggregate_with_ci(group, gpu_df, slo)
        row["config_id"] = config_id
        row["n_requests"] = len(group)
        row["n_completed"] = int((group["status"] == "completed").sum())
        results.append(row)

    agg_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(output_dir / "agg_metrics.csv", index=False)

    try:
        agg_df.to_parquet(output_dir / "agg_metrics.parquet", index=False)
    except Exception:
        pass

    return agg_df
