"""Metric computation functions — full coverage of the specification.

Every public function takes a pandas DataFrame of RequestLog rows (and
optionally a GPU-log DataFrame) and returns a flat dict of metric values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from benchmarks.serving_benchmark.core.schema import SLOConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return np.nan
    return float(series.quantile(q))


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.mean())


# ---------------------------------------------------------------------------
# 1. End-to-End latency
# ---------------------------------------------------------------------------


def compute_e2e_latency(df: pd.DataFrame) -> Dict[str, float]:
    """E2E latency = t_finish - t_arrive (seconds)."""
    lat = df["t_finish"] - df["t_arrive"]
    return {
        "e2e_mean": _safe_mean(lat),
        "e2e_p50": _safe_quantile(lat, 0.50),
        "e2e_p95": _safe_quantile(lat, 0.95),
        "e2e_p99": _safe_quantile(lat, 0.99),
    }


# ---------------------------------------------------------------------------
# 2. Time to First Token
# ---------------------------------------------------------------------------


def compute_ttft(df: pd.DataFrame) -> Dict[str, float]:
    ttft = df["t_first_token"] - df["t_arrive"]
    return {
        "ttft_mean": _safe_mean(ttft),
        "ttft_p50": _safe_quantile(ttft, 0.50),
        "ttft_p95": _safe_quantile(ttft, 0.95),
        "ttft_p99": _safe_quantile(ttft, 0.99),
    }


# ---------------------------------------------------------------------------
# 3. TPOT / ITL (per-output-token latency)
# ---------------------------------------------------------------------------


def compute_tpot(df: pd.DataFrame) -> Dict[str, float]:
    """TPOT = (t_last_token - t_first_token) / (out_tokens - 1).

    Requests with out_tokens <= 1 are excluded.
    """
    mask = df["out_tokens"] > 1
    sub = df.loc[mask].copy()
    if sub.empty:
        return {
            "tpot_mean": np.nan,
            "tpot_p50": np.nan,
            "tpot_p95": np.nan,
            "tpot_p99": np.nan,
        }
    tpot = (sub["t_last_token"] - sub["t_first_token"]) / (sub["out_tokens"] - 1)
    return {
        "tpot_mean": _safe_mean(tpot),
        "tpot_p50": _safe_quantile(tpot, 0.50),
        "tpot_p95": _safe_quantile(tpot, 0.95),
        "tpot_p99": _safe_quantile(tpot, 0.99),
    }


# ---------------------------------------------------------------------------
# 4. Latency breakdown (queueing / scheduling / inference / post)
# ---------------------------------------------------------------------------


def compute_breakdown(df: pd.DataFrame) -> Dict[str, float]:
    queue = df["t_enqueue"] - df["t_arrive"]
    sched = df["t_schedule"] - df["t_enqueue"]
    infer = df["t_last_token"] - df["t_schedule"]
    post = df["t_finish"] - df["t_last_token"]
    return {
        "breakdown_queue_mean": _safe_mean(queue),
        "breakdown_sched_mean": _safe_mean(sched),
        "breakdown_infer_mean": _safe_mean(infer),
        "breakdown_post_mean": _safe_mean(post),
    }


# ---------------------------------------------------------------------------
# 5. Throughput (QPS, tokens/s)
# ---------------------------------------------------------------------------


def compute_throughput(df: pd.DataFrame) -> Dict[str, float]:
    completed = df[df["status"] == "completed"]
    if completed.empty:
        return {"qps": 0.0, "tokens_per_sec": 0.0}
    duration = completed["t_finish"].max() - completed["t_arrive"].min()
    if duration <= 0:
        return {"qps": 0.0, "tokens_per_sec": 0.0}
    qps = len(completed) / duration
    tps = completed["out_tokens"].sum() / duration
    return {"qps": float(qps), "tokens_per_sec": float(tps)}


# ---------------------------------------------------------------------------
# 6. Goodput (SLO-satisfying AND quality_ok)
# ---------------------------------------------------------------------------


def compute_goodput(
    df: pd.DataFrame, slo: Optional[SLOConfig] = None
) -> Dict[str, float]:
    slo = slo or SLOConfig()
    completed = df[df["status"] == "completed"]
    if completed.empty:
        return {"goodput_rate": 0.0, "goodput_count": 0}

    ttft = completed["t_first_token"] - completed["t_arrive"]
    tpot_mask = completed["out_tokens"] > 1
    tpot = pd.Series(np.nan, index=completed.index)
    if tpot_mask.any():
        sub = completed.loc[tpot_mask]
        tpot.loc[tpot_mask] = (sub["t_last_token"] - sub["t_first_token"]) / (
            sub["out_tokens"] - 1
        )

    slo_ok = (ttft <= slo.ttft_threshold) & (tpot.fillna(0) <= slo.tpot_threshold)
    good = slo_ok & completed["quality_ok"]
    return {
        "goodput_rate": float(good.sum() / len(completed)),
        "goodput_count": int(good.sum()),
    }


# ---------------------------------------------------------------------------
# 7. Cost (GPU-seconds per response)
# ---------------------------------------------------------------------------


def compute_cost(
    df: pd.DataFrame, gpu_df: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    completed = df[df["status"] == "completed"]
    if completed.empty or gpu_df is None or gpu_df.empty:
        return {"gpu_seconds_per_resp": np.nan}

    total_time = gpu_df["ts"].max() - gpu_df["ts"].min()
    if total_time <= 0:
        return {"gpu_seconds_per_resp": np.nan}

    sample_interval = total_time / max(len(gpu_df) - 1, 1)
    gpu_seconds = (gpu_df["gpu_util"] / 100.0).sum() * sample_interval
    return {"gpu_seconds_per_resp": float(gpu_seconds / len(completed))}


# ---------------------------------------------------------------------------
# 8. SLO violation rate
# ---------------------------------------------------------------------------


def compute_slo_violation(
    df: pd.DataFrame, slo: Optional[SLOConfig] = None
) -> Dict[str, float]:
    slo = slo or SLOConfig()
    completed = df[df["status"] == "completed"]
    if completed.empty:
        return {"slo_violation_rate": 0.0}

    e2e = completed["t_finish"] - completed["t_arrive"]
    ttft = completed["t_first_token"] - completed["t_arrive"]
    violated = (ttft > slo.ttft_threshold) | (e2e > slo.e2e_threshold)
    return {"slo_violation_rate": float(violated.sum() / len(completed))}


# ---------------------------------------------------------------------------
# 9. QoE — piecewise linear mapping (thresholds 未指定, using defaults)
# ---------------------------------------------------------------------------


def _piecewise_qoe(value: float, good: float, bad: float) -> float:
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return float(1.0 - (value - good) / (bad - good))


def compute_qoe(df: pd.DataFrame, slo: Optional[SLOConfig] = None) -> Dict[str, float]:
    slo = slo or SLOConfig()
    completed = df[df["status"] == "completed"]
    if completed.empty:
        return {"qoe_mean": np.nan}

    ttft = (completed["t_first_token"] - completed["t_arrive"]).values
    e2e = (completed["t_finish"] - completed["t_arrive"]).values

    scores = []
    for t, e in zip(ttft, e2e):
        q_ttft = _piecewise_qoe(t, slo.qoe_ttft_good, slo.qoe_ttft_bad)
        q_e2e = _piecewise_qoe(e, slo.qoe_e2e_good, slo.qoe_e2e_bad)
        scores.append(0.5 * q_ttft + 0.5 * q_e2e)

    return {"qoe_mean": float(np.mean(scores))}


# ---------------------------------------------------------------------------
# 10. GPU utilization (time-weighted average)
# ---------------------------------------------------------------------------


def compute_gpu_util(gpu_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    if gpu_df is None or gpu_df.empty:
        return {"gpu_util_avg": np.nan, "gpu_mem_avg": np.nan}
    return {
        "gpu_util_avg": float(gpu_df["gpu_util"].mean()),
        "gpu_mem_avg": float(gpu_df["gpu_mem_used"].mean()),
    }


# ---------------------------------------------------------------------------
# 11. Fairness — Jain's index across datasets
# ---------------------------------------------------------------------------


def compute_fairness(df: pd.DataFrame) -> Dict[str, float]:
    completed = df[df["status"] == "completed"]
    if completed.empty:
        return {"jain_fairness": np.nan}

    per_ds = completed.groupby("dataset").apply(
        lambda g: (g["t_finish"] - g["t_arrive"]).mean()
    )
    if per_ds.empty or (per_ds == 0).all():
        return {"jain_fairness": np.nan}

    x = per_ds.values
    n = len(x)
    jain = float((x.sum() ** 2) / (n * (x**2).sum()))
    return {"jain_fairness": jain}


# ---------------------------------------------------------------------------
# 12. Stability — variance of sliding-window P99
# ---------------------------------------------------------------------------


def compute_stability(df: pd.DataFrame, window_s: float = 60.0) -> Dict[str, float]:
    """Variance of P99 latency computed over rolling *window_s*-second windows."""
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        return {"stability_p99_var": np.nan}

    completed = completed.sort_values("t_finish")
    e2e = (completed["t_finish"] - completed["t_arrive"]).values
    ts = completed["t_finish"].values

    if len(ts) == 0:
        return {"stability_p99_var": np.nan}

    t_start = ts[0]
    t_end = ts[-1]
    if t_end - t_start < window_s:
        return {"stability_p99_var": 0.0}

    p99_values = []
    win_start = t_start
    while win_start + window_s <= t_end:
        mask = (ts >= win_start) & (ts < win_start + window_s)
        if mask.any():
            p99_values.append(float(np.percentile(e2e[mask], 99)))
        win_start += window_s / 2  # 50% overlap

    if len(p99_values) < 2:
        return {"stability_p99_var": 0.0}

    return {"stability_p99_var": float(np.var(p99_values))}


# ---------------------------------------------------------------------------
# 13. Scalability — placeholder (未指定)
# ---------------------------------------------------------------------------


def compute_scalability(**kwargs: Any) -> Dict[str, float]:
    """Scale-out metrics — NOT SPECIFIED. Returns placeholder values."""
    return {
        "scalability_placeholder": np.nan,
        "_note": "scalability metrics not specified; interface reserved",
    }


# ---------------------------------------------------------------------------
# Master aggregator — computes ALL metrics at once
# ---------------------------------------------------------------------------


def compute_all_metrics(
    df: pd.DataFrame,
    gpu_df: Optional[pd.DataFrame] = None,
    slo: Optional[SLOConfig] = None,
    window_s: float = 60.0,
) -> Dict[str, Any]:
    """Compute every metric family and return a flat dict."""
    result: Dict[str, Any] = {}
    result.update(compute_e2e_latency(df))
    result.update(compute_ttft(df))
    result.update(compute_tpot(df))
    result.update(compute_breakdown(df))
    result.update(compute_throughput(df))
    result.update(compute_goodput(df, slo))
    result.update(compute_cost(df, gpu_df))
    result.update(compute_slo_violation(df, slo))
    result.update(compute_qoe(df, slo))
    result.update(compute_gpu_util(gpu_df))
    result.update(compute_fairness(df))
    result.update(compute_stability(df, window_s))
    result.update(compute_scalability())
    return result
