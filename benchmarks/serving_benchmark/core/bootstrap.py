"""Non-parametric bootstrap for 95% confidence intervals.

Algorithm (written into the final report):
    1. From N samples, draw N samples with replacement, repeat B=10000 times.
    2. Compute the target statistic (mean / P50 / P95 / P99) for each resample.
    3. Take the 2.5-th and 97.5-th percentiles of the B statistics as the 95% CI.
    4. For repeat=3 experiments, merge all raw logs before bootstrapping.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

DEFAULT_B = 10_000
CI_LOWER = 2.5
CI_UPPER = 97.5


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    B: int = DEFAULT_B,
    seed: int = 0,
    ci_lower: float = CI_LOWER,
    ci_upper: float = CI_UPPER,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high).

    Parameters
    ----------
    data : 1-D array of observations
    statistic_fn : callable(array) -> scalar
    B : number of bootstrap resamples
    seed : RNG seed for reproducibility
    ci_lower / ci_upper : percentile bounds (default 2.5 / 97.5 for 95% CI)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    point = float(statistic_fn(data))
    boot_stats = np.empty(B)
    for i in range(B):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(resample)

    lo = float(np.percentile(boot_stats, ci_lower))
    hi = float(np.percentile(boot_stats, ci_upper))
    return (point, lo, hi)


# Convenience wrappers for common statistics


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def _median(x: np.ndarray) -> float:
    return float(np.median(x))


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95))


def _p99(x: np.ndarray) -> float:
    return float(np.percentile(x, 99))


STAT_FUNCTIONS: Dict[str, Callable[[np.ndarray], float]] = {
    "mean": _mean,
    "p50": _median,
    "p95": _p95,
    "p99": _p99,
}


def bootstrap_all_stats(
    data: np.ndarray,
    B: int = DEFAULT_B,
    seed: int = 0,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute bootstrap CI for mean, P50, P95, P99 in one pass.

    Returns dict mapping stat_name -> (point, ci_lo, ci_hi).
    """
    results = {}
    for name, fn in STAT_FUNCTIONS.items():
        results[name] = bootstrap_ci(data, fn, B=B, seed=seed)
    return results


CI_METHOD_DESCRIPTION = """
置信区间采用非参数 Bootstrap 方法:
1. 从 N 个样本中有放回抽取 N 个样本，重复 B=10000 次
2. 每次计算目标统计量 (mean/P50/P95/P99)
3. 取 B 个统计量的 2.5% 和 97.5% 分位数作为 95% CI 边界
4. 对 repeat=3 的实验，先合并所有重复的原始日志再做 bootstrap
""".strip()
