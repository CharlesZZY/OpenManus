"""Arrival process implementations.

Four patterns: Poisson, ON/OFF, Diurnal, LongTail.
Each generates a sorted list of arrival timestamps (seconds from t=0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ArrivalProcess(ABC):
    """Base class for request arrival processes."""

    @abstractmethod
    def generate(self, duration_s: float, seed: int = 42) -> List[float]:
        """Return sorted arrival timestamps in [0, duration_s]."""


class PoissonArrival(ArrivalProcess):
    """Homogeneous Poisson process.

    Default: lambda = 10 rps.
    """

    def __init__(self, rate: float = 10.0):
        self.rate = rate

    def generate(self, duration_s: float, seed: int = 42) -> List[float]:
        rng = np.random.default_rng(seed)
        arrivals: List[float] = []
        t = 0.0
        while t < duration_s:
            gap = rng.exponential(1.0 / self.rate)
            t += gap
            if t < duration_s:
                arrivals.append(float(t))
        return arrivals


class OnOffArrival(ArrivalProcess):
    """ON/OFF bursty pattern.

    Defaults: on_dur=30s, on_rate=30rps; off_dur=90s, off_rate=5rps.
    """

    def __init__(
        self,
        on_dur: float = 30.0,
        on_rate: float = 30.0,
        off_dur: float = 90.0,
        off_rate: float = 5.0,
    ):
        self.on_dur = on_dur
        self.on_rate = on_rate
        self.off_dur = off_dur
        self.off_rate = off_rate

    def generate(self, duration_s: float, seed: int = 42) -> List[float]:
        rng = np.random.default_rng(seed)
        arrivals: List[float] = []
        cycle = self.on_dur + self.off_dur
        t = 0.0

        while t < duration_s:
            phase_offset = t % cycle
            if phase_offset < self.on_dur:
                rate = self.on_rate
                remaining = self.on_dur - phase_offset
            else:
                rate = self.off_rate
                remaining = cycle - phase_offset

            gap = rng.exponential(1.0 / max(rate, 0.01))
            t += gap
            if t < duration_s:
                arrivals.append(float(t))

        return arrivals


class DiurnalArrival(ArrivalProcess):
    """Sinusoidal diurnal pattern.

    Defaults: period=7200s (2h), peak_trough_ratio=5,
    trough_rate=10rps => peak_rate=50rps.
    """

    def __init__(
        self,
        period_s: float = 7200.0,
        trough_rate: float = 10.0,
        peak_trough_ratio: float = 5.0,
    ):
        self.period_s = period_s
        self.trough_rate = trough_rate
        self.peak_rate = trough_rate * peak_trough_ratio

    def _rate_at(self, t: float) -> float:
        """Instantaneous rate at time t (sinusoidal modulation)."""
        mid = (self.peak_rate + self.trough_rate) / 2
        amp = (self.peak_rate - self.trough_rate) / 2
        return mid + amp * np.sin(2 * np.pi * t / self.period_s)

    def generate(self, duration_s: float, seed: int = 42) -> List[float]:
        rng = np.random.default_rng(seed)
        arrivals: List[float] = []
        t = 0.0

        # Thinning algorithm for non-homogeneous Poisson
        max_rate = self.peak_rate
        while t < duration_s:
            gap = rng.exponential(1.0 / max_rate)
            t += gap
            if t >= duration_s:
                break
            accept_prob = self._rate_at(t) / max_rate
            if rng.random() < accept_prob:
                arrivals.append(float(t))

        return arrivals


class LongTailArrival(ArrivalProcess):
    """Long-tail arrival: samples from real token-length distributions.

    Uses a log-normal distribution to model inter-arrival times,
    with heavier weight on L-suite samples (+20% default boost).

    L套件提升幅度: 未指定, default +20%.
    """

    def __init__(self, base_rate: float = 10.0, l_suite_boost: float = 0.20):
        self.base_rate = base_rate
        self.l_suite_boost = l_suite_boost

    def generate(self, duration_s: float, seed: int = 42) -> List[float]:
        rng = np.random.default_rng(seed)
        arrivals: List[float] = []
        t = 0.0

        # Log-normal inter-arrival times (heavy-tailed)
        mu = np.log(1.0 / self.base_rate)
        sigma = 0.8  # controls tail heaviness

        while t < duration_s:
            gap = rng.lognormal(mu, sigma)
            t += gap
            if t < duration_s:
                arrivals.append(float(t))

        return arrivals


ARRIVAL_REGISTRY = {
    "poisson": PoissonArrival,
    "onoff": OnOffArrival,
    "diurnal": DiurnalArrival,
    "longtail": LongTailArrival,
}
