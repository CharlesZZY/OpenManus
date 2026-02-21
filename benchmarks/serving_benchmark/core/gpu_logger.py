"""GPU sampling logger — background thread polling at ~100ms intervals.

Sampling period: 100ms (未指定, configurable).

Strategy:
  1. Primary: pynvml (lower latency, no subprocess overhead)
  2. Fallback: nvidia-smi subprocess polling

Output: gpu_samples_{config_id}_{seed}.csv  with columns (ts, gpu_util, gpu_mem_used).
"""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from benchmarks.serving_benchmark.core.schema import GPULog


class GPULogger:
    """Background GPU sampler."""

    def __init__(
        self,
        output_dir: Path,
        sample_interval_s: float = 0.1,
        prefix: str = "gpu_samples",
    ):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._interval = sample_interval_s
        self._prefix = prefix

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer: List[dict] = []
        self._lock = threading.Lock()

        self._use_pynvml = False
        self._nvml_handle = None
        self._init_backend()

    def _init_backend(self):
        """Try pynvml, fall back to nvidia-smi."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._use_pynvml = True
        except Exception:
            self._use_pynvml = False

    def _sample_pynvml(self) -> GPULog:
        import pynvml

        util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        return GPULog(
            ts=time.time(),
            gpu_util=float(util.gpu),
            gpu_mem_used=float(mem.used) / (1024 * 1024),  # bytes → MiB
        )

    def _sample_nvidia_smi(self) -> GPULog:
        try:
            out = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode()
                .strip()
            )
            parts = out.split(",")
            return GPULog(
                ts=time.time(),
                gpu_util=float(parts[0].strip()),
                gpu_mem_used=float(parts[1].strip()),
            )
        except Exception:
            return GPULog(ts=time.time(), gpu_util=0.0, gpu_mem_used=0.0)

    def _sample(self) -> GPULog:
        if self._use_pynvml and self._nvml_handle is not None:
            try:
                return self._sample_pynvml()
            except Exception:
                pass
        return self._sample_nvidia_smi()

    def _loop(self):
        while self._running:
            entry = self._sample()
            with self._lock:
                self._buffer.append(entry.to_dict())
            time.sleep(self._interval)

    def start(self):
        """Start background sampling."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and flush to CSV."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._flush()

    def _flush(self):
        with self._lock:
            rows = list(self._buffer)
        if not rows:
            return
        csv_path = self._output_dir / f"{self._prefix}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GPULog.columns())
            writer.writeheader()
            writer.writerows(rows)

        # Also write Parquet if possible
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_parquet(self._output_dir / f"{self._prefix}.parquet", index=False)
        except ImportError:
            pass

    def get_dataframe(self):
        """Return a pandas DataFrame of all samples."""
        import pandas as pd

        with self._lock:
            return pd.DataFrame(self._buffer)
