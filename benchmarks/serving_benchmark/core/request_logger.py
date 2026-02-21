"""Thread-safe request-level logger — writes Parquet + CSV.

Every completed request is appended immediately.  At shutdown the buffer
is flushed to both formats.
"""

from __future__ import annotations

import csv
import io
import threading
from pathlib import Path
from typing import List, Optional

from benchmarks.serving_benchmark.core.schema import RequestLog


class RequestLogger:
    """Append-only request log writer (thread-safe)."""

    def __init__(self, output_dir: Path, prefix: str = "request_logs"):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._lock = threading.Lock()
        self._buffer: List[dict] = []

        # Prepare CSV file with header
        self._csv_path = self._output_dir / f"{prefix}.csv"
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RequestLog.columns())
            writer.writeheader()

    def log(self, entry: RequestLog) -> None:
        """Append one request log entry."""
        row = entry.to_dict()
        with self._lock:
            self._buffer.append(row)
            # Append to CSV immediately
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RequestLog.columns())
                writer.writerow(row)

    def log_many(self, entries: List[RequestLog]) -> None:
        for e in entries:
            self.log(e)

    def flush_parquet(self) -> Optional[Path]:
        """Write buffered rows to a Parquet file. Returns path or None."""
        with self._lock:
            if not self._buffer:
                return None
            rows = list(self._buffer)

        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            parquet_path = self._output_dir / f"{self._prefix}.parquet"
            df.to_parquet(parquet_path, index=False)
            return parquet_path
        except ImportError:
            return None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_dataframe(self):
        """Return a pandas DataFrame of all logged entries."""
        import pandas as pd

        with self._lock:
            return pd.DataFrame(self._buffer)
