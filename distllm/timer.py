"""Timing utilities.

Code adapted from:
https://github.com/proxystore/proxystore/blob/ed774b6b6e26ccf58e90d761a68b099f5c6f90dc/proxystore/utils/timer.py#L15
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any
from typing import NamedTuple
from typing import Sequence
from typing import Union

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self

PathLike = Union[Path, str]


class TimeStats(NamedTuple):
    """Time statistics for different events."""

    tags: Sequence[str]
    elapsed_s: float
    start_unix: float
    end_unix: float


class Timer:
    """Performance timer with nanosecond precision.

    Example:
        ```python
        from distllm.timer import Timer

        with Timer() as timer:
            ...

        print(timer.elapsed_ms)
        ```

    Example:
        ```python
        from distllm.timer import Timer

        timer = Timer("my_run")
        timer.start()
        ...
        timer.stop()

        print(timer.elapsed_ms)
        ```

    Raises
    ------
        RuntimeError: If the elapsed time is accessed before the timer is
            stopped or the context block is exited.
    """

    def __init__(self, *tags: Any) -> None:
        self.tags = tags
        self._start = 0
        self._end = 0
        self._start_unix = 0.0
        self._end_unix = 0.0
        self._running = False

    def __enter__(self) -> Self:
        """Start the timer."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Stop the timer."""
        self.stop()

    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        if self._running:
            raise RuntimeError('Timer is still running!')
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1e6

    @property
    def elapsed_s(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1e9

    def start(self) -> Self:
        """Start the timer."""
        self._running = True
        self._start = time.perf_counter_ns()
        self._start_unix = time.time()
        return self

    def stop(self) -> None:
        """Stop the timer."""
        self._end = time.perf_counter_ns()
        self._end_unix = time.time()
        self._running = False
        time_stats = TimeStats(
            tags=self.tags,
            elapsed_s=self.elapsed_s,
            start_unix=self._start_unix,
            end_unix=self._end_unix,
        )
        TimeLogger().log(time_stats)


class TimeLogger:
    """Log times for different events."""

    def parse_logs(self, log_path: PathLike) -> list[TimeStats]:
        """Parse the logs into a pandas DataFrame."""
        # Read each line of the file
        lines = Path(log_path).read_text().strip().split('\n')

        # Parse out any lines that don't contain the timer information
        # lines = [line for line in lines if line.startswith('[timer]')]
        lines = [line for line in lines if '[timer]' in line]

        # Regex pattern to extract items in square brackets []
        regex_pattern = r'\[([^\[\]]+)\]'

        time_stats = []
        # Extracted items from all print statements
        for line in lines:
            match = re.findall(regex_pattern, line)
            time_stats.append(
                TimeStats(
                    tags=match[1].split(),
                    elapsed_s=match[2],
                    start_unix=match[3],
                    end_unix=match[4],
                ),
            )

        return time_stats

    def log(self, ts: TimeStats) -> None:
        """Log the timer information."""
        print(
            f'[timer] [{" ".join(map(str, ts.tags))}]'
            f' in [{ts.elapsed_s:.2f}] seconds.',
            f' start: [{ts.start_unix:.2f}], end: [{ts.end_unix:.2f}]',
        )
