from __future__ import annotations

import time
from pathlib import Path


def test_distllm() -> None:
    """Test distllm."""
    import distllm

    assert distllm.__version__


def test_time_logger(capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test time logger."""
    import pandas as pd

    from distllm.timer import TimeLogger
    from distllm.timer import Timer

    # Run some arbitrary code blocks
    with Timer('func1', Path('/my/path1')):
        time.sleep(1)

    with Timer('func2', Path('/my/path2')):
        time.sleep(0.5)

    # Capture the output
    captured = capsys.readouterr()

    # Write the time log to a temporary file
    log_path = tmp_path / 'time.log'
    log_path.write_text(captured.out)

    # Parse the time log
    time_logger = TimeLogger()
    time_stats = time_logger.parse_logs(log_path)

    # Check the output
    assert time_stats[0].tags == ['func1', '/my/path1']
    assert time_stats[1].tags == ['func2', '/my/path2']

    # Check conversion to pandas
    df = pd.DataFrame(time_stats)
    num_rows = 2
    assert len(df) == num_rows
    print(df.keys())
    assert list(df.keys()) == ['tags', 'elapsed_s', 'start_unix', 'end_unix']
