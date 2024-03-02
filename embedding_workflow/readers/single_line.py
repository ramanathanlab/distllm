"""Single sequence per line file reader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from embedding_workflow.readers.base import BaseReader
from embedding_workflow.readers.base import BaseReaderConfig


class SingleSequencePerLineReaderConfig(BaseReaderConfig):
    """Configuration for the SingleSequencePerLineReader."""

    # The name of the reader
    name: Literal['single_sequence_per_line'] = 'single_sequence_per_line'  # type: ignore[assignment]

    # The number of header lines to skip
    header_lines: int = 1


class SingleSequencePerLineReader(BaseReader):
    """Single sequence per line file reader."""

    config: SingleSequencePerLineReaderConfig

    def read(self, data_file: Path) -> list[str]:
        """Read a file with one sequence per line.

        Parameters
        ----------
        data_file : Path
            The file to read.
        """
        return data_file.read_text().splitlines()[self.config.header_lines :]
