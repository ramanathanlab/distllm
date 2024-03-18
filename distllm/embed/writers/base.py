"""Interface for writers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from distllm.embed.embedders.base import EmbedderResult
from distllm.utils import BaseConfig


class Writer(Protocol):
    """Writer protocol for all writers to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the writer with the configuration."""
        ...

    def write(self, output_dir: Path, result: EmbedderResult) -> None:
        """Write the result to disk.

        Parameters
        ----------
        output_dir : Path
            The output directory to write the result to.
        result : EmbedderResult
            The result to write to disk.
        """
        ...

    def merge(self, dataset_dirs: list[Path], output_dir: Path) -> None:
        """Merge the datasets from multiple directories.

        Parameters
        ----------
        dataset_dirs : list[Path]
            The dataset directories to merge.
        output_dir : Path
            The output directory to write the merged dataset to.
        """
        ...
