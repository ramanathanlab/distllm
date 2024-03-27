"""Writer protocol for all writers to follow."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from distllm.utils import BaseConfig


class Writer(Protocol):
    """Writer protocol for all writers to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the writer with the configuration."""
        ...

    def write(
        self,
        output_dir: Path,
        paths: list[str],
        text: list[str],
        responses: list[str],
    ) -> None:
        """Write the embeddings to disk.

        Parameters
        ----------
        output_dir : Path
            The output directory to write the dataset to.
        paths : list[str]
            The paths for the dataset.
        text : list[str]
            The text for the dataset.
        responses : list[str]
            The responses for the dataset.
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
