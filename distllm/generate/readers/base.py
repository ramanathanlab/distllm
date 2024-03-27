"""Reader protocol for all readers to follow."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from distllm.utils import BaseConfig


class Reader(Protocol):
    """Reader protocol for all readers to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the reader with the configuration."""
        ...

    def read(self, input_path: Path) -> tuple[list[str], list[str]]:
        """Read the dataset.

        Parameters
        ----------
        input_path : Path
            The path to the dataset.

        Returns
        -------
        tuple[list[str], list[str]]
            The text and paths from the dataset.
        """
        ...
