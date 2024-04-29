"""Jsonl reader for reading text from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from distllm.utils import BaseConfig


class JsonlReaderConfig(BaseConfig):
    """Configuration for the jsonl reader."""

    name: Literal['jsonl'] = 'jsonl'  # type: ignore[assignment]
    # The field in the jsonl file that contains the text data
    text_field: str = 'text'
    # The field in the jsonl file that contains the path data
    path_field: str = 'path'


class JsonlReader:
    """Hugging face reader for reading text from disk."""

    def __init__(self, config: JsonlReaderConfig) -> None:
        """Initialize the reader with the configuration."""
        self.config = config

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
        # Read the jsonl file
        lines = input_path.read_text().strip().split('\n')
        # Parse the jsonl content into a list of dictionaries
        content = [json.loads(line) for line in lines]

        # Extract the text data
        text = [item[self.config.text_field] for item in content]

        # Extract the path data
        paths = [item[self.config.path_field] for item in content]

        return text, paths
