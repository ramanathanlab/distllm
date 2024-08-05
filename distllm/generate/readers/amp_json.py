"""Jsonl reader for reading text from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from distllm.utils import BaseConfig


class AMPJsonReaderConfig(BaseConfig):
    """Configuration for the jsonl reader."""

    name: Literal['amp_json'] = 'amp_json'  # type: ignore[assignment]
    # The field in the jsonl file that contains the text data


class AMPJsonReader:
    """Reader for the AMP json dataset."""

    def __init__(self, config: AMPJsonReaderConfig) -> None:
        """Initialize the reader with the configuration."""
        self.config = config

    def read(self, input_path: Path) -> tuple[list[str], list[str]]:
        """Read the dataset.

        We need the path to be the entry here, because text will be
        subsumed by the prompt, and we need a way to keep track of all
        the other metadata.

        Parameters
        ----------
        input_path : Path
            The path to the dataset.

        Returns
        -------
        tuple[list[str], list[str]]
            The text and paths from the dataset.
        """
        with open(input_path) as f:
            data = json.load(f)

        text, paths = [], []
        for _, entries in data.items():
            for entry in entries:
                entry_text = json.dumps(entry)
                text.append(entry_text)
                paths.append(entry_text)

        return text, paths
