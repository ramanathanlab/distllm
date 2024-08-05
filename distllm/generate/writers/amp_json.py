"""Hugging face dataset writer for saving embeddings to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from typing import Optional

from distllm.utils import BaseConfig


class AMPJSONLWriterConfig(BaseConfig):
    """Configuration for the hugging face writer."""

    name: Literal['amp_jsonl'] = 'amp_jsonl'  # type: ignore[assignment]

    # Number of examples before making a new jsonl file
    chunk_size: Optional[int] = 500  # noqa: UP007
    # Base names for the chunks
    base_name: str = 'question_set'


class AMPJSONLWriter:
    """Hugging face writer for saving embeddings to disk."""

    def __init__(self, config: AMPJSONLWriterConfig) -> None:
        """Initialize the writer with the configuration."""
        self.config = config
        self.current_chunk = 0

    def write(
        self,
        output_dir: Path,
        paths: list[str],
        text: list[str],
        responses: list[str],
    ) -> None:
        """Write the embeddings to disk.

        In this case I don't want to do anything with text, because it
        is duplicate of paths (which are the full JSON entries).
        I want to dump a dictionary with the results and the
        responses to disk.

        Parameters
        ----------
        output_dir : Path
            The output directory to write the dataset to.
        paths : list[str]
            The paths for the dataset. In this case this is the original
            JSON object loaded from the input file
        text : list[str]
            The text for the dataset.
        responses : list[str]
            The responses for the dataset.
        """
        print(f'Total number of entries (paths) in writer: {len(paths)}')
        print(
            f'Total number of entries (responses) in writer: {len(responses)}',
        )
        with open(
            output_dir / f'{self.config.base_name}_{self.current_chunk}.jsonl',
            'w',
        ) as f:
            for original_entry, outputs in zip(paths, responses):
                output_entry = json.loads(original_entry)
                output_entry.update(json.loads(outputs))
                f.write(json.dumps(output_entry) + '\n')

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
