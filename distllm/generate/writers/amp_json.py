"""Hugging face dataset writer for saving embeddings to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Optional

from datasets import concatenate_datasets
from datasets import Dataset
from tqdm import tqdm
import json

from distllm.utils import BaseConfig


class AMPJSONLWriterConfig(BaseConfig):
    """Configuration for the hugging face writer."""

    name: Literal['amp_jsonl'] = 'amp_jsonl'  # type: ignore[assignment]

    # Number of examples before making a new jsonl file
    chunk_size: Optional[int] = 500
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

        In this case I don't want to do anything with text, because it is duplicate of
        paths (which are the full JSON entries). I want to dump a dictionary with the results
        and the responses to disk.

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

        for i in range(0, len(paths), self.config.chunk_size):
            with open(output_dir / f'{self.config.base_name}_{self.current_chunk}.jsonl', 'w') as f:
                for j in range(i, min(i + self.config.chunk_size, len(paths))):
                    entry = json.loads(paths[j])
                    entry.update(json.loads(responses[j]))
                    f.write(json.dumps(entry) + '\n')
