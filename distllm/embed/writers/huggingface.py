"""Hugging face dataset writer for saving embeddings to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterator
from typing import Literal
from typing import Optional

import numpy as np
from datasets import concatenate_datasets
from datasets import Dataset

from distllm.embed.embedders.base import EmbedderResult
from distllm.utils import BaseConfig


def _generate_dataset(
    embeddings: np.ndarray,
    text: list[str],
    metadata: list[dict[str, Any]] | None,
) -> Iterator[dict[str, str | np.ndarray | Any]]:
    """Generate a dataset from the embeddings, text, and metadata."""
    for idx, (text_, embedding) in enumerate(zip(text, embeddings)):
        # Always include the text and the embeddings
        item = {'text': text_, 'embeddings': embedding}

        # Add metadata if available
        if metadata is not None:
            for key, value in metadata[idx].items():
                item[key] = value

        yield item


class HuggingFaceWriterConfig(BaseConfig):
    """Configuration for the hugging face writer."""

    name: Literal['huggingface'] = 'huggingface'  # type: ignore[assignment]

    # The number of processes to use for writing the dataset
    num_proc: Optional[int] = None  # noqa: UP007


class HuggingFaceWriter:
    """Hugging face writer for saving embeddings to disk."""

    def __init__(self, config: HuggingFaceWriterConfig) -> None:
        """Initialize the writer with the configuration."""
        self.config = config

    def write(self, output_dir: Path, result: EmbedderResult) -> None:
        """Write the embeddings to disk.

        Parameters
        ----------
        result : EmbedderResult
            The result to write to disk.
        """
        # TODO: Use Dataset.from_dict instead of Dataset.from_generator
        # Create a dataset from the generator
        dataset = Dataset.from_generator(
            _generate_dataset,
            gen_kwargs={
                'embeddings': result.embeddings,
                'text': result.text,
                'metadata': result.metadata,
            },
        )

        # Write the dataset to disk
        dataset.save_to_disk(output_dir)

    def merge(self, dataset_dirs: list[Path], output_dir: Path) -> None:
        """Merge the datasets from multiple directories.

        Parameters
        ----------
        dataset_dirs : list[Path]
            The dataset directories to merge.
        output_dir : Path
            The output directory to write the merged dataset to.
        """
        # Load all the datasets
        all_datasets = [Dataset.load_from_disk(p) for p in dataset_dirs]

        # Concatenate the datasets
        dataset = concatenate_datasets(all_datasets)

        # Write the dataset to disk
        dataset.save_to_disk(output_dir, num_proc=self.config.num_proc)
