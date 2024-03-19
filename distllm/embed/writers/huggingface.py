"""Hugging face dataset writer for saving embeddings to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterator
from typing import Literal

import numpy as np
from torch.utils.data import Dataset

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


# TODO: Test this function. If we can't yield directly from the dataset, we
#       we can load each column entry one by one and yield the dictionary.
def _generate_merged_dataset(
    dataset_dirs: list[Path],
) -> Iterator[dict[str, str | np.ndarray | Any]]:
    """Generate a merged dataset from multiple directories."""
    for dataset_dir in dataset_dirs:
        yield from Dataset.load_from_disk(dataset_dir)


class HuggingFaceWriterConfig(BaseConfig):
    """Configuration for the hugging face writer."""

    name: Literal['huggingface'] = 'huggingface'  # type: ignore[assignment]


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
        # Create a dataset from the generator
        dataset = Dataset.from_generator(
            _generate_merged_dataset,
            gen_kwargs={'dataset_dirs': dataset_dirs},
        )

        # Write the dataset to disk
        dataset.save_to_disk(output_dir)
