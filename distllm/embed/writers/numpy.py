"""Numpy writer for saving embeddings to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from distllm.embed.embedders.base import EmbedderResult
from distllm.utils import BaseConfig


class NumpyWriterConfig(BaseConfig):
    """Configuration for the numpy writer."""

    name: Literal['numpy'] = 'numpy'  # type: ignore[assignment]


class NumpyWriter:
    """Numpy writer for saving embeddings to disk."""

    def __init__(self, config: NumpyWriterConfig) -> None:
        """Initialize the writer with the configuration."""
        self.config = config

    def write(self, output_dir: Path, result: EmbedderResult) -> None:
        """Write the embeddings to disk.

        Parameters
        ----------
        result : EmbedderResult
            The result to write to disk.
        """
        np.save(output_dir / 'embeddings.npy', result.embeddings)
        np.save(output_dir / 'text.npy', result.text)
        if result.metadata is not None:
            np.save(
                output_dir / 'metadata.npy',
                result.metadata,
                allow_pickle=True,
            )

    def merge(self, dataset_dirs: list[Path], output_dir: Path) -> None:
        """Merge the datasets from multiple directories.

        Parameters
        ----------
        dataset_dirs : list[Path]
            The dataset directories to merge.
        output_dir : Path
            The output directory to write the merged dataset to.
        """
        # TODO: For now, we just concatenate the arrays. We should add a
        # to handle larger datasets, we could use NpyAppendArray.

        # Merge the embeddings
        embeddings = [np.load(p / 'embeddings.npy') for p in dataset_dirs]
        np.save(output_dir / 'embeddings.npy', np.concatenate(embeddings))

        # Merge the text
        text = [np.load(p / 'text.npy') for p in dataset_dirs]
        np.save(output_dir / 'text.npy', np.concatenate(text))

        # Merge the metadata
        paths = [p / 'metadata.npy' for p in dataset_dirs]
        if all(p.exists() for p in paths):
            metadata = [np.load(p, allow_pickle=True) for p in paths]
            np.save(output_dir / 'metadata.npy', np.concatenate(metadata))
