"""Hugging face dataset writer for saving embeddings to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from typing import Optional

from datasets import concatenate_datasets
from datasets import Dataset
from tqdm import tqdm

from distllm.utils import BaseConfig


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
        # Create a dataset
        dataset = Dataset.from_dict(
            mapping={
                'path': paths,
                'text': text,
                'response': responses,
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
        all_datasets = []
        for p in tqdm(dataset_dirs):
            # TODO: Debug why for some datasets, we have missing data
            try:
                dset = Dataset.load_from_disk(p)
            except FileNotFoundError:
                print(f'Skipping dataset {p} as it is missing.')
                continue
            all_datasets.append(dset)

        # Concatenate the datasets
        dataset = concatenate_datasets(all_datasets)

        # Write the dataset to disk
        dataset.save_to_disk(output_dir, num_proc=self.config.num_proc)
