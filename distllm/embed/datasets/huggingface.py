"""Hugging face dataset for reading HF dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from datasets import Dataset
from pydantic import Field
from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import DataCollator
from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.encoders.base import Encoder
from distllm.utils import BaseConfig


class HuggingFaceDatasetConfig(BaseConfig):
    """Configuration for the hugging face dataset."""

    name: Literal['huggingface'] = 'huggingface'  # type: ignore[assignment]

    # The name of the text field in the jsonl file
    text_field: str = 'text'
    # The name of the metadata fields in the jsonl file
    metadata_fields: list[str] = Field(default_factory=list)
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Whether to pin memory for the dataloader.
    pin_memory: bool = True


class HuggingFaceDataset:
    """Hugging face dataset for reading text from disk."""

    def __init__(self, config: HuggingFaceDatasetConfig) -> None:
        """Initialize the reader with the configuration."""
        self.config = config

    def get_dataloader(
        self,
        data_file: Path,
        encoder: Encoder,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset.

        Parameters
        ----------
        data_file : Path
            The file to read.
        encoder : Encoder
            The encoder instance.

        Returns
        -------
        DataLoader
            The dataloader instance.
        """
        # Load the dataset from disk
        dataset = Dataset.load_from_disk(data_file)

        # Load the text and paths from the dataset
        texts: list[str] = dataset[self.config.text_field]

        # Load the metadata if available
        metadatas = None
        if self.config.metadata_fields:
            # Load the metadata fields from the dataset
            columns = dataset.select_columns(self.config.metadata_fields)

            # Convert the metadata to a list of dictionaries
            metadatas = [dict(row) for row in columns]

        # Instantiate the dataloader
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(texts, metadatas),
            collate_fn=DataCollator(encoder.tokenizer),
        )
