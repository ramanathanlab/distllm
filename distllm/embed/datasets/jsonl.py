"""Single sequence per line file dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import DataCollator
from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.encoders.base import Encoder
from distllm.utils import BaseConfig


class JsonlDatasetConfig(BaseConfig):
    """Configuration for the JsonlDataset."""

    # The name of the dataset
    name: Literal['jsonl'] = 'jsonl'  # type: ignore[assignment]

    # The name of the text field in the jsonl file
    text_field: str = 'text'
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Whether to pin memory for the dataloader.
    pin_memory: bool = True


class JsonlDataset:
    """Jsonl file dataset."""

    def __init__(self, config: JsonlDatasetConfig):
        """Initialize the dataset."""
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
        # Read the jsonl file
        lines = data_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        # Extract the text data
        data = [item[self.config.text_field] for item in content]

        # Instantiate the dataloader
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(data),
            collate_fn=DataCollator(encoder.tokenizer),
        )
