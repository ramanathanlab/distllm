"""Single sequence per line file dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import DataCollator
from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.encoders.base import Encoder
from distllm.utils import BaseConfig


class SequencePerLineDatasetConfig(BaseConfig):
    """Configuration for the SequencePerLineDataset."""

    # The name of the dataset
    name: Literal['sequence_per_line'] = 'sequence_per_line'  # type: ignore[assignment]

    # The number of header lines to skip
    header_lines: int = 1
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Whether to pin memory for the dataloader.
    pin_memory: bool = True


class SequencePerLineDataset:
    """Sequence per line file dataset."""

    def __init__(self, config: SequencePerLineDatasetConfig):
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
        # Read the file and skip the header lines
        data = data_file.read_text().splitlines()[self.config.header_lines :]

        # Instantiate the dataloader
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(data),
            collate_fn=DataCollator(encoder.tokenizer),
        )
