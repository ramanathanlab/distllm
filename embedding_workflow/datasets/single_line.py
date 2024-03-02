"""Single sequence per line file dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader

from embedding_workflow.datasets.base import BaseDataset
from embedding_workflow.datasets.base import BaseDatasetConfig
from embedding_workflow.datasets.utils import DataCollator
from embedding_workflow.datasets.utils import InMemoryDataset
from embedding_workflow.embedders import BaseEmbedder


class SingleSequencePerLineDatasetConfig(BaseDatasetConfig):
    """Configuration for the SingleSequencePerLineDataset."""

    # The name of the dataset
    name: Literal['single_sequence_per_line'] = 'single_sequence_per_line'  # type: ignore[assignment]

    # The number of header lines to skip
    header_lines: int = 1
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Whether to pin memory for the dataloader.
    pin_memory: bool = True


class SingleSequencePerLineDataset(BaseDataset):
    """Single sequence per line file dataset."""

    config: SingleSequencePerLineDatasetConfig

    def get_dataloader(
        self,
        data_file: Path,
        embedder: BaseEmbedder,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset.

        Parameters
        ----------
        data_file : Path
            The file to read.

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
            collate_fn=DataCollator(embedder.tokenizer),
        )
