"""Dataset interface for all datasets to inherit from."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from torch.utils.data import DataLoader

from embedding_workflow.embedders import Embedder
from embedding_workflow.utils import BaseConfig


class Dataset(Protocol):
    """Dataset protocol for all datasets to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the dataset with the configuration."""
        ...

    def get_dataloader(
        self,
        data_file: Path,
        embedder: Embedder,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset.

        Parameters
        ----------
        data_file : Path
            The file to read.
        embedder : Embedder
            The embedder to use.

        Returns
        -------
        DataLoader
            The dataloader instance.
        """
        ...
