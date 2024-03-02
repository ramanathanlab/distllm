"""Dataset interface for all datasets to inherit from."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader

from embedding_workflow.embedders import BaseEmbedder
from embedding_workflow.utils import BaseModel


class BaseDatasetConfig(BaseModel, ABC):
    """Base config for all datasets."""

    # The name of the dataset
    name: Literal[''] = ''


class BaseDataset(ABC):
    """Base dataset class for all datasets to inherit from."""

    def __init__(self, config: BaseDatasetConfig) -> None:
        """Initialize the dataset with the configuration."""
        self.config = config

    @abstractmethod
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
        embedder : BaseEmbedder
            The embedder to use.

        Returns
        -------
        DataLoader
            The dataloader instance.
        """
        ...
