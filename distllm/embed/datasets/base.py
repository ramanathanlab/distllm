"""Dataset interface for all datasets to follow."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from torch.utils.data import DataLoader

from distllm.embed.encoders.base import Encoder
from distllm.utils import BaseConfig


class Dataset(Protocol):
    """Dataset protocol for all datasets to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the dataset with the configuration."""
        ...

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
            The encoder to use.

        Returns
        -------
        DataLoader
            The dataloader instance.
        """
        ...
