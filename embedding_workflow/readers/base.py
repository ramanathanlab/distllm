"""Reader interface for all readers to inherit from."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Literal

from embedding_workflow.utils import BaseModel


class BaseReaderConfig(BaseModel, ABC):
    """Base config for all readers."""

    # The name of the Reader
    name: Literal[''] = ''


class BaseReader(ABC):
    """Base reader class for all readers to inherit from."""

    def __init__(self, config: BaseReaderConfig) -> None:
        """Initialize the reader with the configuration."""
        self.config = config

    @abstractmethod
    def read(self, data_file: Path) -> list[str]:
        """Read the data file.

        Parameters
        ----------
        data_file : Path
            The file to read.
        """
        ...
