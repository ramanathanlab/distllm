"""Embedder interface for all embedder to inherit from."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from abc import abstractproperty
from typing import Literal

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer
from utils import BaseModel


class BaseEmbedderConfig(BaseModel, ABC):
    """Base config for all embedders."""

    # The name of the embedder
    name: Literal[''] = ''


class BaseEmbedder(ABC):
    """Base embedder class for all embedders to inherit from."""

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """Get the data type of the embedder."""
        ...

    @abstractproperty
    def device(self) -> torch.device:
        """Get the device of the embedder."""
        ...

    @abstractproperty
    def embedding_size(self) -> int:
        """Get the embedding size of the embedder."""
        ...

    @abstractproperty
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the embedder."""
        ...

    @abstractmethod
    def embed(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Embed the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence.

        Returns
        -------
        torch.Tensor
            The embeddings of the sequence
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        ...
