"""Embedder interface for all embedder to inherit from."""

from __future__ import annotations

from typing import Protocol

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig


class Embedder(Protocol):
    """Embedder protocol for all embedders to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the embedder with the configuration."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the embedder."""
        ...

    @property
    def device(self) -> torch.device:
        """Get the device of the embedder."""
        ...

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the embedder."""
        ...

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the embedder."""
        ...

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
