"""Encoder interface for all encoders to follow."""

from __future__ import annotations

from typing import Protocol

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig


class Encoder(Protocol):
    """Encoder protocol for all encoders to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the encoder with the configuration."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        ...

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        ...

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        ...

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        ...

    def encode(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Encode the sequence.

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
