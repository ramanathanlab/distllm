"""Pooler interface for all pooling methods to follow."""

from __future__ import annotations

from typing import Protocol

import torch

from distllm.utils import BaseConfig


class Pooler(Protocol):
    """Pooler protocol for all poolers to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the pooler with the configuration."""
        ...

    def pool(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            The embeddings to pool.
            (shape: [num_sequences, sequence_length, embedding_size])

        attention_mask : torch.Tensor
            The attention mask.
            (shape: [num_sequences, sequence_length])

        Returns
        -------
        torch.Tensor
            The pooled embeddings.
            (shape: [num_sequences, embedding_size])
        """
        ...
