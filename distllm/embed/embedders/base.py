"""Embedder interface for all embedding methods to follow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol

import numpy as np
from torch.utils.data import DataLoader

from distllm.embed.encoders.base import Encoder
from distllm.embed.poolers.base import Pooler
from distllm.utils import BaseConfig


@dataclass
class EmbedderResult:
    """Embedder result dataclass."""

    # The pooled embeddings (shape: [num_sequences, embedding_size])
    embeddings: np.ndarray
    # The original text (shape: [num_sequences])
    text: list[str]
    # The optional metadata (shape: [num_sequences])
    metadata: list[dict[str, Any]] | None = None


class Embedder(Protocol):
    """Embedder protocol for all embedder to follow."""

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the pooler with the configuration."""
        ...

    def embed(
        self,
        dataloader: DataLoader,
        encoder: Encoder,
        pooler: Pooler,
    ) -> EmbedderResult:
        """Embed the sequences.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader to use for batching the data.
        encoder : Encoder
            The encoder to use for inference.
        pooler : Pooler
            The pooler to use for pooling the embeddings.

        Returns
        -------
        EmbedderResult
            Dataclass with the embeddings, text, and optional metadata.
        """
        ...
