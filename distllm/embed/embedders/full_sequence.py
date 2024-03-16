"""Full sequence Embedder."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from distllm.embed import EmbedderResult
from distllm.embed import Encoder
from distllm.embed import Pooler
from distllm.utils import BaseConfig


@torch.no_grad()
def compute_embeddings(
    dataloader: DataLoader,
    encoder: Encoder,
    pooler: Pooler,
) -> np.ndarray:
    """Compute pooled hidden embeddings.

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
    np.ndarray
        A numpy array of pooled hidden embeddings.
    """
    # Get the number of embeddings and the embedding size
    num_embeddings = len(dataloader.dataset)

    # Initialize a torch tensor for storing embeddings in host memory
    all_embeddings = torch.empty(
        (num_embeddings, encoder.embedding_size),
        dtype=encoder.dtype,
    )

    # Index for storing embeddings
    idx = 0

    for batch in tqdm(dataloader):
        # Move the batch to the model device
        inputs = batch.to(encoder.device)

        # Get the model outputs with a forward pass
        embeddings = encoder.encode(inputs)

        # Compute the pooled embeddings
        pooled_embeds = pooler.pool(embeddings, inputs.attention_mask)

        # Get the batch size
        batch_size = inputs.attention_mask.shape[0]

        # Store the pooled embeddings in the output buffer
        all_embeddings[idx : idx + batch_size, :] = pooled_embeds.cpu()

        # Increment the output buffer index by the batch size
        idx += batch_size

    return all_embeddings.numpy()


class FullSequenceEmbedderConfig(BaseConfig):
    """Configuration for the full sequence embedder."""

    name: Literal['full_sequence'] = 'full_sequence'  # type: ignore[assignment]


class FullSequenceEmbedder:
    """Embedder for full sequence embeddings."""

    def __init__(self, config: FullSequenceEmbedderConfig) -> None:
        """Initialize the embedder with the configuration."""
        self.config = config

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
        embeddings = compute_embeddings(dataloader, encoder, pooler)

        # Return the result
        return EmbedderResult(
            embeddings=embeddings,
            text=dataloader.dataset.data,
            metadata=dataloader.dataset.metadata,
        )
