"""Mean Pooler."""

from __future__ import annotations

from typing import Literal

import torch

from distllm.utils import BaseConfig


# TODO: We might want to configure whether to include the start and end tokens
def average_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Get the sequence lengths
    seq_lengths = attention_mask.sum(axis=1)

    # Set the attention mask to 0 for start and end tokens
    attention_mask[:, 0] = 0
    attention_mask[:, seq_lengths - 1] = 0

    # Create a mask for the pooling operation (B, SeqLen, HiddenDim)
    pool_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape)

    # Sum the embeddings over the sequence length (use the mask to avoid
    # pad, start, and stop tokens)
    sum_embeds = torch.sum(embeddings * pool_mask, 1)

    # Avoid division by zero for zero length sequences by clamping
    sum_mask = torch.clamp(pool_mask.sum(1), min=1e-9)

    # Compute mean pooled embeddings for each sequence
    return sum_embeds / sum_mask


class MeanPoolerConfig(BaseConfig):
    """Configuration for the MeanPooler."""

    name: Literal['mean'] = 'mean'  # type: ignore[assignment]


class MeanPooler:
    """Mean Pooler.

    Pooler that averages the hidden states using the attention mask.
    Does not include the pad, start, or end tokens.
    """

    def __init__(self, config: BaseConfig) -> None:
        """Initialize the pooler with the configuration."""
        self.config = config

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
        return average_pool(embeddings, attention_mask)
