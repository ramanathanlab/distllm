"""Last token pooler."""

from __future__ import annotations

from typing import Literal

import torch

from distllm.utils import BaseConfig


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the last hidden states using the attention mask.

    Parameters
    ----------
    last_hidden_states : torch.Tensor
        The last hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


class LastTokenPoolerConfig(BaseConfig):
    """Configuration for the LastTokenPooler."""

    name: Literal['last_token'] = 'last_token'  # type: ignore[assignment]


class LastTokenPooler:
    """Last Token Pooler.

    Pooler that uses the last token of the hidden states.
    """

    def __init__(self, config: LastTokenPoolerConfig) -> None:
        """Initialize the pooler with the configuration."""
        self.config = config

    def pool(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the last hidden states using the attention mask.

        Parameters
        ----------
        embeddings : torch.Tensor
            The last hidden states to pool (B, SeqLen, HiddenDim).
        attention_mask : torch.Tensor
            The attention mask for the hidden states (B, SeqLen).

        Returns
        -------
        torch.Tensor
            The pooled embeddings (B, HiddenDim).
        """
        return last_token_pool(embeddings, attention_mask)
