"""Encoder for the ESM-Cambrian models."""

from __future__ import annotations

from typing import Literal

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig


class EsmCambrianEncoderConfig(BaseConfig):
    """Config for the ESM-Cambrian encoder."""

    # The name of the encoder
    name: Literal['esmc'] = 'esmc'  # type: ignore[assignment]
    # The model id, options [esmc_330m, esmc_600m]
    pretrained_model_name_or_path: str = 'esmc_300m'


class EsmCambrianEncoder:
    """Encoder for the ESM-Cambrian model."""

    def __init__(self, config: EsmCambrianEncoderConfig):
        """Initialize the encoder."""
        from faesm.esmc import ESMC

        # Load model and auto set to device and dtype
        self.model = ESMC.from_pretrained(
            config.pretrained_model_name_or_path,
            use_flash_attn=True,
        )

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        return self.model.config.hidden_size

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self.model.tokenizer

    def encode(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Encode the sequence.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence (containing the input_ids,
            attention_mask, and token_type_ids).

        Returns
        -------
        torch.Tensor
            The embeddings of the sequence extracted from the last hidden state
            (shape: [num_sequences, sequence_length, embedding_size])
        """
        # Get the model outputs with a forward pass
        outputs = self.model(**batch_encoding)

        # Return the last hidden state
        return outputs.embeddings
