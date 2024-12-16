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
    """Encoder for the ESM-Cambrian model.

    For more information on the ESM-Cambrian model, see:
    https://www.evolutionaryscale.ai/blog/esm-cambrian
    """

    def __init__(self, config: EsmCambrianEncoderConfig):
        """Initialize the encoder."""
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer

        # Loads model and auto set device to cuda and dtype to bfloat16
        model = ESMC.from_pretrained(config.pretrained_model_name_or_path)

        # Set the model to evaluation mode
        model.eval()

        # Load the tokenizer
        tokenizer = EsmSequenceTokenizer()

        # Set the model max length for proper truncation
        tokenizer.model_max_length = 2048

        # Get the embedding size
        if config.pretrained_model_name_or_path == 'esmc_600m':
            embedding_size = 1152
        else:
            assert config.pretrained_model_name_or_path == 'esmc_300m'
            embedding_size = 960

        # Set persistent attributes
        self.model = model
        self._tokenizer = tokenizer
        self._embedding_size = embedding_size

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        # NOTE: The model is set to bfloat16 in the ESMC class
        # but we cast to float16 in the encode function to avoid
        # issues with casting in the calling code
        return torch.float16

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder."""
        return self._embedding_size

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self._tokenizer

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
        outputs = self.model(sequence_tokens=batch_encoding['input_ids'])

        # Return the last hidden state (cast from bfloat16 to float16)
        return outputs.embeddings.to(torch.float16)
