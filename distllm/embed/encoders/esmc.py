"""Encoder for the ESM-Cambrian models."""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import model_validator
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer
from typing_extensions import Self

from distllm.utils import BaseConfig


class EsmCambrianEncoderConfig(BaseConfig):
    """Config for the ESM-Cambrian encoder."""

    # The name of the encoder
    name: Literal['esmc'] = 'esmc'  # type: ignore[assignment]
    # The model id, options [EvolutionaryScale/esmc-300m-2024-12,
    # EvolutionaryScale/esmc-600m-2024-12]
    pretrained_model_name_or_path: str = 'EvolutionaryScale/esmc-300m-2024-12'
    # The model embedding size (if you are using a fine-tuned model
    # you should explicitly set this value)
    embedding_size: int | None = None

    @model_validator(mode='after')
    def set_embedding_size(self) -> Self:
        """Set the embedding size based on the model name."""
        # If the embedding size is explicitly set, return the config
        if self.embedding_size is not None:
            return self

        # The embedding size based on the model name
        sizes = {
            'EvolutionaryScale/esmc-300m-2024-12': 960,
            'EvolutionaryScale/esmc-600m-2024-12': 1152,
        }

        # Get the embedding size
        embedding_size = sizes.get(self.pretrained_model_name_or_path, None)

        # If the embedding size is not found, raise an error
        if embedding_size is None:
            raise ValueError(
                f'Invalid model name for ESMC: '
                f'{self.pretrained_model_name_or_path} '
                f'Valid model names are: {", ".join(sizes.keys())}.',
                'Or you can set the embedding_size parameter explicitly ',
                'if you are using a fine-tuned model.',
            )

        # Set the embedding size
        self.embedding_size = embedding_size

        return self


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

        # Ensure the embedding size is set
        assert config.embedding_size is not None

        # Set persistent attributes
        self.model = model
        self._tokenizer = tokenizer
        self._embedding_size = config.embedding_size

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
