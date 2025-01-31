"""Encoder for the ESM-2 model."""

from __future__ import annotations

import warnings
from typing import Literal

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig


class Esm2EncoderConfig(BaseConfig):
    """Config for the ESM-2 encoder."""

    # The name of the encoder
    name: Literal['esm2'] = 'esm2'  # type: ignore[assignment]
    # The model id
    pretrained_model_name_or_path: str = 'facebook/esm2_t6_8M_UR50D'
    # The model tokenizer (if different from pretrained_model_name_or_path)
    tokenizer_path: str | None = None
    # Use the model in half precision
    half_precision: bool = True
    # Set the model to evaluation mode
    eval_mode: bool = True
    # Compile the model for faster inference
    compile_model: bool = False
    # Use faesm implementation (faster)
    faesm: bool = False


class Esm2Encoder:
    """Encoder for the ESM-2 model."""

    def __init__(self, config: Esm2EncoderConfig):
        """Initialize the encoder."""
        import torch
        from transformers import EsmTokenizer

        # Check if faesm is enabled
        if config.faesm:
            try:
                from faesm.esm import FAEsmForMaskedLM as EsmForMaskedLM

                print('Using faesm implementation.')
            except ImportError:
                warnings.warn(
                    'faesm is not installed. Falling back to transformers.',
                    stacklevel=2,
                )
                from transformers import EsmForMaskedLM
        else:
            from transformers import EsmForMaskedLM

        # Load model and tokenizer
        model = EsmForMaskedLM.from_pretrained(
            config.pretrained_model_name_or_path,
        )
        if config.tokenizer_path is None:
            config.tokenizer_path = config.pretrained_model_name_or_path
        tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_path)

        # Set the model max length for proper truncation
        tokenizer.model_max_length = model.config.max_position_embeddings

        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load the model onto the device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Compile the model for faster inference
        if config.compile_model:
            model = torch.compile(model, fullgraph=True)

        # Set persistent attributes
        self.faesm = config.faesm
        self.model = model
        self._tokenizer = tokenizer

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
        outputs = self.model(
            **batch_encoding,
            output_hidden_states=not self.faesm,
        )

        # Return the last hidden state
        if self.faesm:
            return outputs['last_hidden_state']

        return outputs.hidden_states[-1]
